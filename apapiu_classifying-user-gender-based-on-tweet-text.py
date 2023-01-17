import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# we'll want this for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# we'll want this for text manipulation

import re



# for quick and dirty counting

from collections import defaultdict



# the Naive Bayes model

from sklearn.naive_bayes import MultinomialNB

# function to split the data for cross-validation

from sklearn.model_selection import train_test_split

# function for transforming documents into counts

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# function for encoding categories

from sklearn.preprocessing import LabelEncoder



# have to use latin1 even though it results in a lot of dead characters

twigen = pd.read_csv("../input/gender-classifier-DFE-791531.csv", encoding='latin1')

twigen.head()
def normalize_text(s):

    # just in case

    s = str(s)

    s = s.lower()

    

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W\s',' ',s)

    

    # make sure we didn't introduce any double spaces

    s = re.sub('\s+',' ',s)

    

    return s



twigen['text_norm'] = [normalize_text(s) for s in twigen['text']]

twigen['description_norm'] = [normalize_text(s) for s in twigen['description']]
twigen.shape


# how many observations are gold standard?

gold_values = defaultdict(int)

for val in twigen._golden:

    gold_values[val] += 1

print(gold_values)



# what does the confidence look like?

print(np.any(np.isnan(twigen['gender:confidence'])))

# we've got at least one NaN, so let's remove

gender_confidence = twigen['gender:confidence'][np.where(np.invert(np.isnan(twigen['gender:confidence'])))[0]]

print(len(gender_confidence))

gender_nonones = gender_confidence[np.where(gender_confidence < 1)[0]]

print(len(gender_nonones))
twigen_confident = twigen[twigen['gender:confidence']==1]

twigen_confident.shape
gender_counts= twigen_confident['gender'].value_counts()

gender_counts/sum(gender_counts)
# pull the data into vectors

vectorizer = TfidfVectorizer(min_df=3)

x = vectorizer.fit_transform(twigen_confident['text_norm'])



encoder = LabelEncoder()

y = encoder.fit_transform(twigen_confident['gender'])

encoder.classes_
x.shape
# split into train and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y,

                                                    test_size=0.2,

                                                    stratify = y,

                                                    random_state = 4)



# take a look at the shape of each of these

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
def eval_accuracy(model):

    model.fit(x_train, y_train)

    return model.score(x_test, y_test)
models = [LogisticRegression(),

          MultinomialNB(),

          RandomForestClassifier(n_estimators=50),

          KNeighborsClassifier()]
results = pd.Series([eval_accuracy(model) for model in models],

                    index = ["logit", "nb", "rf", "knn"])
results.plot(kind = "barh", title="Accuracy by Model")
vectorizer = CountVectorizer(min_df=5) #5 here to get only actual words.

x = vectorizer.fit_transform(twigen_confident['text_norm'])
model = LogisticRegression()

model.fit(x, y)
encoder.classes_
coeffs_male = pd.Series(model.coef_[2], index = vectorizer.get_feature_names())

coeffs_male.sort_values(ascending=False)[:10].plot.barh()
### Highest female coefficients:
coeffs_female = pd.Series(model.coef_[1], index = vectorizer.get_feature_names())

coeffs_female.sort_values(ascending=False).head(10).plot.barh()
(pd.Series(model.coef_[0], index = vectorizer.get_feature_names())

        .sort_values(ascending=False)

        .head(10)

        .plot.barh())