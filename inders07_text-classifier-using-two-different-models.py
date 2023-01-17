# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # graphs and plotting



# import various required models and modules from scikitlearn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn import metrics







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
# load data

df = pd.read_excel('../input/PERC_mendelly.xlsx')

df.head()
# extract content and type columns only

df = df.drop_duplicates()

df.describe(include='all')

# check for null values

df.isnull().any().describe()
# take a insight of data

df['Poem'][0]
# remove unneccesary new line characters and extra spaces



import re



poems = []

for poem in df.Poem:

    poem = poem.replace('\n',' ')

    poem = poem.replace('\n\n',' ')

    poem = poem.replace('\n\'',' ')

    poem = poem.replace('\'','')

    poem = re.sub(' +',' ',poem)

    poems.append(poem)

df["Poem"] = poems

df.head()
#data after preprocessing

df.Poem[0]


df.groupby('Emotion').Emotion.count().plot.bar()

plt.show()
# extract top 4 most contrasting emotion categories

df = df.loc[df.Emotion.isin(['anger','love','joy','courage'])]

df.groupby('Emotion').Poem.count()
# assign an unique integer category to each emotion category 

df['category_id'] = df['Emotion'].factorize()[0]

emotion_id_df = df[['Emotion','category_id']].drop_duplicates().sort_values('category_id')

emotion_to_id = dict(emotion_id_df.values)

id_to_emotion = dict(emotion_id_df[['category_id','Emotion']].values)

df.head()    
# extract tf-idf features from the data

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',

                     ngram_range = (1,2), stop_words='english')

features = tfidf.fit_transform(df.Poem).toarray()

labels = df.category_id

features.shape
# find most corelated terms to each category

N = 3

for typ, category_id in sorted(emotion_to_id.items()):

    features_chi2 = chi2(features, labels == category_id)

    indices = np.argsort(features_chi2[0])

    feature_names = np.array(tfidf.get_feature_names())[indices]

    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    print('# {}:'.format(typ))

    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))



# split data into training and test set

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 7,

                                                    test_size = 0.2)

# fit the training data into a multinomial naive bayes classifier

model1 = MultinomialNB(alpha=0.5)



model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)



# cofusion matrix

conf_mat = confusion_matrix(y_test, y_pred1)

sns.heatmap(conf_mat, xticklabels=emotion_id_df.Emotion.values, 

            yticklabels=emotion_id_df.Emotion.values,annot = True)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
#training accuracy of classifier 1

model1.score(X_train,y_train)
#test accuracy of classifier 1

model1.score(X_test,y_test)
# final evlauation of classifier 1

print(metrics.classification_report(y_test, y_pred1, 

                                    target_names=df['Emotion'].unique()))

# fit the training data into a linear support vector classifier

model2 = LinearSVC(C = 0.1)



model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)



# confusion matrix for model 2

conf_mat = confusion_matrix(y_test, y_pred2)

sns.heatmap(conf_mat, xticklabels=emotion_id_df.Emotion.values, 

            yticklabels=emotion_id_df.Emotion.values,annot = True)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

# training accuracy of classifier 2

model2.score(X_train, y_train)
# test accuracy of classifier 2

model2.score(X_test, y_test)
# final evaluation of classifier 2

print(metrics.classification_report(y_test, y_pred2, 

                                    target_names=df['Emotion'].unique()))
