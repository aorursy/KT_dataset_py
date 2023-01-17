# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
for i in train.columns:

    print(train[i].value_counts())
train.describe()
train.info()
train.shape
train.location.fillna('Not Available', inplace = True)

train.info()
def handle(text, pattern):

    return ' '.join(text.split(pattern))
train.keyword.fillna('Not Available', inplace = True)

train.keyword = train.keyword.apply(lambda x : handle(x, '%20'))
train.keyword.value_counts()
train.drop('id', inplace = True, axis = 1)
train.head()
from nltk.corpus import stopwords 

from nltk.tokenize import WordPunctTokenizer

from string import punctuation

from nltk.stem import WordNetLemmatizer

import regex



wordnet_lemmatizer = WordNetLemmatizer()



stop = stopwords.words('english')



for punct in punctuation:

    stop.append(punct)



def text_process(text, stop_words):

    word_tokens = WordPunctTokenizer().tokenize(text.lower())

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha()]

    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words]

    return " ".join(filtered_text)

    
train.text = train.text.apply(lambda x : text_process(x, stop))
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(lowercase=False)

ml_data = tfidf.fit_transform(train.iloc[:, 2]).toarray()
ml_data.shape
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



train.keyword = le.fit_transform(train.keyword)

train.location = le.fit_transform(train.location)
list(ml_data)
train
from sklearn.model_selection import RandomizedSearchCV, train_test_split



train_x, test_x, train_y, test_y = train_test_split(ml_data,train.target, stratify=train.target, test_size=0.2, random_state = 1)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix



estimator = LogisticRegression(

    random_state = 1,

    penalty = 'l2'

)

estimator.fit(train_x, train_y)
pred = estimator.predict(test_x)

accuracy_score(pred, test_y)
max_iter = range(100, 500)

solver = ['lbfgs', 'newton-cg', 'liblinear']

warm_start = [True, False]

C = np.arange(0, 1, 0.01)

random_grid ={

    'max_iter' : max_iter,

    'warm_start' : warm_start,

    'solver' : solver,

    'C' : C,

}
random_estimator = RandomizedSearchCV(estimator = estimator,

                                   param_distributions = random_grid,

                                   n_iter = 100,

                                   scoring = 'accuracy',

                                   n_jobs = -1,

                                   verbose = 1, 

                                   random_state = 1,

                                  )



random_estimator.fit(train_x, train_y)
random_estimator.best_params_



best_estimator = random_estimator.best_estimator_



best_estimator.fit(train_x, train_y)



pred = best_estimator.predict(test_x)



accuracy_score(pred, test_y)
import seaborn as sns

import matplotlib.pyplot as plt



predicted = estimator.predict(test_x)

print("Test score: {:.2f}".format(accuracy_score(test_y,predicted)))

print("Cohen Kappa score: {:.2f}".format(cohen_kappa_score(test_y,predicted)))

plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(test_y,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['True', 'False']),

            yticklabels=(['True', 'False']))

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head()
Id = test.id

test.drop('id', inplace = True, axis = 1)
test.text = test.text.apply(lambda x : text_process(x, stop))
ml_test = tfidf.transform(test.iloc[:, 2]).toarray()

ml_test.shape
estimator.fit(ml_data, train.target)

pred = estimator.predict(ml_test)
dic = {

    'id' : Id,

    'target' : pred



}

out = pd.DataFrame(dic)

out.to_csv('submissions.csv', index = False)