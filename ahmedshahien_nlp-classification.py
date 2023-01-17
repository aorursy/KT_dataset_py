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
import numpy as np 

import pandas as pd 



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# XGBoost

import xgboost as xgb

from xgboost import XGBClassifier



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("../input/deepnlp/Sheet_1.csv",encoding='latin-1')

train.shape

train=pd.DataFrame(train)

train.head()
train.info()
train=train.drop(['response_id','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7'],axis=1)

train.head()
test = pd.read_csv("../input/deepnlp/Sheet_2.csv",encoding='latin-1')

test=test.drop(['resume_id'],axis=1)

test.shape

test.head()
train['class'].value_counts()
sns.barplot(train['class'].value_counts().index,train['class'].value_counts(),palette='rocket')
# Applying a first round of text cleaning techniques



def clean_text(response_text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    response_text = response_text.lower()

    response_text = re.sub('\[.*?\]', '', response_text)

    response_text = re.sub('https?://\S+|www\.\S+', '', response_text)

    response_text = re.sub('<.*?>+', '', response_text)

    response_text = re.sub('[%s]' % re.escape(string.punctuation), '', response_text)

    response_text = re.sub('\n', '', response_text)

    response_text = re.sub('\w*\d\w*', '', response_text)

     

    return response_text



# Applying the cleaning function to both test and training datasets

train['response_text'] = train['response_text'].apply(lambda x: clean_text(x))

#test['response_text'] = test['response_text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train.head()
# Applying a first round of text cleaning techniques



def clean_text(resume_text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    resume_text = resume_text.lower()

    resume_text = re.sub('\[.*?\]', '', resume_text)

    resume_text = re.sub('https?://\S+|www\.\S+', '', resume_text)

    resume_text = re.sub('<.*?>+', '', resume_text)

    resume_text = re.sub('[%s]' % re.escape(string.punctuation), '', resume_text)

    resume_text = re.sub('\n', '', resume_text)

    resume_text = re.sub('\w*\d\w*', '', resume_text)

    

    return resume_text



# Applying the cleaning function to   test 

 

test['resume_text'] = test['resume_text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

test.head()
flagged_values = train[train['class']=='flagged']['response_text']

not_flagged_values=train[train['class']=='not_flagged']['response_text']
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(flagged_values))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('flagged resonse',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(not_flagged_values))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non flagged response',fontsize=40);
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



le.fit(train['class'])

#le.classes_

train['class']=le.transform(train['class']) 

train.head()

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



le.fit(test['class'])

#le.classes_

test['class']=le.transform(test['class']) 

test.head()

X=train['response_text']

y=train['class']
VecModel = TfidfVectorizer()

X = VecModel.fit_transform(X)



print(f'The new shape for X is {X.shape}')
X
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=402)
#Applying LogisticRegression Model 



'''

linear_model.LogisticRegression(penalty='l2’,dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,

                                class_weight=None,random_state=None,solver='warn’,max_iter=100,

                                multi_class='warn’, verbose=0,warm_start=False, n_jobs=None)

'''



LogisticRegressionModel = LogisticRegression(penalty='l2',tol=0.00001,solver='sag',C=1.0,max_iter=10000,random_state=33)

LogisticRegressionModel.fit(X_train, y_train)



#Calculating Details

print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))

print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))

 
from sklearn.model_selection import cross_val_score

#Applying Cross Validate Score :  

'''

model_selection.cross_val_score(estimator,X,y=None,groups=None,scoring=None,cv=’warn’,n_jobs=None,verbose=0,

                                fit_params=None,pre_dispatch=‘2*n_jobs’,error_score=’raise-deprecating’)

'''



#  don't forget to define the model first !!!

CrossValidateScoreTrain = cross_val_score(LogisticRegressionModel, X_train, y_train, cv=3)

CrossValidateScoreTest = cross_val_score(LogisticRegressionModel, X_test, y_test, cv=3)



# Showing Results

print('Cross Validate Score for Training Set: \n', CrossValidateScoreTrain)

print('Cross Validate Score for Testing Set: \n', CrossValidateScoreTest)