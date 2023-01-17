# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pyspellchecker
import numpy as np

import pandas as pd

import tensorflow as tf

import nltk

import keras

import os

import seaborn as sns

import matplotlib.pyplot as plt

from nltk import word_tokenize

from nltk import clean_html

import re

import string

from nltk.corpus import stopwords

from bs4 import BeautifulSoup

from spellchecker import SpellChecker

from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
dataTrain = pd.read_csv('../input/nlp-getting-started/train.csv') 
dataTrain.head()
dataTrain.info()
for col in dataTrain.columns:

    print(col+ ' - ' + format(dataTrain[col].isnull().sum()))
dataTrain.drop(['id','keyword','location'],inplace=True,axis=1)

dataTrain.head()


#data
def remove_html(w):

    soup = BeautifulSoup(w)

    text = soup.get_text()

    return w
def remove_url(text):

    # remove urls

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)
def cleanData(data):

    # remove urls

    data['text'] = data['text'].apply(lambda x:remove_url(x))

    print('urls removed')

    

    # remove emojis

    data['text'] = data['text'].apply(lambda x:remove_emoji(x))

    print('emojis removed')

    # correct spellings

    # commenting this out because this takes very long

    #data['text'] = data['text'].apply(lambda x:correct_spellings(x))

    

    # tokenizing words

    data['text'] = data['text'].apply(lambda x:word_tokenize(x))

    print('tokenization done')

    

    # convert all text to lowercase

    data['text'] = data['text'].apply(lambda x:[w.lower() for w in x ])

    print('lowercase done')

    

    # remove html tags

    data['text'] = data['text'].apply(lambda x:[remove_html(w) for w in x])

    print('html tags removed')

    

    # prepare regex for char filtering

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    

    

    # removing puncutations

    data['text'] = data['text'].apply(lambda x:[re_punc.sub('',w) for w in x])

    print('punctuations removed')

    

    # removing non alphabetic words 

    data['text'] = data['text'].apply(lambda x:[w for w in x if w.isalpha()])

    print('numeric removed')

    

    # removing stopwords

    data['text'] = data['text'].apply(lambda x:[w for w in x if w not in stopwords.words('english')])

    print('stopwords removed')

    

    # removing short words

    data['text'] = data['text'].apply(lambda x:[w for w in x if len(w)>2])

    print('shortwords removed')

    

    return data
backup = dataTrain.copy()
dataTrain = backup
#data = pd.DataFrame(['How 123 are you doing Today major qr ?is this corect','fine thank you <b>colonel</b>! https://www.kaggle.com/c/nlp-getting-started'],columns=['text'])

data = cleanData(dataTrain)
dataTrain.head()
#dataTrain.drop(['keyword','location'],inplace=True,axis=1)

#dataTrain.head()
#dataTrain.drop(['id'],inplace=True,axis=1)

#dataTrain.head()
backup = dataTrain.copy()
dataTrain = backup

dataTrain.head()
lem = WordNetLemmatizer()

dataTrain['text'] = dataTrain['text'].apply(lambda x:[lem.lemmatize(w) for w in x])
# join text

dataTrain['text'] = dataTrain['text'].apply(lambda x:' '.join(x))
dataTrain.sample(5)
X = dataTrain['text']

Y = dataTrain['target']
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)
X = X.toarray()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.15)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import r2_score

from sklearn.metrics import precision_score,recall_score

from sklearn.metrics import f1_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold, learning_curve
conMatList = []

prcList = []

clRep= []

rocDet = []

preScore = []

recScore = []

f1Score = []

yPred = []



def getClassModel(model):

    model = model()

    model_name = model.__class__.__name__

    model.fit(x_train,y_train)

    

    #getting prediction

    y_pred = model.predict(x_test)

    yPred.append([model_name,y_pred])

    

    # getting scores

    

    pre_score = precision_score(y_test,y_pred)

    rec_score= recall_score(y_test,y_pred)

    f1score = f1_score(y_test,y_pred)

    

    preScore.append([model_name,pre_score])

    recScore.append([model_name,rec_score])

    f1Score.append([model_name,f1score])

    

    ## getting confusion matrix

    cm = confusion_matrix(y_test,y_pred)

    matrix = pd.DataFrame(cm,columns=['predicted 0','predicted 1'],

                         index=['Actual 0','Actual 1'])

    conMatList.append([model_name,matrix])

    

     ## getting precision recall curve values

    

    precision, recall, thresholds = precision_recall_curve(y_test,y_pred)

    prcList.append([model_name,precision,recall,thresholds])

    

    ## roc details

    

    fpr,tpr,thresholds = roc_curve(y_test,y_pred)

    rocDet.append([model_name,fpr,tpr,thresholds])

    

    ## classification report

    

    classRep = classification_report(y_test,y_pred)

    clRep.append([model_name,classRep])
from sklearn.naive_bayes import MultinomialNB
kfold = StratifiedKFold(n_splits=10)

#classModelList = [LogisticRegression,SVC,GaussianNB,DecisionTreeClassifier

#                 ,RandomForestClassifier,KNeighborsClassifier]

classModelList = [MultinomialNB,LogisticRegression,GaussianNB]

i = 0

for model in classModelList:

    

    getClassModel(model)

    print(i)

    i = i+1
#getting cross validation scores for each model

cv_results = []

for model in classModelList:

    cv_results.append(cross_val_score(model(),x_train,y_train,scoring='accuracy',

                                     cv=kfold,n_jobs=4))

cv_means = []

cv_std = []



for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

model_name = []

for model in classModelList:

    modelIns = model()

    model_name.append(modelIns.__class__.__name__)

    

cv_res = pd.DataFrame({

    "CrossValMeans":cv_means,

    "CrossValErrors":cv_std,

    "Model":model_name

})

  

cv_res
for mat in conMatList:

    print(mat[0])

    print(' ')

    print(mat[1])

    print('-----------------------------------------------')
precisionDf = pd.DataFrame(preScore,columns=['model','precisionScore'])

recallDf = pd.DataFrame(recScore,columns=['model','recallScore'])

f1Df = pd.DataFrame(f1Score,columns=['model','f1Score'])

precisionDf['f1Score'] = f1Df['f1Score']

precisionDf['recallScore'] = recallDf['recallScore']

precisionDf
for roc in rocDet:

    print(roc[0])

    fpr = roc[1]

    tpr = roc[2]

    plt.plot(fpr,tpr,label=roc[0])

    plt.legend()
for prc in prcList:

    precision = prc[1]

    recall = prc[2]

    plt.plot(precision,recall,label=prc[0])

    plt.legend()
logreg = LogisticRegression()

logreg.fit(x_train,y_train)
import pickle

pkl_Filename = "regModel"



with open(pkl_Filename, 'wb') as file:

    pickle.dump(logreg,file)


from IPython.display import FileLink

FileLink('regModel.pkl')
#cheking if model saved works

with open(pkl_Filename, 'rb') as file: 

    print(file)

    Pickled_LR_Model = pickle.load(file)
ty = x_train[2]
y_pred = Pickled_LR_Model.predict(np.reshape(ty,(1,ty.shape[0])))

y_pred
y_train[:3]
dataTrial = pd.DataFrame(['several casualties as death result in millions'],columns = ['text'])

dataTrial = cleanData(dataTrial)
dataTrial['text'] = dataTrial['text'].apply(lambda x:[lem.lemmatize(w) for w in x])

dataTrial['text'] = dataTrial['text'].apply(lambda x:' '.join(x))
Xtest = dataTrial['text']

Xtest = tfidf.transform(Xtest)
Xtest = Xtest.toarray()
Xtest.shape
y_pred = Pickled_LR_Model.predict(Xtest)

y_pred
ty.shape[0]
ty = np.reshape(ty,(1,ty.shape[0]))
ty.shape