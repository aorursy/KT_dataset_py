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
!pip install nltp
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltp import Preprocessor
df = pd.read_csv('../input/fraud-email-dataset/fraud_email_.csv')

df.head()
df.rename(columns={'Text':'Emails'}, inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df['Class'].value_counts()
df.shape
#VISUALIZING OUR TARGET VALUES

sns.countplot(df['Class'])

plt.title("Plot of Target Variable")

plt.show()
pre = Preprocessor(df['Emails']).text_cleaner()
pre[2]
words = pre

plt.figure(figsize = (15,15))

word_cloud  = WordCloud(max_words = 1000 , width = 1600 , height = 800,

               collocations=False).generate(" ".join(words))

plt.imshow(word_cloud,interpolation='bilinear')

plt.axis('off')

plt.show()
X = pre

y = df['Class']
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import BernoulliNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print(f'Spliting Completed')

print(f'X Train: {len(X_train)} X Test: {len(X_test)} y Train: {len(y_train)} y Test: {len(y_test)}')
#creating a function that can be used to train and test our models

def fit_predict(model):   

    #CREATING A PIPELINE TO PROCESSING THE REVIEWS INTO O's AND 1's WITH Tf idf VECTORIZER

    clf = Pipeline([('tfidf',TfidfVectorizer()),

                   ('clf',model)])



    #training model

    clf.fit(X_train, y_train)

    print(f'Fitting Model Completed.')

    

    #USING THE TEST DATA TO EVALUATED THE MODEL CREATED

    Score = clf.score(X_test,y_test)

    print(f'Accuracy: {Score*100}') 

    

    return clf

    
class Metrics():

    

    def __init__(self, pred):

        self.pred = pred

        

    def cm(self):

        cm = confusion_matrix(y_test, self.pred)

        labels = ['Not Spam','Spam']



        f, ax = plt.subplots(figsize=(5,5))

        sns.heatmap(cm,annot =True, linewidth=.6, linecolor="r", fmt=".0f", ax = ax)



        ax.set_xticklabels(labels)

        ax.set_yticklabels(labels)

        plt.show()



    def report(self):

        class_report = classification_report(y_test, self.pred)

        print(class_report)

  
LR_model = fit_predict(LogisticRegression())



LR_pred = LR_model.predict(X_test)
Metrics(LR_pred).cm()



Metrics(LR_pred).report()
SVC_model = fit_predict(LinearSVC())



SVC_pred = SVC_model.predict(X_test)
Metrics(SVC_pred).cm()



Metrics(SVC_pred).report()
NB_model = fit_predict(BernoulliNB())



NB_pred = NB_model.predict(X_test)




Metrics(NB_pred).cm()



Metrics(NB_pred).report()
import joblib



filename = 'model.joblib'

joblib.dump(LR_model,open(filename,'wb'))
with open('model.joblib','rb') as f:

    model = joblib.load(f)
model