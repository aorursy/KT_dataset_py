# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_file="/kaggle/input/train.csv"
df=pd.read_csv(input_file)

df.head()
df.isnull().sum()
df= df.drop(['location','keyword'],axis=1)
import seaborn as sns



sns.distplot(df['target']);
#to lowercase



def to_lowercase(row):

    return(row.lower())

df['text_lower']=df['text'].apply(to_lowercase)

df.head()
#Removing Punctuations using Regex

df['text_punc']=df['text_lower'].str.replace('[^\w\s]','')

df.head()

    
#tokenising

def tokensie(row):

    _row=row.split()

    return _row



df['text_tokens']=df['text_punc'].apply(tokensie)



df.head()
#Removing the stop words



import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 



def remove_stopwords(row):

    _row=[x for x in row if x not in stop_words]

    return _row



df['text_stopwords']= df['text_tokens'].apply(remove_stopwords)



df.head()
#Stemming the words



from nltk.stem import PorterStemmer

ps=PorterStemmer()



def stemming(row):

    _row=[ps.stem(x) for x in row]

    return _row



df['text_stemmed']=df['text_stopwords'].apply(stemming)

df.head()
def detokenise(row):

    _row=" ".join([x for x in row])

    return _row



df['ready_text']=df['text_stemmed'].apply(detokenise)

df.head()
from sklearn.model_selection import train_test_split

x=df['ready_text']

y= df['target']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=43)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

vectorizer.fit(df['ready_text'])

x_train_vec = vectorizer.transform(x_train)

x_test_vec=vectorizer.transform(x_test)
from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(x_train_vec.todense(),y_train)
from sklearn.naive_bayes import MultinomialNB

model1= MultinomialNB(alpha=1.0, fit_prior=True)

model1.fit(x_train_vec.todense(),y_train)
from sklearn.naive_bayes import BernoulliNB

model2= BernoulliNB()

model2.fit(x_train_vec.todense(),y_train)
#Extension of MultionomialNB



from sklearn.naive_bayes import ComplementNB

model3= ComplementNB()

model3.fit(x_train_vec.todense(),y_train)

import numpy as np

y_pred= model2.predict((x_test_vec).todense())

y_pred
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(f1_score(y_test,y_pred))

print(recall_score(y_test,y_pred))

print(precision_score(y_test,y_pred))
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_test,y_pred)

roc_auc = metrics.auc(fpr, tpr)

print(plt.xlabel('False Positive Rate')

,fpr)

print('True Positive Rate',tpr)

print('Threshold',threshold)

print('roc_auc',roc_auc)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))