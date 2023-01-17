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
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
df.head()
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.columns=['labels','data']
df.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labels=le.fit_transform(df['labels'])
df['labels']=labels
df.head()
y=labels
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from wordcloud import WordCloud
cv=CountVectorizer(decode_error='ignore')
X=cv.fit_transform(df['data'])
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.33)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,y_train)
print('Train Accuracy:',model.score(X_train,y_train))
print('Test Accuracy:',model.score(X_test,y_test))
import matplotlib.pyplot as plt
def visualize(label):
    words=''
    for msg in df[df['labels']==label]['data']:
        msg=msg.lower()
        words+=msg+' '
    wordcloud=WordCloud(width=600,height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
visualize(0)
df['predictions']=model.predict(X)
sneaky_spam=df[(df['predictions']==0) & (df['labels']==1)]['data']
for msg in sneaky_spam:
    print(msg)
not_actually_spam=df[(df['predictions']==1) & (df['labels']==0)]['data']
for msg in sneaky_spam:
    print(msg)