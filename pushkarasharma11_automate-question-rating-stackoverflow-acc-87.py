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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')
df.head()
df.shape
df.columns
df = df.drop(['Id','Tags','CreationDate'],axis=1)

df.head()
df['Num_words_body'] = df['Body'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

df['Num_words_title'] = df['Title'].apply(lambda x:len(str(x).split())) #Number Of words in main text

df['Total_words'] = abs(df['Num_words_body'] + df['Num_words_title']) #Total  Number of words text and Selected Text

plt.figure(figsize=(12,6))

p = sns.kdeplot(df['Num_words_body'],shade=True).set_title('Distribution of Body text')

p = sns.kdeplot(df['Num_words_title'],shade=True).set_title('Distribution of Body text')

plt.xlim(0,300)
plt.figure(figsize=(12,6))

p1=sns.kdeplot(df[df['Y']=='HQ']['Total_words'], shade=True,).set_title('Distribution of Total No.Of words Per Category')

p2=sns.kdeplot(df[df['Y']=='LQ_CLOSE']['Total_words'], shade=True)

p2=sns.kdeplot(df[df['Y']=='LQ_EDIT']['Total_words'], shade=True)

plt.legend(labels=['HQ','LQ_CLOSE','LQ_EDIT'])

plt.xlim(-20,500)
df['Y'] = df['Y'].map({'LQ_CLOSE':0,'LQ_EDIT':1,'HQ':2})

df.head()
df.isnull().sum()
values = [len(df[df['Y']==0]),len(df[df['Y']==1]),len(df[df['Y']==2])]

plt.bar(['LQ_CLOSE','LQ_EDIT','HQ'],values)

plt.show()
df['All_text'] = df['Title']+' '+df['Body']

new_df = df.copy()

new_df = new_df.drop(['Title','Body'],axis=1)

new_df.head()
from nltk.corpus import stopwords

import re
stop_words = stopwords.words('english')
def data_cleaning(data):

    data = data.lower()

    data = re.sub(r'[^(a-zA-Z)\s]','',data)

    data = data.split()

    temp = []

    for i in data:

        if i not in stop_words:

            temp.append(i)

    data = ' '.join(temp)

    return data
new_df['All_text'] = new_df['All_text'].apply(data_cleaning)
new_df['All_text']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(new_df['All_text'],new_df['Y'],test_size=0.20)
print(x_train.size,x_test.size,y_train.size,y_test.size)
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

x_train = vec.fit_transform(x_train)

x_test = vec.transform(x_test)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,plot_confusion_matrix

predictions = xgb.predict(x_test)

acc = accuracy_score(predictions,y_test)
acc
plot_confusion_matrix(xgb,x_test,y_test)