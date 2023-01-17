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
df=pd.read_csv("../input/stock-sentiment-analysis/Stock_Dataa.csv",encoding="ISO-8859-1")
df.head()
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
train.head()
data=train.iloc[:,2:27]
data.replace('[^a-zA-Z]',' ',regex=True,inplace=True)
data.head()
a=[i for i in range(25)]
index=[str(i) for i in (a)]
data.columns=index
data.head()
for i in index:
    data[i]=data[i].str.lower()
data.head()
data.index
headlines=[]

for i in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[i,0:25]))
headlines[0]
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
countvector=CountVectorizer(ngram_range=(2,2))
traindata=countvector.fit_transform(headlines)
type(traindata)
random_for=RandomForestClassifier(n_estimators=200,criterion='entropy')
random_for.fit(traindata,train['Label'])
test_data=[]
for i in range(0,len(test.index)):
    test_data.append(' '.join(str(x) for x in test.iloc[i,2:27]))

test_data=countvector.transform(test_data)
test_pred=random_for.predict(test_data)
from sklearn.metrics import confusion_matrix
con=confusion_matrix(test_pred,test['Label'])
con
from sklearn.metrics import accuracy_score
acc=accuracy_score(test_pred,test['Label'])
acc*100
from sklearn.metrics import classification_report
rep=classification_report(test_pred,test['Label'])
print(rep)
sen=['The stock prices fall dramatic due to the pandemic situation']
ab=['The stock prices decrease due to wrap up of lockdown']
data=countvector.transform(ab)
pred=random_for.predict(data)
pred
