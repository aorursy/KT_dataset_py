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
df=pd.read_csv(r'/kaggle/input/phishing-data/combined_dataset.csv')
df.head()
df.shape
df.isnull().any()
df.describe()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(12,7))
df.boxplot()
plt.show()
df['nosOfSubdomain'].value_counts()
plt.figure(figsize=(10,6))
sns.catplot(x="nosOfSubdomain", y="domainLen", data=df)
plt.show()
plt.figure(figsize=(5,5))
sns.barplot(x="valid", y="domainLen", data=df)
plt.show()
sns.countplot(df.label)
from wordcloud import WordCloud
train_qs = pd.Series(df['domain'].tolist()).astype(str)
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(15, 12))
plt.imshow(cloud)
plt.axis('off')
plt.figure(figsize=(15,10))
sns.lineplot(y=df.urlLen,x=df.domainLen,data=df)
plt.show()
sns.countplot(df['isIp'],color="r")
cor=df.corr()
cor
sns.heatmap(cor)
x=df.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
y=df.iloc[:,[11]]
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
import xgboost
xgb=xgboost.XGBClassifier()
xgb.fit(x_train,y_train)
pred=xgb.predict(x_test)
pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
from sklearn import svm
s=svm.SVC()
s.fit(x_train,y_train)
pre=s.predict(x_test)
pre
print(accuracy_score(pre,y_test))
print(confusion_matrix(pre,y_test))
print(classification_report(pre,y_test))
