# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df['Average Score']=((df['math score']+df['reading score']+df['writing score'])/3).astype(int)
df.head()
sns.countplot(x='test preparation course',data=df,hue='parental level of education')
df=df.replace(['female','male'],[0,1])
df=df.replace(["bachelor's degree", 'some college', "master's degree",
       "associate's degree", 'high school', 'some high school'],[0,1,2,3,4,5])
df=df.replace(['group A','group B','group C','group D','group E'],[0,1,2,3,4])
df=df.replace(['standard','free/reduced'],[0,1])
df=df.replace(['none','completed'],[0,1])
df
sns.pairplot(df,palette='rainbow')
sns.barplot(x='test preparation course',y='Average Score',data=df,palette='spring')
sns.barplot(x='gender',y='Average Score',data=df,palette='spring')
sns.barplot(x='parental level of education',y='Average Score',data=df,hue='gender',palette='spring')
sns.distplot(df['Average Score'],bins=40)
plt.figure(figsize=(10,6))
(df[df['gender']==0]['Average Score']).hist(bins=40,label='Female',alpha=0.7,color='red')
(df[df['gender']==1]['Average Score']).hist(bins=40,label='Male',alpha=0.7,color='blue')
plt.legend()
#as you xcan see female has higher average score than male
#I want to find, given total score, can I find its gender
from sklearn.model_selection import  train_test_split
X=df.drop('Average Score',axis=1)
y=df['Average Score']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
predict=lm.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(y_test,predict))

print(r2_score(y_test,predict))

