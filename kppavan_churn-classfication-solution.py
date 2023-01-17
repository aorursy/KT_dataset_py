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
#import the dataset

df=pd.read_csv("/kaggle/input/churn/churn.csv")
df.head()
df.dtypes
df.isna().any()
df.duplicated()
df.state.nunique()
import re

df['area_code']=df.area_code.str.extract('(\d+)')

df.area_code.astype(str).astype(int)
df['area_code']=df.area_code.astype(str).astype(int)
df.head()
import seaborn as sns
sns.countplot(x='international_plan',data=df,hue='churn')
pd.crosstab(df.international_plan,df.churn,normalize='index').round(4)*100
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

df['state']=lb_make.fit_transform(df['state'])

df['international_plan']=lb_make.fit_transform(df['international_plan'])

df['voice_mail_plan']=lb_make.fit_transform(df['voice_mail_plan'])

df['churn']=lb_make.fit_transform(df['churn'])



col=list(df.columns)

predictors=col[:-1]

result=col[-1]
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
d=sc.fit_transform(df[predictors])

new_df=pd.DataFrame(d)
new_df.head()
#import package for PCA

from sklearn.decomposition import PCA

pca=PCA()

pc=pca.fit_transform(new_df)
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(pc,df[result])
from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

model1=lg.fit(xtrain,ytrain)

pred1=model1.predict(xtest)
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

print(f1_score(ytest,pred1))

print(precision_score(ytest,pred1))

print(recall_score(ytest,pred1))

print(confusion_matrix(ytest,pred1))

print(np.mean(ytest==pred1))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

model2=knn.fit(xtrain,ytrain)

pred2=model2.predict(xtest)
print(f1_score(ytest,pred2))

print(precision_score(ytest,pred2))

print(recall_score(ytest,pred2))

print(confusion_matrix(ytest,pred2))

print(np.mean(ytest==pred2))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

model3=rf.fit(xtrain,ytrain)

pred3=model3.predict(xtest)
print(f1_score(ytest,pred3))

print(precision_score(ytest,pred3))

print(recall_score(ytest,pred3))

print(confusion_matrix(ytest,pred3))

print(np.mean(ytest==pred3))