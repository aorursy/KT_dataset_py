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
df=pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
df
df.columns
df=df.drop(['id'],axis=1)
df['target'].unique()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['target']=le.fit_transform(df['target'])
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(12,5))

sns.countplot(df['target'])
#here we can observe that classes 1,5, and 7 dominates
#now prepare the model for the training and testing

y=df['target']

x=df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

list_models=[]

list_scores=[]

lr=LogisticRegression(max_iter=100000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)

list_models.append('logistic regression')

list_scores.append(score_1)
fig,axes=plt.subplots(1,2)

fig.set_size_inches(11.7, 8.27)

sns.countplot(pred_1,ax=axes[0])

sns.countplot(y_test,ax=axes[1])
#from the above plots we can point out that the model mainly predicted the dominant classes i.e 1,7 and 5
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)

list_scores.append(score_2)

list_models.append('random forest classifier')
score_2
fig,axes=plt.subplots(1,2)

fig.set_size_inches(11.7, 8.27)

sns.countplot(pred_2,ax=axes[0])

sns.countplot(y_test,ax=axes[1])
#lets create a comparison bw predictions of logistic regression model and random forest model

fig,axes=plt.subplots(1,2)

fig.set_size_inches(11.7, 8.27)

sns.countplot(pred_1,ax=axes[0])

axes[0].legend(title='predictions by logistic regression')

sns.countplot(pred_2,ax=axes[1])

axes[1].legend(title='predictions by random forest')
#from above observations we can conclude that the only major difference bw those predictions  is count of 1 class is fewer and count of class 2 is higer in random forest as compare to  predictions in lagistic regression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('support vector machines')
score_3
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(x_test)

score_4=accuracy_score(y_test,pred_4)

list_models.append('xgboost classifier')

list_scores.append(score_4)
score_4
plt.figure(figsize=(12,5))

plt.bar(list_models,list_scores,width=0.3)

plt.xlabel('classifictions models')

plt.ylabel('accuracy scores')

plt.show()