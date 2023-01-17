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
import pandas as pd

import matplotlib.pyplot as plt

import re

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
agegroup=pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

covid_19_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospital_beds=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details=pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
agegroup.head()
hospital_beds=hospital_beds[:-2]

hospital_beds.fillna(0,inplace=True)

for col in hospital_beds.columns[2:]:

    if hospital_beds[col].dtype=='object':

        hospital_beds[col]=hospital_beds[col].astype('int64')
covid_19_india['Date']=pd.to_datetime(covid_19_india['Date'])

covid_19_india['ConfirmedForeignNational'].replace('-',0,inplace=True)

covid_19_india['ConfirmedIndianNational'].replace('-',0,inplace=True)

covid_19_india['ConfirmedIndianNational']=covid_19_india['ConfirmedIndianNational'].astype('int64')

covid_19_india['ConfirmedForeignNational']=covid_19_india['ConfirmedForeignNational'].astype('int64')

lbl=LabelEncoder()

covid_19_india['State/UnionTerritory']=lbl.fit_transform(covid_19_india['State/UnionTerritory'])

covid_19_india['date']=covid_19_india['Date'].dt.day

covid_19_india['month']=covid_19_india['Date'].dt.month
tree=DecisionTreeRegressor()

linear=LinearRegression()

logistic=LogisticRegression()

nb=GaussianNB()

forest=RandomForestClassifier()
x=covid_19_india[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]

y=covid_19_india['Confirmed']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
tree.fit(x_train,y_train)

linear.fit(x_train,y_train)

logistic.fit(x_train,y_train)

nb.fit(x_train,y_train)

forest.fit(x_train,y_train)
from sklearn.metrics import r2_score

prediction=tree.predict(x_test)

score1=r2_score(y_test,prediction)
prediction=logistic.predict(x_test)

score2=r2_score(y_test,prediction)
prediction=linear.predict(x_test)

score3=r2_score(y_test,prediction)
prediction=forest.predict(x_test)

score4=r2_score(y_test,prediction)
prediction=nb.predict(x_test)

score5=r2_score(y_test,prediction)
scores=[score1,score2,score3,score4,score5]

models=['DecisionTreeRegressor','LogisticRegression','LinearRegression','RandomForestClassifier','GaussianNB']

plt.figure(figsize=(20,10))

plt.title('Comparing Accuracy of different models',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('models',fontsize=30)

plt.ylabel('Accuracy',fontsize=30)

plt.bar(models,scores,color=['red','magenta','cyan','blue','green'],alpha=0.5,linewidth=3,edgecolor='black')

for i,v in enumerate(scores):

    plt.text(i-.15,v+.03,format(scores[i],'.2f'),fontsize=20)