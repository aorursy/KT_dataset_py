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

framingham = pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")
framingham.shape
framingham.columns
framingham.head()
# Basic Packages:

import statsmodels.api as sm
framingham.corr()
framingham.info()
framingham.isnull().sum()
df=framingham.dropna()

df.shape
na=framingham.shape[0]-df.shape[0]

na_percentage= (na/framingham.shape[0])*100

na_percentage
df.rename(columns={'male':'gender'},inplace=True)
df.corr()
df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



pd.crosstab(df['gender'],df['TenYearCHD']).plot.bar(stacked=True)
sns.catplot(x='TenYearCHD',y='age',hue='gender',kind='box',data=df)
#Statistical Test:

# gender vs disease - 2 sample propotion test

# age vs disease - 2 sample t test

# education vs disaease - chi-sq

# current smoker - 2 sample propotion test 

# cigsperday - 2 sample t test

# BPMeds vs disease - 2 sample propotion test

# prevalentStroke, prevalentHyp, diabetes, totChol,sysBP, diaBP, BMI - 2 sample propotion test.



X=df.drop('TenYearCHD',axis=1)

Y=df['TenYearCHD']
X.shape,Y.shape
from statsmodels.tools import add_constant as add_constant

df_constant=add_constant(df)

df_constant.head()
cols=df_constant.columns[:-1]

model=sm.Logit(df.TenYearCHD,df_constant[cols]) # ('y~x',df)

result=model.fit()

result.summary()
X_final=df[['gender','age','cigsPerDay','totChol','sysBP','glucose']]

X_final.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

model=LogisticRegression()

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_final,Y,test_size=0.30,random_state=2)

model.fit(Xtrain,Ytrain)

y_pred=model.predict(Xtest)

acc=metrics.accuracy_score(Ytest,y_pred)

cm=metrics.confusion_matrix(Ytest,y_pred)

print('Overall Accuracy=',acc*100)

print('Confusion Matrix=\n',cm)
tpr=cm[1,1]/cm[1,:].sum()# Sensitivity

print(tpr)

print('Senstivity error (%) =',(1-tpr)*100)
tnr=cm[0,0]/cm[0,:].sum() #Specivicity

tnr
np.round(model.coef_,4)