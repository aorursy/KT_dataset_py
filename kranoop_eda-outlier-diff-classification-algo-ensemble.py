# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import time

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.shape
data.describe()
pd.DataFrame(data.isna().sum())
data=data.drop("Unnamed: 32", axis =1)

data=data.drop("id", axis =1)
sns.countplot(data['diagnosis'],label="Count")
data["diagnosis"].replace("M",0,inplace = True)

data["diagnosis"].replace("B",1,inplace = True)
data.boxplot()
data.boxplot(column=(['radius_mean','texture_mean','perimeter_mean']),vert=False)
data.boxplot(column=(['compactness_mean','concavity_mean','concave points_mean']),vert=False)
data.boxplot(column=(['area_mean']),vert=False)
q25, q75 = np.percentile(data['radius_mean'], 25), np.percentile(data['radius_mean'], 75)

iqr = q75 - q25

cut_off = iqr * 1.5

lower, upper = q25 - cut_off, q75 + cut_off



outliers = [x for x in data['radius_mean'] if x < lower or x > upper]

len(outliers)
data['radius_mean'] = np.where(data['radius_mean']< lower, np.NaN, data['radius_mean'])

data['radius_mean'] = np.where(data['radius_mean']> upper, np.NaN, data['radius_mean'])

data['radius_mean'].isna().sum()

data['radius_mean'].replace(to_replace =np.NaN, 

                 value =data['radius_mean'].median(),inplace=True)
# Nomally above statement is coded as below, but if we do so it will identify the medean of data which 

# includes the outliers, which will be wrong hence we need to replace the outliers with 0 or NaN then idenfy the median



# outliers = [x for x in data['radius_mean'] if x < lower or x > upper]

# data['radius_mean'].replace(to_replace =[x for x in data['radius_mean'] if x > lower and x < upper], 

#                 value =data['radius_mean'].median(),inplace=True)



#Same need to be performed for all the columns
for column in ['texture_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']:

    q25, q75 = np.percentile(data[column], 25), np.percentile(data[column], 75)

    iqr = q75 - q25

    cut_off = iqr * 1.5

    lower, upper = q25 - cut_off, q75 + cut_off

    data[column].replace(to_replace =[x for x in data[column] if x > lower and x < upper], 

                 value =data[column].median(),inplace=True)

print( 'Outliers are replaced with Median')
data.head()
corr = data.corr()

plt.figure(figsize=(18,18))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},

           cmap= 'coolwarm')
data_backup=data

data=data.drop(["radius_se"], axis =1)

data.head()

prediction_var = list(['texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',  'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se' ,'radius_worst', 'texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])

#prediction_var 
corr = data.corr()

plt.figure(figsize=(18,18))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},

           cmap= 'coolwarm')
from sklearn.model_selection import train_test_split



#convert the datset into Test and Train data

X_train, X_test, Y_train, Y_test = train_test_split(data.drop('diagnosis', axis=1), data['diagnosis'],\

                                                    test_size=0.2, random_state=156)
from sklearn.linear_model import LogisticRegression



lgr = LogisticRegression(max_iter = 200)

lgr.fit(X_train,Y_train)

ypred=lgr.predict(X_test)

ypred_Log=ypred

print('Accuracy score - ' ,lgr.score(X_test,Y_test))
from sklearn.ensemble import RandomForestClassifier

RFmodel=RandomForestClassifier()

result=RFmodel.fit(X_train, Y_train)

ypred=result.predict(X_test)

ypred_RandomForest=ypred

print('Accuracy score - ' ,accuracy_score(Y_test,ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier() 

result=gb.fit(X_train, Y_train)

ypred=result.predict(X_test)



print('Accuracy score - ' ,accuracy_score(Y_test,ypred))
from xgboost import XGBClassifier

XGB = XGBClassifier() 

result=XGB.fit(X_train, Y_train)

ypred=result.predict(X_test)



print('Accuracy score - ' ,accuracy_score(Y_test,ypred))
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB() 

result=NB.fit(X_train, Y_train)

ypred=result.predict(X_test)



print('Accuracy score - ' ,accuracy_score(Y_test,ypred))
lr_diagnosis=lgr.predict(data.drop('diagnosis',axis=1))

rf_diagnosis=RFmodel.predict(data.drop('diagnosis',axis=1))

gb_diagnosis=gb.predict(data.drop('diagnosis',axis=1))

xgb_diagnosis=XGB.predict(data.drop('diagnosis',axis=1))

nb_diagnosis=NB.predict(data.drop('diagnosis',axis=1))
data['lr_diagnosis']=lr_diagnosis

data['rf_diagnosis']=rf_diagnosis

data['gb_diagnosis']=gb_diagnosis

data['xgb_diagnosis']=xgb_diagnosis

data['nb_diagnosis']=nb_diagnosis
data.head()
X_Train2, X_Test2, Y_Train2, Y_Test2 = train_test_split(data.drop("diagnosis",axis=1),data["diagnosis"],test_size=0.25,random_state=123)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier() 

result=gb.fit(X_Train2, Y_Train2)

ypred2=result.predict(X_Test2)

accuracy_score(Y_Test2,ypred2)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix( Y_Test2 ,gb.predict(X_Test2))



f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax)