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
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
data=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data.info()
data.describe()
data.isnull().any() #Check if any Nulls
data1=data.drop(['PassengerId','Firstname','Lastname'],axis=1) #drop columns that will have no effect on model predictions
colss=['Country','Sex','Age','Category'] #Columns for Model
cols=['Country','Sex','Category'] #Columns for categorical label encoder
label_encoder=LabelEncoder()

for col in cols:

    data1[col]=label_encoder.fit_transform(data1[col]).reshape(-1,1)
X_train,X_valid,y_train,y_valid=train_test_split(data1[colss],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)
forest_model=RandomForestRegressor(n_estimators=10000,random_state=1)

forest_model.fit(X_train,y_train)

prediction=forest_model.predict(X_valid)

print(accuracy_score(y_valid,prediction.round()))
#Check collinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.Series([variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])],index=X_train.columns)
#Check feature importance

import eli5

from eli5.sklearn import PermutationImportance

perm=PermutationImportance(forest_model, random_state=1).fit(X_valid,y_valid)

eli5.show_weights(perm, feature_names=X_valid.columns.tolist())
#Using results from feature importance

col2=['Age','Sex','Country']

X_train2,X_valid2,y_train2,y_valid2=train_test_split(data1[col2],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model2=RandomForestRegressor(n_estimators=1000,random_state=1)

forest_model2.fit(X_train2,y_train2)

prediction2=forest_model2.predict(X_valid2)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid2,prediction2.round())*100,2),'%')
col3=['Age','Sex']

X_train3,X_valid3,y_train3,y_valid3=train_test_split(data1[col3],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model3=RandomForestRegressor(n_estimators=1000,random_state=1)

forest_model3.fit(X_train3,y_train3)

prediction3=forest_model3.predict(X_valid3)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid3,prediction3.round())*100,2),'%')
col4=['Age']

X_train4,X_valid4,y_train4,y_valid4=train_test_split(data1[col4],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model4=RandomForestRegressor(n_estimators=1000,random_state=1)

forest_model4.fit(X_train4,y_train4)

prediction4=forest_model4.predict(X_valid4)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid4,prediction4.round())*100,2),'%')
col5=['Age','Country']

X_train5,X_valid5,y_train5,y_valid5=train_test_split(data1[col5],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model5=RandomForestRegressor(n_estimators=1000,random_state=1)

forest_model5.fit(X_train5,y_train5)

prediction5=forest_model5.predict(X_valid5)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid5,prediction5.round())*100,2),'%')
col6=['Category','Country']

X_train6,X_valid6,y_train6,y_valid6=train_test_split(data1[col6],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model6=RandomForestRegressor(n_estimators=1000,random_state=1)

forest_model6.fit(X_train6,y_train6)

prediction6=forest_model6.predict(X_valid6)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid6,prediction6.round())*100,2),'%')
from sklearn.metrics import mean_absolute_error

def get_est(n_estimators, train_X,val_X,train_y,val_y):

    model=RandomForestRegressor(n_estimators=n_estimators, random_state=1)

    model.fit(train_X,train_y)

    preds=model.predict(val_X)

    mae=accuracy_score(val_y,preds.round())

    return(mae)
estimates=range(1000,10000,1000)
sizes=[]

for estimate in estimates:

    esti=get_est(estimate,X_train,X_valid,y_train,y_valid)

    sizes.append(esti)

best_size=estimates[sizes.index(min(sizes))]
best_size
estimates=range(100,3000,100)
sizes=[]

for estimate in estimates:

    esti=get_est(estimate,X_train,X_valid,y_train,y_valid)

    sizes.append(esti)

fine_best_size=estimates[sizes.index(min(sizes))]
fine_best_size
#Rerun the best results with estimators=400, but still the same

col2=['Age','Sex','Country']

X_train2,X_valid2,y_train2,y_valid2=train_test_split(data1[col2],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)

forest_model2=RandomForestRegressor(n_estimators=400,random_state=1)

forest_model2.fit(X_train2,y_train2)

prediction2=forest_model2.predict(X_valid2)

#print(mean_absolute_error(y_valid,prediction))

print(round(accuracy_score(y_valid2,prediction2.round())*100,2),'%')