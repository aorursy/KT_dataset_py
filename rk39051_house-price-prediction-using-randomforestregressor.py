# let's pridict the house price using simple RandomForestRegressor with all features
#importing the basic library

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

pd.set_option('max_column',None)

pd.set_option('max_row',None)
#loading the data and viewing the shape

data1=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data2=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

y=data1["SalePrice"] #load the dependent variable

data1=data1.drop(columns=["SalePrice"])

dataset=pd.concat([data1,data2])

dataset.head(2)

dataset.shape
dataset.isnull().sum()
#remove some unwanted columns from dataset

unwanted_columns=[]

for i in dataset.columns:

    if (dataset[i].isnull().sum()>=0.999*len(dataset) or i=="Id"):

        unwanted_columns.append(i)

        
unwanted_columns
dataset=dataset.drop(columns=unwanted_columns)
#extracting categorical, numerical and temporal variabe(date time year)

categorial_features=[]

numerical_features=[]

temporal_features=[]





for i in dataset.columns:

    if "Year" in i or "Yr" in i:

        temporal_features.append(i)

        

for i in dataset.columns:

    if dataset[i].dtype!='O':

        numerical_features.append(i)

    if dataset[i].dtype =='O' and i not in (temporal_features):

        categorial_features.append(i)

dataset[temporal_features].head()
dataset[categorial_features].head(10)
#dealing with categorical missing value

def categorical_missed_value(dataset,feature):

    data=dataset.copy()

    data[feature]=data[feature].fillna("missing")

    return data

        

dataset=categorical_missed_value(dataset,categorial_features)  





#dealing with numerical missing value

   

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])
dataset=pd.get_dummies(drop_first=True,data=dataset,columns=categorial_features)
dataset.head(5)
train_data,test_data=dataset.iloc[:1460,:],dataset.iloc[1460:,:]
test_data.head()
#split the data

X=train_data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)
# Fit Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor 

rfr = RandomForestRegressor(n_estimators =400)#n_estimator is no of decision tree

rfr.fit(X_train, y_train)
#test your model on X_test (splitted dataset)

y_pred=rfr.predict(X_test)
y_pred
y_test
#predict the result of test_data for submission

result=rfr.predict(test_data)
result
#load the sample file of submission.csv

temp=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
temp.head()
#load the predicted result as dataframe

result=pd.DataFrame(result)
result.head()
#rename the result column name as SalePrice

result=result.rename(columns={0:"SalePrice"})
#drop SalePrice from temp dataframe because we will add our own predicted SalePrice

temp=temp.drop(columns="SalePrice")
#add temp dataframe with result

final=pd.concat([temp,result],axis=1)
#final submission dataset

final.head()
#save to csv to make submission

final.to_csv("sample_submission.csv",index=False)