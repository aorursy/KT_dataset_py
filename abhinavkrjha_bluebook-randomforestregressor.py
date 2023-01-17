import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
LOCATION = "../input/"

PATH = os.path.join(LOCATION,"bluebook-for-bulldozers/","train/")
data = pd.read_csv(PATH+"Train.csv",low_memory=False,parse_dates = ['saledate'])

data.head()
print("Number of rows",data.shape[0])

print("Number of columns",data.shape[1])
print('List of columns \n\n')

print(data.columns)
data['SalePrice'] = np.log(data['SalePrice']) #as kaggle is using rmse error function 
data['saleYear'] = data.saledate.apply(lambda x:int(str(x).split('-')[0])) #extract sale year from saleYear parsed

data = data.drop(columns='saledate',axis=1)
data.head().T
def replace_category(s):#replacing the categorical varible with numeric values

    if s.lower()=="high":

        return 0

    elif s.lower()=="medium":

        return 1

    else:

        return 2

        

data['UsageBand'] = data['UsageBand'].apply(lambda x:replace_category(str(x)))
for column in data.columns:#replacing all strings which have unique categories to numeric categorical values

    if not np.issubdtype(data[column],np.number):

        data[column] = data[column].astype('category')

        data[column] = data[column].cat.codes+1

    data[column] = data[column].fillna(data[column].median())
x = data.loc[:,data.columns!='SalePrice']

y = data.loc[:,'SalePrice']# as we have to predict the salePrice
rg = RandomForestRegressor(n_jobs=-1,n_estimators=10,oob_score=True,min_samples_leaf=3,max_features=0.5)#various model parameters got after hyperparameter tuning

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1)
%time rg.fit(xtrain,ytrain)
y_pred = rg.predict(xtest)
print('Testing accuracy is ',rg.score(xtest,ytest))

print('Training accuracy is ',rg.score(xtrain,ytrain))