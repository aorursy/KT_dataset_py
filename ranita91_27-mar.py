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
train_data=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.shape,test_data.shape
#train_data1=train_data[train_data.describe().columns]
#test_data1=test_data[test_data.describe().columns]
#train_data1.shape,test_data1.shape
#NV_Treatment:
def null_value_treatment(data,threshold_value):
    data_nvp = (data.isna().sum()/data.shape[0]) * 100
    drop_column_names_list = data_nvp[data_nvp>float(threshold_value)].index
    data.drop(drop_column_names_list,axis = 1,inplace = True)
    num_columns = data.describe().columns
    cat_columns = [i for i in data.columns if i not in num_columns]
    for i in num_columns:
        data[i].fillna(data[i].median(),inplace = True)
    for i in cat_columns:
        data[i].fillna(data[i].value_counts().index[0],inplace = True)
    data,num_columns,cat_columns = data,num_columns,cat_columns
    return [data,num_columns,cat_columns]

[train_data1,num_columns,cat_columns]=null_value_treatment(train_data,20)
[test_data1,test_num_columns,test_cat_columns]=null_value_treatment(test_data,20)

#label encoders:
from sklearn.preprocessing import LabelEncoder
for i in cat_columns:
    Le=LabelEncoder()
    Le.fit(train_data1[i])
    x=Le.transform(train_data1[i])
    train_data1[i]=x
for i in test_cat_columns:
    Le=LabelEncoder()
    Le.fit(test_data1[i])
    x=Le.transform(test_data1[i])
    test_data1[i]=x    
#Outlier_Treatment:
import numpy as np
def iqr_outlier_treatment(df,numerical_columns):
	print()
	print('Before iqr outler treatment the stats are as below')
	print(df.describe())
	for col in df.columns:
		if col in numerical_columns:
			q1 = np.quantile(df[col].values,0.25)
			q3 = np.quantile(df[col].values,0.75)
			iqr = q3 - q1
			ltv = q1 - (1.5 * iqr)
			utv = q3 + (1.5 * iqr)
			fillval = []
			for val in df[col].values:
				if (val < ltv) or (val > utv):
					fillval.append(df[col].median())
				else:
					fillval.append(val)
			df[col] = fillval
	print('After iqr outler treatment the stats are as below')
	print(df.describe())
	return df
iqr_outlier_treatment(train_data1,num_columns)
iqr_outlier_treatment(test_data1,test_num_columns)
#histogram in individual window:
import matplotlib.pyplot as plt
for i in num_columns:
    train_data1[i].plot.hist()
    plt.title(i)
    plt.show()
    

#histogram in same window:  
#plt.figure(figsize=(50,50))
#train_data1.hist(layout=(6,13))
#plt.show()    
train_data1['1stFlrSF_trf']=np.log(train_data1['1stFlrSF'])
#train_data1['2ndFlrSF_trf']=np.log(train_data1['2ndFlrSF'])
train_data1['GrLivArea_trf']=np.log(train_data1['GrLivArea'])
train_data1['SalePrice_trf']=np.log(train_data1['SalePrice'])
train_data1.drop(columns=['1stFlrSF','GrLivArea','SalePrice'],inplace=True)
train_data1.columns

#plt.hist(np.log(train_data1['BsmtFinSF1']))
#plt.hist(np.log(BsmtUnfSF))
#plt.hist(np.log(TotalBsmtSF))

#plt.hist(train_data1['1stFlrSF_trf'])
#plt.hist(train_data1['2ndFlrSF_trf'])
#plt.hist(train_data1['GrLivArea_trf'])
#plt.hist(train_data1['SalePrice_trf'])

    
   
y=train_data1['SalePrice_trf']
x=train_data1.drop('SalePrice_trf',axis=1)

## 9 algorithms we are going to learn 

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.neural_network import multilayer_perceptron


def model_validation(modelname):
    lr=modelname()
    lr.fit(x,y)
    lr.predict(x)
    actual=y
    predicted=lr.predict(x)
    error=abs(actual-predicted).sum()
    error
    mae=error/x.shape[0]
    mse=np.square(error).sum()/x.shape[0]
    rmse=np.sqrt(mse)
    mape= (((actual-predicted).sum()/actual.sum())*100)/x.shape[0]
    return [mae,mse,rmse,mape]

models=[LinearRegression,Lasso,Ridge,DecisionTreeRegressor,KNeighborsRegressor]
model_results=[]
for i in models:
    mae,mse,rmse,mape=model_validation(i)
    model_results.append([i,mae,mse,rmse,mape])

model_results=pd.DataFrame(model_results)
model_results.columns=["model_name","mae","mse","rmse","mape"]

model_results


from sklearn.tree import DecisionTreeRegressor
dr=DecisionTreeRegressor()
dr.fit(x,y)
dr.predict(x)
predicted_values=dr.predict(test_data1)
predicted_values=pd.DataFrame(predicted_values)
submission_file=predicted_values.set_index(test_data['Id'])
submission_file.columns=['sales_price']
submission_file.head(20)


submission_file.to_
