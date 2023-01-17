import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor



%matplotlib inline

sns.set(style="darkgrid")
idata = pd.read_csv('../input/india-trade-data/2018-2010_import.csv') 

edata = pd.read_csv('../input/india-trade-data/2018-2010_export.csv') 
def max_counts( feature , number, data, return_rest = False ) : 

    counts = data[feature].value_counts()

    values_list = list(counts[:number].values)

    rest_value =  sum(counts.values) - sum (values_list)

    index_list = list(counts[:number].index)

    

    if return_rest : 

        values_list.append(rest_value )

        index_list.append('rest items')

    

    result = pd.Series(values_list, index=index_list)

    if len(data[feature]) <= number : 

        result = None

    return result
def series_pie(series) : 

    plt.pie(series.values,labels=list(series.index),autopct ='%1.2f%%',labeldistance = 1.1,explode = [0.05 for i in range(len(series.values))] )

    plt.show()
def series_bar(series) : 

    plt.bar(list(series.index),series.values )

    plt.show()
def make_label_encoder(original_feature , new_feature,data) : 

    enc  = LabelEncoder()

    enc.fit(data[original_feature])

    data[new_feature] = enc.transform(data[original_feature])

    data.drop([original_feature],axis=1, inplace=True)
idata.head()
idata.shape
len(idata['HSCode'].unique())
idata['HSCode'].value_counts()[:10]
len(idata['HSCode'].unique())
len(idata['Commodity'].unique())
new_data = idata[idata['HSCode']==5]['Commodity']

new_data
new_data.unique()
for x in range(idata['HSCode'].max()): 

    new_data = idata[idata['HSCode']==x]['Commodity']

    print(len(new_data.unique()))

    print('---------------')
idata.drop(['Commodity'],axis=1, inplace=True)
idata.head()
idata['country'].unique()
len(idata['country'].unique())
counts = idata['country'].value_counts()
counts[:10]
max_countries = max_counts('country',8,idata)
max_countries
series_pie(max_countries)
series_bar(max_countries)
make_label_encoder('country','country_code',idata)
idata.head()
idata.info()
new_data  = idata[idata['value'] > 0]
new_data.shape
new_data.info()
X = new_data.drop(['value'], axis=1, inplace=False)

y = new_data['value']
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=10,learning_rate = 0.1 ,random_state=33)

GBRModel.fit(X_train, y_train)
print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))

print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
y_pred = GBRModel.predict(X_test)

print('Predicted Value for GBRModel is : ' , y_pred[:10])
len(edata['HSCode'].unique())
edata['HSCode'].value_counts()[:10]
len(edata['HSCode'].value_counts())
edata.drop(['Commodity'],axis=1, inplace=True)
edata.head()
edata['country'].unique()
len(edata['country'].unique())
counts = edata['country'].value_counts()
counts[:10]
list(counts[:10].index)
max_countries = max_counts('country',8,edata)

max_countries
series_pie(max_countries)
series_bar(max_countries)
make_label_encoder('country','country_code',edata)
edata.head()
edata.info()
new_data  = edata[edata['value'] > 0]
new_data.shape
new_data.info()
X = new_data.drop(['value'], axis=1, inplace=False)

y = new_data['value']
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=10,learning_rate = 0.1 ,random_state=33)

GBRModel.fit(X_train, y_train)
print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))

print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
y_pred = GBRModel.predict(X_test)

print('Predicted Value for GBRModel is : ' , y_pred[:10])