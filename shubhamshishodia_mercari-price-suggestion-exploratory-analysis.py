# Importing necessary packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

import re

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
# Read Training Data
trainData = pd.read_csv('../input/train.tsv',sep='\t',na_values={'brand_name':'NaN'})

# Looking up for basic information of the dataset
display(trainData.dtypes)
display(trainData.info())

# Viewing the first 10 rows of the dataset
display(trainData.head(10))

# Looking at distribution of price
display(trainData['price'].describe())
sns.distplot(trainData['price'])
plt.title('Histogram of prices')
plt.show()
plt.close()
display(sum(trainData['price']==0))

display(trainData[trainData['price']==0].head())
# Define a function to perform pre-processing
def dataPreprocess(input_data, train = True):
    data = input_data.copy()


    ## Creating individual categories from category_name
    categoryNames = data['category_name'].str.split('/',expand = True)

    data['category1'] = categoryNames[0]
    data['category2'] = categoryNames[1]
    data['category3'] = categoryNames[2]
        
    ## Converting item_condition_id, shipping and category_name to categorical variables
    data['shipping'] = pd.Categorical(['Free' if x==1 else 'Paid' for x in data['shipping']])
    data['item_condition_id'] = pd.Categorical(data['item_condition_id'])
    data['category1'] = pd.Categorical(['No category name present' if x!=x else x for x in data['category1']])
    data['category2'] = pd.Categorical(['No category name present' if x!=x else x for x in data['category2']])
    data['category3'] = pd.Categorical(['No category name present' if x!=x else x for x in data['category3']])
    data['brand_name_present'] = pd.Categorical(['Yes' if x==False else 'No' for x in data['brand_name'].isnull()])
    data['item_description_present'] = pd.Categorical(['Yes' if x==False else 'No' for x in data['item_description'].isnull()])
    
    ## Creating a column storing log (base 10) of prices
    if train==True:
        ## Dropping rows with 0 prices
        data = data[data['price']!=0]
        data['log_price'] = np.log(data['price']+1)
        return data
    else:
        return data
    
    

trainDataProcessed = dataPreprocess(trainData)
display(trainDataProcessed.head())
display(trainDataProcessed.dtypes)
display(trainDataProcessed.info())
display(sum(trainDataProcessed['price']==0))
print(set(trainDataProcessed['category1']))
print('Number of unique Category 1 Labels: '+str(len(set(trainDataProcessed['category1'])))) # Accounting for missing values
print('Number of unique Category 2 Labels: '+str(len(set(trainDataProcessed['category2'])))) # Accounting for missing values
print('Number of unique Category 3 Labels: '+str(len(set(trainDataProcessed['category3'])))) # Accounting for missing values

print('Category 1 Missing Values: ',trainDataProcessed['category1'].isna().sum())
print('Category 2 Missing Values: ',trainDataProcessed['category2'].isna().sum())
print('Category 3 Missing Values: ',trainDataProcessed['category3'].isna().sum())
fig, ax = plt.subplots(figsize=(50, 20))
ax = sns.boxplot(x='category1',y='log_price', data=trainDataProcessed)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.title('Boxplot of Log Prices wrt Category 1',fontsize=26)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(50, 20))
ax = sns.boxplot(x='category2',y='log_price', data=trainDataProcessed)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.title('Boxplot of Log Prices wrt Category 2',fontsize=26)
plt.suptitle('')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(50, 20))
ax = sns.boxplot(x='category3',y='log_price', data=trainDataProcessed)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.title('Boxplot of Log Prices wrt Category 3',fontsize=26)
plt.suptitle('')
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
ax=sns.boxplot(x='shipping',y='log_price', data=trainDataProcessed)
plt.title('Effect of free shipping on prices',fontsize=26)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
ax=sns.boxplot(x='category1',y='log_price',hue='shipping', data=trainDataProcessed)
plt.title('Effect of free shipping on prices across different categories',fontsize=26)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
ax=sns.boxplot(x='item_condition_id',y='log_price', data=trainDataProcessed)
plt.title('Effect of item condition  on prices',fontsize=26)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
ax=sns.boxplot(x='brand_name_present',y='log_price', data=trainDataProcessed)
plt.title('Effect of presence of brand names  on prices',fontsize=26)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
ax=sns.boxplot(x='item_description_present',y='log_price', data=trainDataProcessed)
plt.title('Effect of presence of item description  on prices',fontsize=26)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
def word_count(string):
    try:
        return len(re.findall("[a-zA-Z_]+", string))
    except TypeError:
        return 0
trainDataProcessed['word_count'] = [word_count(x) for x in trainDataProcessed['item_description']]
trainDataProcessed.head()


sns.scatterplot(x='word_count',y='log_price',data=trainDataProcessed)
xData = trainDataProcessed[['category1','category2','shipping','brand_name_present']]

xOneHotEncoded = pd.get_dummies(xData,
                                  columns = ['category1','category2','shipping','brand_name_present'],
                                  prefix= ['cat1','cat2','shipping','brand'])
yData = trainDataProcessed['log_price']

display(xData.head())
display(xOneHotEncoded.shape)
display(yData.head())
x_train, x_test, y_train, y_test = train_test_split(xOneHotEncoded,yData,test_size=0.2, random_state = 1)
modelRegression = LinearRegression()
modelRegression.fit(x_train,y_train)
minPrice = trainDataProcessed.groupby(['category1','category2'], as_index=False)['price'].min()
minPrice['price'] = np.log(minPrice['price']+1)
minPrice

maxPrice = trainDataProcessed.groupby(['category1','category2'], as_index=False)['price'].max()
maxPrice['price'] = np.log(maxPrice['price']+1)
maxPrice

medPrice = trainDataProcessed.groupby(['category1','category2'], as_index=False)['price'].median()
medPrice['price'] = np.log(medPrice['price']+1)
medPrice

trainDataModel2 = pd.merge(trainDataProcessed,minPrice,on=['category1','category2'],suffixes=('','_min'))
trainDataModel2 = pd.merge(trainDataModel2,maxPrice,on=['category1','category2'],suffixes=('','_max'))
trainDataModel2 = pd.merge(trainDataModel2,medPrice,on=['category1','category2'],suffixes=('','_med'))
trainDataModel2
xData2 = trainDataModel2[['category1','category2','price_min','price_max','price_med','shipping','brand_name_present']]

xOneHotEncoded2 = pd.get_dummies(xData2,
                                  columns = ['category1','category2','shipping','brand_name_present'],
                                  prefix= ['cat1','cat2','shipping','brand'])
yData2 = trainDataProcessed['log_price']

display(xData2.head())
display(xOneHotEncoded2.shape)
display(yData2.head())
x_train2, x_test2, y_train2, y_test2 = train_test_split(xOneHotEncoded2,yData2,test_size=0.2, random_state = 1)
modelRegression2 = LinearRegression()
modelRegression2.fit(x_train2,y_train2)
xData3 = trainDataProcessed[['category1','category2','shipping','brand_name_present']]

xOneHotEncoded3 = pd.get_dummies(xData,
                                  columns = ['category1','category2','shipping','brand_name_present'],
                                  prefix= ['cat1','cat2','shipping','brand'])
yData3 = trainDataProcessed['log_price']

x_train3, x_test3, y_train3, y_test3 = train_test_split(xOneHotEncoded3,yData3,test_size=0.2, random_state = 1)

modelRegression3 = GradientBoostingRegressor()
modelRegression3.fit(x_train3,y_train3)
y_test1 = np.asarray(y_test)
y_test1 = y_test1.reshape(-1,1)

y_train1 = np.asarray(y_train)
y_train1 = y_train1.reshape(-1,1)

y_test12 = np.asarray(y_test2)
y_test12 = y_test12.reshape(-1,1)

y_train12 = np.asarray(y_train2)
y_train12 = y_train12.reshape(-1,1)

y_test13 = np.asarray(y_test3)
y_test13 = y_test12.reshape(-1,1)

y_train13 = np.asarray(y_train3)
y_train13 = y_train12.reshape(-1,1)
print('RMSLE for Model 1 on Training Set:',mean_squared_error(modelRegression.predict(x_train),y_train1))
print('RMSLE for Model 1 on Validation Set:',mean_squared_error(modelRegression.predict(x_test),y_test1))
      
print('RMSLE for Model 2 on Training Set:',mean_squared_error(modelRegression2.predict(x_train2),y_train12))
print('RMSLE for Model 2 on Validation Set:',mean_squared_error(modelRegression2.predict(x_test2),y_test12))

print('RMSLE for Model 3 on Training Set:',mean_squared_error(modelRegression3.predict(x_train3),y_train13))
print('RMSLE for Model 3 on Validation Set:',mean_squared_error(modelRegression3.predict(x_test3),y_test13))
fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_train, label = 'Actual')
sns.distplot(modelRegression.predict(x_train), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 1 on Training Set', fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_test, label = 'Actual')
sns.distplot(modelRegression.predict(x_test), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 1 on Validation Set', fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_train2, label = 'Actual')
sns.distplot(modelRegression2.predict(x_train2), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 2 on Training Set',fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_test2, label = 'Actual')
sns.distplot(modelRegression2.predict(x_test2), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 2 on Validation Set',fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_train3, label = 'Actual')
sns.distplot(modelRegression3.predict(x_train3), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 3 on Training Set',fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(50, 20))
sns.distplot(y_test3, label = 'Actual')
sns.distplot(modelRegression3.predict(x_test3), label = 'Predicted')
plt.title('Histogram of Actual vs Predicted Prices for Model 3 on Validation Set',fontsize = 26)
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='22')
plt.setp(ax.get_legend().get_title(), fontsize='32')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel(ax.get_xlabel(), fontsize=24)
ax.set_ylabel(ax.get_ylabel(), fontsize=24)
plt.show()
plt.close()
testData = pd.read_csv('../input/test_stg2.tsv',sep='\t',na_values={'brand_name':'NaN','item_description':'No description yet'})

testDataProcessed = dataPreprocess(testData, train = False)
testDataProcessed = pd.merge(testDataProcessed,minPrice,on=['category1','category2'],suffixes=('','_min'))
testDataProcessed = pd.merge(testDataProcessed,maxPrice,on=['category1','category2'],suffixes=('','_max'))
testDataProcessed = pd.merge(testDataProcessed,medPrice,on=['category1','category2'],suffixes=('','_med'))

testDataProcessed.rename(columns={'price':'price_min'},inplace=True)
display(testDataProcessed.head())
## Checking whether categories in the train and test set are the same or not

print('Category 1 is same for train and test sets?: ',set(testDataProcessed['category1'])==set(trainDataProcessed['category1']))
print('Category 1 is same for train and test sets?: ',set(testDataProcessed['category2'])==set(trainDataProcessed['category2']))
print('Category 1 is same for train and test sets?: ',set(testDataProcessed['category3'])==set(trainDataProcessed['category3']))
testXData = testDataProcessed[['category1','category2','shipping','brand_name_present']]
xTestDataHotEncoded = pd.get_dummies(testXData,
                                  columns = ['category1','category2','shipping','brand_name_present'],
                                  prefix= ['cat1','cat2','shipping','brand'])
predictedPrices = modelRegression.predict(xTestDataHotEncoded)
predictedPrices = np.exp(predictedPrices)+1
submission1 = pd.DataFrame({'test_id':testDataProcessed['test_id'],
                          'price':predictedPrices})
submission1.to_csv('submission1.csv',index=False)
testXData2 = testDataProcessed[['category1','category2','price_min','price_max','price_med','shipping','brand_name_present']]
xTestDataHotEncoded2 = pd.get_dummies(testXData2,
                                  columns = ['category1','category2','shipping','brand_name_present'],
                                  prefix= ['cat1','cat2','shipping','brand'])

predictedPrices2 = modelRegression2.predict(xTestDataHotEncoded2)
predictedPrices2 = np.exp(predictedPrices2)+1

submission2 = pd.DataFrame({'test_id':testDataProcessed['test_id'],
                          'price':predictedPrices2})
submission2.to_csv('submission2.csv',index=False)
predictedPrices3 = modelRegression3.predict(xTestDataHotEncoded)
predictedPrices3 = np.exp(predictedPrices3)+1
submission3 = pd.DataFrame({'test_id':testDataProcessed['test_id'],
                          'price':predictedPrices3})
submission3.to_csv('submission3.csv',index=False)
submission3.head()