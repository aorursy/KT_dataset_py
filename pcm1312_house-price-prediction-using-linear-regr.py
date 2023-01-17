import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv("../input/train.csv", sep=",", na_values=".")

test=pd.read_csv("../input/test.csv", sep=",", na_values=".")
train.describe()
#finding the missing values in each field in the train dataset

for col in train:

    if train[col].isnull().sum() > 0:

         

        print('number of missing values in {0} is {1}'.format(col,train[col].isnull().sum()))

    #else:

         #print("There are no missing values in {0}".format(col))

    

#Finding the missing values in the test dataset

for col in test:

    if test[col].isnull().sum() > 0:

        print('the number of missing values in {0} is {1}'.format(col,test[col].isnull().sum()))

        #print('number of missing values in {0} is {1}'.format(col,test[col].isnull().sum()))

    
train.LotFrontage.fillna(train["LotFrontage"].mean(), inplace=True)
#filling the missing values in the GarageType

train.GarageType.fillna('Attchd', inplace=True)
train.Fence.fillna('MnPrv',inplace=True)
train.PoolQC.fillna('Gd',inplace='True')
train.PoolArea.hist(color='green')#Analysing individual variables graphically
train.OverallQual.hist(stacked=True,color='Blue')
train.MoSold.hist(stacked=True,color='Brown')
train.corr() #Finding correlation
#ANOVA

import scipy.stats as stats

#Preparing data by grouping various categories

df_saleprice_street_Grvl = train['SalePrice'][train.Street == 'Grvl']

df_saleprice_street_Pave = train['SalePrice'][train.Street == 'Pave']



print (len(df_saleprice_street_Grvl))

print (len(df_saleprice_street_Pave))



# Use the ANOVA function available in Stats library

F, p = stats.f_oneway(df_saleprice_street_Grvl,df_saleprice_street_Pave)

print (F,p)

#Tukey's Post hoc analysis

from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.stats.multicomp import MultiComparison



mc = MultiComparison(train['SalePrice'], train['Condition1'])

result = mc.tukeyhsd()

 

print(result)

print(mc.groupsunique)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.stats.multicomp import MultiComparison



mc = MultiComparison(train['SalePrice'], train['Fence'])

result = mc.tukeyhsd()

 

print(result)

print(mc.groupsunique)

trace = plt.scatter(train.LotFrontage,train.SalePrice,color='Magenta')

print (trace)
trace = plt.scatter(train.LotArea,train.SalePrice,color='green')

print (trace)
plt.subplots(1,1, figsize=(5,5))

average_age = train[['OverallQual', 'SalePrice']].groupby(['OverallQual'], as_index=False).mean()

sb.barplot('OverallQual', 'SalePrice', data=average_age)

plt.xticks(rotation=90)

print ('OverallQual SalePrice relation: ')
plt.subplots(1,1, figsize=(5,5))

average_age = train[['PoolQC', 'SalePrice']].groupby(['PoolQC'], as_index=False).mean()

sb.barplot('PoolQC', 'SalePrice', data=average_age)

plt.xticks(rotation=90)

print ('PoolQC SalePrice relation: ')
#lABEL ENCODING

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

categorical_data = pd.DataFrame()

categorical_data['Street']=train['Street']

categorical_data['LandContour'] = train['LandContour']

categorical_data['Utilities']=train['Utilities']

categorical_data['Neighborhood']=train['Neighborhood']

categorical_data['Condition1']=train['Condition1']

categorical_data['Condition2']=train['Condition2']

categorical_data['Fireplaces']=train['Fireplaces']

#categorical_data['GarageType']=train['GarageType']

categorical_data['PoolQC']=train['PoolQC']

categorical_data['SaleType']=train['SaleType']

categorical_data['SaleCondition']=train['SaleCondition']

categorical_data.head(5)

encoded_cat_df = pd.DataFrame()

encoded_cat_df1 = pd.DataFrame()

for column in categorical_data.columns:

    le.fit(categorical_data[column])

    encoded_cat_df[column] = le.transform(categorical_data[column]) 

encoded_cat_df.head()

num_feature_list = [ 'OverallQual','MoSold','YrSold','SalePrice','GrLivArea','OverallCond','KitchenAbvGr','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','MSSubClass']

num_df = (train[num_feature_list])
mydata = pd.concat([encoded_cat_df, num_df], axis=1)
test.PoolQC.fillna('Ex',inplace=True)

cat_data = pd.DataFrame()

#cat_data['Street']=test['Street']

#cat_data['LandContour'] = test['LandContour']

#cat_data['Utilities']=test['Utilities']

cat_data['Neighborhood'] = test['Neighborhood']

#cat_data['Condition1']=test['Condition1']

#cat_data['Condition2'] = test['Condition2']



cat_data['PoolQC'] = test['PoolQC']

#cat_data['SaleType']=test['SaleType']

cat_data['SaleCondition'] = test['SaleCondition']

cat_data.head(5)
encoded_cat_df2 = pd.DataFrame()

encoded_cat_df3 = pd.DataFrame()

for column in cat_data.columns:

    le.fit(cat_data[column])

    encoded_cat_df2[column] = le.transform(cat_data[column]) 

encoded_cat_df2.head()
num_feature_list1 = [ 'OverallQual','MoSold','YrSold','Fireplaces','GrLivArea','OverallCond','KitchenAbvGr','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','MSSubClass']

num_df1 = (test[num_feature_list1])
testdata = pd.concat([encoded_cat_df2, num_df1], axis=1)
#lINEAR REGRESSION

from sklearn.cross_validation import train_test_split

import numpy as np

from sklearn import linear_model

X_train = mydata[['Neighborhood', 'OverallQual','SaleCondition','GrLivArea','OverallCond','KitchenAbvGr','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','MSSubClass','PoolQC']]

y_train=mydata['SalePrice']

X_test = testdata[['Neighborhood', 'OverallQual', 'SaleCondition','GrLivArea','OverallCond','KitchenAbvGr','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','MSSubClass','PoolQC']]

regr=linear_model.LinearRegression()

# Train the model using the training sets

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

output = pd.DataFrame({

        

        'Id': test.Id,

        'SalePrice': y_pred

    })

output.to_csv('SalePricePrediction.csv', index=False)



 








