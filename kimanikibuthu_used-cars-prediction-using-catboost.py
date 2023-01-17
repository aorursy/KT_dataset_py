# Data manipulation libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport



%matplotlib inline



#Model development and feature engineering libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

import lightgbm as lgb

from sklearn import metrics
train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')



test = pd.read_csv('../input/used-cars-price-prediction/test-data.csv')



print('Train Size : {} \nTest Size : {}'.format(train.shape,test.shape) )
combined = [train,test]
train.head()
test.head()
# info on the dataset

train.info()



print('-'*45)



test.info()
ProfileReport(train)
ProfileReport(test)
#dropping Unnamed and New price columns



for dataset in combined:

    dataset.drop(['Unnamed: 0','New_Price'],axis=1,inplace = True)
#The highly skewed km driven. 



train['Kilometers_Driven'].skew()
test['Kilometers_Driven'].skew()
# Apply log transform and see how the skew changes

#train['Kilometers_Driven']=np.log(train['Kilometers_Driven'])



# Apply box cox

from scipy import stats



km = stats.boxcox(train['Kilometers_Driven'])[0]

train['Kilometers_Driven'] = pd.Series(km)
train['Kilometers_Driven'].skew()
# Year to date time

for dataset in combined:

    dataset['Year']=pd.to_datetime(dataset['Year'])
# Let's split the data and change the datatype



# Train data set

new_train = train['Mileage'].str.split(' ',n = 1, expand=True)



# Test dataset

new_test = test['Mileage'].str.split(' ',n=1,expand=True)



print(new_train)



print(new_test)
#Change data in the column mileage

train['Mileage']=new_train[0].astype(float)



test['Mileage']=new_test[0].astype(float)
# Now let's categorize these column

for dataset in combined:

    dataset.loc[(dataset ['Mileage']>0 ) & (dataset['Mileage']<=10) , 'Mileage'] = 0

    dataset.loc[(dataset ['Mileage']>10 ) & (dataset['Mileage']<=20) , 'Mileage'] = 1

    dataset.loc[(dataset ['Mileage']>20 ) & (dataset['Mileage']<=30) , 'Mileage'] = 2

    dataset.loc[(dataset ['Mileage']>30 ),'Mileage']=3

    
#Splitting the values

#Train set

engine_new = train['Engine'].str.split(" ", n = 1 , expand = True)



#Test

engine_newer = test['Engine'].str.split(" ", n = 1 , expand = True)
#Replace the values in the engine column

train['Engine']= engine_new[0]



test['Engine']= engine_newer[0]



train['Engine'].isnull().sum()
# Position of missing values in train



train.loc[np.nonzero(pd.isnull(train['Engine']))]
test['Engine'].isnull().sum()

#drop missing values

for dataset in combined:

    dataset.dropna(how='any', inplace=True)

    
test.isnull().sum()
train.shape
test.shape
test.sample(6)
#Change values to int

for dataset in combined:

    dataset['Engine']=dataset['Engine'].astype(np.int64)
train.info()
# Let's categorize the values

for dataset in combined:

    dataset.loc[(dataset['Engine']<=1000 ),'Engine']=0

    dataset.loc[(dataset['Engine']>1000 ) & (dataset['Engine']<=2000) , 'Engine'] = 1

    dataset.loc[(dataset['Engine']>2000 ) & (dataset['Engine']<=3000) , 'Engine'] = 2

    dataset.loc[(dataset['Engine']>3000 ),'Engine']=3

    
test.sample(6)
# Split the data

#train



power_train = train['Power'].str.split(' ', n = 1,expand = True)



#Test

power_test = test['Power'].str.split(' ', n = 1,expand = True)
train['Power']= power_train[0]



test['Power']=power_test[0]
#Length of null values



len(train[train['Power']=='null'])

#Length of null values



len(test[test['Power']=='null'])
# Dealing with these null values. We label them as unknown and give them a value of zero

for dataset in combined:

    dataset['Power'].replace({'null':'0'}, inplace=True)

# Change values to float

for dataset in combined:

    dataset['Power']=dataset['Power'].astype(float)
#Categorize the data

for dataset in combined:

    dataset.loc[(dataset ['Power']>0 ) & (dataset['Power']<=50) , 'Power'] = 1

    dataset.loc[(dataset ['Power']>50 ) & (dataset['Power']<=100) , 'Power'] = 2

    dataset.loc[(dataset ['Power']>100 ),'Power']=3

    
## View the data



train.head()
test.sample(6)
#split



#Train

name_train = train['Name'].str.split(' ',n = 1, expand = True)



# Test

name_test =test['Name'].str.split(' ',n = 1, expand = True)
name_train.sample(6)
#Change the data in Name

train['Name']=name_train[0]



test['Name']=name_test[0]
train.head()
train.Name.value_counts()
# Categorize the data

for dataset in combined:

    for x in dataset['Name']:

        if (x !='Maruti') and (x!='Hyundai') and (x!='Honda') and (x !='Toyota') and (x !='Mercedes-Benz'):

            dataset['Name'].replace(x,'Other', inplace= True)  



            

            
train.head()
# Extract year from that column



train['Year'] = pd.DatetimeIndex(train['Year']).year



test['Year'] = pd.DatetimeIndex(test['Year']).year
train.head()
train.Year.value_counts()
sns.set(rc={'figure.figsize':(20, 15)})
sns.countplot('Location', hue = 'Name', data = train)
sns.countplot('Fuel_Type', hue = 'Name', data = train)
sns.countplot('Transmission', hue = 'Name', data = train)
sns.countplot('Mileage', hue = 'Name', data = train)
sns.countplot('Power', hue = 'Name', data = train)
sns.countplot('Seats', hue = 'Name', data = train)
sns.countplot('Location', hue = 'Seats', data = train)
sns.countplot('Owner_Type', hue = 'Name', data = train)
ax = sns.pointplot(x='Name', y='Price', hue='Location', ci=None, data=train)
ax = sns.pointplot(x='Seats', y='Price', hue='Transmission', ci=None, data=train)
ax = sns.pointplot(x='Name', y='Price', hue='Power', ci=None, data=train)
train.head()
# Scale the data

#from sklearn.preprocessing import StandardScaler



#StandardScaler.fit()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for dataset in combined:

    dataset['Name']=le.fit_transform(dataset['Name'])

    dataset['Fuel_Type']=le.fit_transform(dataset['Fuel_Type'])

    dataset['Transmission']=le.fit_transform(dataset['Transmission'])

    dataset['Owner_Type']=le.fit_transform(dataset['Owner_Type'])

    dataset['Location']=le.fit_transform(dataset['Location'])
train.head()
test.head()
for dataset in combined:

    dataset['Mileage'] = dataset['Mileage'].astype(int)

    dataset['Power'] = dataset['Power'].astype(int)

    dataset['Seats'] = dataset['Seats'].astype(int)

    
test.head()
train.head()
for dataset in combined:

    dataset.drop('Year', axis = 1, inplace =True)
# Correlation Matrix

x = train.iloc[:,-1:]  #independent columns

y = train.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
x = train.drop(['Price','Name','Location'], axis = 1)

y = train['Price']
# Scale the data

#from sklearn.preprocessing import StandardScaler



#ss = StandardScaler()

#ss.fit(x)

#x = ss.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 7)
#Decision Tree



dt = DecisionTreeRegressor()



#Train

dt.fit(x_train,y_train)



#Predict



dt_pred = dt.predict(x_test)



# Evaluate

dt_rmse = metrics.mean_squared_error(y_test,dt_pred,squared=False)

dt_R2= metrics.r2_score(y_test,dt_pred)





print('rmse : ', dt_rmse)



print('R2 Score: ', dt_R2)



sns.regplot(x=y_test, y = dt_pred)
#Random Forest



rf = RandomForestRegressor()



#Train

rf.fit(x_train,y_train)



#Predict



rf_pred = rf.predict(x_test)



# Evaluate

rf_rmse = metrics.mean_squared_error(y_test,rf_pred,squared=False)

rf_R2= metrics.r2_score(y_test,rf_pred)





print('rmse : ', rf_rmse)



print('R2 Score: ', rf_R2)



sns.regplot(x=y_test, y = rf_pred)
lgbr = lgb.LGBMRegressor()



#Train

lgbr.fit(x_train,y_train)



#Predict



lgbr_pred = lgbr.predict(x_test)



# Evaluate

lgbr_rmse = metrics.mean_squared_error(y_test,lgbr_pred,squared=False)

lgbr_R2= metrics.r2_score(y_test,lgbr_pred)





print('rmse : ', lgbr_rmse)



print('R2 Score: ', lgbr_R2)



sns.regplot(x=y_test, y = lgbr_pred)
cat = CatBoostRegressor()



#Train

cat.fit(x_train,y_train)



#Predictcat



cat_pred = cat.predict(x_test)



# Evaluate

cat_rmse = metrics.mean_squared_error(y_test,cat_pred,squared=False)

cat_R2= metrics.r2_score(y_test,cat_pred)





print('rmse : ', cat_rmse)



print('R2 Score: ', cat_R2)



sns.regplot(x=y_test, y = cat_pred)