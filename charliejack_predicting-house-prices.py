#Importing



import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import warnings

import os



print('File paths:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

       

for dirname, _, filenames in os.walk('Data/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Converting csv files to pandas dataframes

x = '/kaggle/input/house-prices-advanced-regression-techniques/'

train_data = pd.read_csv(x + 'train.csv')

test_data = pd.read_csv(x + 'test.csv')
train_data.tail()
test_data.shape
train_data.columns
#Obtaining correlation matrix

correlation_matrix = train_data.corr()

fig, ax1 = plt.subplots(figsize=(11,9))

sns.heatmap(correlation_matrix,square=True)

plt.show()
variables = 20 #amount of features we're interested in

columns = correlation_matrix['SalePrice'].sort_values(

          ascending=False).iloc[:variables].index

cm = train_data[columns].corr()

sns.set(font_scale=1.25)

fig,ax = plt.subplots(figsize=(14,10))

hm = sns.heatmap(cm,annot=True,fmt='.2f',square=True,annot_kws={'fontsize':11})

plt.show()
fig,axes = plt.subplots(figsize=(14,12),ncols=4,nrows=5)

fig.tight_layout()

for i in range(variables):

    ax = axes[i%5,i%4]

    sns.scatterplot(x=train_data[columns[i]], y=train_data['SalePrice'],ax=ax)

    ax.set_ylabel('')

    

print('Please note, the "Y" axis for all these plots is "SalePrice"')
df_train = train_data.loc[:,columns] #restricting train data to only our columns of interest, finally

df_test = test_data[columns[1:]] #Getting the relevant test data

df_combo = pd.concat([df_train.iloc[:,1:],df_test],axis=0) #Combining the df's so we can remove columns correctly

df_combo = df_combo.reset_index().drop('index',axis=1) #Making index proper, just incase





n_train = train_data.shape[0]

n_test = test_data.shape[0]
total_mv = df_train.isnull().sum().sort_values(ascending=False)

percentage = (df_train.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)





fig, ax = plt.subplots(figsize=(9, 7))

sns.barplot(x=percentage.index[:5],y=percentage[:5])



sns.set(font_scale=1.2)

plt.xlabel('Features')

plt.ylabel('Percent of missing values')

plt.title('Percent missing data by feature')

plt.show()



percentage = percentage.map('{:,.2f}%'.format)

combined = pd.concat([total_mv,percentage],axis=1,keys=['Total','Percentage'])

combined.head(5)
df_combo = df_combo.drop(['LotFrontage','MasVnrArea','GarageYrBlt'],axis=1)
plt.figure(figsize=(10,6))

sns.scatterplot(y=df_train['SalePrice'],x=df_train['OpenPorchSF'])
df_combo = df_combo.drop(['OpenPorchSF'],axis=1)



plt.figure(figsize=(10,6))

sns.scatterplot(y=df_train['SalePrice'],x=df_train['GarageArea'])
df_combo = df_combo.drop(df_train[df_train['GarageArea'] > 1220].index,axis=0)

df_train = df_train.drop(df_train[df_train['GarageArea'] > 1220].index,axis=0)

n_train -= 4



plt.figure(figsize=(10,6))

sns.scatterplot(y=df_train['SalePrice'],x=df_train['WoodDeckSF'])
df_combo = df_combo.drop('WoodDeckSF',axis=1)



print(df_train.shape)
df_combo = df_combo.drop(['GarageCars','1stFlrSF'],axis=1)
from scipy import stats



sns.distplot(df_train['SalePrice'] ,kde=True);





#ploting the distribution

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
df_train['SalePrice'] = np.log(df_train['SalePrice']+1)



sns.distplot(df_train['SalePrice'] ,kde=True);

#ploting the distribution

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
for col in ['GrLivArea','TotalBsmtSF','GarageArea']:

    df_combo[col] = np.log(df_combo[col]+1)
df_combo['YearBuilt'] = np.log(df_combo['YearBuilt'])/np.log(10)
df_combo = pd.get_dummies(df_combo)
train = df_combo.iloc[:n_train,:]

test = df_combo.iloc[n_train:,:]

test = test.fillna(0) 

y_train = np.array(df_train['SalePrice'])
import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor



model_rfr = RandomForestRegressor()

model_gbr = GradientBoostingRegressor()

model_xgb = xgb.XGBRegressor(n_estimators=2000)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction_seed=9, bagging_seed=9)
model_xgb.fit(train,y_train)

model_lgb.fit(train,y_train)

model_rfr.fit(train,y_train)

model_gbr.fit(train,y_train)
submission  = pd.DataFrame(data={'Id': np.arange(1461,1459+1461)})
prediction_1 = model_xgb.predict(test)

submission['xgb'] = np.exp(prediction_1) - 1

prediction_2 = model_lgb.predict(test)

submission['lgb'] = np.exp(prediction_2) - 1

prediction_4 = model_rfr.predict(test)

submission['rfr'] = np.exp(prediction_4) - 1

prediction_5 = model_gbr.predict(test)

submission['gbr'] = np.exp(prediction_5) - 1
submission['final'] = submission['xgb']*0.3+submission['lgb']*0.3 + submission['rfr']*0.3+submission['gbr']*0.1
submission.head()
submission_best = submission[['Id','final']]

submission_best = submission_best.rename(columns={'final': 'SalePrice'})

submission_best.to_csv('submission_final.csv',index=False)
submission_best