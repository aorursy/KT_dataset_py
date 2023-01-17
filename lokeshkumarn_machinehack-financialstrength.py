import os

import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

import seaborn as sns
os.listdir('../input')
df_train = pd.read_csv('../input/final_train_data.csv',index_col=False)

df_test = pd.read_csv('../input/final_test_data.csv',index_col=False)
df_train = df_train[['Country Name','Country Code','Year','Balance','Inflation','GDP','Exports','Trade']]

df_train.head()
print('Number of countries:',len(df_train['Country Name'].unique()))
#Years of data we have for each country

df_train.groupby(by=['Country Name'])['Year'].count().sort_values(ascending=False)[:30].plot(kind='bar',figsize=(15,5))
#Count of records/years

df_train.groupby(by=['Country Name'])['Year'].count().plot(kind='hist',figsize=(15,5))
print("Min Year:",df_train['Year'].min())

print("Max Year:",df_train['Year'].max())

print("Range:",df_train['Year'].max()- df_train['Year'].min())#Expecting 56 rows for each country but No
#Which year saw big growth

df_train.groupby(['Year']).mean()['GDP'].sort_values(ascending=False)[:56].plot(kind='bar',figsize=(18,5))
#Country which had the highest GDP in 1968

df_train[df_train['Year']==1968].sort_values(['GDP'],ascending=False).plot('Country Name','GDP',kind='bar',figsize=(5,3))
#Country which had the lowesst GDP in 2009

df_train[df_train['Year']==2009].sort_values(['GDP'])[:150].plot('Country Name','GDP',kind='bar',figsize=(20,5))
df_train[df_train['Year']==2009].sort_values(['GDP'])['GDP'][:150].plot(kind='hist')
#Intereseting to See Syria and Mynammar where having higher GDP on the year - 2009

df_train[(df_train['Year']==2009) & (df_train['Country Code'].isin(['SYR','MMR','IND','ZMB','ETH']))]#Oh NaN for Syria
df_train.info() #Missing values in Inflation,GDP,Exports,Trade
df_train.isnull().sum()
#Lets start with the Big one - Inflation\

df_train[df_train['Inflation'].isnull()].groupby(['Country Name'])['Year'].count().sort_values(ascending=False)
#Lets take Libya missing values

df_train[df_train['Country Name']=='Libya'].sort_values(['Year'])
#Before fixing the Missing values lets look into the rows which has null inflation,export,trade,gdp

indexes = df_train[(df_train['Inflation'].isnull()) & (df_train['Exports'].isnull()) &

         (df_train['GDP'].isnull()) & (df_train['Trade'].isnull())].index



print(indexes)



#Drop those Rows

df_train.drop(indexes,inplace=True)
#again check info

df_train.info()
df_train.isnull().sum()
#Next we check which attribute in [Inflation, Trade,Exports, GDP] is highly correlated to Balance

sns.heatmap(df_train.dropna()[['Inflation','Trade','Exports','GDP','Balance']].corr(),annot=True)#No relation
#Coming back to Libya - Inflation  - From correlation its clear that Inflation doesn't have relation to GDP,Trade,Exports also

df_train[df_train['Country Name']=='Libya'].sort_values(['Year'],ascending=True)
df_train[df_train['Country Name']=='Libya'].sort_values(['Year'],ascending=True).describe()
df_train[df_train['Country Name']=='Libya'].sort_values(

    ['Year'],ascending=True).plot('Year','Balance',kind='bar',figsize=(15,5))
df_train[(df_train['Inflation'].isnull()) & (df_train['GDP'].isnull()) & (df_train['Trade'].isnull())]
#Lets discard rows in which Trade,GDP,inflation is NaN and Impute mean to GDP and Inflation if Export and trade has value

indexes = df_train[(df_train['Inflation'].isnull()) & (df_train['GDP'].isnull()) & (df_train['Trade'].isnull())].index

print(indexes)

#Drop those Rows

df_train.drop(indexes,inplace=True)
df_train.info()
df_train.isnull().sum()
#Again Libya to Check

df_train[df_train['Country Name']=='Libya'].sort_values(['Year'],ascending=True)
df = df_train.copy()
df[(df['Country Name'] == 'Libya') & (df['Inflation'].isnull())].loc[:,'Inflation']
for ctry_name in df['Country Name'].unique():

    print(ctry_name)

    df.loc[(df['Country Name'] == ctry_name) & 

           (df['Inflation'].isnull()),'Inflation'] = df[(df['Country Name'] == ctry_name) & 

                                                        (df_train['Inflation'].notnull())]['Inflation'].mean()

    df.loc[(df['Country Name'] == ctry_name) & 

           (df['GDP'].isnull()),'GDP'] = df[(df['Country Name'] == ctry_name) & (df_train['GDP'].notnull())]['GDP'].mean()

    

    df.loc[(df['Country Name'] == ctry_name) & 

           (df['Exports'].isnull()),'Exports'] = df[(df['Country Name'] == ctry_name) & 

                                                        (df_train['Exports'].notnull())]['Exports'].mean()

    

    df.loc[(df['Country Name'] == ctry_name) & 

           (df['Trade'].isnull()),'Trade'] = df[(df['Country Name'] == ctry_name) & 

                                                (df_train['Trade'].notnull())]['Trade'].mean()
df.isnull().sum()
df_train = df
df_train.isnull().sum()
print('Mean balance:',df_train['Balance'].mean())

print('Median balance:',df_train['Balance'].median())

print('STD balance:',df_train['Balance'].std())

df_train['Balance'].plot(kind='hist')
print('Mean Inflation:',df_train['Inflation'].mean())

print('Median Inflation:',df_train['Inflation'].median())

print('STD Inflation:',df_train['Inflation'].std())

df_train['Inflation'].plot(kind='hist')
print('Mean GDP:',df_train['GDP'].mean())

print('Median GDP:',df_train['GDP'].median())

print('STD GDP:',df_train['GDP'].std())

df_train['GDP'].plot(kind='hist')
from sklearn.preprocessing import LabelEncoder
country_name_encoder = LabelEncoder()

country_name_encoder.fit(df_train['Country Name'])
country_name_encoder.transform(df_train['Country Name'])
len(country_name_encoder.transform(df_train['Country Name']))
df_train['CountryId'] = country_name_encoder.transform(df_train['Country Name'])
df_train = df_train[['CountryId','Year','Exports','Inflation','GDP','Trade','Balance']]
df_train = df_train.sort_values(by=['CountryId','Year']).reset_index(drop=True)
df_train[df_train['CountryId']==0]
#from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
features=['CountryId','Year','Exports','Inflation','GDP','Trade']

target=['Balance']



X = df_train[features]

y=df_train[target].values.reshape(df_train[target].values.shape[0])
#imp =SimpleImputer(missing_values=np.nan,strategy='mean')

#imp.fit(X)

#X = imp.fit_transform(X)
X.shape,y.shape
scaler = StandardScaler()

X = scaler.fit_transform(X)
X.shape
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor



import xgboost as xgb

from xgboost import XGBRegressor



from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
random_seed = 2019
X_B,X_val,y_B,y_val = train_test_split(X,y,test_size=0.1,random_state=random_seed)

X_train,X_test,y_train,y_test = train_test_split(X_B,y_B,test_size=0.1,random_state=random_seed)

print(X_B.shape,X_val.shape,y_B.shape,y_val.shape,X_train.shape,X_test.shape,y_train.shape,y_test.shape)
regressors =[XGBRegressor(),GradientBoostingRegressor(),AdaBoostRegressor()]
regression_params = {

    'XGBRegressor':{

        'min_child_weight':[4,5], 

        'gamma':[i/10.0 for i in range(3,6)],

        'subsample':[0.5,0.7,0.8],

        'learning_rate':np.linspace(0.1,0.2,5),

        'n_estimators':np.arange(100,300,100),

        'colsample_bytree':[i/10.0 for i in range(6,11)], 

        'max_depth': [3,5,7]

    },

    'GradientBoostingRegressor' : 

    {

        'loss':['huber', 'quantile'],

        'learning_rate':np.linspace(0.1,0.2,5),

        'n_estimators':np.arange(100,400,100),

        'min_samples_split':[0.1,0.3,0.5],

        'min_samples_leaf':[0.1,0.3,0.5],

        'subsample':[0.5,0.7,0.8],

        'max_depth':[3,5,7],

        'max_features':['sqrt'],

        'n_iter_no_change':[3],

        'random_state':[random_seed]

    },

    'AdaBoostRegressor':

    {

        'loss' : ['linear', 'square', 'exponential'],

        'learning_rate':np.linspace(0.1,0.2,5),

        'n_estimators':np.arange(100,400,100),

        'random_state':[random_seed]       

    }    

}
regression_results =[]

for i,reg in enumerate(regressors):

    reg_name = type(reg).__name__

    print(reg_name)

    

    regressor = GridSearchCV(reg,regression_params[reg_name],

                             scoring='neg_mean_squared_error',verbose=1,cv=5,n_jobs=-1)

    regressor.fit(X_train,y_train)

    y_pred_val = regressor.predict(X_val)  

    regression_results.append({'Reg':regressor,'y_pred_val':y_pred_val})
print(regression_results[0]['Reg'].best_params_)

print(regression_results[1]['Reg'].best_params_)

print(regression_results[2]['Reg'].best_params_)
xgb_params =regression_results[0]['Reg'].best_params_



gdc_params =regression_results[1]['Reg'].best_params_



ab_params =regression_results[2]['Reg'].best_params_



print(xgb_params)

print()

print(gdc_params)

print()

print(ab_params)
from sklearn.metrics import mean_squared_error as mse
xgb_clf = XGBRegressor(**xgb_params)

xgb_clf.fit(X_train,y_train)



y_pred_val = xgb_clf.predict(X_val)

y_pred_test = xgb_clf.predict(X_test)

print(mse(y_val,y_pred_val))

print(mse(y_test,y_pred_test))
clf = GradientBoostingRegressor(**gdc_params)

clf.fit(X_train,y_train)



y_pred_val = clf.predict(X_val)

y_pred_test = clf.predict(X_test)

print(mse(y_val,y_pred_val))

print(mse(y_test,y_pred_test))
clf_ab = AdaBoostRegressor(**ab_params)

clf_ab.fit(X_train,y_train)



y_pred_val = clf_ab.predict(X_val)

y_pred_test = clf_ab.predict(X_test)

print(mse(y_val,y_pred_val))

print(mse(y_test,y_pred_test))
df_test['CountryId'] = country_name_encoder.transform(df_test['Country Name'])
df_test = df_test[['CountryId','Year','Exports','Inflation','GDP','Trade']]
df_test.head()
df_test.isnull().sum()
df_test.values.shape
X_test = df_test.values

X_test = scaler.fit_transform(X_test)
balance = xgb_clf.predict(X_test)



df_out = pd.read_csv('../input/final_test_data.csv',index_col=False)



df_out = df_out[['Country Name','Country Code','Year','Inflation','GDP','Exports','Trade']]



df_out['Balance']=balance



df_out = df_out[['Country Name','Country Code','Year','Balance','Inflation','GDP','Exports','Trade']]

df_out.head()



df_out.to_csv('XGB_Submission_04022019.csv',index=False)
balance = clf.predict(X_test)



df_out = pd.read_csv('../input/final_test_data.csv',index_col=False)



df_out = df_out[['Country Name','Country Code','Year','Inflation','GDP','Exports','Trade']]



df_out['Balance']=balance



df_out = df_out[['Country Name','Country Code','Year','Balance','Inflation','GDP','Exports','Trade']]

df_out.head()



df_out.to_csv('GB_Submission_04022019.csv',index=False)
balance_ab = clf_ab.predict(X_test)



df_out = pd.read_csv('../input/final_test_data.csv',index_col=False)



df_out = df_out[['Country Name','Country Code','Year','Inflation','GDP','Exports','Trade']]



df_out['Balance']=balance_ab



df_out = df_out[['Country Name','Country Code','Year','Balance','Inflation','GDP','Exports','Trade']]

df_out.head()



df_out.to_csv('AB_Submission_04022019.csv',index=False)

