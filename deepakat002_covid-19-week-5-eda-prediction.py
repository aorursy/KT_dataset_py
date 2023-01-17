# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



# to show whole column and rows 

pd.set_option('display.max_columns',5400)

pd.set_option('display.max_rows',5400)



import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px







from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading datas

train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
train
test
submission
print ('train dataset shape : ', train.shape,'\n', 'test dataset shape : ',test.shape)
#General information about train data set



train.info()
#General information about train data set



test.info()
# Checking null values 



train.isnull().sum()
# Checking % null values 



round(100*(train.isnull().sum() )/ train.shape[0],3)
round(100*(test.isnull().sum() )/ train.shape[0],3)
train.columns
#checking values where country is not null



train.loc[~train['County'].isnull()]
#checking values where country is not null



train.loc[train['County'].isnull()]
# Here we are considering only country wise. Same can be performed for county as well as province_state.

# So dropping 'County', 'Province_State'



train = train.drop(['County', 'Province_State'],axis = 1)

test  = test.drop(['County', 'Province_State'],axis = 1)

train
# converting date column from object type to date time 



train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train.info()
# Creating separate df for confirmed cases & Fatalities



by_tv = train.groupby('Target')

confirmed_df = by_tv.get_group('ConfirmedCases')

confirmed_df
fatality_df = by_tv.get_group('Fatalities')

fatality_df
# Plotting mean confirmed cases country wise 



plt.figure(figsize=(30,100))

ax0=sns.barplot(x = 'TargetValue',y= 'Country_Region', data = confirmed_df,estimator = np.mean, ci =None)



for p in ax0.patches:

  val = p.get_width() # height of each bar

  x = p.get_x() + p.get_width() + 10.0 #x-cordinate of the text

  y = p.get_y() + p.get_height()/2 # y-coordinate of the text

  ax0.annotate(round(val,2),(x,y)) # attaching bar height to each bar of the barplot



plt.show()



# Plotting mean fatalities country wise



plt.figure(figsize=(30,100))



a = sns.barplot(x = 'TargetValue', y = 'Country_Region', estimator = np.mean, data = fatality_df,ci =None)



for p in a.patches:

  val = p.get_width()

  x = p.get_x() + p.get_width() + 10

  y = p.get_y() + p.get_height()/2

  a.annotate(round(val,2),(x,y))



plt.show()
#country vs targetValue



fig = px.pie(train, values='TargetValue', names='Country_Region')



fig.show()

# ploting confirmed cases country wise with time 



countries =set( confirmed_df['Country_Region'])



len(countries)
#Plotting confirmed cases with date for all countries



country_group =train.groupby('Target')

df = country_group.get_group('ConfirmedCases')



# df= country_group.get_group('US') # as we want to see the trend for US

df = pd.DataFrame(df.groupby(['Country_Region','Date'])['TargetValue'].sum()) # as multiple values present for single date

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25'] # As cases ae reported in high number after 25th Feb,2020



fig = px.line(df, x='Date', y ='TargetValue', color = 'Country_Region', title = 'Confirmed cases of all countries with date')

fig.show()





#Plotting fatality cases with date for all ocuntries



country_group =train.groupby('Target')

df = country_group.get_group('Fatalities')



# df= country_group.get_group('US') # as we want to see the trend for US

df = pd.DataFrame(df.groupby(['Country_Region','Date'])['TargetValue'].sum()) # as multiple values present for single date

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25'] # As cases ae reported in high number after 25th Feb,2020



fig = px.line(df, x='Date', y ='TargetValue', color = 'Country_Region', title = 'Fatalities cases of all countries with date')

fig.show()





#Plotting confirmed cases with date for India





country_group =confirmed_df.groupby('Country_Region')



df= country_group.get_group('India')

df = pd.DataFrame(df.groupby(['Date'])['TargetValue'].sum())

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25']



fig = px.line(df, x='Date', y ='TargetValue',title ='confirmed cases with date for India')



fig.show()



#Plotting fatalities cases with date for India



country_group = fatality_df.groupby('Country_Region')



df= country_group.get_group('India')

df = pd.DataFrame(df.groupby(['Date'])['TargetValue'].sum())

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25']



fig = px.line(df, x='Date', y ='TargetValue', title ='fatalities cases with date for India')





fig.show()





#Plotting confirmed cases with date for 'United Arab Emirates'



country_group =confirmed_df.groupby('Country_Region')



df= country_group.get_group('United Arab Emirates') # as we want to see the trend for US

df = pd.DataFrame(df.groupby(['Date'])['TargetValue'].sum()) # as multiple values present for single date

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25'] # As cases ae reported in high number after 25th Feb,2020



fig = px.line(df, x='Date', y ='TargetValue',title ="confirmed cases with date for 'United Arab Emirates'")

fig.show()





#Plotting fatalities with date for 'United Arab Emirates'



country_group =fatality_df.groupby('Country_Region')



df= country_group.get_group('United Arab Emirates') # as we want to see the trend for US

df = pd.DataFrame(df.groupby(['Date'])['TargetValue'].sum()) # as multiple values present for single date

df.reset_index(inplace = True)

df = df.loc[df['Date'] >= '2020-02-25'] # As cases ae reported in high number after 25th Feb,2020



fig = px.line(df, x='Date', y ='TargetValue',title ="fatalities with date for 'United Arab Emirates'")

fig.show()


#Creating Features from date columns



def date_feature(df):

  df['day'] = df['Date'].dt.day

  df['month'] = df['Date'].dt.month

#   df['dayofweek'] = df['Date'].dt.dayofweek  

#   df['weekofyear'] = df['Date'].dt.weekofyear #these are not selected as they dont give good result -reults were checked

#   df['quarter'] = df['Date'].dt.quarter



  return df

  

train = date_feature(train)

test = date_feature(test)

train
# dropping date column



train.drop(['Date'],axis =1, inplace =True)

test.drop(['Date'],axis =1, inplace =True)
train.columns
# Rearranging columns of train



train = train [['Id', 'Country_Region', 'Population','day', 'month','Weight','Target', 'TargetValue']]

# Rearranging columns of test



test = test [['ForecastId','Country_Region', 'Population','day', 'month','Weight','Target']]



train
country_train = set(train['Country_Region']) #unique countries in train dataset

country_test = set(test['Country_Region']) #unique countries in test dataset



country_list = [i for i in country_train if i in country_test]



print('no. of unique countries in train dataset = ', len(country_train),'\n','no. of unique countries in train dataset = ',len(country_test))

print('no. of unique countries after varification =', len(country_list))
target_train = set(train['Target'])

target_test = set(test['Target'])



target_list = [i for i in target_train if i in target_test]



print('no. of unique Target values in train dataset = ', len(target_train),'\n','no. of unique Target values in train dataset = ',len(target_test))

print('no. of unique Target values after varification =', len(target_list))
# encoding target values 



combine = [train,test]

for dataset in combine:

    dataset['Target'] = dataset['Target'].map({'ConfirmedCases':0,'Fatalities':1}).astype(int)

train
#Encoding Country names



combine = [train,test]

country = train['Country_Region'].unique()

num = [item for item in range(1,len(country)+1)]

country_num = dict(zip(country,num))

for dataset in combine:

    dataset['Country_Region'] = dataset['Country_Region'].map(country_num).astype(int)



train
#Removing id from train dataset

id_train = train.pop('Id')

train
# for test dataset



id_test = test.pop('ForecastId')

test
# Spliting into X and y 



y = train.pop('TargetValue')

X = train

X
# Spliting into train and test 



from sklearn.model_selection import train_test_split



X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.10,random_state =7)

X_train
X_test
print('X_train shape : ',X_train.shape, '\n','X_test shape : ',X_test.shape)
print('y_train shape : ',y_train.shape, '\n','y_test shape : ',y_test.shape)
col = X_train.columns
# # Standardising for faster convergence



# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler()



# X_train[col] = scaler.fit_transform(X_train[col])





# X_train
# X_test[col] = scaler.transform(X_test[col])

# X_test
# # Scaling test data set



# test[col] = scaler.transform(test[col])

# test
# Searching for best parameters by Gridsearch



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold





# Hyperparameter tuning for random forest



# param_rf = {

#     'max_depth': [8,10],

#     'min_samples_leaf': range(50, 450, 50),

#     'min_samples_split':range(50, 300, 50),

#     'n_estimators': [100,150,200],

    

# }



# rf = RandomForestRegressor(n_jobs=-1, max_features='auto',random_state=105)



# folds= KFold(n_splits = 3, shuffle = True, random_state = 90)



# grid_rf = GridSearchCV(estimator = rf, param_grid = param_rf, 

#                           cv = folds, n_jobs = -1,verbose = 1,scoring = 'r2')





# # Fitting

# grid_rf.fit(X_train, y_train)
# #best params

# grid_rf.best_params_
# Random forest



rf = RandomForestRegressor(n_jobs = -1,random_state=7)



rf.fit(X_train,y_train)
#Predicting



y_train_pred = rf.predict(X_test)

pd.DataFrame({'y_train_test':y_test, 'y_train_pred': y_train_pred})
# importing metrics



from sklearn.metrics import r2_score



r2_score(y_test,y_train_pred)

#Predicting on test data for submission



test_pred = rf.predict(test)

test_pred
# #Hyperparameters tuning



# hyper={'learning_rate': [0.1,0.2,0.3,0.5],

#           'n_estimators':[100,500,1000,1500,2000],

#           'subsample':[0.3,0.50,.75],

#           'reg_alpha':[0,1]

#           }

# xgb=XGBRegressor(max_depth=20,tree_method='gpu_hist', gpu_id=0)



# folds= KFold(n_splits=10,shuffle=True,random_state=100)



# xcv=GridSearchCV(estimator=xgb,

#                     param_grid=hyper,

#                     cv=folds,

#                     verbose=1,

#                     n_jobs=-1,

#                     return_train_score=True,

#                     scoring='neg_mean_absolute_error'

#                     )

# xcv.fit(X_train,y_train)
# Using XGboost



# trying with xgboost



xgb=XGBRegressor(max_depth=20,random_state=7,learning_rate=0.3,n_estimators=2000,n_jobs = -1,tree_method='gpu_hist', gpu_id=0)





xgb.fit(X_train,y_train)

y_pred_xgb = xgb.predict(X_test)



pd.DataFrame({'y_train_test':y_test, 'y_train_pred': y_pred_xgb})
#Predicting on test data for submission



test_predx = xgb.predict(test)

test_predx
#Creatin submission file xgb



sub = pd.DataFrame({'Id': id_test , 'TargetValue': test_predx})

sub
# #Creatin submission file rf



# sub = pd.DataFrame({'Id': id_test , 'TargetValue': test_pred})

# sub
m=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

n=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
m.columns = ['Id' , 'q0.05']

n.columns = ['Id' , 'q0.5']

q.columns = ['Id' , 'q0.95']
m = pd.concat([m,n['q0.5'] , q['q0.95']],1)

m
id_list = []

variable_list = []

value_list = []

for index, row in m.iterrows():

  id_list.append(row['Id'])

  variable_list.append('q0.05')

  value_list.append(row['q0.05'])



  id_list.append(row['Id'])

  variable_list.append('q0.5')

  value_list.append(row['q0.5'])



  id_list.append(row['Id'])

  variable_list.append('q0.95')

  value_list.append(row['q0.95'])



sub = pd.DataFrame({'Id':id_list, 'variable': variable_list, 'value':value_list})

sub
sub = sub.astype({'Id':int})

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub