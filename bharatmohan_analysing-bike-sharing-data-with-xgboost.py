import seaborn as sns
#!pip install pandas_profiling

import pandas_profiling as pf 
import pandas as pd



day_bike = pd.read_csv("../input/day-bike-csv/day.csv",parse_dates=[1])



#report=day_bike.profile_report(style={'full_width':True})
day_bike_visualistion=day_bike.copy()
report.to_file(output_file="D:/day_bike_eda1.html")

%matplotlib inline 
day_bike["mnth"] =day_bike['mnth'] .astype('category')

day_bike ["season"] =day_bike['season'] .astype('category')



day_bike ["weathersit"] =day_bike["weathersit"] .astype('category')



day_bike ['weekday'] = day_bike['weekday'] .astype('category')
day_bike['yr']=day_bike['yr'].astype('category')
day_bike['holiday']=day_bike['holiday'].astype('category')



day_bike['workingday']=day_bike['workingday'].astype('category')
day_bike.drop(columns=['instant','casual','registered','dteday'],inplace=True)


day_bike_visualistion ["season"] =day_bike_visualistion .season.map({1: 'Winter', 2 : 'Spring', 3 : 'Summer', 4 : 'Fall' }).astype('category')



day_bike_visualistion ["weathersit"] =day_bike_visualistion .weathersit.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " }).astype('category')



day_bike_visualistion ['weekday'] = day_bike_visualistion .weekday.map({0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}).astype('category')

day_bike_visualistion ['mnth'] =day_bike_visualistion .mnth.map({1:'January', 2:'February', 3:'March', 4:'April', 5:'May',

                                      6: 'June' , 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}).astype('category')

day_bike_visualistion['yr']=day_bike_visualistion['yr'].map({0:2011,1:2012}).astype('category')



day_bike_visualistion['holiday']=day_bike_visualistion['holiday'].astype('category')



day_bike_visualistion['workingday']=day_bike_visualistion['workingday'].astype('category')
day_bike_visualistion.drop(columns=['instant','casual','registered','dteday','yr'],inplace=True)
import numpy as np

numerical_columns = day_bike_visualistion.select_dtypes(include=[np.number]).columns
numerical_columns


sns.pairplot(day_bike_visualistion[numerical_columns],kind='scatter',diag_kind='kde')
import missingno as msno

msno.matrix(day_bike.sample(250))
msno.heatmap(day_bike[numerical_columns])
msno.bar(day_bike)
import seaborn as sns

sns.set(rc={'figure.figsize':(15,15)})
numerical_columns=day_bike.select_dtypes(include=[np.number]).columns

sns.boxplot(day_bike[numerical_columns[0]],orient='v')

sns.boxplot(day_bike[numerical_columns[1]],orient='v')





sns.boxplot(day_bike[numerical_columns[2]],orient='v')
sns.boxplot(day_bike[numerical_columns[3]],orient='v')
Q1 = day_bike[numerical_columns].quantile(0.25)

Q3 = day_bike[numerical_columns].quantile(0.75)

IQR = Q3 - Q1

print(IQR)

((day_bike[numerical_columns] < (Q1 - 1.5 * IQR)) |(day_bike[numerical_columns] > (Q3 + 1.5 * IQR))).sum()
day_bike = day_bike[~((day_bike[numerical_columns] < (Q1 - 1.5 * IQR)) |(day_bike[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
day_bike.head()
import matplotlib.pyplot as plt

sns.boxplot(data=day_bike[numerical_columns[1]],orient='v')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.xlabel('humidity')
sns.boxplot(data=day_bike[numerical_columns[2]],orient='v')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.xlabel('windspeed')
sns.set(rc={'figure.figsize':(8,8)})

data=day_bike_visualistion.groupby('season')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.season,y=data.cnt,order=list(data['season']))

sns.set(rc={'figure.figsize':(20,20)})

data=day_bike_visualistion.groupby('weathersit')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.weathersit,y=data.cnt,order=list(data['weathersit']))

sns.set(rc={'figure.figsize':(8,8)})

data=day_bike_visualistion.groupby('holiday')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.holiday,y=data.cnt,order=list(data['holiday']))

sns.set(rc={'figure.figsize':(8,8)})

data=day_bike_visualistion.groupby('workingday')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.workingday,y=data.cnt,order=list(data['workingday']))

sns.barplot(data=day_bike_visualistion, x='workingday', y= 'cnt')

ax=sns.barplot(data=day_bike_visualistion, x='workingday', y= 'cnt')
sns.set(rc={'figure.figsize':(8,8)})

data=day_bike.groupby('yr')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.yr,y=data.cnt,order=list(data['yr']))

sns.set(rc={'figure.figsize':(15,15)})

data=day_bike_visualistion.groupby('mnth')['cnt'].sum().reset_index().sort_values(by='cnt',ascending=False)

sns.barplot(x=data.mnth,y=data.cnt,order=list(data['mnth']))
sns.regplot(x=numerical_columns[0],y='cnt',data=day_bike)
sns.regplot(x=numerical_columns[1],y='cnt',data=day_bike)
sns.regplot(x=numerical_columns[2],y='cnt',data=day_bike)
sns.regplot(x=numerical_columns[3],y='cnt',data=day_bike)
correlation = day_bike[numerical_columns].corr()


k_value= 11

columns = correlation.nlargest(k_value,'temp')['temp'].index

print(columns)

cmap1 = np.corrcoef(day_bike[columns].values.T)

f_a , axes = plt.subplots(figsize = (14,12))

sns.heatmap(cmap1, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = columns.values ,annot_kws = {'size':12},yticklabels = columns.values)
day_bike.drop(columns=['atemp'],inplace=True)
target=day_bike['cnt']

day_bike.drop(columns=['cnt'],inplace=True)
day_bike.columns
columns=[i for i in day_bike.columns if str(day_bike[i].dtype)=='category']
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(day_bike, target, test_size=0.33, random_state=123)


data=day_bike

selected_columns=day_bike.columns

import statsmodels.api as sm

def backwardElimination(x, Y, sl, columns):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(Y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j, 1)

                    columns = np.delete(columns, j)

                    

    regressor_OLS.summary()

    return x, columns

SL = 0.05

data_modeled, selected_columns = backwardElimination(Xtrain.iloc[:,:].astype('float').values, ytrain.values, SL, selected_columns)
print(list(selected_columns))

print('total_columns:',list(Xtrain.columns))
selected_columns_for_regression=selected_columns
#!pip install boruta
from sklearn.ensemble import RandomForestRegressor

import xgboost

from boruta import BorutaPy
rf = RandomForestRegressor()
feat_selector = BorutaPy(rf)



feat_selector.fit(Xtrain.astype('float').values,ytrain)
columns_imp=[]

for i in range(0,len(Xtrain.columns)):

    if(feat_selector.support_[i]):

        columns_imp.append(Xtrain.columns[i])

columns_imp
selected_column_xgb=columns_imp
data_regression=day_bike[selected_columns_for_regression]

data_regression=pd.get_dummies(columns=[i for i in data_regression.columns if str(data_regression[i].dtype)=='category'],data=data_regression,drop_first=True)
data_xgb_regression=day_bike[selected_column_xgb]

data_xgb_regression=pd.get_dummies(columns=[i for i in data_xgb_regression.columns if str(data_xgb_regression[i].dtype)=='category'],data=data_xgb_regression,drop_first=True)



Xtrain_r, Xtest_r, ytrain_r, ytest_r= train_test_split(data_regression, target, test_size=0.33, random_state=123)

    
Xtrain_x, Xtest_x, ytrain_x, ytest_x= train_test_split(data_xgb_regression, target, test_size=0.33, random_state=42)

    


import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

model = sm.OLS(ytrain_r.iloc[:].astype(float), Xtrain_r.iloc[:].astype(float)).fit()
print(model.summary())
selected_columns_for_regression
preds=model.predict(Xtest_r)

dataframe=pd.DataFrame({'actual':ytest_r,'predicted':(preds)})

dataframe


#Function for Mean Absolute Percentage Error

def MAPE(y_actual,y_pred):

    mape = np.mean(np.abs((y_actual - y_pred)/y_actual))

    return mape
MAPE(ytest_r,preds)
SS_Residual = sum((ytest_r-preds)**2)

SS_Total = sum((ytest_r-np.mean(ytest_r))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(ytest_r)-1)/(len(ytest_r)-Xtest_r.shape[1]-1)

print (r_squared, adjusted_r_squared)
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(ytest_r, preds))

rmse_train= np.sqrt(mean_squared_error(ytrain_r,model.predict(Xtrain_r)))

print("RMSE: %f" % (rmse))

print("RMSE: %f" % (rmse_train))
from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(Xtrain_r, ytrain_r)

MAPE(ytest_r,preds)
SS_Residual = sum((ytest_r-preds)**2)

SS_Total = sum((ytest_r-np.mean(ytest_r))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(ytest_r)-1)/(len(ytest_r)-Xtest_r.shape[1]-1)

print (r_squared, adjusted_r_squared)
MAPE(ytest_r,preds)
import xgboost

model = xgboost.XGBRegressor()
model
model.fit(Xtrain_x,ytrain_x)
preds=model.predict(Xtest_x)
SS_Residual = sum((ytest_x-preds)**2)

SS_Total = sum((ytest_x-np.mean(ytest_x))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(ytest_x)-1)/(len(ytest_x)-Xtest_x.shape[1]-1)

print (r_squared, adjusted_r_squared)
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(ytest_x, preds))

rmse_train= np.sqrt(mean_squared_error(ytrain_x,model.predict(Xtrain_x)))

print("RMSE: %f" % (rmse))

print("RMSE: %f" % (rmse_train))
MAPE(ytest_x,preds)
from sklearn.model_selection import cross_validate

#Additional scklearn functions



from sklearn.model_selection import GridSearchCV
import datetime

start_time  = datetime.datetime.now()

xgb_clf=xgboost.XGBRegressor()

parameters = {'n_estimators': [120, 100, 140], 'max_depth':[3,5,7,9],'n_rounds':[100,200,500,1000],'learning_rate':[0.05,0.08,0.1,0.2,0.3]}

grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1)

print(parameters)

grid_search.fit(Xtrain_x, ytrain_x)

print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters=grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))

end_time = datetime.datetime.now()

print ('Select Done..., Time Cost: %d' % ((end_time - start_time).seconds) )
