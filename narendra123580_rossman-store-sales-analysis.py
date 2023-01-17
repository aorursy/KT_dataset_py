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
store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

data=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

submission=pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
data.head()
store.head()
print(data.shape)

print(store.shape)

print(test.shape)

print(submission.shape)
test.head()
data.dtypes  # date and stateholidays as categorical columns
#lets dive into the statistical description of data

data.describe(include='object')  # we have 942 similar dates and 5 similar etries in state holidays
data.describe()
#we have sales and customer column in which we have to focus more on the customer segment as this will also be one of the major factor affecting the sales 



data.describe()[['Sales','Customers']].loc['max']
data.describe()[['Sales','Customers']].loc['mean']
data.Store.nunique()



data.Store.value_counts().head(20).plot.bar()

data.Store.value_counts().tail(20).value_counts()



data.Store.value_counts()
# lets check for the store to be open

data.Open.value_counts()
#record of dayof week for the store

data.DayOfWeek.value_counts()



#record for the store

data.groupby(['DayOfWeek'])[['Sales']].mean()  # the first day of the week ,the sales is higher
#checking the null values

data.isnull().sum()

store.isnull().sum()   # the store data has more null values so for furthur imputations we have to fill it 

test.isnull().sum()   # there is some missing values in test data too
# convert datetime column to date

data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')
# lets take one store and make visualisations to  se the pattern 

store_id=data.Store.unique()[0]

print(store_id)



store_rows=data[data['Store']==store_id]

store_rows.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(10,8))



#plotting the 1day sales for the selected store_id over past 2 years before 2015
#carry on our analysis over the picked store



store_rows[store_rows['Sales']==0]  # showing the analysis of no sales for the store 1
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

store_test_rows=test[test['Store']==store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_test_rows['Open'].value_counts() # the shop will be closed(7days )means the sales will be zero for the given days 

                                       #lets proceed ahead to the exploration for predictig the sales for the dates on test data
store_rows['Sales'].plot.hist()



#it is slightly skewed  in target column means there are certain rows where slaes data is missing
store.head()

store[store['Store']==store_id].T  # we dont know what value imputaion we can do to fill the missinng values so we are takinng a single store and anlaysing the data
store[~store['Promo2SinceYear'].isna()].iloc[0]
store[~store['Promo2SinceYear'].isna()].iloc[0]
#missing value treatment

store.isna().sum()

# its obvious technically to say that the week that have zero promos so we can fill it with zero

store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

#technically its wrong to say promos missed for 2 years but for current scenrio we can fill it with mode



store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])



#its obvious that there is no competitor

store['CompetitionDistance']=store['CompetitionDistance'].fillna(0)

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
store['Promo2SinceYear'].mode()

data_merged=data.merge(store,on='Store',how='left')

print(data.shape)

print(data_merged.shape)

#just have a  look over the missing value after merging the data over store data/sometimes there is missing value after the merge

print(data_merged.isna().sum().sum())

#lets do the  decision tree regresssion  



#Encoding

#3 cat_cols,1date_col,rest are numerical

data_merged.dtypes



data_merged['day']=data_merged['Date'].dt.day

data_merged['month']=data_merged['Date'].dt.month

data_merged['year']=data_merged['Date'].dt.year



# will give the day of week extracion

# data_merged['dayofweek']=data_merged['Date'].dt.strftime('%A')
data_merged.dtypes

#stateholiday,assortment,promointerval,storetype-cat_cols

data_merged['StateHoliday'].unique()



data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)
#check the assortment columns for missing value imputation

data_merged['Assortment'].unique()
#data_merged.dtypes

#stateholiday,assortment,promointerval,storetype-cat_cols

data_merged['Assortment'].unique()



data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment']=data_merged['Assortment'].astype(int)
data_merged['StoreType']=data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType']=data_merged['StoreType'].astype(int)
map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)
#trainn and test validate



features=data_merged.columns.drop(['Sales','Date'])

from sklearn.model_selection import train_test_split

train_x,validate_x,train_y,validate_y=train_test_split(data_merged[features],np.log(data_merged['Sales']+1),test_size=0.2,random_state=1)



train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
#apply decision tree regressor



from sklearn.tree import DecisionTreeRegressor

    

model_dt=DecisionTreeRegressor(max_depth=11,random_state=1).fit(train_x,train_y)

validate_y_pred=model_dt.predict(validate_x)
validate_y_pred=model_dt.predict(validate_x)



from sklearn.metrics import mean_squared_error



def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe= np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe



validate_y_inv=np.exp(validate_y)-1

validate_y_pred_inv=np.exp(validate_y_pred)-1



rmse_val=np.sqrt(mean_squared_error(validate_y_inv,validate_y_pred_inv))

rmspe_val=rmspe(validate_y_inv,validate_y_pred_inv)

print(rmse_val,rmspe_val)
# from sklearn.model_selection import GridSearchCV



# parameters={'max_depth':list(range(5,20))}   # parmeters{'max_depth':list(range(5,20),'min_sample_split':[5,10,20])}

# base_model=DecisionTreeRegressor()

# cv_model=GridSearchCV(base_model,param_grid=parameters,cv=5,return_train_score=True).fit(train_x,train_y)
# cv_model.best_params_
# df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]

# plt.figure(figsize=(10,5))

# df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

# df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

# print(df_cv_results)

pd.Series(model_dt.feature_importances_,index=features)
#checking the most important feature for test data from the merged data



import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

yvalues=model_dt.feature_importances_

xvalues=features

plt.bar(xvalues,yvalues)

plt.xticks(rotation=90,color='red')

plt.show()
#to check the importance of each column over the merged data on sales (target variable)

data_merged.corr().loc['Sales'].sort_values(ascending=False)
#we rae finding the average number of customers as it will help in addition of customers  column in test data will help in prediction of customers

store_avg_cust=data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)
test1=test.merge(store_avg_cust,on='Store',how='left')



test1.shape,test.shape
test_merged=test1.merge(store,on='Store',how='left')
test_merged['Open']=test_merged['Open'].fillna(1)

test_merged['Date']=pd.to_datetime(test_merged['Date'],format='%Y-%m-%d')

test_merged['day']=test_merged['Date'].dt.day

test_merged['month']=test_merged['Date'].dt.month

test_merged['year']=test_merged['Date'].dt.year





test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})

test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)

test_merged.isna().sum()
test_merged['Assortment']=test_merged['Assortment'].map({'a':1,'b':2,'c':3})

test_merged['Assortment']=test_merged['Assortment'].astype(int)

test_merged['StoreType']=test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merged['StoreType']=test_merged['StoreType'].astype(int)

map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

test_merged['PromoInterval']=test_merged['PromoInterval'].map(map_promo)
test_pred=model_dt.predict(test_merged[features])

test_pred_inv=np.exp(test_pred)-1





submission_predicted=pd.DataFrame({'Id':test['Id'],'Sales':test_pred_inv})

submission_predicted
submission_predicted.to_csv('submission.csv',index=False)
def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe

validate_y_inv=np.exp(validate_y)-1

validate_y_pred_inv=np.exp(validate_y_pred)-1



rmse_val=np.sqrt(mean_squared_error(validate_y_inv,validate_y_pred_inv))

rmspe_val=rmspe(validate_y_inv,validate_y_pred_inv)

print(rmse_val,rmspe_val)
import numpy as np

from sklearn.model_selection import GridSearchCV

def get_rmspe_score(model,input_values,y_actual):

    y_predicted=model.predict(input_values)

    

    Y_actual=np.exp(y_actual)-1

    y_predicted=np.exp(y_predicted)-1

    score=rmspe(y_actual,y_predicted)

    return score

parameter={'max_depth':list(range(5,8))}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=parameter,cv=5,return_train_score=True,scoring=get_rmspe_score).fit(train_x,train_y)

pd.DataFrame(cv_model.cv_results_)[['params','mean_test_score','mean_train_score']]