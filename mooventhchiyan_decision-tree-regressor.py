# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def read_all_data():

    data= pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv",low_memory=False)

    store=pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv",low_memory=False)

    test=pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv",low_memory=False)

    sample= pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv",low_memory=False)

    return data,test,store,sample
# Preprocessing for getting the datas ready for modelling

def calc_iqr(ps):

    q1= ps.quantile(0.25)

    q3= ps.quantile(0.75)

    IQR=q3-q1

    return IQR



def get_customers_info():

    data= pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv",low_memory= False)

    store_avg_cust= data.groupby(['Store']).agg(avg_cust=('Customers','mean'),

                                               median_cust=("Customers","median"),

                                               IQR_cust= ("Customers",calc_iqr)).reset_index()

    store_avg_cust_dow =data.groupby(['Store','DayOfWeek']).agg(avg_cust_dow=('Customers','mean'),

                                               median_cust_dow=("Customers","median"),

                                                IQR_cust= ("Customers",calc_iqr)).reset_index()

    store_avg_cust_promo =data.groupby(['Store','Promo']).agg(avg_cust_promo=('Customers','mean'),

                                               median_cust_promo=("Customers","median"),

                                                IQR_cust= ("Customers",calc_iqr)).reset_index()



    return store_avg_cust,store_avg_cust_dow,store_avg_cust_promo

store_avg_cust,store_avg_cust_dow,store_avg_cust_promo=get_customers_info()



def fill_store_data(store):

    store["CompetitionDistance"]= store["CompetitionDistance"].fillna(store['CompetitionDistance'].median())

    store["PromoInterval"] =store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

    store["CompetitionOpenSinceMonth"]= store['CompetitionOpenSinceMonth'].fillna(0)

    store["CompetitionOpenSinceYear"]= store["CompetitionOpenSinceYear"].fillna(store["CompetitionOpenSinceYear"].mode().iloc[0])

    store["Promo2SinceWeek"]= store['Promo2SinceWeek'].fillna(store["Promo2SinceWeek"].fillna(0))

    store["Promo2SinceYear"]= store['Promo2SinceYear'].fillna(store["Promo2SinceYear"].mode().iloc[0])

    return store

                                                             

def feature_engg(df,store):

    df['is_december']= df['month'].apply(lambda v:1 if v==12 else 0)

    df["is_state_holiday"]=df['StateHoliday'].apply(lambda v:0 if v==0 else 1)

    df["is_holiday"]=df["is_state_holiday"] * df['SchoolHoliday']

    df= pd.merge(left=df,right=store_avg_cust,on='Store',how='left')

    df= pd.merge(left=df,right=store_avg_cust_dow,on=['Store','DayOfWeek'],how='left')

    df= pd.merge(left=df,right=store_avg_cust_promo,on=['Store','Promo'],how='left')

    return df

      

    

def preprocess(df,store):

    store = fill_store_data(store)

    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

    df['day']  = df.Date.dt.day

    df['month']=df.Date.dt.month

    df['year']=df.Date.dt.year

    df['StateHoliday']=df['StateHoliday'].map({ "0":0,0:0,'a':1,'b':2,'c':3 })

    df= feature_engg(df,store)

    return df



 

calc_rmse= lambda mse:np.sqrt(mse)
data,test,store,sample= read_all_data()

data= preprocess(data,store)

data= data[data['Open']!=0]

test= preprocess(test,store)



data_dummies= pd.get_dummies(data.drop(['Customers','Date'], axis=1),drop_first=True)

test_dummies= pd.get_dummies(test.drop(['Date','Id'],axis=1), drop_first=True)

data_dummies.shape, test_dummies.shape
#np.setdiff1d(data_dummies.columns,test_dummies.columns)
target_col ="Sales"

input_cols= data_dummies.columns.drop([target_col])

X_train,X_validate,y_train,y_validate=train_test_split(data_dummies[input_cols],

                                                  data_dummies[target_col],

                                                  test_size=0.2,

                                                  random_state=1)

X_train.shape,y_train.shape,X_validate.shape,y_validate.shape
model= DecisionTreeRegressor(max_depth=10,random_state=1).fit(X_train,y_train)

y_validate_pred= model.predict(X_validate)

mse = mean_squared_error(y_validate,y_validate_pred)

rmse= calc_rmse(mse)

print(rmse)
y_test_pred= model.predict(test_dummies.fillna(0))

test['Sales']=y_test_pred

closed_dates_indexes= test[test['Open']==0].index

test.loc[closed_dates_indexes,'Sales']==0

test[['Id','Sales']].to_csv('submission.csv',index=False)
#df_feature_importance= pd.DataFrame(model.feature_importances_, index= X_train.columns,columns=["score"])

#df_feature_importance.sort_values('score',ascending=False).plot.bar(figsize=(20,4))
'''## Hyper Parameter Tuning

from sklearn.model_selection import GridSearchCV

base_model= DecisionTreeRegressor()

param_grid= {"max_depth": [7,9,11,13,15]}

tuner= GridSearchCV(estimator=base_model,cv=5,param_grid=param_grid,return_train_score=True).fit(X_train,y_train)'''
#tuner.best_params_
##df_cv= pd.DataFrame(tuner.cv_results_)

##df_cv[['param_max_depth','mean_test_score','rank_test_score']].sort_values('mean_test_score',ascending=False)
'''import seaborn as sns

import matplotlib.pyplot as plt

sns.lineplot(data=df_cv, x='param_max_depth',y='mean_train_score')

sns.lineplot(data=df_cv, x='param_max_depth',y='mean_test_score')

plt.legend(['Training Score','Testing Score'])'''