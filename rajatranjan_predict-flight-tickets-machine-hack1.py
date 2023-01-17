# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/flight_ticket_participant_datasets-20190305t100527z-001/Flight_Ticket_Participant_Datasets"))



# Any results you write to the current directory are saved as output.
train1=pd.read_excel('../input/flight_ticket_participant_datasets-20190305t100527z-001/Flight_Ticket_Participant_Datasets/Data_Train.xlsx')

test1=pd.read_excel('../input/flight_ticket_participant_datasets-20190305t100527z-001/Flight_Ticket_Participant_Datasets/Test_set.xlsx')

s=pd.read_excel('../input/flight_ticket_participant_datasets-20190305t100527z-001/Flight_Ticket_Participant_Datasets/Sample_submission.xlsx')

train1.head()
# train1

train1.iloc[6474, 7] = "24h 5m"

test1.iloc[2660, 7] = "24h 5m"



train1.iloc[1478, 0]

train1.iloc[2618, 0] = "Jet Airways Business"

train1.iloc[5439, 0] = "Jet Airways Business"

train1.drop(index = 2924, inplace = True, axis = 0)
print(train1[train1['Price']<=60000].shape)

print(train1.shape)

train=train1.copy()

train=train[train['Price']<=60000]

train.Additional_Info.replace(['No Info'], ['No info'], inplace=True)

train['Route'].fillna('DEL → COK',inplace=True)

train['Total_Stops'].fillna('non-stop',inplace=True)

train=train[~train['Additional_Info'].isin(['Red-eye flight','1 Short layover','2 Long layover'])]

train["Airline"]=train.Airline.replace("Trujet", "SpiceJet")

print(train.shape)



# train=train1.copy()

train.describe(include='all').T

# train['Airline'].value_counts()

# np.where(train["Airline"] == "Multiple carriers Premium economy", "Multiple carriers", train["Airline"])

# np.where(train['Airline']=='Jet Airways Business', 'High', 'Low')

train['Additional_Info'].value_counts()
train['Airline'].value_counts()
test1.describe(include='all').T

# train['Airline'].value_counts()
# len(np.intersect1d(train['Duration'],test1['Duration']))

# np.setdiff1d(train['Duration'],test1['Duration'])
train=train[(train['Duration'].isin(test1['Duration'])) & (train['Route'].isin(test1['Route'])) & (train['Dep_Time'].isin(test1['Dep_Time'])) & (train['Arrival_Time'].isin(test1['Arrival_Time']))]
train.shape
train=pd.DataFrame(data=train.values,columns=train.columns).reset_index().drop('index',axis=1)
print(train.shape)

main_df=train.append(test1,ignore_index=True)
main_df.reset_index().drop('index',axis=1,inplace=True)
main_df.Airline.value_counts()

main_df.replace({'New Delhi': 'Delhi'},inplace=True)
# main_df[((main_df.Arrival_Time>'18:00') & (main_df.Arrival_Time>'18:00') )& ((main_df.Dep_Time>'18:00') & (main_df.Dep_Time>'18:00'))]




# Banglore Kolkata 1,879 3

# 'Cochin Delhi', 2705 5



#        'Kolkata Chennai', 1676 2

#     'Delhi Banglore', 2157 4

#     'Hyderabad Mumbai' 711 1



# travelmap={'Banglore Kolkata':3, 'Cochin Delhi':5, 'Delhi Banglore':4,

#        'Kolkata Chennai':2, 'Hyderabad Mumbai':1}

# (main_df['Destination']+' '+main_df['Source']).replace(travelmap)
from sklearn.preprocessing import MultiLabelBinarizer

def routeMap(val):

    val=val.split(" → ")

    val2=[i[:3] for i in val]

    return val2

def other_day_arr(val):

    if len(val.split(" "))>1:

        return 1

    else:

        return 0



def dura_hr(val):

    p=val.split(" ")[0]

    return int(p[:-1])



    

def dura_min(val):

    try:

        p=val.split(" ")[1]

    except:

        p='0h'

    return int(p[:-1])

def stop_count(val):

    if val=='non-stop':

        return 0

    else:

        return int(val.split(" ")[0])

    

def isweekday(v):

    if v in [5,6]:

        return 0

    else:

        return 1

    

def ismultiple(v):

    if v in ['Multiple carriers Premium economy','Multiple carriers']:

        return 1

    else:

        return 0

def isprem(v):

    if v.find('Premium')!=-1:

        return 1

    else:

        return 0

    

def time_bins(v):

    if v>='00:00' and v<'06:00':

        return 'Morning'

    elif v>='06:00' and v<'12:00':

        return 'Afternoon'

    elif v>='12:00' and v<'18:00':

        return 'Evening'

    elif v>='18:00' and v<='23:60':

        return 'Night'

    else:

        return 'Else'

    

df=main_df.copy()

df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])

df['day']=df['Date_of_Journey'].dt.day.astype(str)

df['month']=df['Date_of_Journey'].dt.month.astype(str)

df['week']=df['Date_of_Journey'].dt.week.astype(str)

df['weekday']=df['Date_of_Journey'].dt.weekday.astype(str)

df['isweekday']=df['Date_of_Journey'].dt.weekday.apply(isweekday)



df['Dest_Source']=df['Destination']+' '+df['Source']

print(df['Dest_Source'].unique())

# df['travel_mp']=df['Dest_Source'].replace(travelmap)

df['ismultipleCarr']=df['Airline'].apply(ismultiple)

df['isprem']=df['Airline'].apply(isprem)

df['Route_stops']=df['Route'].astype(str).apply(lambda x :len(x.split(" → ")))

df['dep_time_hour']=pd.to_datetime(df['Dep_Time']).dt.hour.astype(str)

df['dep_time_sec']=pd.to_datetime(df['Dep_Time']).dt.minute.astype(str)



df['arr_time_bins']=df['Arrival_Time'].apply(time_bins)

df['dept_time_bins']=df['Dep_Time'].apply(time_bins)

df['arr_time_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour.astype(str)

df['arr_time_sec']=pd.to_datetime(df['Arrival_Time']).dt.minute.astype(str)

df['arr_next_day']=df['Arrival_Time'].apply(other_day_arr)

# df['dur_min']=df['Duration'].apply(dura_min)

# df['dur_hr']=df['Duration'].apply(dura_hr)



df['Total_Stops'].fillna('non-stop',inplace=True)

df['stop_count']=df['Total_Stops'].apply(stop_count)

df["class"] = np.where(df['Airline']=='Jet Airways Business', 'High', 'Low')

df["meal"] = np.where(df['Additional_Info']=='In-flight meal not included', 'High', 'Low')

df["checkin"] = np.where(df['Additional_Info']=='No check-in baggage included', 'High', 'Low')

print(df.columns)





tr_route=df[df['Price'].isnull()==False]['Route'].apply(lambda x: " ".join(str(x).split(" → ")))

ts_route=df[df['Price'].isnull()==True]['Route'].apply(lambda x: " ".join(str(x).split(" → ")))

df['total']=df['Airline']+' '+df['Source']+' '+df['Destination']+' '+df['Total_Stops']+' '+df['Additional_Info']

add_info_tr=df[df['Price'].isnull()==False]['total'].values

add_info_ts=df[df['Price'].isnull()==True]['total'].values

df.drop(['total'],axis=1,inplace=True)



import seaborn as sns



# sns.heatmap(df[df['Price'].isnull()==False].corr())





df=pd.get_dummies(df,columns=['Date_of_Journey','Additional_Info', 'Airline', 'Destination', 'Duration', 'Route', 'Source','Dest_Source',

       'Total_Stops','Route_stops','arr_time_bins','dept_time_bins','weekday','class','meal','checkin','month','day','week','dep_time_hour','dep_time_sec'

                             ,'arr_time_hour','arr_time_sec'],drop_first=True)

# df.drop(['Date_of_Journey','Dep_Time','Duration','Arrival_Time','Total_Stops'],axis=1,inplace=True)

# ['Additional_Info', 'Airline', 'Arrival_Time', 

#        'Dep_Time', 'Destination', 'Duration', 'Route', 'Source',

#        'Total_Stops','Route_stops']

df.drop(['Dep_Time','Arrival_Time'],axis=1,inplace=True)

df['Price']









# from sklearn.preprocessing import MultiLabelBinarizer

# m=MultiLabelBinarizer()

# route_bin=pd.DataFrame(data=m.fit_transform(main_df['Route'].astype(str).apply(routeMap)),columns=m.classes_)

# merged = df.merge(route_bin, how='inner',left_index=True, right_index=True)

# merged.head()

# merged.drop(['Route'],axis=1,inplace=True)





df.head()
df_train = df[df['Price'].isnull()==False]

df_test = df[df['Price'].isnull()==True]

# df_test = df_test.drop(['Price'], axis =1)

print(df_train.shape,df_test.shape)
df['meal_Low'].value_counts()

# df.head()
# df_train = merged[merged['Price'].isnull()==False]

# df_test = merged[merged['Price'].isnull()==True]

# # df_test = df_test.drop(['Price'], axis =1)

# print(df_train.shape,df_test.shape)

import matplotlib.pyplot as plt

%matplotlib inline

df_train['Price'].plot()
# main_df

# maindf_train = main_df[main_df['Price'].isnull()==False].drop(['Price'],axis=1)

# maindf_test = main_df[main_df['Price'].isnull()==True].drop(['Price'],axis=1)

# # maindf_test = maindf_test.drop(['Price'], axis =1)

# y=main_df[main_df['Price'].isnull()==False]['Price'].astype(np.float64)

# print(maindf_train.shape,maindf_test.shape)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vec_cr = TfidfVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

v_train =vec_cr.fit_transform(tr_route)

v_test =vec_cr.transform(ts_route)



# vect_cr = TfidfVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

# vt_train =vec_cr.fit_transform(tr_route)

# vt_test =vec_cr.transform(ts_route)



vec_char = TfidfVectorizer(ngram_range=(1,7),stop_words="english", analyzer='char')

v_trainC =vec_char.fit_transform(tr_route)

v_testC =vec_char.transform(ts_route)





vec_ad = TfidfVectorizer(ngram_range=(1,7),stop_words="english", analyzer='word')

vad_train =vec_ad.fit_transform(add_info_tr)

vad_test =vec_ad.transform(add_info_ts)



# vec_adC = TfidfVectorizer(ngram_range=(3,16),stop_words="english", analyzer='char')

# vad_trainC =vec_adC.fit_transform(add_info_tr)

# vad_testC =vec_adC.transform(add_info_ts)
# vect_cr.vocabulary_

# v_testC
v_test
df_train.head()
df_train.drop(['Price'],axis=1).shape
print(df_train.shape,df_test.shape)
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((df_train.drop(['Price'],axis=1), v_train,v_trainC,vad_train)).tocsr()

final_features1 = sparse.hstack((df_test.drop(['Price'],axis=1), v_test,v_testC,vad_test)).tocsr()
final_features

# from sklearn.decomposition import TruncatedSVD

# t=TruncatedSVD(n_components=1000)

# fftr=t.fit_transform(final_features)
from sklearn.model_selection import train_test_split

import math

from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_squared_log_error

X=final_features

y=np.log1p(df_train['Price'].astype(np.float64))

# y=train['Fees']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994)
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,BayesianRidge

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.metrics import mean_squared_log_error

import math

def rmsle(real, predicted):

    real=np.expm1(real)

    predicted=np.expm1(predicted)

    return np.sqrt(mean_squared_log_error(real,predicted))

    

def rmsle_lgb(labels, preds):

    return 'rmsle', rmsle(preds,labels), False
# from sklearn.decomposition import TruncatedSVD

# t=TruncatedSVD(n_components=200)

# xtr=t.fit_transform(X_train)

# xts=t.transform(X_val)

from xgboost import XGBRegressor

# m=XGBRegressor(n_estimators=6000,learning_rate=0.03,random_state=1994)

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)],eval_metric='rmse', early_stopping_rounds=100,verbose=100)

# p1=m.predict(X_val)

# # m=Rid

# # print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# # print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,p1))


m=LGBMRegressor(n_estimators=4000,learning_rate=0.05,random_state=1994)

# m=RidgeCV(cv=4)

m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)],eval_metric=rmsle_lgb, early_stopping_rounds=100,verbose=100)

p=m.predict(X_val)

# m=Rid

# print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# print(np.sqrt(mean_squared_log_error(y_val.values,p)))

print(rmsle_lgb(y_val.values,p))
# from sklearn_pandas import DataFrameMapper, FunctionTransformer, CategoricalImputer

# from sklearn.preprocessing import LabelEncoder, LabelBinarizer, KBinsDiscretizer

# from sklearn.impute import SimpleImputer

# from sklearn.base import TransformerMixin



# from sklearn.model_selection import train_test_split

# from sklearn.model_selection import StratifiedShuffleSplit



# import xgboost as xgb

# import lightgbm as lgb

# from sklearn.ensemble import GradientBoostingRegressor



# from sklearn.metrics import mean_squared_error

# from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import GridSearchCV

# import lightgbm as lgb

# lgbm_model = lgb.LGBMRegressor()

# lgbm_model.fit(X_train.astype(float), y_train.astype(float))



# print( np.sqrt( -cross_val_score(lgbm_model, X_train.astype(float), y_train, cv = 5, scoring = "neg_mean_squared_error") ) )

# np.sqrt(mean_squared_error(y_val, lgbm_model.predict(X_val)))
# lgbm_engine = lgb.LGBMRegressor()

# lgbm_params = {'max_depth' : [7,9], 'learning_rate' : [ 0.1,0.05], 'n_estimators': [2000]}



# lgbm_grid = GridSearchCV(lgbm_engine, lgbm_params, cv = 5, n_jobs = -1, verbose = 1)

# lgbm_grid.fit(X_train, y_train)

# print( np.sqrt( -cross_val_score(lgbm_grid.best_estimator_, X_train, y_train, cv = 5, scoring = "neg_mean_squared_error") ) )

# print( np.sqrt( mean_squared_error(y_val, lgbm_grid.predict(X_val)) ) )
# # for a, b, c in zip( lgbm_grid.cv_results_['params'], lgbm_grid.cv_results_['mean_test_score'], lgbm_grid.cv_results_['mean_train_score'] ):

# #     print(a, b, c)



# base_model = xgb.XGBRegressor(learning_rate=0.05)

# base_model.fit(X_train, y_train)

# print( np.sqrt( -cross_val_score(base_model, X_train, y_train, cv = 5, scoring = "neg_mean_squared_error") ) )
# xgb_engine = xgb.XGBRegressor(n_estimator=2000) #n_estimator not used

# xgb_params = {'max_depth' : np.arange(5, 12, 3), 'gamma' : [0.01, 1], 'learning_rate' : [0.1, 0.02, 0.05]}



# xgb_grid = GridSearchCV(xgb_engine, xgb_params, cv = 5, n_jobs = -1, verbose = 1)

# xgb_grid.fit(X_train, y_train)



# print( np.sqrt( -cross_val_score(xgb_grid.best_estimator_, X_train, y_train, cv = 5, scoring = "neg_mean_squared_error") ) )
# from mlxtend.regressor import StackingRegressor

# from mlxtend.data import boston_housing_data

# from sklearn.linear_model import LinearRegression

# from sklearn.linear_model import Ridge

# from sklearn.svm import SVR

# import matplotlib.pyplot as plt

# import numpy as np







# lgb = LGBMRegressor(n_estimators=4000,learning_rate=0.05,random_state=1994)

# rf = RandomForestRegressor(n_estimators=100)

# xgb=XGBRegressor(n_estimators=4000,learning_rate=0.05,random_state=1994)

# # ridge = Ridge(random_state=1)

# # svr_rbf = SVR(kernel='rbf')



# stregr = StackingRegressor(regressors=[rf,xgb ,rf,xgb], 

#                            meta_regressor=lgb,use_features_in_secondary=False,verbose=1)



# # Training the stacking classifier

# stregr.fit(X_train,y_train)

# p3=stregr.predict(X_val)

# # m=Rid

# # print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# # print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,p3))
from sklearn.ensemble import RandomForestRegressor

m=RandomForestRegressor(n_estimators=150)

# m=RidgeCV(cv=4)

m.fit(X_train,y_train)

p2=m.predict(X_val)

# m=Rid

# print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# print(np.sqrt(mean_squared_log_error(y_val.values,p)))

print(rmsle_lgb(y_val.values,p2))
print(rmsle_lgb(y_val,(p*0.7+p2*0.3)))

print(rmsle_lgb(y_val,(p+p2)/2))
# m=RidgeCV(cv=5,random_state=1994)

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)],eval_metric=rmsle_lgb, early_stopping_rounds=100,verbose=100)

# p=m.predict(X_val)

# # m=Rid

# # print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# # print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,p))
errlgb=[]

y_pred_totlgb=[]

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=25,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 4000,

                   'n_jobs': -1,'learning_rate':0.01,'random_state':1994}

    rf=LGBMRegressor(**lgbm_params)

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

         eval_metric=rmsle_lgb,

#            eval_metric='rmse',

#            categorical_feature=cat_cols,

         verbose=200

         , early_stopping_rounds=100

          )

    pr=rf.predict(X_test)

    print("errlgb: ",rmsle_lgb(y_test.values,pr)[1])

    

    errlgb.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(final_features1)

    y_pred_totlgb.append(p)
errxgb=[]

y_pred_totxgb=[]

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=25,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 6000,

                   'n_jobs': -1}

    rf=XGBRegressor(**lgbm_params)

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

#          eval_metric=rmsle_lgb,

           eval_metric='rmse',

#            categorical_feature=cat_cols,

         verbose=200

         , early_stopping_rounds=100

          )

    pr=rf.predict(X_test)

    print("errxgb: ",rmsle_lgb(y_test.values,pr)[1])

    

    errxgb.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(final_features1)

    y_pred_totxgb.append(p)
errrf=[]

y_pred_totrf=[]

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=25,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    rf=RandomForestRegressor(n_estimators=150,random_state=1994)

    rf.fit(X_train,y_train)

    pr=rf.predict(X_test)

    print("errxgb: ",rmsle_lgb(y_test.values,pr)[1])

    

    errrf.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(final_features1)

    y_pred_totrf.append(p)
print(np.mean(errlgb,0),np.mean(errrf,0),np.mean(errxgb,0))
# np.expm1(np.mean(y_pred_totlgb,0)*0.7+np.mean(y_pred_totrf,0)*0.3)
s['Price']=np.expm1((np.mean(y_pred_totlgb,0)+np.mean(y_pred_totrf,0)+np.mean(y_pred_totxgb,0))/3)

s.to_excel('MH-flight_price_sol25v24.xlsx',index=False)



s['Price']=np.expm1(np.mean(y_pred_totlgb,0))

s.to_excel('MH-flight_price_sol25LGBv24.xlsx',index=False)



s['Price']=np.expm1(np.mean(y_pred_totrf,0))

s.to_excel('MH-flight_price_sol25RFv24.xlsx',index=False)



s['Price']=np.expm1(np.mean(y_pred_totxgb,0))

s.to_excel('MH-flight_price_sol25XGBv24.xlsx',index=False)
# m=LGBMRegressor(n_estimators=1500,random_state=1994)

# # m=RidgeCV(cv=4)

# m.fit(X,y,eval_set=[(X, y.values)],eval_metric=rmsle_lgb, early_stopping_rounds=100,verbose=100)

# p=m.predict(final_features1)

# m=Rid

# print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,p))

s.head()
# s['Price']=np.expm1(p)

# s.to_excel('MH-flight_price_lgbmonly.xlsx',index=False)