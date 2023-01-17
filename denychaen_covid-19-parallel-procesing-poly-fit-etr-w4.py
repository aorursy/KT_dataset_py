import pandas as pd

import numpy as np

import gc

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train.rename(columns={'Country_Region':'Country','Province_State':'State','ConfirmedCases':'Confirmed'}, inplace=True)



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test.rename(columns={'Country_Region':'Country','Province_State':'State','ConfirmedCases':'Confirmed', 'ForecastId':'Id'},inplace=True)

train['Type']='train'

test['Type']='test'



test['Confirmed']=0

test['Fatalities']=0



print(train['Date'].min(),train['Date'].max())

print(test['Date'].min(),test['Date'].max())
import pandas as pd

import numpy as np

import os, gc





train['id_x']=train['Date'].astype(str).values+'_'+train['State'].astype(str).values+'_'+train['Country'].astype(str).values+'_'+train['Type'].astype(str).values

test['id_x']=test['Date'].astype(str).values+'_'+test['State'].astype(str).values+'_'+test['Country'].astype(str).values+'_'+test['Type'].astype(str).values



raw = pd.concat([train,test], axis=0, sort=False)

raw['Date'] = pd.to_datetime(raw["Date"])

raw.sort_values('Date',inplace=True)



raw.fillna(0, inplace=True)

Country_State = raw.Country+'_'+raw.State.astype(str)

raw['Country_State'] = Country_State

raw['Country_State_id'] = Country_State.astype('category').cat.codes

raw['Day']=raw['Date'].astype('category').cat.codes+1

raw.set_index('Country_State_id', inplace=True)

raw.Day=raw.Day.astype(np.int32)

raw.reset_index(inplace=True)

raw['Country_']=raw.Country.astype('category').cat.codes

raw['State_']=raw.State.astype('category').cat.codes



features = ['id_x','Day','Id','Country_State','Country_','State_','Country_State_id']



train = train.merge(raw[features], on=['id_x'], how='left')

test = test.merge(raw[features], on=['id_x'],  how='left')



print(train.shape, test.shape)
import os, gc

from multiprocessing.pool import ThreadPool

from sklearn.feature_selection import SelectFromModel

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import StandardScaler



os.environ['OMP_NUM_THREADS'] = '1'  

gc.enable()



features = ['id_x','Day']



X_train = [np.array(train[train.Country_State_id== x][features]) for x in list(train.Country_State_id.unique())]



X_test = [np.array(test[test.Country_State_id== x][features]) for x in list(train.Country_State_id.unique())]



y_target_c = [np.array(train[train.Country_State_id == x][['Confirmed']]) for x in list(train.Country_State_id.unique())]



y_target_f = [np.array(train[train.Country_State_id == x][['Fatalities']]) for x in list(train.Country_State_id.unique())]



poly=PolynomialFeatures(10)



out_ = pd.DataFrame({'id_x': [], 'Confirmed': [], 'Fatalities': []})



def fit_model(xtrain, xtest, ytrain, ytrain1, idx) -> np.array:

        

    X = xtrain[idx][:,1]

    x_test = xtest[idx][:,1]

    Y = ytrain[idx]

    Y1 = ytrain1[idx]

    

    X_transf=poly.fit_transform(X.reshape(-1,1))

    x_test_transf=poly.fit_transform(x_test.reshape(-1,1))

    

        

    f = linear_model.Lasso()

    model = ExtraTreesRegressor(n_estimators=300)

    sel_ = SelectFromModel(linear_model.Lasso(alpha=0.05))

    

    sel_.fit(X_transf,Y)

    if sum(sel_.get_support())>3:        

        model.fit(X_transf[:,sel_.get_support()],Y)

        conf_p = model.predict(x_test_transf[:,sel_.get_support()])

    else:

        f.fit(np.array(X_transf),Y)

        conf_p = f.predict(x_test_transf)

        



    sel_.fit(X_transf,Y1)    

    if sum(sel_.get_support())>3:        

        model.fit(X_transf[:,(sel_.get_support())],Y1)

        conf_f = model.predict(x_test_transf[:,(sel_.get_support())])

    else:

        f.fit(X_transf,Y1)

        conf_f = f.predict(x_test_transf)



    res = pd.DataFrame({'id_x':xtest[idx][:,0] , 'Confirmed': conf_p, 'Fatalities': conf_f})    

    return res



with ThreadPool(processes=4) as pool:

    args = [(X_train, X_test, y_target_c, y_target_f, idx) for idx in test.Country_State_id.unique()]

    out_ = pd.concat(pool.starmap(fit_model, args))

    out_ = test[['id_x']].merge(out_, on='id_x', how='left')

    pool.close()





sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

sub_new = sub[["ForecastId"]]

result = pd.concat([out_.reset_index().Confirmed,out_.reset_index().Fatalities,sub_new],axis=1)

result.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

result = result[['ForecastId','ConfirmedCases', 'Fatalities']]

result.to_csv("submission.csv",index=False)

result.head()