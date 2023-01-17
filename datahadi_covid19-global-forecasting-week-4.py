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
X_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

X_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')



X_train.rename(columns={'Country_Region':'Country'}, inplace=True)

X_test.rename(columns={'Country_Region':'Country'}, inplace=True)



X_train.rename(columns={'Province_State':'State'}, inplace=True)

X_test.rename(columns={'Province_State':'State'}, inplace=True)



X_train.Date = pd.to_datetime(X_train.Date, infer_datetime_format=True)



X_test.Date = pd.to_datetime(X_test.Date, infer_datetime_format=True)



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



X_xTrain = X_train.copy()



X_xTrain.State.fillna(EMPTY_VAL, inplace=True)

X_xTrain.State = X_xTrain.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTrain.loc[:, 'Date'] = X_xTrain.Date.dt.strftime("%m%d")

X_xTrain.Date  = X_xTrain.Date.astype(int)



X_xTest = X_test.copy()



X_xTest.State.fillna(EMPTY_VAL, inplace=True)

X_xTest.State = X_xTest.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTest.loc[:, 'Date'] = X_xTest.Date.dt.strftime("%m%d")

X_xTest.Date  = X_xTest.Date.astype(int)
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



X_xTrain.Country = le.fit_transform(X_xTrain.Country)

X_xTrain.State = le.fit_transform(X_xTrain.State)



X_xTest.Country = le.fit_transform(X_xTest.Country)

X_xTest.State = le.fit_transform(X_xTest.State)
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn import preprocessing



le = preprocessing.LabelEncoder()



from xgboost import XGBRegressor

#from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor



countries = X_xTrain.Country.unique()
xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



y1_xTrain = X_xTrain.loc[:, 'ConfirmedCases']

y2_xTrain = X_xTrain.loc[:, 'Fatalities']

X_xTest_CS_Id = X_xTest.loc[:, 'ForecastId']



X_xtrain_Cs=X_xTrain.drop(columns=['Id','ConfirmedCases','Fatalities'])

X_xtest_Cs=X_xTest.drop(columns=['ForecastId'])



#xmodel1 = XGBRegressor(n_estimators=1000)

#xmodel1 = DecisionTreeClassifier()

xmodel1 = DecisionTreeRegressor()

xmodel1.fit(X_xtrain_Cs, y1_xTrain)

y1_xpred = xmodel1.predict(X_xtest_Cs)

        

#xmodel2 = XGBRegressor(n_estimators=1000)

#xmodel2 = DecisionTreeClassifier()

xmodel2 = DecisionTreeRegressor()

xmodel2.fit(X_xtrain_Cs, y2_xTrain)

y2_xpred = xmodel2.predict(X_xtest_Cs)
#from sklearn.linear_model import ElasticNet



#best_alpha=100

#best_iter=3500

#final_reg_confirmed = ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_iter)

#final_reg_confirmed.fit(X_xtrain_Cs, y1_xTrain)

#y1_xpred=final_reg_confirmed.predict(X_xtest_Cs)



#best_alpha=100

#best_iter=3500

#final_reg_fatal = ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_iter)

#final_reg_fatal.fit(X_xtrain_Cs, y2_xTrain)

#y2_xpred=final_reg_fatal.predict(X_xtest_Cs)
xdata = pd.DataFrame({'ForecastId': X_xTest_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})

xout = pd.concat([xout, xdata], axis=0)
xout.ForecastId = xout.ForecastId.astype('int')

xout.ConfirmedCases = xout.ConfirmedCases.astype('int')

xout.Fatalities = xout.Fatalities.astype('int')



xout.tail()

xout.to_csv('submission.csv', index=False)