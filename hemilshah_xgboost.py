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

import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, GridSearchCV

import lightgbm as lgb

import xgboost as xgb

from tqdm import tqdm



train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')
train['Date'] = pd.to_datetime(train['Date']).dt.strftime("%m%d").astype(int)

test['Date'] = pd.to_datetime(test['Date']).dt.strftime("%m%d").astype(int)
train['Province_State'] = train['Province_State'].fillna('unknown')

test['Province_State'] = test['Province_State'].fillna('unknown')
train['Province_State'] = train['Province_State'].astype('category')

train['Country_Region'] = train['Country_Region'].astype('category')



test['Province_State'] = test['Province_State'].astype('category')

test['Country_Region'] = test['Country_Region'].astype('category')
FEATURES = ['Date']

submission = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])



for i in tqdm(train.Country_Region.unique()):

    z_train = train[train['Country_Region'] == i]

    z_test = test[test['Country_Region'] == i]

    for k in z_train.Province_State.unique():

        p_train = z_train[z_train['Province_State'] == k]

        p_test = z_test[z_test['Province_State'] == k]

        x_train = p_train[FEATURES]

        y1 = p_train['ConfirmedCases']

        y2 = p_train['Fatalities']

        model = xgb.XGBRegressor(n_estimators=1300)

        model.fit(x_train, y1)

        ConfirmedCasesPreds = model.predict(p_test[FEATURES])

        model.fit(x_train, y2)

        FatalitiesPreds = model.predict(p_test[FEATURES])

        

        p_test['ConfirmedCases'] = ConfirmedCasesPreds

        p_test['Fatalities'] = FatalitiesPreds

        submission = pd.concat([submission, p_test[['ForecastId', 'ConfirmedCases', 'Fatalities']]], axis=0)
train_df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', index_col="Id")

test_df = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv', index_col="ForecastId")
y_train_cc = np.array(train_df['ConfirmedCases'].astype(int))

y_train_ft = np.array(train_df['Fatalities'].astype(int))

cols = ['ConfirmedCases', 'Fatalities']



full_df = pd.concat([train_df.drop(cols, axis=1), test_df])

index_split = train_df.shape[0]

full_df = pd.get_dummies(full_df, columns=full_df.columns)



x_train = full_df[:index_split]

x_test= full_df[index_split:]
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=1)

dt.fit(x_train,y_train_cc)

y_pred_cc = dt.predict(x_test)

y_pred_cc = y_pred_cc.astype(int)

y_pred_cc[y_pred_cc <0]=0
dt_f = DecisionTreeRegressor()

dt_f.fit(x_train,y_train_ft)
y_pred_ft = dt_f.predict(x_test)

y_pred_ft = y_pred_ft.astype(int)

y_pred_ft[y_pred_ft <0]=0

predicted_df_dt = pd.DataFrame([y_pred_cc, y_pred_ft], index = ['ConfirmedCases','Fatalities'], columns= np.arange(1, y_pred_cc.shape[0] + 1)).T
from sklearn.tree import DecisionTreeClassifier



dtcla = DecisionTreeClassifier()

# We train model

dtcla.fit(x_train, y_train_cc)

predictions = dtcla.predict(x_test)

dtcla.fit(x_train,y_train_ft)

predictions1 = dtcla.predict(x_test)
submission1 = pd.DataFrame({'ForecastId': test_df.index,'ConfirmedCases':predictions,'Fatalities':predictions1})
submission['ConfirmedCases']=np.round((submission['ConfirmedCases']+submission1['ConfirmedCases'])/2)

submission['Fatalities']=np.round((submission['Fatalities']+submission1['Fatalities'])/2)
submission.to_csv('submission.csv',index=False)