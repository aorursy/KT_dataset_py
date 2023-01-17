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
!pip install pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima import auto_arima    



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train.info()
test.info()
submission.info()
train['Country/Province'] = np.where(train['Province_State'].isna() == False, train['Country_Region'] + '/' + train['Province_State'], train['Country_Region'])

test['Country/Province'] = np.where(test['Province_State'].isna() == False, test['Country_Region'] + '/' + test['Province_State'], test['Country_Region'])
countries = train['Country/Province'].unique()

submission=pd.DataFrame(columns=submission.columns)

for country in countries:

    train_df = train[train['Country/Province'] == country]

    test_df = test[test['Country/Province'] == country]

    

########### Farecasting ConfirmedCases...........



    X_train_conf = train_df['ConfirmedCases'].values

    p,d,q = auto_arima(X_train_conf).order

    model_conf = SARIMAX(X_train_conf,order=(p,d,q),seasonal_order=(0,0,0,0))

    results = model_conf.fit()

    fcast_conf = results.predict(len(X_train_conf)-13,len(X_train_conf)+len(test_df)-14,typ='levels')





########### Farecasting Fatalities.............



    X_train_fat = train_df['Fatalities'].values

    p,d,q = auto_arima(X_train_fat).order

    model_fat = SARIMAX(X_train_fat,order=(p,d,q),seasonal_order=(0,0,0,0),initialization='approximate_diffuse')

    results = model_fat.fit()

    fcast_fat = results.predict(len(X_train_fat)-13,len(X_train_fat)+len(test_df)-14,typ='levels')



    test_id = test_df['ForecastId'].values.tolist()

    test_res = pd.DataFrame(columns=submission.columns)

    test_res['ForecastId'] = test_id

    test_res['ConfirmedCases'] = np.rint(fcast_conf)

    test_res['Fatalities'] = np.rint(fcast_fat)

    submission = submission.append(test_res)
submission.to_csv('submission.csv',index=False)