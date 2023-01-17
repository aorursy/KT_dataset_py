!pip install pmdarima
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

from pmdarima.arima import auto_arima

import pmdarima as pm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
dates=["2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01","2020-04-02","2020-04-03","2020-04-04","2020-04-05","2020-04-06","2020-04-07"]

for date in dates:

    test_d1 = test[test.Date==date].index

    train_d1 = train[train.Date==date].index

    test.loc[test_d1,'ConfirmedCases'] = np.array(train.loc[train_d1,'ConfirmedCases'])

    test.loc[test_d1,'Fatalities'] = np.array(train.loc[train_d1,'Fatalities'])
for i in train['Country_Region'].unique():

    test_country=test[test['Country_Region']==i]

    alpha = 1.2#error

    beta = 0.5#trend

    df=pd.DataFrame()

    df['ConfirmedCases']=train[train['Country_Region']==i]['ConfirmedCases']

    df.index=pd.to_datetime(train[train['Country_Region']==i]['Date'],infer_datetime_format=True)

    model = ExponentialSmoothing(df,trend='add').fit(smoothing_level=alpha, smoothing_slope=beta)

    model1= pm.auto_arima(train[train['Country_Region']==i]['Fatalities'])

    index=test_country[test_country['ConfirmedCases'].isna()==True].index

    length=len(index)

    ypred=np.array(model.forecast(length))

    ypred1=np.array(model1.predict(length))

    j=0

    for k in index:

        fatal=ypred1[j]

        if(fatal<=0):

            fatal=0

        test_country.loc[k,'ConfirmedCases']=ypred[j]

        test_country.loc[k,'Fatalities']=fatal

        j=j+1

    test[test['Country_Region']==i]=test_country
submission=test[['ForecastId','ConfirmedCases','Fatalities']]

submission['ConfirmedCases']=submission['ConfirmedCases']

submission['Fatalities']=submission['Fatalities']
submission.to_csv('submission.csv',index=False)