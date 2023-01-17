!pip install pmdarima
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pmdarima as pm

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
if(len(train[train.Date=="2020-04-01"])==0):

    dates=["2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31"]

    for date in dates:

        test_d1 = test[test.Date==date].index

        train_d1 = train[train.Date==date].index

        test.loc[test_d1,'ConfirmedCases'] = np.array(train.loc[train_d1,'ConfirmedCases'])

        test.loc[test_d1,'Fatalities'] = np.array(train.loc[train_d1,'Fatalities'])

else:

    dates=["2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01"]

    for date in dates:

        test_d1 = test[test.Date==date].index

        train_d1 = train[train.Date==date].index

        test.loc[test_d1,'ConfirmedCases'] = np.array(train.loc[train_d1,'ConfirmedCases'])

        test.loc[test_d1,'Fatalities'] = np.array(train.loc[train_d1,'Fatalities'])
for i in train['Country_Region'].unique():

    test_country=test[test['Country_Region']==i]

    model = pm.auto_arima(train[train['Country_Region']==i]['ConfirmedCases'])

    model1= pm.auto_arima(train[train['Country_Region']==i]['Fatalities'])

    index=test_country[test_country['ConfirmedCases'].isna()==True].index

    length=len(index)

    ypred=np.array(np.round(model.predict(length)))

    ypred1=np.array(np.round(model1.predict(length)))

    j=0

    for k in index:

        test_country.loc[k,'ConfirmedCases']=ypred[j]

        test_country.loc[k,'Fatalities']=ypred1[j]

        j=j+1

    test[test['Country_Region']==i]=test_country
submission=test[['ForecastId','ConfirmedCases','Fatalities']]

submission['ConfirmedCases']=submission['ConfirmedCases'].astype(int)

submission['Fatalities']=submission['Fatalities'].astype(int)
submission.to_csv('submission.csv',index=False)