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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
train.info()
submission.head()
for data in [train,test]:

    data['Date'] = data['Date'].apply(lambda date : pd.to_datetime(date))

    data['month'] = data['Date'].apply(lambda date: date.month)

    data['day'] = data['Date'].apply(lambda date: date.day)

    day_from_jan_first = data['Date'] - pd.to_datetime('2020-01-01')

    data['day_from_jan_first'] = day_from_jan_first.apply(lambda day_obj : day_obj.days + 1)
train['Country/Province'] = np.where(train['Province_State'].isna() == False, train['Country_Region'] + '/' + train['Province_State'], train['Country_Region'])

test['Country/Province'] = np.where(test['Province_State'].isna() == False, test['Country_Region'] + '/' + test['Province_State'], test['Country_Region'])
countries = train['Country/Province'].unique()
from xgboost import XGBRegressor

submission=pd.DataFrame(columns=submission.columns)

for country in countries:

    train_df=train[train['Country/Province']==country]

    x=train_df[['day_from_jan_first','month', 'day']]

    x_fat = train_df[['day_from_jan_first','month', 'day','ConfirmedCases']]

    y1=train_df[['ConfirmedCases']]

    y2=train_df[['Fatalities']]

    model_1=XGBRegressor()

    model_2=XGBRegressor()

    model_1.fit(x,y1)

    model_2.fit(x_fat,y2)

    test_df=test[test['Country/Province']==country]

    test_id=test_df['ForecastId'].values.tolist()

    test_x=test_df[['day_from_jan_first','month', 'day']]

    test_x_fat = test_df[['day_from_jan_first','month', 'day']]

    test_y1=model_1.predict(test_x)

    test_x_fat['ConfirmedCases']=np.rint(test_y1)

    test_y2=model_2.predict(test_x_fat)

    test_res=pd.DataFrame(columns=submission.columns)

    test_res['ConfirmedCases']=np.rint(test_y1)

    test_res['ForecastId']=test_id

    test_res['Fatalities']=np.rint(test_y2)

    submission=submission.append(test_res)
submission.head()
submission.to_csv('submission.csv',index=False)