import pandas as pd

import numpy as np
train = pd.read_csv('../input/covid-clean/clean_train.csv')
test = pd.read_csv('../input/covid-clean/clean_test.csv')
submission = pd.read_csv('../input/covid-clean/submission.csv')
countries = train['Country/Province'].unique()
countries[0]
from xgboost import XGBRegressor

submission=pd.DataFrame(columns=submission.columns)

for country in countries:

    train_df=train[train['Country/Province']==country]

    x=train_df[['day_from_jan_first','month', 'day', 'days_from_firstcase']]

    x_fat = train_df[['day_from_jan_first','month', 'day', 'days_from_firstcase','ConfirmedCases']]

    y1=train_df[['ConfirmedCases']]

    y2=train_df[['Fatalities']]

    model_1=XGBRegressor()

    model_2=XGBRegressor()

    model_1.fit(x,y1)

    model_2.fit(x_fat,y2)

    test_df=test[test['Country/Province']==country]

    test_id=test_df['ForecastId'].values.tolist()

    test_x=test_df[['day_from_jan_first','month', 'day', 'days_from_firstcase']]

    test_x_fat = test_df[['day_from_jan_first','month', 'day', 'days_from_firstcase']]

    test_y1=model_1.predict(test_x)

    test_x_fat['ConfirmedCases']=np.rint(test_y1)

    test_y2=model_2.predict(test_x_fat)

    test_res=pd.DataFrame(columns=submission.columns)

    test_res['ConfirmedCases']=np.rint(test_y1)

    test_res['ForecastId']=test_id

    test_res['Fatalities']=np.rint(test_y2)

    submission=submission.append(test_res)
submission.to_csv('submission.csv',index=False)