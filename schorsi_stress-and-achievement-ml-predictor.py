import numpy as np

import pandas as pd

import os

wellbeing = pd.read_csv('/kaggle/input/lifestyle-and-wellbeing-data/Wellbeing_and_lifestyle_data.csv')

wellbeing = wellbeing.drop('Timestamp', axis=1)

wellbeing = wellbeing.drop([10005]) # This entry contained errors that needed to be corrected or erased

age_dict = {'Less than 20' : 1, '21 to 35' : 2, '36 to 50' : 3, '51 or more' : 4}

wellbeing['AGE'] = pd.Series([age_dict[x] for x in wellbeing.AGE], index=wellbeing.index)

gender_dict = {'Female' : 1, 'Male' : 0}

wellbeing['GENDER'] = pd.Series([gender_dict[x] for x in wellbeing.GENDER], index=wellbeing.index)

wellbeing['DAILY_STRESS'] = wellbeing['DAILY_STRESS'].astype(int)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

X = wellbeing.drop(['DAILY_STRESS', 'ACHIEVEMENT'], axis=1)

y = wellbeing[['DAILY_STRESS', 'ACHIEVEMENT']]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)

my_model = RandomForestRegressor(n_estimators=100).fit(train_X, train_y)

_ = X.columns
df_pred = {}

print('For each of the following enter an integer value representing your answer to the survey questions')

for col in _:

    print('\n',col, end='\t')

    df_pred[col] = [int(input())]



df = pd.DataFrame.from_dict(df_pred, orient='columns')
_ = my_model.predict(df)

print('Prediction of Daily Stress: ', float(_[:,0]), "\t\tPrediction for Achievement: ", float(_[:,1]))