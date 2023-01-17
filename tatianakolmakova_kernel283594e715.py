import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import ShuffleSplit, cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_log_error

from sklearn.pipeline import Pipeline
train_df = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv', low_memory=False)

test_df = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv', low_memory=False)
train_df.head()
train_df.describe()
train_df.drop(["Province_State", "County"], inplace=True, axis=1)

train_df.Date = pd.to_datetime(train_df.Date).dt.strftime("%Y%m%d").astype(int)
train_df.head()
fig = px.pie(train_df[train_df.Target == 'ConfirmedCases'], values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')
fig = px.pie(train_df[train_df.Target == 'Fatalities'], values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')
fatalities_df = train_df[train_df.Target == 'Fatalities']

fatalities_df = fatalities_df[['Country_Region', 'TargetValue']].groupby(['Country_Region']).sum()
confirmed_df = train_df[train_df.Target == 'ConfirmedCases']

confirmed_df = confirmed_df[['Country_Region', 'TargetValue']].groupby(['Country_Region']).sum()
medical_care_2017_df = pd.read_excel('../input/who-health-2017/who_2017.xlsx')

medical_care_2017_df = medical_care_2017_df.rename(columns={'Countries': 'Country_Region'})
medical_care_2017_df.head()
info_fatalities_df = fatalities_df.merge(medical_care_2017_df, on='Country_Region', how='left')

info_fatalities_df.fillna(value=-1, inplace=True)
print(info_fatalities_df.corrwith(info_fatalities_df.TargetValue))
info_confirmed_df = confirmed_df.merge(medical_care_2017_df, on='Country_Region', how='left')

info_confirmed_df.fillna(value=-1, inplace=True)
print(info_confirmed_df.corrwith(info_confirmed_df.TargetValue))
train_df = train_df.merge(medical_care_2017_df, on='Country_Region', how='left')

train_df.fillna(value=-1, inplace=True)
def to_train(train_df):

    le = LabelEncoder()



    df = train_df.copy()

    df.Country_Region = le.fit_transform(df.Country_Region)

    df.Target = le.fit_transform(df.Target)

    

    X_train = df.drop(["Id", "TargetValue"], axis=1)

    Y_train = df.TargetValue

    return X_train, Y_train
def get_train_score(train_dt, model):

    X_train, Y_train = to_train(train_df)



    pipeline = Pipeline([

        ('scale' , StandardScaler()),

        ('model ', model)

    ])

    score = cross_val_score(

        pipeline, 

        X_train, Y_train, 

        cv=ShuffleSplit(n_splits=3), 

        scoring='neg_mean_squared_error'

    )

    return score
# get_train_score(info_confirmed_df, RandomForestRegressor()).mean()
# get_train_score(train_df, RandomForestRegressor()).mean()
test_df.drop(["Province_State", "County"], inplace=True, axis=1)

test_df.Date = pd.to_datetime(test_df.Date).dt.strftime("%Y%m%d").astype(int)

test_df = test_df.merge(medical_care_2017_df, on='Country_Region', how='left')

test_df.fillna(value=-1, inplace=True)

test_df.drop(['ForecastId'],axis=1,inplace=True)

test_df.index.name = 'Id'
pipeline = Pipeline([

    ('scale' , StandardScaler()),

    ('model ', RandomForestRegressor(n_jobs=-1))

])



le_region = LabelEncoder()

le_target = LabelEncoder()



X_train = train_df.drop(["Id", "TargetValue"], axis=1)

y_train = train_df.TargetValue



X_train.Country_Region = le_region.fit_transform(X_train.Country_Region)

X_train.Target = le_target.fit_transform(X_train.Target)



pipeline.fit(X_train, y_train)
X_test = test_df

X_test.Country_Region = le_region.transform(X_test.Country_Region)

X_test.Target = le_target.transform(X_test.Target)



y_pred = pipeline.predict(X_test)
test_df.head()
pred_list = [int(x) for x in y_pred]



output = pd.DataFrame({'Id': test_df.index, 'TargetValue': pred_list})



a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()



a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05'].clip(0,10000)

a['q0.5']=a['q0.5'].clip(0,10000)

a['q0.95']=a['q0.95'].clip(0,10000)



a['Id'] =a['Id']+ 1
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.head()
sub.to_csv("submission.csv",index=False)