import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import plotly.express as px

from sklearn.model_selection import train_test_split,validation_curve

from sklearn.ensemble import RandomForestRegressor

%matplotlib inline
train_df = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test_df = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
train_df.head()
train_df.shape
train_df.info()
train_df.tail()
l = train_df[['Country_Region','TargetValue']].groupby(['Country_Region'], as_index = False).sum().sort_values(by = 'TargetValue',ascending=False)

w = pd.DataFrame(l)

data1 = l.head(15)
fig = px.bar(data1,y = 'Country_Region',x = 'TargetValue',orientation = 'h',color = 'TargetValue')

fig.show()
print(train_df[['Population','TargetValue']].groupby(['Population'], as_index = False).mean().sort_values(by = 'TargetValue',ascending=False))
last_date = train_df.Date.max()

df_countries = train_df[train_df['Date']==last_date]

df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()

df_countries = df_countries.nlargest(10,'TargetValue')

df_trend = train_df.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()

df_trend = df_trend.merge(df_countries, on='Country_Region')

df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)

px.line(df_trend, x='Date', y='Cases', color='Country')
q = train_df[['Date','TargetValue']].groupby(['Date'], as_index = False).sum().sort_values(by = 'TargetValue',ascending=False)

#q = pd.DataFrame(q).head(30)

fig = px.bar(q,x = 'Date',y = 'TargetValue')

fig.show()
test_df['date_1'] = pd.to_datetime(test_df['Date'])

train_df['date_1'] = pd.to_datetime(train_df['Date'])
test_df['month'] = 0

list1=[]

for i in test_df['date_1']:

    list1.append(i.month)

test_df['month'] = list1



train_df['month'] = 0

list1=[]

for i in train_df['date_1']:

    list1.append(i.month)

train_df['month'] = list1
test_df['date'] = 0

list1=[]

for i in test_df['date_1']:

    list1.append(i.day)

test_df['date'] = list1



train_df['date'] = 0

list1=[]

for i in train_df['date_1']:

    list1.append(i.day)

train_df['date'] = list1
plt.figure(figsize =(10,10))

sea.heatmap(train_df.corr(),annot=True)
train_df = train_df.drop(['Date'],axis=1)

test_df = test_df.drop(['Date'],axis=1)
train_df = train_df.drop(['date_1'],axis=1)

test_df = test_df.drop(['date_1'],axis=1)
train_df = train_df.drop(['County'],axis=1)

test_df = test_df.drop(['County'],axis=1)
train_df = train_df.drop(['Province_State'],axis=1)

test_df = test_df.drop(['Province_State'],axis=1)
target_dict = {'ConfirmedCases':0,'Fatalities':1}

combine = [train_df,test_df]

for dataset in combine:

    dataset['Target'] = dataset['Target'].map(target_dict).astype(int)
combine = [train_df,test_df]

country = train_df['Country_Region'].unique()

len(country)

num = [item for item in range(1,188)]

country_num = dict(zip(country,num))

for dataset in combine:

    dataset['Country_Region'] = dataset['Country_Region'].map(country_num).astype(int)
train_df = train_df.drop(['Id'],axis=1)
test_df = test_df.drop(['ForecastId'],axis=1)
test_df.head()
train_df.head()
x = train_df.drop(['TargetValue'],axis=1)

y = train_df['TargetValue']
train_x,validate_x,train_y,validate_y = train_test_split(x,y,test_size=0.15,random_state=7)
reg = RandomForestRegressor()
reg.fit(train_x,train_y)
pred_y = reg.predict(validate_x)
validation_data = pd.DataFrame({'Predicted' : pred_y, 'Actual' : validate_y})
from sklearn.metrics import r2_score

print(r2_score(validate_y,pred_y))

test_df.head()
my_prediction = reg.predict(test_df)
my_prediction
test =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
output = pd.DataFrame({'Id': test.ForecastId  , 'TargetValue': my_prediction})
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()