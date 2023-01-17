import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import cufflinks as cf

cf.go_offline()

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn import metrics

import plotly.io as pio

pio.renderers.default='notebook'
train=pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
train.shape
train.head(3)
train.info()
test.shape
test.head(3)
test.info()
100*(train.isnull().sum())/(train.shape[0])
train.drop(['County','Province_State'],axis=1,inplace=True)
test.drop(['County','Province_State'],axis=1,inplace=True)
train['Date']=pd.to_datetime(train['Date'])
train['Month']=train['Date'].apply(lambda x :x.month)
train['Day']=train['Date'].apply(lambda x :x.day)
train.drop(['Date'],axis=1,inplace=True)
train.head(3)
test['Date']=pd.to_datetime(test['Date'])
test['Month']=test['Date'].apply(lambda x :x.month)
test['Day']=test['Date'].apply(lambda x :x.day)
test.drop(['Date'],axis=1,inplace=True)
test.head(3)
confirmed=train[train['Target']=='ConfirmedCases']
fig = px.treemap(confirmed, path=['Country_Region'], values='TargetValue',width=900, height=600)

fig.update_traces(textposition='middle center', textfont_size=15)

fig.update_layout(

    title={

        'text': 'Total Share of Worldwide COVID19 Confirmed Cases',

        'y':0.92,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
dead=train[train['Target']=='Fatalities']
fig = px.treemap(dead, path=['Country_Region'], values='TargetValue',width=900,height=600)

fig.update_traces(textposition='middle center', textfont_size=15)

fig.update_layout(

    title={

        'text': 'Total Share of Worldwide COVID19 Fatalities',

        'y':0.92,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
pop=train.groupby('Country_Region').max().sort_values(by='Population',ascending=False).head(20)
plt.figure(figsize=(10,6))

sns.barplot(x='Population',y=list(pop.index), data=pop)

plt.xlabel('Population',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Top 20 Countries in Population',fontsize=20)

plt.show()
top_confirmed=confirmed.groupby('Country_Region').sum().sort_values(by='TargetValue',ascending=False).head(20)
plt.figure(figsize=(10,6))

sns.barplot(x='TargetValue',y=list(top_confirmed.index), data=top_confirmed)

plt.xlabel('Number of Confirmed Cases',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Top 20 Countries in the Number of Confirmed Cases',fontsize=20)

plt.show()
top_dead=dead.groupby('Country_Region').sum().sort_values(by='TargetValue', ascending=False).head(20)
plt.figure(figsize=(10,6))

sns.barplot(x='TargetValue',y=list(top_dead.index), data=top_dead)

plt.xlabel('Number of Daed Cases',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Top 20 Countries in the Number of Dead Cases',fontsize=20)

plt.show()
confirmed['TargetValue'].sum()
dead['TargetValue'].sum()
plt.figure(figsize = (8,8))

plt.pie(x=[11528819,653271],labels=['Confirmed Cases','Fatalities'], autopct='%1.1f%%',pctdistance=0.7,labeldistance=1.1, explode=(0,0.2),shadow=True,startangle=90,colors= ['teal','paleturquoise'], data=train)

plt.title('\nPercentage Distribution of Fatalities and Confirmed Cases in the World\n',loc='center',fontsize=15)

plt.show()
x=confirmed['TargetValue'].max()

y=confirmed[confirmed['TargetValue']==confirmed['TargetValue'].max()]['Day'].values[0]

z=confirmed[confirmed['TargetValue']==confirmed['TargetValue'].max()]['Month'].values[0]

v=confirmed[confirmed['TargetValue']==confirmed['TargetValue'].max()]['Country_Region'].values[0]

print(f'The highest number of confirmed cases in the world is {x} which was recorded in day {y} of month {z} in {v}.')
confirmed_month_value=confirmed[['TargetValue','Month']]
sum_confirmed_month_value=confirmed_month_value.groupby('Month').sum()
sum_confirmed_month_value.columns=['Confirmed Cases']
dead_month_value=dead[['TargetValue','Month']]
sum_dead_month_value=dead_month_value.groupby('Month').sum()
sum_dead_month_value.columns=['Dead Cases']
pd.merge(sum_confirmed_month_value,sum_dead_month_value, on='Month').iplot(kind='bar',color=['yellowgreen','green'], title='Worldwide Confirmed/Death Cases Over Time')

plt.show()
india=train[train['Country_Region']=='India']
india.shape
india.head(3)
india.info()
conf_ind=india[india['Target']=='ConfirmedCases']
conf_ind.reset_index(inplace=True)
conf_ind.drop(['index'],axis=1,inplace=True)
conf_ind.head(2)
x=conf_ind['TargetValue'].max()

y=conf_ind[conf_ind['TargetValue']==conf_ind['TargetValue'].max()]['Day'].values[0]

z=conf_ind[conf_ind['TargetValue']==conf_ind['TargetValue'].max()]['Month'].values[0]

print(f'The highest number of confirmed cases in India is {x} which was recorded in day {y} of month {z}.')
dd_ind=india[india['Target']=='Fatalities']
dd_ind.reset_index(inplace=True)
dd_ind.drop(['index'],axis=1,inplace=True)
dd_ind.head(2)
x=dd_ind['TargetValue'].max()

y=dd_ind[dd_ind['TargetValue']==dd_ind['TargetValue'].max()]['Day'].values[0]

z=dd_ind[dd_ind['TargetValue']==dd_ind['TargetValue'].max()]['Month'].values[0]

print(f'The highest number of confirmed cases in India is {x} which was recorded in day {y} of month {z}.')
mon_conf_ind=conf_ind.groupby('Month').sum()
mon_dd_ind=dd_ind.groupby('Month').sum()
plt.figure(figsize=(18,6))



ax1=plt.subplot(1,2,1)

sns.barplot(x=mon_conf_ind.index, y='TargetValue', data=mon_conf_ind,palette='viridis')

plt.xlabel('Month',fontsize=12)

plt.ylabel('Number of Confirmed Cases',fontsize=12)

plt.title('Monthly Number of Confirmed Cases in India',fontsize=15)



ax2=plt.subplot(1,2,2,sharey=ax1)

sns.barplot(x=mon_dd_ind.index, y='TargetValue', data=mon_dd_ind,palette='viridis')

plt.xlabel('Month',fontsize=12)

plt.ylabel('')

ax2.set_yticks([])

plt.title('Monthly Number of Dead Cases in India',fontsize=15)



plt.show()
conf_ind['TargetValue'].sum()
dd_ind['TargetValue'].sum()
plt.figure(figsize = (8,8))

plt.pie(x=[276583,7745], labels=['Confirmed Cases','Fatalities'],pctdistance=0.7,labeldistance=1.1,autopct='%1.1f%%', explode=(0,0.3), shadow=True,colors=['c','pink'],startangle=90, data=india)

plt.title('\nPercentage Distribution of Fatalities and Confirmed Cases in India\n\n',loc='center',fontsize=15)

plt.show()
le1=LabelEncoder()
le1.fit(train['Country_Region'])
train['Encoded_Country']=le1.transform(train['Country_Region'])
le2=LabelEncoder()
le2.fit(train['Target'])
train['Encoded_Target']=le2.transform(train['Target'])
train.head(3)
y=train['TargetValue']

X=train[['Encoded_Country','Encoded_Target','Weight','Month','Day']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
rf=RandomForestRegressor()
rf.fit (X_train,y_train)
predictions_rf=rf.predict(X_test)
print('RMSE_Random Forest Regression=', np.sqrt(metrics.mean_squared_error(y_test,predictions_rf)))

print('R2 Score_Random Forest Regression=',metrics.r2_score(y_test,predictions_rf))
plt.figure(figsize=(8,6))

plt.plot(y_test,y_test,color='deeppink')

plt.scatter(y_test,predictions_rf,color='dodgerblue')

plt.xlabel('Actual Target Value',fontsize=15)

plt.ylabel('Predicted Target Value',fontsize=15)

plt.title('Random Forest Regressor (R2 Score= 0.95)',fontsize=14)

plt.show()
xgbr= xgb.XGBRegressor(n_estimators=800, learning_rate=0.01, gamma=0, subsample=.7,

                       colsample_bytree=.7, max_depth=10,

                       min_child_weight=0, 

                       objective='reg:squarederror', nthread=-1, scale_pos_weight=1,

                       seed=27, reg_alpha=0.00006, n_jobs=-1)
xgbr.fit(X_train,y_train)
prediction_xgbr=xgbr.predict(X_test)
print('RMSE_XGBoost Regression=', np.sqrt(metrics.mean_squared_error(y_test,prediction_xgbr)))

print('R2 Score_XGBoost Regression=',metrics.r2_score(y_test,prediction_xgbr))
plt.figure(figsize=(8,6))

plt.scatter(x=y_test, y=prediction_xgbr, color='dodgerblue')

plt.plot(y_test,y_test, color='deeppink')

plt.xlabel('Actual Target Value',fontsize=15)

plt.ylabel('Predicted Target Value',fontsize=15)

plt.title('XGBoost Regressor (R2 Score= 0.89)',fontsize=14)

plt.show()
le3=LabelEncoder()
le3.fit(test['Country_Region'])
test['Encoded_Country']=le3.transform(test['Country_Region'])
le4=LabelEncoder()
le4.fit(test['Target'])
test['Encoded_Target']=le4.transform(test['Target'])
test=test[['Encoded_Country','Encoded_Target','Weight','Month','Day']]
test.head(3)
pred=xgbr.predict(test)
submission=pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
submission.head(3)
test_1=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
output=pd.DataFrame({'Id':test_1['ForecastId'], 'TargetValue':pred})
a=output.groupby(['Id']).quantile(q=0.05).reset_index()

b=output.groupby(['Id']).quantile(q=0.5).reset_index()

c=output.groupby(['Id']).quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']
a['q0.5']=b['q0.5']

a['q0.95']=c['q0.95']
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].apply(lambda x: x.replace('q',''))
sub['var']=sub['variable'].apply(lambda x: str(x))
sub['id']=sub['Id'].apply(lambda x: str(x))
sub['ForecastId_Quantile']=sub['id']+'_'+sub['var']
sub.drop(['Id','variable','var','id'],axis=1,inplace=True)
sub.columns=['TargetValue','ForecastId_Quantile']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.head(3)
sub.to_csv("submission.csv",index=False)