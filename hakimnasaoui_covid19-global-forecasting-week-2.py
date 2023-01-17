DATASET_DIR = "../input/covid19-global-forecasting-week-2"

TRAIN_FILE = DATASET_DIR + "/train.csv"

TEST_FILE = DATASET_DIR + "/test.csv"
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from datetime import datetime
df_train = pd.read_csv(TRAIN_FILE)

df_train.head()
df_test = pd.read_csv(TEST_FILE)

df_test.head(5)
df_train.info()
df_train["Province_State"].isnull().sum()
df_train.isnull().sum()
df_train[df_train["Province_State"].notnull()]
# reformat dates

df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train.loc[:2,:'Date']
pd.plotting.register_matplotlib_converters()

grouped_data = df_train.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

grouped_data = grouped_data.sort_values(by=['Date'], ascending=True) 

grouped_data['ConfirmedCases'] = grouped_data['ConfirmedCases'].astype(int)

grouped_data['Fatalities'] = grouped_data['Fatalities'].astype(int)

grouped_data.head()
df_train['Date'] = pd.to_datetime(df_train['Date'])

grouped_data = df_train.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'max','Fatalities': 'max'})

grouped_data['ConfirmedCases'] = grouped_data['ConfirmedCases'].astype(int)

grouped_data['Fatalities'] = grouped_data['Fatalities'].astype(int)

display(grouped_data.head())
grouped_data = df_train.groupby('Date').sum()[['ConfirmedCases', 'Fatalities']]

sns.set(style = 'whitegrid')

grouped_data.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)

plt.bar(grouped_data.index, grouped_data['ConfirmedCases'],alpha=0.3,color='g')

plt.xlabel('Days', fontsize=15)

plt.ylabel('Number of cases', fontsize=15)

plt.title('Worldwide Covid-19 cases - Confirmed & Fatalities',fontsize=20)

plt.legend()

plt.show()
grouped_data = df_train[df_train['Country_Region']=='Morocco'].groupby('Date').sum()[['ConfirmedCases', 'Fatalities']]

sns.set(style = 'whitegrid')

grouped_data.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)

plt.bar(grouped_data.index, grouped_data['ConfirmedCases'],alpha=0.3,color='g')

plt.xlabel('Days', fontsize=15)

plt.ylabel('Number of cases', fontsize=15)

plt.title('Morocco Covid-19 cases - Confirmed & Fatalities',fontsize=20)

plt.legend()

plt.show()
grouped_data = df_train[(df_train['Country_Region']=='Morocco') & (df_train['ConfirmedCases']>0)].groupby('Date').sum()[['ConfirmedCases', 'Fatalities']]

sns.set(style = 'whitegrid')

grouped_data.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)

plt.bar(grouped_data.index, grouped_data['ConfirmedCases'],alpha=0.3,color='g')

plt.xlabel('Days', fontsize=15)

plt.ylabel('Number of cases', fontsize=15)

plt.title('Morocco Covid-19 cases - Confirmed & Fatalities',fontsize=20)

plt.legend()

plt.show()
grouped_data = df_train.groupby('Country_Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()

grouped_data = grouped_data.sort_values(by=['ConfirmedCases'], ascending=False) 

grouped_data['ConfirmedCases'] = grouped_data['ConfirmedCases'].astype(int)

grouped_data['Fatalities'] = grouped_data['Fatalities'].astype(int)

grouped_data.head(10)
grouped_data = df_train.groupby('Country_Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()

grouped_data_sort_confirmed_cases = grouped_data.sort_values(by=['ConfirmedCases', 'Fatalities'], ascending=False)[:10] 

grouped_data_sort_fatalities = grouped_data.sort_values(by=['Fatalities', 'ConfirmedCases'], ascending=False)[:10] 
sns.barplot(x="ConfirmedCases", y="Country_Region", data=grouped_data_sort_confirmed_cases)

plt.xticks(rotation=45)

plt.xlabel("Confirmed Cases")

plt.ylabel("Country")

plt.title("Confirmed Cases By Countries")

plt.show()
sns.barplot(x="Fatalities", y="Country_Region", data=grouped_data_sort_fatalities)

plt.xticks(rotation=45)

plt.xlabel("Fatalities")

plt.ylabel("Country")

plt.title("Fatalities By Countries")

plt.show()
df_china = df_train[df_train['Country_Region'] == "China"]

df_china = df_china[df_china['Date'] == max(df_china['Date'])]

grouped_data = df_china.groupby('Province_State')['ConfirmedCases', 'Fatalities'].max().reset_index()



grouped_data_sort = grouped_data.sort_values(by=['ConfirmedCases', 'Fatalities'], ascending=False)[:10] 



fig = sns.barplot(x="ConfirmedCases", y="Province_State", data=grouped_data_sort)



plt.xticks(rotation=45)

plt.xlabel("Confirmed Cases")

plt.ylabel("Province")

plt.title("Confirmed Cases By Provinces In China")

plt.show()
df_china = df_train[df_train['Country_Region'] == "US"]

df_china = df_china[df_china['Date'] == max(df_china['Date'])]

grouped_data = df_china.groupby('Province_State')['ConfirmedCases', 'Fatalities'].max().reset_index()



grouped_data_sort = grouped_data.sort_values(by=['ConfirmedCases', 'Fatalities'], ascending=False)[:10] 



fig = sns.barplot(x="ConfirmedCases", y="Province_State", data=grouped_data_sort, label="Confirmed Fatalities In China")



plt.xticks(rotation=45)

plt.xlabel("Confirmed Cases")

plt.ylabel("State")

plt.title("Confirmed Cases By States In US")

plt.show()
train_data = df_train.copy()

# train_data.drop('Province_State',axis=1,inplace=True)



test_data = df_test.copy()

# test_data.drop('Province_State',axis=1,inplace=True)



train_data['Date'] = train_data['Date'].dt.strftime("%m%d").astype(int)

test_data['Date'] = test_data['Date'].dt.strftime("%m%d").astype(int)

train_data['Province_State'] = train_data['Province_State'].fillna("N/D")

test_data['Province_State'] = test_data['Province_State'].fillna("N/D")
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()



train_data['Province_State']=LE.fit_transform(train_data['Province_State'])

test_data['Province_State']=LE.transform(test_data['Province_State'])



train_data['Country_Region']=LE.fit_transform(train_data['Country_Region'])

test_data['Country_Region']=LE.transform(test_data['Country_Region'])
train_data.tail()
x_cols = ['Date','Province_State', 'Country_Region']

y_cols = ['ConfirmedCases', 'Fatalities']



train_data[x_cols].head(10)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV



model_dtr = DecisionTreeRegressor(max_depth=None) 

param_grid = {

    'n_estimators': [100, 200, 300],

    'learning_rate': [0.1, 0.01, 0.001]

             }

model_abr = AdaBoostRegressor(base_estimator=model_dtr)



model = RandomizedSearchCV(estimator = model_abr, param_distributions = param_grid,

                    n_iter = 100, cv =10, verbose=0, random_state=2020, n_jobs = -1)

model.fit(train_data[x_cols], train_data[y_cols[0]])

predictions1 = model.predict(test_data[x_cols])



model = RandomizedSearchCV(estimator = model_abr, param_distributions = param_grid,

                    n_iter = 100, cv = 10, verbose=0, random_state=2020, n_jobs = -1)

model.fit(train_data[x_cols], train_data[y_cols[1]])

predictions2 = model.predict(test_data[x_cols])
predictions1[:50]
predictions2[:50]
submission = pd.DataFrame({'ForecastId':test_data['ForecastId'],'ConfirmedCases':predictions1,'Fatalities':predictions2})

submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int)

submission['Fatalities'] = submission['Fatalities'].astype(int)

submission.head(50)
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)