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
import numpy as np   #linear algebra(Scientific computing)

import pandas as pd  #data processing
## import python libraries for data visualization



import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
train.head()
train.isnull().any()
train_confm_date=train.groupby('Date')['ConfirmedCases','Fatalities'].sum()
train_confm_date.head()
plt.figure(figsize=(16,10))

train_confm_date.plot()

plt.title('globabally confirmed case & fatalities')

plt.xticks(rotation=60)
#From the above graph,we can see that 12th march onwards,the rate of confirmed cases increases tremendously world-wide 

train_confirm_country=train.groupby('Country/Region')['ConfirmedCases','Fatalities'].sum().reset_index().sort_values('ConfirmedCases',ascending=False)
train_confirm_country.head()
plt.figure(figsize=(12,6))

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['ConfirmedCases'][:10])

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatalities'][:10])

plt.legend(['Blue Color: Confirmed case','yellow Color:Fatality'])
#from above graph,we can see that highest no.of  confirmed cases has been registered in China and followed by Italy and Iran.
train_confirm_country['Fatality rate in %']=train_confirm_country['Fatalities']/train_confirm_country['ConfirmedCases']*100
#replacing infinite value with column(min)

train_confirm_country['Fatality rate in %'].replace([np.inf],min(train_confirm_country['Fatality rate in %']),inplace=True)
train_confirm_country.sort_values('Fatality rate in %',ascending=False).head(10)
#from the above table we found that Sudan is having high Fatality rate that is 10 death oout of 15 confirmed case
plt.figure(figsize=(12,6))

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatality rate in %'][:10])

df_top_10=train_confirm_country[:10]

df_top_10.head(5)
sns.barplot(y='Country/Region',x='Fatality rate in %',data=df_top_10)
#Observation: We see that the fatality rate in China(3.5%) is very less even though the no. of confirmed is very high.

#Itality is having Fatality rate of around 8%.
#Daily report of china

train_daily_report_china=train[train['Country/Region']=='China']

train_daily_report_china_sort=train_daily_report_china.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(18,8))

train_daily_report_china_sort.plot()

plt.ylabel('No of confirmed cases')

plt.legend(['China: Confirmed cases till 2020-03-22'])

plt.xticks(rotation=60)
#from above graph,we can see that no of Confirmed cases increases rapidly(in thousands) from 1 february onwards

#and virus affect get normal from 21 february onwards
train_daily_report_india=train[train['Country/Region']=='India']

train_daily_report_india_sort=train_daily_report_india.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_india_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['India: Confirmed cases till 2020-03-22'])

plt.xticks(rotation=60)

#from above graph,we can see that no of Confirmed cases increases rapidly from 12th march onwards
train_daily_report_italy=train[train['Country/Region']=='Italy']

train_daily_report_italy_sort=train_daily_report_italy.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_italy_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['Italy: Confirmed cases till 2020-03-22'])

plt.xticks(rotation=60)

#from above graph,we can see that no of Confirmed cases increases rapidly from 12th march onwards
train_daily_report_iran=train[train['Country/Region']=='Iran']

train_daily_report_iran_sort=train_daily_report_iran.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_iran_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['Iran: COnfirmed cases till 2020-03-22'])

plt.xticks(rotation=60)
#Applying machine learning algorithmb
test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
train.isnull().sum()
test.isnull().sum()
#we saw that from both tain & test dataset null value is present only in 'Province/State' column
##as we have Lat & long we can drop 'Province/State' & 'Country/Region' from dataser
train.drop(['Province/State','Country/Region'],axis=1,inplace=True)

test.drop(['Province/State','Country/Region'],axis=1,inplace=True)

train.isnull().sum()
test.isnull().sum()
train.info()
#convert date type object to int

train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train['Date']=train['Date'].astype(int)

train.info()
test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

test.info()
#creatr train & test data

X_train=train.drop(['Id','ConfirmedCases','Fatalities'],axis=1)

X_train.head()
y_confrm=train[['ConfirmedCases']]

y_fat=train[['Fatalities']]

y_confrm.head()
y_fat.head()
X_test=test.drop('ForecastId',axis=1)

X_test.head()
#linear Regression
from sklearn.linear_model import LinearRegression

my_model=LinearRegression()

confirmCase_result=my_model.fit(X_train,y_confrm)

confirmCase_pred_1=confirmCase_result.predict(X_test)

confirmCase_pred_1 = pd.DataFrame(confirmCase_pred_1).round()#round dataframe to decimal value

confirmCase_pred_1.columns=["confirmedCases_prediction"]

confirmCase_pred_1.head()
fatality_result=my_model.fit(X_train,y_fat)

fatalit_pred_1=confirmCase_result.predict(X_test)

fatalit_pred_1 = pd.DataFrame(fatalit_pred_1).round()

fatalit_pred_1.columns=["confirmedCases_prediction"]

fatalit_pred_1.head()
#Random Forest Regressior
from sklearn.ensemble import RandomForestRegressor

rand_reg = RandomForestRegressor(random_state=42)

rand_reg.fit(X_train,y_confrm)



ConfirmCase_pred_2 = rand_reg.predict(X_test)

ConfirmCase_pred_2 = pd.DataFrame(ConfirmCase_pred_2).round()

ConfirmCase_pred_2.columns = ["ConfirmedCases_prediction"]

ConfirmCase_pred_2.head()
rand_reg.fit(X_train,y_fat)



fatalities_pred_2 = rand_reg.predict(X_test)

fatalities_pred_2 = pd.DataFrame(fatalities_pred_2).round()

fatalities_pred_2.columns = ["Fatality_prediction"]

fatalities_pred_2.head()
#decision tree regressior
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train,y_confrm)



y_tree_conf=tree_reg.predict(X_test)

y_tree_conf=pd.DataFrame(y_tree_conf).round()

y_tree_conf.columns=['Confrmed_prediction']

y_tree_conf.head()
tree_reg.fit(X_train,y_fat)



y_tree_fat=tree_reg.predict(X_test)

y_tree_fat=pd.DataFrame(y_tree_fat).round()

y_tree_fat.columns=['fatality_prediction']

y_tree_fat.head()
#final submission based on 'ForecastID'
sample=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

sample.head()
submission=sample[['ForecastId']]

submission.head()
final_submission=pd.concat([submission,ConfirmCase_pred_2,fatalities_pred_2],axis=1)

final_submission.head()
final_submission.columns=[['ForecastId','ConfirmedCases', 'Fatalities']]

final_submission.head()
final_submission.to_csv("final_submission.csv",index=False)
final_submission
