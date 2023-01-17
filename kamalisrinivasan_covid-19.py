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
import seaborn as sns

from sklearn import linear_model

from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
train_df.describe()
# Dropping NaN

train_df.drop(['Province/State'], axis=1, inplace=True)

test_df.drop(['Province/State'], axis=1, inplace=True)
# Duplicating the data for model fitting

temp_train_df = train_df.copy()

temp_test_df = test_df.copy()
train_df.head()
# Finding Unique Countries for better interpretation and plotting

Unique_countries = list(train_df['Country/Region'].unique())

group_by_country = train_df.groupby(['Country/Region'])
print("No of countries affected by COVID-19 =",len(Unique_countries))
# Finding country-wise confirmed cases and death cases

Count = []

Country = []

Death = []



for country in Unique_countries:

    Country.append(country)

    Count.append(int(group_by_country.get_group(country).ConfirmedCases.sum()))

    Death.append(group_by_country.get_group(country).Fatalities.sum())
Confirmed_Cases_df = pd.DataFrame()



Confirmed_Cases_df['Country'] = Country

Confirmed_Cases_df['Confirmed_Cases'] = Count

Confirmed_Cases_df['Death'] = Death

Confirmed_Cases_df['Death_rate'] = Confirmed_Cases_df['Death'] / Confirmed_Cases_df['Confirmed_Cases']
Confirmed_Cases_df.head()
print("Top 10 countries affected by COVID-19")

print(Confirmed_Cases_df.sort_values(by = 'Confirmed_Cases', ascending=False)[['Country','Confirmed_Cases']].head(10))
print("Top 10 countries which has max no of death")

print(Confirmed_Cases_df.sort_values(by = 'Death', ascending=False)[['Country','Death']].head(10))
print("Top 10 countries which has highest Death rate")

print(Confirmed_Cases_df.sort_values(by = 'Death_rate', ascending=False)[['Country','Death_rate']].head(10))
month_df = pd.DataFrame()

month_df["Date"] = train_df["Date"].apply(lambda x: x.split('-')[1])

month_df["Date"]  = month_df["Date"].astype(int)
def NumToMonth(x):

    if x == 1:

        return 'Jan'

    if x == 2:

        return 'Feb'

    if x == 3:

        return 'Mar'
month_df['Date'] = month_df['Date'].apply(lambda x: NumToMonth(x))
plt.bar(month_df['Date'],train_df['ConfirmedCases'])

plt.xlabel('Month')

plt.ylabel('No of Confirmed cases')

plt.show()
plt.bar(month_df['Date'],train_df['Fatalities'])

plt.xlabel('Month')

plt.ylabel('No of Death cases')

plt.show()
date_df = pd.DataFrame()

date_df['Date'] = train_df["Date"].apply(lambda x: x.replace("-",""))
Confirmed_Country = Confirmed_Cases_df.sort_values(by = 'Confirmed_Cases', ascending=False)[['Country']].head(10)
fig,axes= plt.subplots(nrows=2, ncols=5)

plt.subplots_adjust(left=4, right=5)

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



for country in range(Confirmed_Country.shape[0]):

    temp = pd.DataFrame()

    temp['ConfirmedCases'] = group_by_country.get_group(Confirmed_Country.iloc[country,0])['ConfirmedCases']

    if country < 5:

        axes[0][country].plot(temp['ConfirmedCases'])

        axes[0][country].set_title(Confirmed_Country.iloc[country,0])

    else:

        axes[1][country-5].plot(temp['ConfirmedCases'])

        axes[1][country-5].set_title(Confirmed_Country.iloc[country,0])

plt.show()
Death_Country = Confirmed_Cases_df.sort_values(by = 'Death', ascending=False)[['Country','Death']].head(10)
fig,axes= plt.subplots(nrows=2, ncols=5)

plt.subplots_adjust(left=4, right=5)

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



for country in range(Death_Country.shape[0]):

    temp = pd.DataFrame()

    temp['Death'] = group_by_country.get_group(Death_Country.iloc[country,0])['Fatalities']

    if country < 5:

        axes[0][country].plot(temp['Death'])

        axes[0][country].set_title(Death_Country.iloc[country,0])

    else:

        axes[1][country-5].plot(temp['Death'])

        axes[1][country-5].set_title(Death_Country.iloc[country,0])

plt.show()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Worldwide Confirmed Cases Over Time" , fontsize = 30)

total_cases = train_df.groupby('Date')['Date', 'ConfirmedCases'].sum().reset_index()

total_cases['date'] = pd.to_datetime(total_cases['Date'])





ax = sns.pointplot( x = total_cases.date.dt.date ,y = total_cases.ConfirmedCases , color = 'r')

ax.set(xlabel='Dates', ylabel='Total cases')
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Worldwide Death cases over time" , fontsize = 30)

total_cases = train_df.groupby('Date')['Date', 'Fatalities'].sum().reset_index()

total_cases['date'] = pd.to_datetime(total_cases['Date'])





ax = sns.pointplot( x = total_cases.date.dt.date ,y = total_cases.Fatalities , color = 'r')

ax.set(xlabel='Dates', ylabel='Total cases')
Confirmed_Country_Count = Confirmed_Cases_df.sort_values(by = 'Confirmed_Cases', ascending=False)[['Country','Confirmed_Cases']].head(10)
plt.figure(figsize= (15,10))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total cases",fontsize = 30)

plt.ylabel('Country',fontsize = 30)

plt.title("Top 10 countries having most confirmed cases" , fontsize = 30)

ax = sns.barplot(x = Confirmed_Country_Count.Confirmed_Cases, y = Confirmed_Country_Count.Country)

for i, (value, name) in enumerate(zip(Confirmed_Country_Count.Confirmed_Cases,Confirmed_Country_Count.Country)):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Total cases', ylabel='Country')
test_Unique_countries = list(temp_test_df['Country/Region'].unique())
def TrainConvertCountryToNum(country):

    index = Unique_countries.index(country)

    return index



def TestConvertCountryToNum(country):

    index = test_Unique_countries.index(country)

    return index
temp_train_df['Date'] = temp_train_df["Date"].apply(lambda x: x.replace("-",""))

temp_test_df['Date'] = temp_test_df["Date"].apply(lambda x: x.replace("-",""))
temp_train_df["Country/Region"] = temp_train_df["Country/Region"].apply(lambda x: TrainConvertCountryToNum(x))

temp_test_df["Country/Region"] = temp_test_df["Country/Region"].apply(lambda x: TestConvertCountryToNum(x))
x_train = temp_train_df[['Country/Region', 'Lat', 'Long', 'Date']]

y_train = temp_train_df['ConfirmedCases']
x_test = temp_test_df[['Country/Region', 'Lat', 'Long', 'Date']]
fig,axes= plt.subplots(nrows=2, ncols=2)

plt.subplots_adjust(left=4, right=5)

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



# Linear Regression

lm = linear_model.LinearRegression()

lm.fit(x_train,y_train)

lm_y_pred = lm.predict(x_train)



# Lasso Regression

lm = linear_model.Lasso(alpha = 0.01, max_iter=10e5)

lm.fit(x_train,y_train)

lasso_y_pred = lm.predict(x_train)



# Decsion Tree

dtr = DTR()

dtr.fit(x_train,y_train)

DT_y_pred = dtr.predict(x_train)



# Random Forest

RF = RandomForestRegressor(max_depth=2, random_state=0)

RF.fit(x_train, y_train)

RF_y_pred = RF.predict(x_train)



axes[0][0].plot(lm_y_pred, color='red', label='Predicted')

axes[0][0].plot(y_train, color='blue', label = 'Actual')

axes[0][0].set_title("Linear Regression")

axes[0][0].legend()



axes[0][1].plot(lasso_y_pred, color='red', label='Predicted')

axes[0][1].plot(y_train, color='blue', label = 'Actual')

axes[0][1].set_title("Lasso Regression")

axes[0][1].legend()



axes[1][0].plot(DT_y_pred, color='red', label='Predicted')

axes[1][0].plot(y_train, color='blue', label = 'Actual')

axes[1][0].set_title("Decision Tree")

axes[1][0].legend()



axes[1][1].plot(RF_y_pred, color='red', label='Predicted')

axes[1][1].plot(y_train, color='blue', label = 'Actual')

axes[1][1].set_title("Random Forest")

plt.legend()



plt.show()
dtr = DTR()

dtr.fit(x_train,y_train)

y_pred = dtr.predict(x_test)
test_df['ConfirmedCases'] = y_pred
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases predicted by Decision Tree" , fontsize = 30)

total_cases = test_df.groupby('Date')['Date', 'ConfirmedCases'].sum().reset_index()

total_cases['date'] = pd.to_datetime(total_cases['Date'])





ax = sns.pointplot( x = total_cases.date.dt.date ,y = total_cases.ConfirmedCases , color = 'r')

ax.set(xlabel='Dates', ylabel='Total cases')
RF.fit(x_train, y_train)

y_pred = RF.predict(x_test)
test_df['ConfirmedCases'] = y_pred
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases predicted by Random Forest" , fontsize = 30)

total_cases = test_df.groupby('Date')['Date', 'ConfirmedCases'].sum().reset_index()

total_cases['date'] = pd.to_datetime(total_cases['Date'])





ax = sns.pointplot( x = total_cases.date.dt.date ,y = total_cases.ConfirmedCases , color = 'r')

ax.set(xlabel='Dates', ylabel='Total cases')
y_train = temp_train_df['Fatalities']
fig,axes= plt.subplots(nrows=2, ncols=2)

plt.subplots_adjust(left=4, right=5)

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



# Linear Regression

lm = linear_model.LinearRegression()

lm.fit(x_train,y_train)

lm_y_pred = lm.predict(x_train)



# Lasso Regression

lm = linear_model.Lasso(alpha = 0.01, max_iter=10e5)

lm.fit(x_train,y_train)

lasso_y_pred = lm.predict(x_train)



# Decsion Tree

dtr = DTR()

dtr.fit(x_train,y_train)

DT_y_pred = dtr.predict(x_train)





# Random Forest

RF = RandomForestRegressor(max_depth=2, random_state=0)

RF.fit(x_train, y_train)

RF_y_pred = RF.predict(x_train)



axes[0][0].plot(lm_y_pred, color='red', label='Predicted')

axes[0][0].plot(y_train, color='blue', label = 'Actual')

axes[0][0].set_title("Linear Regression")

axes[0][0].legend()



axes[0][1].plot(lasso_y_pred, color='red', label='Predicted')

axes[0][1].plot(y_train, color='blue', label = 'Actual')

axes[0][1].set_title("Lasso Regression")

axes[0][1].legend()



axes[1][0].plot(DT_y_pred, color='red', label='Predicted')

axes[1][0].plot(y_train, color='blue', label = 'Actual')

axes[1][0].set_title("Decision Tree")

axes[1][0].legend()



axes[1][1].plot(RF_y_pred, color='red', label='Predicted')

axes[1][1].plot(y_train, color='blue', label = 'Actual')

axes[1][1].set_title("Random Forest")

plt.legend()



plt.show()
# Decision Tree predicts well 

dtr = DTR()

dtr.fit(x_train,y_train)

DT_y_pred = dtr.predict(x_test)
test_df['Death'] = DT_y_pred
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Death Cases predicted by Decision Tree" , fontsize = 30)

total_cases = test_df.groupby('Date')['Date', 'Death'].sum().reset_index()

total_cases['date'] = pd.to_datetime(total_cases['Date'])





ax = sns.pointplot( x = total_cases.date.dt.date ,y = total_cases.Death , color = 'r')

ax.set(xlabel='Dates', ylabel='Total cases')
import folium
lattitude = list(train_df['Lat'].values)

longitude = list(train_df['Long'].values)
m = folium.Map([44.4604788, -110.8281375], zoom_start=11)
for i in range(len(lattitude)):

    folium.CircleMarker([lattitude[i], longitude[i]],

                            radius=15,

                            popup='country',

                            fill_color="#3db7e4", # divvy color

                           ).add_to(m)
m