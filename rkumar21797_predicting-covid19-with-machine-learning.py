import pandas as pd
covid = pd.read_csv(r'../input/covid-19/covid.csv')
#Having a glance at some of the records
covid.head()
#Looking at the shape
covid.shape
covid.columns
#Looking at the different locations
covid["location"].value_counts()
#Checking if columns have null values
covid.isna().any()
#Getting the sum of null values across each column
covid.isna().sum()
#Getting the cases in India
india_case=covid[covid["location"]=="India"] 
india_case.head()
india_case.tail()
import seaborn as sns
from matplotlib import pyplot as plt
#Total cases per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=india_case)
plt.show()
#Making a dataframe for last 5 days
india_last_5_days=india_case.tail()
#Total cases in last 5 days
sns.set(rc={'figure.figsize':(3,3)})
sns.lineplot(x="date",y="total_cases",data=india_last_5_days)
plt.show()
#Total tests per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_tests",data=india_case)
plt.show()
#Total tests in last 5 days
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_tests",data=india_last_5_days)
plt.show()
#Brazil Case
brazil_case=covid[covid["location"]=="Brazil"] 
brazil_case.head()
brazil_case.tail()
#Making a dataframe for brazil for last 5 days
brazil_last_5_days=brazil_case.tail()
#Total cases in last 5 days
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=brazil_last_5_days)
plt.show()
#Understanding cases of India, China and Japan
india_japan_china=covid[(covid["location"] =="India") | (covid["location"] =="China") | (covid["location"]=="Japan")]
#Plotting growth of cases across China, India and Japan
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="location",y="total_cases",data=india_japan_china,hue="date")
plt.show()
#Understanding cases of germany and spain
germany_spain=covid[(covid["location"] =="Germany") | (covid["location"] =="Spain")]
#Plotting growth of cases across Germany and Spain
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="location",y="total_cases",data=germany_spain,hue="date")
plt.show()
#Getting latest data
last_day_cases=covid[covid["date"]=="2020-05-24"]
last_day_cases
#Sorting data w.r.t total_cases
max_cases_country=last_day_cases.sort_values(by="total_cases",ascending=False)
max_cases_country
#Top 5 countries with maximum cases
max_cases_country[1:6]
#Making bar-plot for countries with top cases
sns.barplot(x="location",y="total_cases",data=max_cases_country[1:6],hue="location")
plt.show()
india_case.head()
#Linear regression
from sklearn.model_selection import train_test_split
#converting string date to date-time
import datetime as dt
india_case['date'] = pd.to_datetime(india_case['date']) 
india_case.head()
india_case.head()
#converting date-time to ordinal
india_case['date']=india_case['date'].map(dt.datetime.toordinal)
india_case.head()
#getting dependent variable and inpedent variable
x=india_case['date']
y=india_case['total_cases']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
import numpy as np
lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
india_case.tail()
y_pred=lr.predict(np.array(x_test).reshape(-1,1))
from sklearn.metrics import mean_squared_error
mean_squared_error(x_test,y_pred)
lr.predict(np.array([[737573]]))
