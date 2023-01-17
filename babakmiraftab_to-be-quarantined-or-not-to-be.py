import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


df = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')





#df = pd.read_csv('/input/novel-corona-virus-2019-dataset/covid_19_data.csv', index_col='Date', parse_dates=True)



newdf= df.groupby(['Date','Country/Region'])['Confirmed','Deaths','Recovered'].sum().reset_index()

newdf
#Define our intented countries:Italy, Iran and UK



df_italy = newdf[:][newdf['Country/Region']=='Italy']

df_iran = newdf[:][newdf['Country/Region']=='Iran']

df_uk = newdf[:][newdf['Country/Region']=='United Kingdom']



#Define growth rate and death rate for Italy, Iran and UK





df_italy["Current Case"]=df_italy["Confirmed"]-df_italy["Deaths"]-df_italy["Recovered"]

df_iran["Current Case"] =df_iran["Confirmed"] -df_iran["Deaths"] -df_iran["Recovered"]

df_uk["Current Case"]=df_uk["Confirmed"]-df_uk["Deaths"]-df_uk["Recovered"]



df_italy["growth"]=df_italy["Current Case"]/df_italy["Current Case"].shift(+1)

df_iran["growth"]=df_iran["Current Case"]/df_iran["Current Case"].shift(+1)

df_uk["growth"]=df_uk["Current Case"]/df_uk["Current Case"].shift(+1)



df_italy["death rate"]=df_italy["Deaths"]/df_italy["Confirmed"]

df_iran["death rate"]=df_iran["Deaths"]/df_iran["Confirmed"]

df_uk["death rate"]=df_uk["Deaths"]/df_uk["Confirmed"]



italy_growth = df_italy[["Date","growth"]]

iran_growth = df_iran[["Date","growth"]]

uk_growth = df_uk[["Date","growth"]]



italy_death = df_italy[["Date","death rate"]]

iran_death = df_iran[["Date","death rate"]]

uk_death = df_uk[["Date","death rate"]]
#plotting the diagram for growth rate

#Italy vs Iran

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



ax=italy_growth.plot(kind='line',x='Date',y='growth',color='red')

y=iran_growth.plot(kind='line',x='Date',y='growth',color='orange',ax=ax)

plt.legend(('Italy', 'Iran',),    loc='upper left')

plt.ylabel("Growth Rate")

plt.xlabel("Date")

plt.title("Growth Rate of Italy vs Iran")

plt.show()
#plotting the diagram for growth rate

#Italy vs UK

ax=italy_growth.plot(kind='line',x='Date',y='growth',color='red')

y=uk_growth.plot(kind='line',x='Date',y='growth',color='blue',ax=ax)

plt.legend(('Italy', 'UK'),    loc='upper left')

plt.ylabel("Growth Rate")

plt.xlabel("Date")

plt.title("Growth Rate of Italy vs UK")

plt.show()

y.figure.savefig('growthukItaly.pdf')
#plotting the diagram for death rate

#Iran vs Italy 

ax = plt.gca()

italy_death.plot(kind='line',x='Date',y='death rate',color='red',ax=ax)

y=iran_death.plot(kind='line',x='Date',y='death rate',color='orange',ax=ax)

plt.legend(('Italy', 'Iran'),    loc='upper left')

plt.ylabel("Death Rate")

plt.xlabel("Date")

plt.title("Death Rate of Italy vs Iran")

y.figure.savefig('deathiranItaly.pdf')

plt.show()
#plotting the diagram for death rate

#UK vs Italy 

ax=italy_death.plot(kind='line',x='Date',y='death rate',color='red')

y=uk_death.plot(kind='line',x='Date',y='death rate',color='blue',ax=ax)

plt.legend(('Italy', 'UK'),    loc='upper left')

plt.ylabel("Death Rate")

plt.xlabel("Date")

plt.title("Death Rate of Italy and UK")

y.figure.savefig('deathukItaly.pdf')



plt.show()
#Preparing the data for Regression

#Italy

df_italy= df_italy.dropna()

df_italy['Day Number'] = range(1,df_italy.shape[0]+1)

#df_italy

X = df_italy.iloc[:, 8].values

X =X.reshape(-1,1)

y = df_italy.iloc[:, 6:7].values

y =y.reshape(-1,1)
# Fitting Linear Regression to the dataset 

#Italy



from sklearn.linear_model import LinearRegression 

lin = LinearRegression() 

  

lin.fit(X, y) 
#calculating RMSE for linear regression

#Italy



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

y_prediction = lin.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
# Fitting Polynomial Regression to the dataset 

#Italy



from sklearn.preprocessing import PolynomialFeatures 

  

poly = PolynomialFeatures(degree = 2) 

X_poly = poly.fit_transform(X) 

  

poly.fit(X_poly, y,) 

lin2 = LinearRegression() 

lin2.fit(X_poly, y)
#calculating RMSE for polynomial regression

#Italy



X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.33)

y_prediction = lin2.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
# Visualising the Linear Regression results

#Italy



fig = plt.figure(figsize=(10, 8), dpi=40)

ax1 = fig.add_subplot(111)

plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin.predict(X), color = 'red') 

plt.title('Linear Regression of Italy') 

plt.xlabel('Day Number') 

plt.ylabel('Growth Rate') 

plt.show() 

ax1.figure.savefig('linregitaly.pdf')
# Visualising the Polynomial Regression results 

#Italy

plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 

plt.title('Polynomial Regression') 

plt.xlabel('Days') 

plt.ylabel('Growth Rate') 

  

plt.show()
#Prediction for the next 10 days

#Italy

df_italy_prediction = df_italy[['growth','Day Number']]



for i in range (58,68):

    df_italy_prediction = df_italy_prediction.append({'Day Number': i,'growth': float(lin.predict(np.array([i]).reshape(-1,1)))

}, ignore_index=True)
# Visualising the predictoin

#Italy

dff=df_italy[['growth','Day Number']]

dfff=df_italy_prediction

fig = plt.figure(figsize=(10, 8), dpi=40)

ax = plt.gca()

dff.plot(kind='line',x='Day Number',y='growth',color='red',ax=ax)

dfff.plot(linestyle='dotted',x='Day Number',y='growth',color='red',ax=ax)

plt.legend(('Real', 'Predication'),    loc='upper left')

plt.xlabel("Day Number")

plt.ylabel("Growth Rate")

plt.title("Prediction of Growth Rate of Italy")

ax.figure.savefig('PredItaly.pdf')

plt.show()
#Preparing the data for Regression

#UK

df_uk= df_uk.dropna()

df_uk['Day Number'] = range(1,df_uk.shape[0]+1)

#df_italy

X = df_uk.iloc[:, 8].values

X =X.reshape(-1,1)

y = df_uk.iloc[:, 6:7].values

y =y.reshape(-1,1)
# Fitting Linear Regression to the dataset 

#UK

lin = LinearRegression() 

lin.fit(X, y)

poly = PolynomialFeatures(degree = 5) 

X_poly = poly.fit_transform(X) 

  

poly.fit(X_poly, y,) 

lin2 = LinearRegression() 

lin2.fit(X_poly, y) 
# Visualising the Linear Regression results

#UK

fig = plt.figure(figsize=(10, 8), dpi=40)

ax1 = fig.add_subplot(111)



plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin.predict(X), color = 'red') 

plt.title('Linear Regression of UK') 

plt.xlabel('Days') 

plt.ylabel('Growth Rate') 

  

plt.show() 
#calculating RMSE for linear regression

#UK



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

y_prediction = lin.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
# Visualising the Polynomial Regression results 

#UK

fig = plt.figure(figsize=(10, 8), dpi=40)

ax1 = fig.add_subplot(111)

plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 

plt.title('Polynomial Regression of UK') 

plt.xlabel('Days') 

plt.ylabel('Growth Rate') 

  

plt.show()
#calculating RMSE for polynomial regression

#UK



X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.33)

y_prediction = lin2.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
#Prediction for the next 10 days

# Visualising the predictoin

#UK

df_uk_prediction = df_uk[['growth','Day Number']]



for i in range (57,67):

    df_uk_prediction = df_uk_prediction.append({'Day Number': i,'growth': float(lin.predict(np.array([i]).reshape(-1,1)))

}, ignore_index=True)

    

dff=df_uk[['growth','Day Number']]

dfff=df_uk_prediction

fig = plt.figure(figsize=(10, 8), dpi=40)

ax = plt.gca()



dff.plot(kind='line',x='Day Number',y='growth',color='blue',ax=ax)

dfff.plot(linestyle='dotted',x='Day Number',y='growth',color='blue',ax=ax)

plt.legend(('Real', 'Predication'),    loc='upper left')

plt.ylabel("Growth rate")

plt.xlabel("Day Number")

plt.title("Prediction of Growth Rate of UK")

ax.figure.savefig('Preduk.pdf')

plt.show()
#Preparing the data for Regression

#Iran

df_iran= df_iran.drop([728]).dropna()

df_iran['Day Number'] = range(1,df_iran.shape[0]+1)

df_iran = df_iran.iloc[15:]



X = df_iran.iloc[:, 8].values

X =X.reshape(-1,1)

y = df_iran.iloc[:, 6:7].values

y =y.reshape(-1,1)
# Fitting Linear Regression to the dataset 

#Iran

liniran = LinearRegression() 

liniran.fit(X, y)

poly = PolynomialFeatures(degree = 2) 

X_poly = poly.fit_transform(X) 

  

poly.fit(X_poly, y,) 

liniranpoly = LinearRegression() 

liniranpoly.fit(X_poly, y)
# Visualising the linear Regression results 

#Iran

fig = plt.figure(figsize=(10, 8), dpi=40)

ax1 = fig.add_subplot(111)



plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, liniran.predict(X), color = 'red') 

plt.title('Linear Regression of Iran') 

plt.xlabel('Days') 

plt.ylabel('Growth Rate') 

  

plt.show()
#Calculating RMSE for linear regression

#Iran



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

y_prediction = liniran.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
# Visualising the Polynomial Regression results 

#Iran

fig = plt.figure(figsize=(10, 8), dpi=40)

ax1 = fig.add_subplot(111)

plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, liniranpoly.predict(poly.fit_transform(X)), color = 'red') 

plt.title('Polynomial Regression of Iran') 

plt.xlabel('Days') 

plt.ylabel('Growth Rate') 

  

plt.show()

ax1.figure.savefig('polyregiran.pdf')
#Calculating RMSE for linear regression

#Iran



X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.33)

y_prediction = liniranpoly.predict(X_test)

RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)
#Prediction for the next 10 days

# Visualising the predictoin

#Iran

df_iran_prediction = df_iran[['growth','Day Number']]



for i in range (37,40):

    df_iran_prediction = df_iran_prediction.append({'Day Number': i,'growth': float(liniran.predict(liniran.predict(np.array([i]).reshape(-1,1))))

}, ignore_index=True)

    

dff=df_iran[['growth','Day Number']]

dfff=df_iran_prediction

fig = plt.figure(figsize=(10, 8), dpi=40)

ax = plt.gca()



dff.plot(kind='line',x='Day Number',y='growth',color='orange',ax=ax)

dfff.plot(linestyle='dotted',x='Day Number',y='growth',color='orange',ax=ax)

plt.legend(('Real', 'Predication'),    loc='upper left')

plt.ylabel("Growth Rate")

plt.xlabel("Day Number")

plt.title("Prediction of Growth Rate of Iran")

ax.figure.savefig('PredIran.pdf')

plt.show()