import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing, svm 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

import datetime
data = pd.read_excel("../input/moroccan-covid-statistics/Covid19-Mrocco (5).xlsx")

data = data.iloc[:216,0:6]

data["index"]=[x for x in range(1,len(data.values)+1)]

data = data.fillna(0)

data.describe()
plt.plot(data["index"],data["death"])

plt.xlabel("Days")

plt.ylabel("death")

plt.show()
plt.plot(data["index"],data["Recovered"])

plt.xlabel("Days")

plt.ylabel("Recovered")

plt.show()
plt.plot(data["index"],data["Cases found"])

plt.xlabel("Days")

plt.ylabel("Cases found")

plt.show()
active_cases = data['Cases found'].sum()-data['death'].sum()-data['Recovered'].sum()

a = [data['death'].sum(),data['Recovered'].sum(),active_cases]

df = pd.DataFrame({'mass': a},index=['death', 'recovered', 'active cases'])

plot = df.plot.pie(y='mass', figsize=(15, 8),autopct='%1.1f%%')
datacum = data.iloc[:,1:6].cumsum()

datacum["index"]=[x for x in range(1,len(datacum.values)+1)]
plt.plot(datacum["index"],datacum["death"])

plt.xlabel("Days")

plt.ylabel("death")

plt.show()
plt.plot(datacum["index"],datacum["Recovered"])

plt.xlabel("Days")

plt.ylabel("Recovered")

plt.show()
plt.plot(datacum["index"],datacum["Cases found"])

plt.xlabel("Days")

plt.ylabel("Cases found")

plt.show()
x = np.array(datacum['index']).reshape(-1,1)

target = np.array(datacum['Cases found']).reshape(-1,1)

 

Input=[('polynomial',PolynomialFeatures(degree=4)),('modal',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(x.reshape(-1,1),target.reshape(-1,1))

poly_pred=pipe.predict(x.reshape(-1,1))

plt.scatter(x,target)

sorted_zip = sorted(zip(x,poly_pred))

x_poly, poly_pred = zip(*sorted_zip)

plt.plot(x_poly,poly_pred,color='red',label='Polynomial Regression')

plt.show()
commingDays = np.array([x for x in range(216,305)])

estimateCases = pipe.predict(commingDays.reshape(-1,1))

estimateCases = estimateCases.astype(int)

 

plt.plot(commingDays,estimateCases,color='red')

plt.title('COVID-19 cases predictor using polynomial regression (Morocco)')

plt.xlabel('next 97 Days from 26-09-2020')

plt.ylabel('cases number estimator')

plt.show()
dates=[]

date = datetime.datetime(2020,10,3)

for i in range(89): 

    date += datetime.timedelta(days=1)

    dates.append(date.strftime("%d/%m/%Y"))

 

predictionCases = pd.DataFrame()

predictionCases["next days"]= dates

predictionCases["Prediction Cases"] = estimateCases

 

predictionCases.to_excel("predictions.xlsx",sheet_name="prediction_cases")

 

predictionCases