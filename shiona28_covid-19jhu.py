#Importing the libraries

import numpy as np, pandas as pd, scipy as sp

import matplotlib.pyplot as plt, seaborn as sns

import operator, datetime, statistics

from math import sqrt

from scipy.integrate import odeint

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

%matplotlib inline
#Read data from csv to dataframes

confirmedDf = pd.read_csv('/kaggle/input/confirmed.csv')

deathsDf = pd.read_csv('/kaggle/input/deaths.csv')

recoveriesDf = pd.read_csv('/kaggle/input/recovered.csv')

confirmedDf.head(10)
# To drop unused columns

#confirmedDf.drop(['Province/State','Lat','Long'], axis=1, inplace=True)
cols = confirmedDf.keys()

confirmed = confirmedDf.loc[:, cols[4]:]

deaths = deathsDf.loc[:, cols[4]:]

recoveries = recoveriesDf.loc[:, cols[4]:]
dates = confirmed.keys()

totalCases, totalDeaths, mortalityRate, recoveryRate, totalRecovered, totalActive  = [], [], [], [], [], []



for i in dates:

    confirmedSum = confirmed[i].sum()

    deathSum = deaths[i].sum()

    recoveredSum = recoveries[i].sum()

    

    # confirmed, deaths, recovered, and active cases

    totalCases.append(confirmedSum)

    totalDeaths.append(deathSum)

    totalRecovered.append(recoveredSum)

    totalActive.append(confirmedSum-deathSum-recoveredSum)

    

    # calculating rates

    mortalityRate.append(deathSum/confirmedSum)

    recoveryRate.append(recoveredSum/confirmedSum)

#Top 10 Countries with confirmed cases

top10Conf = confirmedDf.nlargest(10, '4/5/20')

plt.figure(figsize=(8,4))

graph1 = sns.barplot(x = 'Country/Region', y = '4/5/20', data = top10Conf)

graph1.set_xticklabels(graph1.get_xticklabels(),rotation=45)

plt.ylabel('Cases')

plt.xlabel('Country')

plt.title('Top 10 Countries with confirmed cases')
#Top 10 countries with dead cases

top10Deaths = deathsDf.nlargest(10, '4/5/20')

fig,ax = plt.subplots(figsize=(8,6))

explode = (0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3)#,0.4,0.5,0.6,0.7)

ax.pie(top10Deaths['4/5/20'], labels = top10Deaths['Country/Region'], autopct='%1.1f%%', startangle=45, explode=explode)

ax.set_title('Top 10 countries with dead cases')

ax.legend(bbox_to_anchor=(1,0.5), labels=top10Deaths['Country/Region'])
daysFuture = 10

futureForcast = np.array([i for i in range(len(dates)+daysFuture)]).reshape(-1, 1)

adjustedDates = futureForcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

futureForcastDates = []

for i in range(len(futureForcast)):

    futureForcastDates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
adjustedDates = adjustedDates.reshape(1, -1)[0]

plt.figure(figsize=(10, 6))

plt.plot(adjustedDates, totalCases)

plt.title('No: of Coronavirus Cases Over Time')

plt.xlabel('Days Since 22 Jan 2020')

plt.ylabel('No: of Cases')

plt.show()
startDay = np.array([i for i in range(len(dates))]).reshape(-1, 1)

totalCases = np.array(totalCases).reshape(-1, 1)

totalDeaths = np.array(totalDeaths).reshape(-1, 1)

totalRecovered = np.array(totalRecovered).reshape(-1, 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(startDay, totalCases, test_size=0.2)
# transform data for polynomial regression

poly = PolynomialFeatures(degree=5)

polyXtrain = poly.fit_transform(Xtrain)

polyXtest = poly.fit_transform(Xtest)

polyFutureForcast = poly.fit_transform(futureForcast)
# polynomial regression

linReg = LinearRegression(normalize=True, fit_intercept=False)

linReg.fit(polyXtrain, ytrain)

testlinPred = linReg.predict(polyXtest)

linPred = linReg.predict(polyFutureForcast)

print('RMSE:',sqrt(mean_squared_error(testlinPred, ytest)), '\nr2:', r2_score(testlinPred, ytest))
#Linear Regression

mlr = LinearRegression()

mlr.fit(Xtrain, ytrain)



#predict the test set results

lrPred = mlr.predict(Xtest)

print('RMSE:',sqrt(mean_squared_error(lrPred, ytest)),'\nr2 Score:', r2_score(lrPred, ytest))
# Support Vector Machine

svm = SVR(kernel='poly', gamma=0.01, epsilon=1, degree=5, C=0.1)

svm.fit(Xtrain, ytrain)

svmPred = svm.predict(futureForcast)
# check against testing data

svmtestPred = svm.predict(Xtest)

print('RMSE:',sqrt(mean_squared_error(svmtestPred, ytest)),'\nr2 Score:', r2_score(svmtestPred, ytest))
# Future predictions using PR 

plt.figure(figsize=(8, 4))

plt.plot(adjustedDates, totalCases)

plt.plot(futureForcast, linPred, linestyle='dashed', color='red')

plt.title('No: of Coronavirus Cases Over Time')

plt.xlabel('Days Since 1/22/2020')

plt.ylabel('No: of Cases')

plt.legend(['Confirmed Cases', 'Polynomial Regression predictions'])

plt.show()

linPred = linPred.reshape(1,-1)[0]

print('PR future predictions:')

set(zip(futureForcastDates[-10:], np.round(linPred[-10:])))

#set(zip(futureForcastDates[-10:], np.round(linPred[-10:])))
# Future predictions using SVM 

plt.figure(figsize=(8, 4))

plt.plot(adjustedDates, totalCases)

plt.plot(futureForcast, svmPred, linestyle='dashed', color='red')

plt.title('No: of Coronavirus Cases Over Time')

plt.xlabel('Days Since 1/22/2020')

plt.ylabel('No: of Cases')

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.show()

print('SVM future predictions:')

set(zip(futureForcastDates[-10:], np.round(svmPred[-10:])))
# Total population, N.

N = 78000000

# Number of infected and recovered individuals, I0 and R0.

I0, R0 = confirmedSum, recoveredSum

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

beta, gamma = 0.7, statistics.mean(recoveryRate)

# A grid of time points (in days)

t = np.linspace(0, 160, 160)

# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



# Initial conditions vector

y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, beta, gamma))

s, i, r = ret.T



f = plt.figure(figsize=(8,5)) 

plt.plot(s, 'b', label='susceptible');

plt.plot(i, 'r', label='infected');

plt.plot(r, 'c', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("days", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,35)

plt.show()
  