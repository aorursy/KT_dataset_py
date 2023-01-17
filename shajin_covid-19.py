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
from statsmodels.tsa.stattools import acf

from statsmodels.graphics.tsaplots import plot_acf
# Reading Train and Test Datasets

train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

print("\t\tTrain Data:\n")

display(train.head())

display(train.tail())

print("\t\tTest Data:\n")

display(test.head())

display(test.tail())

print("\t\tSummary of Train Data:\n")

display(train.describe())
import matplotlib.pyplot as plt

import seaborn as sns
confirmed_cases = train.groupby(['Date']).agg({'ConfirmedCases' : ['sum']})

fatalities = train.groupby(['Date']).agg({'Fatalities' : ['sum']})

totalCases = confirmed_cases.join(fatalities)

fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)

confirmed_cases.plot(ax=ax1)

ax1.set_title('Exploration of Global Confirmed Cases', size = 13)

ax1.set_ylabel('Number of Cases', size = 13)

ax1.set_xlabel('Date', size = 13)

fatalities.plot(ax=ax2)

ax2.set_title('Exploration of Fatalities', size = 13)

ax2.set_ylabel('Number of Cases', size = 13)

ax2.set_xlabel('Date', size = 13)

fig.tight_layout()

plt.show()
totalCases.head(10)
import plotly.express as px

countries = list(set(list(train['Country_Region'])))

agg_funcs = {'Date': 'first', 'ConfirmedCases': 'sum', 'Fatalities': 'sum'}
num_conf_cases = []

for country in countries:

    data2 = train.loc[train['Country_Region'] == country]

    num_cases_country = data2.groupby(data2['Date']).aggregate(agg_funcs).max().ConfirmedCases

    num_conf_cases.append(num_cases_country)



# index ordered by num_conf_cases    

idx_top_by_cases = list(reversed(np.argsort(num_conf_cases)))



for i in range(20):

    idx_top = idx_top_by_cases[i]

    print('%d: %s (%d cases)' % (i+1, countries[idx_top], num_conf_cases[idx_top]))
countries_str = '[%s]'% (', '.join(["'%s'"%countries[idx] for idx in idx_top_by_cases[:20]]))  

data_top_countries = train.query("Country_Region == %s" % countries_str) 



fig = px.line(data_top_countries, x="Date", y="ConfirmedCases", color="Country_Region",

              line_group="Country_Region", hover_name="Country_Region",

              title="Top 20 Countries with Most Confirmed Cases")

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fatalities_cases = []

for country in countries:

    data2 = train.loc[train['Country_Region'] == country]

    fatalities_country = data2.groupby(data2['Date']).aggregate(agg_funcs).max().Fatalities

    fatalities_cases.append(fatalities_country)



# index ordered by num_conf_cases    

idx_top_by_fatalities = list(reversed(np.argsort(fatalities_cases)))



for i in range(20):

    idx_top = idx_top_by_fatalities[i]

    print('%d: %s (%d cases)' % (i+1, countries[idx_top], fatalities_cases[idx_top]))
countries_str = '[%s]'% (', '.join(["'%s'"%countries[idx] for idx in idx_top_by_fatalities[:20]]))



data_top_countries = train.query("Country_Region == %s" % countries_str)



fig = px.line(data_top_countries, x="Date", y="Fatalities", color="Country_Region",

              line_group="Country_Region", hover_name="Country_Region",

              title="Top 20 countries with Fatalities")

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
#Calculating percentage increase in Confirmed Cases and Fatalities

totalCases['CasesIncrease'] = totalCases.ConfirmedCases.pct_change()

totalCases['FatalitiesIncrease'] = totalCases.Fatalities.pct_change()
totalCases.head()
fig = px.scatter(totalCases, x = 'CasesIncrease', y = 'FatalitiesIncrease')

fig.show()
# Finding Correlation Between Confirmed Cases and Fatalities

correlation = totalCases.CasesIncrease.corr(totalCases.FatalitiesIncrease)

print("Correlation Between Increase in Fatalities and Confirmed Cases:", correlation)
type(totalCases)
percent_change = pd.DataFrame({"CasesIncrease" : [x for x in totalCases.CasesIncrease],

                              "FatalitiesIncrease" : [y for y in totalCases.FatalitiesIncrease]}, index=totalCases.index)

percent_change.head()
import statsmodels.api as sm
X = pd.DataFrame(percent_change, columns = ['CasesIncrease'])

X = sm.add_constant(X)

X.head()
X.fillna(0, inplace = True)

X.head()
percent_change['FatalitiesIncrease'].fillna(0, inplace = True)

y = percent_change['FatalitiesIncrease']

print(y)
results = sm.OLS(y, X).fit()

results.summary()
percent_change.head()
totalCases2 = totalCases[['ConfirmedCases', 'Fatalities']]

totalCases2.head()
totalCases2.index = pd.to_datetime(totalCases2.index)

totalCases3 = totalCases2.resample(rule='W').last() # Weekly

percent_change3 = totalCases3.pct_change()

percent_change3 = percent_change3.dropna()
percent_change3.head()
# Weekly Autocorrelations

auto_corr_conf = percent_change3['ConfirmedCases']['sum'].autocorr() #Autocorrelation of CasesIncrease 

auto_corr_Fatal = percent_change3['Fatalities']['sum'].autocorr() #Autocorrelation of FatalitiesIncrease

print("The Autocorrelation of CasesIncrease Time Series:", auto_corr_conf)

print("The Autocorrelation of FatalitiesIncrease Time Series:", auto_corr_Fatal)
# Potting ACF of total Confirmed Cases

plot_acf(totalCases2.ConfirmedCases, alpha = 0.05)
# Potting ACF of Total Fatalities

plot_acf(totalCases2.Fatalities, alpha = 0.05)
totalCases2.head()
from statsmodels.tsa.stattools import adfuller



# Running ADF test on ConfirmedCases

results_ADF = adfuller(totalCases2.ConfirmedCases)



# print p-value

print(results_ADF[1])
# Running ADF test on Fatalities

results2_ADF = adfuller(totalCases2.Fatalities)



# print p-value

print(results2_ADF[1])
# We are now testing the same test for Increase in ConfirmedCases and Fatalities

totalCases4 = totalCases2.pct_change()

totalCases4.dropna(inplace = True)

totalCases4.columns = ['CasesIncrease', 'FatalitiesIncrease']

totalCases4.head()
# Running ADF test on FatalitiesIncrease

results3_ADF = adfuller(totalCases4.FatalitiesIncrease)



# print p-value

print(results3_ADF[1])
# Running ADF test on CasesIncrease

results4_ADF = adfuller(totalCases4.CasesIncrease)



# print p-value

print(results4_ADF[1])
india_data = train[train.Country_Region == 'India']

india_data.tail()
confirmed_cases_india = india_data.groupby(['Date']).agg({'ConfirmedCases' : ['sum']})

fatalities_india = india_data.groupby(['Date']).agg({'Fatalities' : ['sum']})

totalCasesIndia = confirmed_cases.join(fatalities)

fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)

confirmed_cases_india.plot(ax=ax1)

ax1.set_title('Exploration of Confirmed Cases in India', size = 13)

ax1.set_ylabel('Number of Cases', size = 13)

ax1.set_xlabel('Date', size = 13)

fatalities_india.plot(ax=ax2)

ax2.set_title('Exploration of Fatalities in India', size = 13)

ax2.set_ylabel('Number of Cases', size = 13)

ax2.set_xlabel('Date', size = 13)

fig.tight_layout()

plt.show()
acf_array_cases = acf(confirmed_cases_india)

print(acf_array_cases)
acf_array_fatalities = acf(fatalities_india)

print(acf_array_fatalities)
# Potting ACF of Confirmed Cases in India

plot_acf(confirmed_cases_india, alpha = 0.05)
# Plotting ACF of Fatalities in India

plot_acf(fatalities_india, alpha = 0.05)
# Running ADF test on Confirmed Cases in India

results_india_conf = adfuller(confirmed_cases_india)



# print p-value

print(results_india_conf[1])
# Running ADF test on Fatalities in India

results_india_fatal = adfuller(fatalities_india)



# print p-value

print(results_india_fatal[1])
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(totalCases2.ConfirmedCases, alpha=0.05)
from statsmodels.tsa.arima_model import ARMA



# ARMA Model for ConfirmedCases

mod = ARMA(totalCases2.ConfirmedCases, order = (1, 0))

result = mod.fit()

result.summary()
result.plot_predict(start=60, end=90)

plt.show()
#ARMA Model for Fatalities

plot_pacf(totalCases2.Fatalities, alpha=0.05)
mod2 = ARMA(totalCases2.Fatalities, order = (1, 0))

result2 = mod2.fit()

result2.summary()
result2.plot_predict(start = 60, end = 90)

plt.show()
totalCases4.head()
plot_pacf(totalCases4.FatalitiesIncrease, alpha=0.05)
BIC = np.zeros(6)

for p in range(6):

    mod3 = ARMA(totalCases4.FatalitiesIncrease, order = (p, 0))

    result3 = mod3.fit()

    # Storing BIC

    BIC[p] = result3.bic
# Plot the BIC as a function of p

plt.plot(range(1,6), BIC[1:6], marker='o')

plt.xlabel('Order of AR Model')

plt.ylabel('Bayesian Information Criterion')

plt.show()
mod4 = ARMA(totalCases4.FatalitiesIncrease, order = (3, 0))

result4 = mod4.fit()

result4.summary()
result4.plot_predict(start = 50, end = 90)

plt.show()
plot_pacf(totalCases4.CasesIncrease, alpha=0.05)
BIC = np.zeros(6)

for p in range(6):

    mod3 = ARMA(totalCases4.CasesIncrease, order = (p, 0))

    result3 = mod3.fit()

    # Storing BIC

    BIC[p] = result3.bic
# Plot the BIC as a function of p

plt.plot(range(1,6), BIC[1:6], marker='o')

plt.xlabel('Order of AR Model')

plt.ylabel('Bayesian Information Criterion')

plt.show()
mod5 = ARMA(totalCases4.CasesIncrease, order = (3, 0))

result5 = mod5.fit()

result5.summary()
result5.plot_predict(start=50, end=90)

plt.show()