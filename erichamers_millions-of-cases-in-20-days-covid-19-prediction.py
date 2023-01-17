import matplotlib.pyplot as plt

import numpy as np 

import os

import pandas as pd

import seaborn as sns

import warnings



from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



# Package configuration #



pd.options.display.max_columns = 999

pd.options.display.max_rows = 999

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

sub = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')



df = pd.concat([train, test])



global_cases = train.groupby('Date')['ConfirmedCases'].sum()

global_deaths = train.groupby('Date')['Fatalities'].sum()
fig, ax = plt.subplots(1, 2, figsize=(18, 6))





## Code for plot 1 ##



sns.lineplot(x=global_cases.index, y=global_cases.values, linewidth=3, ax=ax[0])



sns.despine(top=True, left=True)



ax[0].grid(axis='both', alpha=.2)



every_nth = 10

for n, label in enumerate(ax[0].xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax[0].tick_params(axis='both', which='both', bottom=False, left=False)



ax[0].text(x=0, y=max(global_cases.values)+60000, s='Covid-19 Spread', fontsize=22, fontweight='bold')

ax[0].text(x=0, y=max(global_cases.values)+25000, s='The total number of confirmed cases of covid-19 \nworldwide.', fontsize=16)

ax[0].set_xlabel('')



ax[0].set_ylabel('Total number of cases\n')



## Code for plot 2 ##



sns.lineplot(x=global_cases.index, y=np.log(global_cases.values), linewidth=3, ax=ax[1])



sns.despine(top=True, left=True)



ax[1].grid(axis='both', alpha=.2)



every_nth = 10

for n, label in enumerate(ax[1].xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax[1].tick_params(axis='both', which='both', bottom=False, left=False)



ax[1].text(x=0, y=max(np.log(global_cases.values))+1.5, s='Log-Scaled', fontsize=22, fontweight='bold')

ax[1].text(x=0, y=max(np.log(global_cases.values))+.70, s='The log scale of the total number of confirmed cases \nof covid-19 worldwide.', fontsize=16)

ax[1].set_xlabel('')



plt.show()
g = global_cases/global_cases.shift(1)
fig, ax = plt.subplots(figsize=(18, 6))

                       

sns.lineplot(x=g.index, y=g.values, linewidth=3, ax=ax)

sns.lineplot(x=g.index, y=g.rolling(5).mean().values, linewidth=2, ax=ax, color='blue', alpha=.5)

sns.lineplot(x=g.index, y=g.rolling(15).mean().values, linewidth=2, ax=ax, color='orange')



sns.despine(top=True, left=True)



ax.grid(axis='both', alpha=.2)



every_nth = 10

for n, label in enumerate(ax.xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax.tick_params(axis='both', which='both', bottom=False, left=False)



ax.text(x=0, y=max(g.fillna(0)) + .20, s='Growth Factor', fontsize=22, fontweight='bold')

ax.text(x=0, y=max(g.fillna(0)) + .08, s='The ratio between the number of cases on one day and the\nnumber of cases in the previous day', fontsize=16)



sns.lineplot(x=g.index, y=np.ones(len(g)), ax=ax)

ax.lines[1].set_linestyle('dashed')

ax.lines[2].set_linestyle('dashed')

ax.lines[3].set_linestyle(':')

ax.legend(['Growth Factor', '5 Days Rolling Mean', '15 Days Rolling Mean', 'Possible Point of Inflection'])

ax.set_xlabel('')



plt.show()
x = np.arange(0, len(global_cases)).reshape(-1, 1)

y = np.log(global_cases.values)



model = LinearRegression().fit(x, y)



print('R-Squared: %s' % model.score(x, y))
fig, ax = plt.subplots(1, 2, figsize=(18, 6))



## Code for plot 1 ##



sns.lineplot(global_cases.index, np.exp(y), ax=ax[0])

sns.lineplot(global_cases.index, np.exp(model.predict(x)), ax=ax[0])



ax[0].grid(axis='both', alpha=.2)



sns.despine(top=True, left=True)



every_nth = 10

for n, label in enumerate(ax[0].xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax[0].text(x=0, y=470000, s='Covid-19 Cases Predictions', fontsize=22, fontweight='bold')

ax[0].text(x=0, y=415000, s='Prediction curve of the number of covid-19 cases\nusing a simple OLS.', fontsize=16)

        

ax[0].tick_params(axis='both', which='both', bottom=False, left=False)

ax[0].legend(['True Values', 'Predicted Values'])

ax[0].set_xlabel('')

ax[0].lines[1].set_linestyle(':')



ax[0].set_ylabel('Total number of cases\n')

        

## Code for plot 2 ##



sns.lineplot(global_cases.index, np.log(global_cases.values), ax=ax[1])

sns.lineplot(global_cases.index, model.predict(x), ax=ax[1])



ax[1].grid(axis='both', alpha=.2)



sns.despine(top=True, left=True)



every_nth = 10

for n, label in enumerate(ax[1].xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax[1].text(x=0, y=14.3, s='Covid-19 Cases Log-Predictions', fontsize=22, fontweight='bold')

ax[1].text(x=0, y=13.4, s='Log prediction curve of the number of covid-19\ncases using a simple OLS.', fontsize=16)

        

ax[1].tick_params(axis='both', which='both', bottom=False, left=False)

ax[1].legend(['True Values', 'Predicted Values'])

ax[1].set_xlabel('')

ax[1].lines[1].set_linestyle(':')



plt.show()
x_2 = np.arange(0, 79).reshape(-1, 1)



predictions = model.predict(x_2)
first_date = pd.to_datetime(min(global_cases.index))

last_date =  (first_date + timedelta(days=78)).strftime('%Y-%m-%d')

first_date = first_date.strftime('%Y-%m-%d')

index = pd.date_range(first_date, last_date)
fig, ax = plt.subplots(figsize=(15, 6))

                       

sns.lineplot(x=index, y=predictions, linewidth=3)



sns.despine(top=True, left=True)



ax.grid(axis='both', alpha=.2)



every_nth = 2

for n, label in enumerate(ax.xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)

        

ax.tick_params(axis='both', which='both', bottom=False, left=False)



ax.text(x='2020-01-18', y=16, fontsize=22, fontweight='bold', s='Covid-19 OLS Predictions')

ax.text(x='2020-01-18', y=15.2, fontsize=16, s='Medium term predictions of the spread of covid-19 using a simple\nOLS regression. (Log scaled)')



ax.axvline(x='2020-04-08', ymin=0,ymax=.94, linestyle='dashed', color='red', alpha=.7)

ax.hlines(y=14.34, xmin='2020-01-22', xmax='2020-04-08', linestyle='dashed', color='red', alpha=.7)

ax.text(x='2020-04-09', y=14, s='1.8 Million Cases', fontsize=14)



plt.show()