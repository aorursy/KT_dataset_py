# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import math, datetime, pandas as pd, warnings

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.express as px

import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, f1_score

from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings('ignore')
print('Last update on', pd.to_datetime('now'))
df = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv').groupby('date').sum()[27:].reset_index()

brazil = pd.DataFrame({

    'date': pd.to_datetime(df['date'], format='%Y/%m/%d'),

    'cases': df['cases'], 

    'new_cases': df['cases'].diff().fillna(0).astype(int),

    'growth_cases': df['cases'].diff().fillna(0).astype(int)/df['cases'],

    'deaths': df['deaths'],

    'new_deaths': df['deaths'].diff().fillna(0).astype(int),

    'growth_deaths': df['deaths'].diff().fillna(0).astype(int)/df['deaths'],

    'mortality_rate': df['deaths']/df['cases']

})

brazil.fillna(0).tail()
def poly_reg(x, y, x_test, d):

    poly = PolynomialFeatures(degree = d) 

    poly.fit(poly.fit_transform(x), y)

    model = LinearRegression()

    model.fit(poly.fit_transform(x), y)

    return model.predict(poly.fit_transform(x_test))



def score(y, yhat):

    r2 = r2_score(y,yhat)

    rmse = np.sqrt(mean_squared_error(y,yhat))

    return (r2,rmse)
# Defines the range

start = 1

end = len(brazil)



# Sets the samples

x = np.asarray(range(start,end)).reshape(-1,1)

y = brazil.iloc[start:,1]



# Creates polynomial model and predict

yhat = poly_reg(x, y, x, 4)



# Plot the line chart

fig, ax = plt.subplots(figsize=(14, 10))

plt.scatter(x, y, s=40)

plt.plot(x, yhat, color='magenta', linestyle='solid', linewidth=4, alpha=0.5)

plt.title('Evaluating the model', fontsize=18, fontweight='bold', color='#333333')

plt.legend(labels=['prediction','cases'], fontsize=12)

plt.text(0.01,1.0,s='R2: %.3f RMSE: %.3f' % score(y, yhat), transform=ax.transAxes, fontsize=9)

plt.grid(which='major', axis='y')

ax.set_axisbelow(True)

ax.set_ylim(0)

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

plt.show();
# Até qual data realizar a predição

data_inicial = ''

data_final = '2020-08-25'
dates = pd.date_range(start=brazil.iloc[0,0], end=data_final) #.strftime('%d/%m').to_list()
# Creates polynomial model and predict

x_test = np.asarray(range(start, len(dates))).reshape(-1,1)

yhat = poly_reg(x, y, x_test, 4)

yhat_deaths = [i * 0.06 for i in yhat]



# Plot the line chart

fig, ax = plt.subplots(figsize=(14, 10))

plt.plot(dates[start:end], y, color='limegreen', linewidth=8, alpha=0.5)

plt.plot(brazil['date'][17:], brazil['deaths'][17:], color='magenta', linewidth=8, alpha=0.5)



plt.plot(dates[start:len(dates)], yhat, color='green', linestyle='None', marker='o')

plt.plot(dates[start:len(dates)], yhat_deaths, color='darkorchid', linestyle='None', marker='o')



#plt.bar(brazil['date'][17:], brazil['new_cases'][17:])

plt.title('COVID-19: cases prediction in Brazil', fontsize=18, fontweight='bold', color='#333333')

plt.legend(labels=['cases','deaths', 'cases prediction', 'deaths prediction', 'new cases'], fontsize=14)



plt.text(0.01,1.01,s='R2: %.3f RMSE: %.2f' % score(y, yhat[:len(y)]), transform=ax.transAxes, fontsize=10)

plt.xticks(rotation=90)

plt.tick_params(axis='y', length = 0)

ax.set_axisbelow(True)

#[ax.annotate('%s' % y, xy=(x,y+300), fontsize=10) for x,y in zip(dates[start:len(dates)], yhat.astype(int))]

#[ax.annotate('%s' % y, xy=(x,y+500), fontsize=10) for x,y in zip(dates[len(brazil['date']):len(dates)], yhat[len(brazil['date'][17:]):].astype(int))]

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

ax.yaxis.set_major_locator(plt.MultipleLocator(50000))

ax.yaxis.set_minor_locator(plt.MultipleLocator(10000))

ax.set_ylim(0)

[ax.spines[side].set_visible(False) for side in ['right','top']]

plt.grid(which='major', color='#EEEEEE')

plt.grid(which='minor', color='#EEEEEE', linestyle=':')

plt.show();