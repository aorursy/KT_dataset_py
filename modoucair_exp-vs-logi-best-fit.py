%pip install lmfit
import pandas as pd

import numpy as np

from datetime import datetime

from datetime import timedelta 

from scipy.optimize import curve_fit

import seaborn as sns

from datetime import datetime

from datetime import timedelta 

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from bokeh.plotting import figure

from bokeh.io import output_notebook,show

from lmfit.model import Model

output_notebook()

df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

df.head()
df_it = df[df['Country_Region']=='Italy']
df_it['Date'] = pd.to_datetime(df_it["Date"])

p = figure(x_axis_type="datetime",title="Corona Italy", plot_height=300, plot_width=800,

           background_fill_color='#F6F8F9')

p.line(df_it['Date'],df_it['ConfirmedCases'],legend_label='ConfirmedCases',line_color='#EB19FF',

      line_width=2)

p.line(df_it['Date'],df_it['Fatalities'],legend_label='Fatalities',line_color='red',

       line_width=2)

p.legend.location = "top_left"

show(p, notebook_handle=True)
df_it['ConfirmedCasesChange'] = df_it['ConfirmedCases'].pct_change()

df_it['Fatalities_change'] = df_it['Fatalities'].pct_change()

pop = df_it.pop('Province_State')
df_it.dropna(inplace=True)

df_it.head()
df_it.drop(index=9830,inplace=True)

plt.subplots(1,1,figsize=(10,5))

sns.distplot(df_it['Fatalities_change'],bins=len(df_it),label='Fatalities change')

sns.distplot(df_it['ConfirmedCasesChange'],bins=len(df_it),label='ConfirmedCases change')

plt.legend()

plt.show()
x = []

for i in range(len(df_it.index)):

    x.append(i)

y = df_it['ConfirmedCases']

def exp_func(x,a,b):

    return a*np.exp(b*x)



exponmodel = Model(exp_func)

params = exponmodel.make_params(a=5, b=0.01)

result = exponmodel.fit(y, params, x=x)

print(result.fit_report())
p = figure(x_axis_type="datetime",title="Corona Italy", plot_height=300, plot_width=800,

           background_fill_color='#000000')

p.line(df_it['Date'],result.best_fit,legend_label='fitted',line_color='#F73E5F',

      line_width=4)

p.circle(df_it['Date'],y,legend_label='ConfirmedCases',color='#39E639',

       size=5)

p.legend.location = "top_left"

show(p, notebook_handle=True)
def f(x, a, b, c):

    return a / (1. + np.exp(-c * (x - b))) 

logistic = Model(f)

params = logistic.make_params(a=0, b=0,c=0)

result = logistic.fit(y, params, x=x)

print(result.fit_report())
p = figure(x_axis_type="datetime",title="Corona Italy", plot_height=300, plot_width=800,

           background_fill_color='#000000')

p.line(df_it['Date'],result.best_fit,legend_label='fitted',line_color='#F73E5F',

      line_width=4)

p.circle(df_it['Date'],y,legend_label='ConfirmedCases',color='#39E639',

       size=5)

p.legend.location = "top_left"

show(p, notebook_handle=True)
a =result.best_values['a']

b =result.best_values['b']

c =result.best_values['c']

x_predict = np.arange(start=x[0], stop=x[-1]+60)

y_predict = f(x_predict, a, b, c)

d = datetime(2020, 2, 22)

numdays = 98

date_list = [d - timedelta(days=-x) for x in range(numdays)]

p = figure(x_axis_type="datetime",title="Corona Italy", plot_height=300, plot_width=800,

           background_fill_color='#000000')

p.line(date_list,result.best_fit,legend_label='fitted',line_color='#F73E5F',

      line_width=4)

p.circle(date_list,y,legend_label='Confirmed',color='#39E639',

       size=5)

p.circle(date_list,y_predict,legend_label='Predicted',color="#FFFD40",

       size=2)

p.legend.location = "top_left"

show(p, notebook_handle=True)