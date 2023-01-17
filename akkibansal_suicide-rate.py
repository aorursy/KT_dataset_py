import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))





import plotly.plotly as py

import plotly.graph_objs as go

from scipy import stats

from plotly.offline import iplot, init_notebook_mode

import cufflinks



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

cufflinks.go_offline(connected=True)



# To interactive buttons

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls
data = pd.read_csv("../input/master.csv")
data.head()
data.shape
data.info()
data.rename(columns={" gdp_for_year ($) ": "gdp_for_year ($)"},inplace=True)
country_rate = data.groupby(['country']).sum()[['suicides_no','population','gdp_per_capita ($)','year']].reset_index()

country_rate.sort_values('suicides_no',ascending=False,inplace=True)
data.groupby(['country']).sum()['suicides_no'].nlargest(30).iplot(kind='bar',

                                                                     xTitle='Suicide count', yTitle='Country',

                                                                     title='Countries with most suicide rate')
country_rate['suicides_no'][:3].sum()/country_rate['suicides_no'].sum() * 100
country_rate['% suicides'] = country_rate['suicides_no']/country_rate['population'] * 100

country_rate= country_rate.reset_index()

country_rate.sort_values('% suicides',ascending=False,inplace=True)
#suicide rate for top 30 countries

plt.figure(figsize=(19,12))

plt.barh(country_rate['country'][:30],country_rate['% suicides'][:30])

plt.xlabel('Percentage', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title("Suicides per population size")

plt.show()


trace2 = go.Bar(

    x=country_rate['country'][:30],

    y=country_rate['gdp_per_capita ($)'][:30],

    name='GDP Comparision'

)



d = [trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=d, layout=layout)

iplot(fig)
year_rate = data.groupby(['year']).sum()[['suicides_no','population']].reset_index()

suicides_per_year = int(year_rate['suicides_no'].sum()/year_rate.shape[0])

print("Number of sucicides per year is {}".format(suicides_per_year), " which is ", round(suicides_per_year/365,0), " per day!", " or ", round(suicides_per_year/(365*24),0) , " per hour!")
plt.figure(figsize=(14,10))

#plt.bar(year_rate['year'],year_rate['suicides_no'])

sns.lineplot(x="year", y="suicides_no", data=year_rate)

plt.show()
np.log(data['gdp_per_capita ($)']).iplot(kind="histogram",bins=50, theme="white",histnorm='probability',

                          title="Log Distribuitions of gdp_per_capita ($)",

                          xTitle='Logarithmn Distribuition',

                          yTitle='Probability')
plt.figure(figsize = (19,15))

sns.jointplot(data['suicides_no'],data['gdp_per_capita ($)'],kind="regg")

plt.show()
data['suicides_no'].corr(data['gdp_per_capita ($)'])
def quantiles(columns):

    for name in columns:

        print(name + " quantiles")

        print(data[name].quantile([.01,.25,.5,.75,.99]))

        print("")
quantiles(['suicides_no','gdp_per_capita ($)','population'])
age_data = data.groupby(['age']).sum()['suicides_no']
percent_category = round(age_data, 2)



print("Category percentual: ")

print(percent_category/percent_category.sum() * 100,2)



types = round(age_data/ len(age_data) * 100,2)



labels = list(types.index)

values = list(types.values)



trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']), text=percent_category.values)



layout = go.Layout(title="Percentual of Suicides per Age Bracket", 

                   legend=dict(orientation="h"));



fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
sex_data = data.groupby(['sex']).sum()['suicides_no'].reset_index()
sex_data
plt.figure(figsize=(14,5))



g1 = sns.lvplot(x='sex', y='suicides_no', data=data, palette="Set1")

g1.set_title("Suicides of Male v/s Female", fontsize=20)

g1.set_xlabel("Sex", fontsize=15)

g1.set_ylabel("Total Suicides", fontsize=15)



plt.show()
plt.figure(figsize=(24,10))

g1= sns.barplot(x='year',y='suicides_no',data = data,hue='sex',ci=None)

g1.set_title("Suicides over the years", fontsize=20)

g1.set_xlabel("Years", fontsize=15)

g1.set_ylabel("Total Suicides", fontsize=15)

plt.show()