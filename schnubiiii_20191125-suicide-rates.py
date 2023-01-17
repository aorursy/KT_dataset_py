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
import seaborn as sns # makes nicer plots (personal taste)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from bokeh.plotting import figure
from bokeh.io import output_file, show
df = pd.read_csv(os.path.join(dirname, filename))
df.head()
df.info()
plt.figure(figsize = (25,10))
trace1 = go.Bar(
                x = df.year,
                y = df.suicides_no,
                name = "year",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df.country)
data = [trace1]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
plt.figure(figsize = (20,50))
sns.barplot(x = 'suicides/100k pop', y = 'country', data = df, ci = None, hue = 'year')
plt.title('Total number of suicides sorted by male & female')
plt.xlabel(' ')
plt.ylabel('number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
df2 = df.groupby(['sex']).sum()
df2 = df2.reset_index()
df2.head()
sns.catplot(x = 'sex', y = 'suicides_no', data = df2, kind ='bar', height = 8)
plt.title('Total number of suicides sorted by male & female')
plt.xlabel(' ')
plt.ylabel('number')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize = (20,30))
sns.barplot(x = 'suicides_no' , y = 'country', data = df, ci = None, hue = 'sex')
plt.title('Total number of suicides sorted by male & female')
plt.xlabel('total number')
plt.ylabel('countries')
plt.show()
df3 = df.groupby(['age', 'year']).sum()
df3 = df3.reset_index()
df3.head()
plt.figure(figsize = (20,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'age',y = 'suicides_no', data = df3, hue = 'year', order = order)
plt.title('Total number of suicides sorted age')
plt.xlabel(' ')
plt.ylabel('number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize = (15,55))
sns.barplot(x = 'suicides_no' , y= 'country', data = df, hue = 'age', ci = None) 
plt.title('Total number of suicides per Country & age group')
plt.xticks(rotation = 45)
plt.xlabel('total number')
plt.ylabel('County')
plt.show()
df_ger = df.loc[df['country'] == 'Germany'] #slicing the DataFrame for all columns regarding ,,Germany"
df_ger.reset_index()
plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'suicides_no', data = df_ger, ci = None)
plt.title('Total number of suicides in Germany between 1990 and 2015')
plt.xlabel('year')
plt.ylabel('total number')
plt.show()


plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'gdp_per_capita ($)', data = df_ger, ci = None)
plt.title('GDP per capita in Germany between 1990 and 2015 ')
plt.xlabel('year')
plt.ylabel('gdp per capita in $')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize = (25,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'year', y = 'suicides_no', data = df_ger, ci = None, hue = 'age')
plt.title('Total number of suicides in Germany between 1990 and 2015 by age')
plt.xlabel('year')
plt.ylabel('total number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
df3_ger = df_ger.groupby(['age', 'year']).sum()
df3_ger = df3_ger.reset_index()
df3.head()
plt.figure(figsize = (25,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'age', y = 'suicides_no', data = df3_ger, ci=None, hue = 'year', order = order) 
plt.title('Overview about suicide distribution in Germany between 1990 and 2015 grouped by age')
plt.xlabel('generation')
plt.ylabel('total number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'suicides/100k pop', data = df_ger, ci=None, hue = 'age') 
plt.title('Suicides per 100.000 inhabitants in Germany between 1990 and 2015')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize = (25,6))

sns.barplot(x = 'year', y = 'population', data = df_ger, ci=None, hue = 'age') 
plt.title('Population distribution in Germany between 1990 and 2015')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
sns.set(style="ticks")
sns.pairplot(df_ger)
plt.show()
df_ger.head()
df_ger2 = df_ger
df_ger2.head()
df_ger2 = df.drop(['country', 'country-year','HDI for year'], axis = 1)
df_ger2[' gdp_for_year ($) '] = df_ger2[' gdp_for_year ($) '].str.replace(",","")
df_ger2 = pd.get_dummies(df_ger2, columns = ['sex', 'age', 'generation'])
df_ger2.head()
df_ger2.info()
from sklearn.model_selection import train_test_split
X = df_ger2.drop('suicides_no', axis = 1)
y = df_ger2['suicides_no']

X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 0, test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion = "entropy", n_estimators = 10)
model.fit(X_train, y_train)
print('RandomForrest: ' + str(model.score(X_test,y_test)))
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train, y_train)
print('Linear Regression: ' + str(model2.score(X_test,y_test)))
