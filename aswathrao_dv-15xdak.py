# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.drop('SNo',inplace = True,axis = 1)
data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])
data.columns
import seaborn as sns
sns.pairplot(data)
data['Cases'] = data['Active'] - int(data['Active'].mean())
for i in list(set(data['Country/Region'])):
    temp = data[data['Country/Region'] == i]
    temp['Cases'] = temp['Active'] - int(temp['Active'].mean())
    #print(temp)
    sns.barplot(x=temp['ObservationDate'].tail(30), y=temp['Cases'].tail(30)) 
    break
sns.barplot(x=data['Country/Region'].tail(12), y=data['Cases'].tail(12)) 
plt.show()
import numpy as np
import plotly.graph_objects as go
import colorlover as cl
k = 0
new = pd.DataFrame()
for i in list(set(data['Country/Region'])):
    temp = data[data['Country/Region'] == i]
    df = pd.DataFrame(columns=['Country','Confirmed','Deaths'])
    #print(df)
    df['Country'] = temp['Country/Region'].tail(1)
    df['Confirmed'] = temp['Confirmed'].mean()
    df['Deaths'] = temp['Deaths'].mean()

    new = new.append(df)
print(new[new['Country'] ==  'India'])
category_order = [
    'Deaths',
    'Cases','Confirmed'
]
# sort by desired column
new = new.sort_values(by='Confirmed', ascending = False)

fig = go.Figure()

for column in new.columns:
    fig.add_trace(go.Bar(
        x = new[column],
        y = new.Country,
        name = column,
        orientation = 'h'
    ))

fig.update_layout(
    barmode = 'relative',
    title = 'COVID - Relative graph'
)
fig.show()
for i in list(set(data['Country/Region'])):
    temp = data[data['Country/Region'] == 'India']
    #print(temp)
    temp['Province/State'].value_counts().plot(kind='pie')
    break
data.columns
list(data['Active'])
fig, ax = plt.subplots()

size = 0.3
vals = np.array([data['Active'][:10] , data['Deaths'][:10], data['Recovered'][:10]])
vals
fig, ax = plt.subplots()

size = 0.3
vals = np.array([data['Active'][:20] , data['Deaths'][:20], data['Recovered'][:20]])

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal", title='Pie plot with `ax.pie`')
plt.show()