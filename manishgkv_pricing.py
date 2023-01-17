# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/pricing-data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from mpl_toolkits.basemap import Basemap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
df1=pd.read_excel('../input/pricing-data/AR1979to2019.xlsx')
df2=pd.read_excel('../input/pricing-data/ARRIVAL_REPORT_1979-2020.xlsx')
df3=pd.read_excel('../input/pricing-data/MW_arrivals19-20.xlsx')
df1.head(5)

df2.head(5)

df3.head(5)

df1.head()

df1.describe()
df1.corr()
x_values = list(df1['COMMODITY'])
y_values = list(df1['TOTAL'])

loc_values = []

for index in range(0, len(x_values)):
    temp_value = []

    temp_value.append(x_values[index])
    temp_value.append(y_values[index])
    loc_values.append(temp_value)
apr_values = list(df1['APRIL'])
may_values = list(df1['MAY'])

june_values = list(df1['JUNE'])
july_values = list(df1['JULY'])
august_values = list(df1['AUG'])

september_values = list(df1['SEP'])
october_values = list(df1['OCT'])
nov_values = list(df1['NOV'])

dec_values = list(df1['DEC'])
jan_values = list(df1['JAN'])
feb_values = list(df1['FEB'])
march_values = list(df1['MAR'])
attribute_list = []

for index in range(0, len(x_values)):
    temp_list = []
    
    temp_list.append(x_values[index])
    temp_list.append(y_values[index])
    
    temp_list.append(apr_values[index])
    temp_list.append(may_values[index])

    temp_list.append(june_values[index])
    temp_list.append(july_values[index])
    
    temp_list.append(august_values[index])

    temp_list.append(september_values[index])
    temp_list.append(october_values[index])
    temp_list.append(nov_values[index])
    
    temp_list.append(dec_values[index])
    temp_list.append(jan_values[index])
    temp_list.append(feb_values[index])
    temp_list.append(march_values[index])

    attribute_list.append(temp_list)
def count_points(x_points, y_points, scaling_factor):
    count_array = []
    
    for index in range(0, len(x_points)):
        temp_value = [x_points[index], y_points[index]]
        count = 0
        
        for value in loc_values:
            if(temp_value == value):
                count = count + 1
        count_array.append(count * scaling_factor )

    return count_array
def histogram_plot(dataset, title):
    plt.figure(figsize=(12, 8))    
    
    ax = plt.subplot()    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    plt.title(title, fontsize = 22)
    plt.hist(dataset, edgecolor='black', linewidth=1.2)
plt.figure(figsize=(8, 6))    
    
ax = plt.subplot()    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
    
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left() 
    
plt.title("sabji mandi Distribution plot", fontsize = 22)
plt.scatter(x_values, y_values, s = count_points(x_values, y_values, 100), alpha = 0.9)
plt.show()
histogram_plot(apr_values, title = "apr_values_distribution")
plt.show()
histogram_plot(may_values, title = "may_values_distribution")
plt.show()
histogram_plot(june_values, title = "june_values_distribution")
plt.show()
histogram_plot(july_values, title = "july_values_distribution")
plt.show()
histogram_plot(august_values, title = "august_values_distribution")
plt.show()
histogram_plot(september_values, title = "sep_values_distribution")
plt.show()
histogram_plot(october_values, title = "oct_values_distribution")
plt.show()
histogram_plot(nov_values, title = "nov_values_distribution")
plt.show()
histogram_plot(dec_values, title = "dec_values_distribution")
plt.show()
histogram_plot(jan_values, title = "jan_values_distribution")
plt.show()
histogram_plot(feb_values, title = "feb_values_distribution")
plt.show()
histogram_plot(march_values, title = "march_values_distribution")
plt.show()
df1.info()

df1.describe(include='O') #capital O
# Count of Number of Speicies
plt.figure(figsize=(10,5))
# sns.countplot(data['Species'])
ax = sns.countplot(df1['COMMODITY'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+1,
            '{:1}'.format(h),
           ha="left")
plt.show()
a = df1.groupby(['COMMODITY']).count()
a
sns.barplot(x=a.index, y='APRIL', data=a)
colors = ['cyan',]  *9
trace1 = go.Bar(
    y=df1.COMMODITY,          
    x=df1.index,
    marker_color=colors
   
)

df = [trace1]
layout = go.Layout(
    title='Count of Number',
    font=dict(
        size=16
    ),
    legend=dict(
        font=dict(
            size=6
        )
    )
)
fig = go.Figure(data=df, layout=layout)
py.iplot(fig, filename='barchart')
plt.figure(figsize=(10,5))
ax = sns.barplot(x=df1.index, y='COMMODITY', data=df1)
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{}'.format(h),
           ha="center")
plt.show()
colors = ['cyan','red','green','lightpink','blue','orange'] 
trace1 = go.Bar(
    y=df1.COMMODITY,
    x=df1.index,
    marker_color=colors,
   
)

df = [trace1]
layout = go.Layout(
    title='Average Weight of Species',
    font=dict(
        size=16
    ),
    legend=dict(
        font=dict(
            size=30
        )
    )
)
fig = go.Figure(data=df, layout=layout)
py.iplot(fig, filename='barchart')
df1['TOTAL'] = (df1['APRIL'] + df1['MAY'] + df1['JUNE'] + df1['JULY']+df1['AUG']+df1['SEP']+df1['OCT']+df1['NOV']+df1['DEC']+df1['JAN']+df1['FEB']+df1['MAR']) / 12
plt.plot(df1['APRIL'], label='1')
plt.plot(df1['MAY'], label='2')
plt.plot(df1['JUNE'], label='3')
plt.plot(df1['JULY'], label='4')
plt.plot(df1['AUG'], label='5')
plt.plot(df1['SEP'], label='6')
plt.plot(df1['OCT'], label='7')
plt.plot(df1['NOV'], label='8')
plt.plot(df1['DEC'], label='9')
plt.plot(df1['JAN'], label='10')
plt.plot(df1['FEB'], label='11')
plt.plot(df1['MAR'], label='12')
plt.plot(df1['TOTAL'], label='F')
plt.legend()
import seaborn as sns
plt.figure(figsize=(15,10))
cor_mat= df1.corr(method='spearman')
sns.heatmap(cor_mat,annot=True, square=False)
from pandas_profiling import ProfileReport
ProfileReport(df1)
def box(var):
    # this function take the variable and return a boxplot for each type of fish
    sns.boxplot(x="COMMODITY", y=var, data=df1,palette='rainbow')
fig, ax = plt.subplots(2, 3,figsize=(20,15))
plt.subplot(2,3,1)
box('APRIL')
plt.subplot(2,3,2)
box('MAY')
plt.subplot(2,3,3)
box('JUNE')
plt.subplot(2,3,4)
box('JULY')
plt.subplot(2,3,5)
box('AUG')
plt.subplot(2,3,6)
box('SEP')
plt.subplot(2,3,6)
box('OCT')
plt.subplot(2,3,6)
box('NOV')
plt.subplot(2,3,6)
box('DEC')
plt.subplot(2,3,6)
box('JAN')
plt.subplot(2,3,6)
box('FEB')
plt.subplot(2,3,6)
box('MAR')
sns.pairplot(data=df1)
sns.pairplot(data=df1,hue='COMMODITY')
plt.title('pairwise relationships in a dataset')
dfn =df1.dropna()
dfn.head()
