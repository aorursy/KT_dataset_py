import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

from scipy.stats import norm

from scipy import stats

from time import time

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import plotly.tools as tls
train = pd.read_csv('../input/train.tsv', sep = '\t')

test = pd.read_csv('../input/test.tsv', sep = '\t')
train.describe()
train.head()
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

#Using pandas.DataFrame.plot.hist() to draw histgram

start1 = time()

np.log(train['price']+1).plot.hist(bins=50, ax = ax1, edgecolor='white')

ax1.set_xlabel('log(price+1)', fontsize=12)

ax1.set_ylabel('frequency', fontsize=12)

ax1.tick_params(labelsize=15)

end1 = time()

#Using Seaborn to draw the distribution plot. The shape is the same. Just change frequency to density

start2 = time()

sns.distplot(np.log(train['price']+1),bins = 50, ax = ax2,kde = False)

ax2.tick_params(labelsize=15)

ax2.set_xlabel('log(price+1)', fontsize=12)

ax2.set_ylabel('frequency', fontsize=12)

end2 = time()
start3 = time()

data = [go.Histogram(x = np.log(train['price']+1), nbinsx = 50)]

py.iplot(data, filename = 'price_bar')

end3 = time()
print ('the time to excute plot1 is {}'.format(end1 - start1))

print ('the time to excute plot2 is {}'.format(end2 - start2))

print ('the time to excute plot3 is {}'.format(end3 - start3))
train['cat1'] = train['category_name'].str.extract('([A-Z]\w{0,})',expand = True)

train['cat2'] = train['category_name'].str.extract('/(.*)/', expand = True)

train['cat3'] = train['category_name'].str.extract('/.+/(.*)', expand = True)
train.head()
cat1 = pd.DataFrame(train['cat1'].value_counts())

cat1_mean = pd.DataFrame(train.groupby('cat1').mean()['price'], index = cat1.index)

main_cat_sum =pd.concat([cat1, cat1_mean], axis = 1)

main_cat_sum.head()
start1 = time()

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18,6))

width = 0.8

main_cat_sum['cat1'].plot.bar(ax = ax1, width = width, color = 'b')

ax1.set_ylabel('count')

ax1.set_xlabel('Category')

ax1.set_title('Number of items by Category1')

end1 = time()



start2 = time()

sns.barplot(x= cat1.index, y= cat1['cat1'], ax = ax2)

ax2.set_ylabel('count')

ax2.set_xlabel('Category')

ax2.set_title('Number of items by Category1')

end2 = time()
start3 = time()

trace1 = go.Bar(x= cat1.index, y= cat1['cat1'] )

layout = dict(title= 'Number of Items by Main Category',

              yaxis = dict(title='Count'),

              xaxis = dict(title='Category'))

fig=dict(data=[trace1], layout=layout)

py.iplot(fig)

end3 = time()
print ('the time to run pandas.DataFrame.plot() is {}'.format(end1 - start1))

print ('the time to excute seaborn is {}'.format(end2 - start2))

print ('the time to excute plotly is {}'.format(end3 - start3))
train['log(price+1)'] = np.log(train['price']+1)
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18,6))

current_palette = sns.color_palette()

start1 = time()

train.boxplot(by = 'cat1', column = 'log(price+1)', ax = ax1, rot = 45)

ax1.set_ylabel('log(price+1)')

ax1.set_xlabel('Category')

ax1.set_title('log(price+1) of items by Category1')

end1 = time()

start2 = time()

sns.boxplot(y = 'cat1', x = 'log(price+1)',ax = ax2, data = train)

ax2.set_xlabel('log(price+1)')

ax2.set_ylabel('Category')

ax2.set_title('log(price+1) of items by Category1')

end2 = time()
start3 = time()

general_cats = train['cat1'].unique()

x = [train.loc[train['cat1']==cat, 'price'] for cat in general_cats]

data = [go.Box(x=np.log(x[i]+1), name=general_cats[i]) for i in range(len(general_cats))]

layout = dict(title="Price Distribution by General Category",

              yaxis = dict(title='Frequency'),

              xaxis = dict(title='Category'))

fig = dict(data=data, layout=layout)

py.iplot(fig)

end3 = time()
print ('the time to run pandas.DataFrame.plot() is {}'.format(end1 - start1))

print ('the time to excute seaborn is {}'.format(end2 - start2))

print ('the time to excute plotly is {}'.format(end3 - start3))