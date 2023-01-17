import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
lis = pd.read_csv("../input/listings.csv")

nei = pd.read_csv("../input/neighbourhoods.csv")

rev = pd.read_csv("../input/reviews.csv")

lis_sum = pd.read_csv("../input/listings_summary.csv")

rev_sum = pd.read_csv("../input/reviews_summary.csv")

cal_sum = pd.read_csv("../input/calendar_summary.csv")
lis.head()
lis.dtypes
ng = lis['neighbourhood_group'].value_counts().reset_index()

ng.columns = ['Neighbourhood_Group', 'Count']

ng['Percent'] = ng['Count']/ng['Count'].sum() * 100

ng
import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib import rcParams

rcParams['figure.figsize'] = 13, 10



ax = sns.barplot(x="Neighbourhood_Group", y="Count", data=ng)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ngp = lis['room_type'].value_counts().reset_index()

ngp.columns = ['room_type', 'Count']

ngp['Percent'] = ngp['Count']/ngp['Count'].sum() * 100
import matplotlib.pyplot as plt



labels = ngp.room_type.tolist()

sizes = ngp['Percent'].tolist()

explode = (0.1, 0, 0)



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=45)

ax1.axis('equal')



plt.show()
price = lis['price']

nor = lis['number_of_reviews']
import warnings

warnings.filterwarnings('ignore')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,9))



sns.distplot(price, hist=True, kde=False, hist_kws={'edgecolor':'black'}, ax=ax[0])

sns.distplot(price, hist=True, kde=True, hist_kws={'edgecolor':'black'}, ax=ax[1])



ax[1].set_title('Histogram of Listing Prices With KDE')

ax[1].set_xlim(0,1000)

ax[1].set_ylabel('Frequency')



ax[0].set_ylabel('Frequency')

ax[0].set_xlim(0,1000)

ax[0].set_title('Histogram of Listing Prices Without KDE')
lis['price'].describe()
import seaborn as sns

sns.set(style="whitegrid")

ax = sns.boxplot(x=lis["availability_365"])
lis_sum['security_deposit'] = lis_sum['security_deposit'].str.replace('$', '')

lis_sum['security_deposit'] = lis_sum['security_deposit'].str.replace(',', '')

lis_sum['security_deposit'] = lis_sum['security_deposit'].fillna(0)

lis_sum['security_deposit'] = lis_sum['security_deposit'].astype(str).astype(float)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)
import plotly.express as px

#tips = px.data.tips()

fig = px.box(lis_sum, x="room_type", y="security_deposit")

fig.show()
lis_sum.groupby('room_type')['security_deposit'].mean()
lis_sum_t = lis_sum[['neighbourhood_group_cleansed', 'room_type']]

two_cls = pd.crosstab(lis_sum_t.neighbourhood_group_cleansed, lis_sum_t.room_type)

two_cls
two_cls.plot.bar(stacked=True)

#plt.legend(title='mark')

plt.show()
from matplotlib import rcParams

rcParams['figure.figsize'] = 12, 9



sns.scatterplot(x="number_of_reviews", y="security_deposit", data=lis_sum)
pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



lis_sum.head(3)
sns.FacetGrid(lis_sum, col="room_type", size = 4).map(plt.scatter, "review_scores_cleanliness", "bedrooms").add_legend()
g = sns.jointplot(x="price", y="availability_365", kind='kde', data=lis)



g.ax_marg_x.set_xlim(0, 800)

g.ax_marg_y.set_ylim(0, 500)