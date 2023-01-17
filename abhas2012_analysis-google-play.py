# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
app=pd.read_csv('../input/googleplaystore.csv')
review=pd.read_csv('../input/googleplaystore_user_reviews.csv')
app.head()
review.head()
app.describe()
type(app)
app.count()
app.dropna(how='any',inplace=True)
app.count()
app.groupby('Type').size()
app.groupby('Type').size()
app.groupby('Genres').size()
import matplotlib.pyplot as plt
apptype=app['Type'].value_counts()
apptype
import matplotlib.patches as mpat
app['Type'].value_counts().plot(kind='bar')
plt.legend()
plt.show()
app.groupby('Rating').size()
r=app['Rating'].mean()
r=round(r,1)
app['Rating'].replace(
    to_replace=19.0,
    value=r,
    inplace=True
)
print(r)
app.groupby('Rating').size()
app.groupby('Category').size()
app.groupby('Content Rating').size()
app['RatingG']=pd.cut(app['Rating'],bins=[1,2,3,4,5],include_lowest=True)
app.head()
ax=app['RatingG'].value_counts(sort=False).plot.bar(rot='0',color='r',figsize=(6,6))
o=app['RatingG']
plt.xlabel('Ratings')
plt.ylabel('No. of Apps')
plt.title('Ratings Distribution')
# ax.set_xticklabels([c[1:-1].replace(","," to") for c in o.cat.categories])
plt.legend()
plt.show()
app['RatingG'].value_counts(sort=False).plot(kind='bar')
plt.xlabel('Ratings')
plt.ylabel('No. of Apps')
plt.title('Ratings Distribution using Barplot')
plt.legend(['Different Groups of Ratings'])
plt.show()
import scipy.stats as sc
h=app['Rating'].values
h.sort()
hmean=np.mean(h)
hstd=np.std(h)
f=sc.norm.pdf(h,hmean,hstd)
plt.plot(h,f)
hmean
import seaborn as sns
p=sns.kdeplot(app['Rating'],color='green',shade=True)
p.set_xlabel('Rating')
p.set_ylabel('Common')
plt.title('Rating Distribution',size=10)
plt.legend(loc=4)
plt.show()
cplot=sns.catplot(x='Category',y='Rating',data=app,kind='box',palette='Set2',height=10)
cplot.despine(left=True)
cplot.set_xticklabels(rotation=90)
cplot.set(xticks=range(0,35))
cplot.set_ylabels("Rating")
plt.title("Category v/s Rating")
app.groupby('Category').mean()
app.head()
app['Installs']=app['Installs'].map(lambda x:x.rstrip('+'))
app.head()
plo=sns.catplot(x='Installs',data=app,hue='Category',palette='deep',height=10,kind='count')
plo.set_xticklabels(rotation=90)
plo.despine(left=True)
plo.set(xticks=range(0,20))
plo.set_ylabels('Frequency By Category')
plo.set_xlabels('Installs')

