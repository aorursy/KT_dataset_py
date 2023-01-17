import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/visualization/iris.csv')

df.head()
fig,ax = plt.subplots(figsize=(12,4))

ax.plot('Sepal length', data=df, label='Sepal length', marker='o')

ax.plot('Petal length', data=df, label='Petal length', marker='^')

ax.legend()

plt.show()
fig, (ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(12,4))

ax1.plot('Sepal length', data=df, label='Sepal length', marker='o')

ax2.plot('Petal length', data=df, label='Petal length', marker='o', color='orange')

plt.show()
df[['Sepal length','Petal length']].plot.line(subplots=True, layout=(1,2), figsize=(12,4))
df1 = pd.read_csv('../input/visualization/barcharts-us-music-sales-by-genre.csv')

df1.head()
fig,ax=plt.subplots(figsize=(12,4))

ax.bar(x='Year', height='Rock', data=df1, label='Rock')

ax.bar(x='Year', height='Jazz', data=df1, label='Jazz')

ax.set_xticks(df1.Year)

plt.show()
fig,ax=plt.subplots(figsize=(12,4))

width=0.3

ax.bar('Year', 'Rock', data=df1, label='Rock',width=width)

ax.bar(df1.Year.values+width, 'Jazz', data=df1, label='Jazz',width=width)

ax.set_xticks(df1.Year)

ax.set_xticks(df1.Year.values+width/2)

ax.set_xticklabels(df1.Year.values)

plt.show()
fig2,ax2=plt.subplots(figsize=(12,4))

df1[['Rock','Jazz']].plot.bar(ax=ax2)

ax2.set_xticklabels(df1.Year.values)

plt.show()
for p in ax2.patches:

    an=ax2.annotate(str(p.get_height())+'%', xy=(p.get_x(),p.get_height()))

    an.set_size(9)

fig2
fig,ax=plt.subplots(figsize=(12,4))

df1[['Country', 'Jazz', 'Rock']].plot.bar(stacked=True, ax=ax, rot=0)

ax.set_xticklabels(df1.Year.values)

plt.show()
df2 = pd.read_csv('../input/visualization/index.csv')

df2.head()
fig,ax=plt.subplots(figsize=(12,4))



ax.scatter('Federal Funds Target Rate', 'Unemployment Rate', data=df2)

ax.set_xlabel('Interest rage')

ax.set_ylabel('Unemployment rate')

plt.show()
fig,ax=plt.subplots(figsize=(12,4))



ax.scatter('Federal Funds Target Rate', 'Unemployment Rate', data=df2)

ax.scatter('Federal Funds Target Rate', 'Inflation Rate', data=df2)

ax.set_xlabel('Rate')

ax.set_ylabel('Unemployment rate')

plt.show()
df3 = pd.read_csv('../input/visualization/barchart-divorcerates.csv')

df3
fig,ax=plt.subplots()

wedget, label, value = ax.pie(labels='Age', x='Women', data=df3, autopct='%1.1f%%', radius=2)
segment1 = wedget[0]

segment1.set_facecolor('y')

fig
fig,ax=plt.subplots()

wedget, label, value = ax.pie(labels='Age', x='Women', data=df3, autopct='%1.1f%%', radius=2)
from matplotlib.patches import Circle



circle = Circle((0,0), 0.9, facecolor='white')

ax.add_artist(circle)

fig
df4 = pd.read_csv('../input/visualization/df_weed.csv')

df4
fig,ax=plt.subplots(figsize=(12,4))



ax.hist('HighQ', data=df4, label='HighQ')

plt.show()
fig,ax=plt.subplots(figsize=(12,4))

df4.plot.hist(ax=ax)

plt.show()
df3.head()
fig,ax=plt.subplots(figsize=(12,4), subplot_kw=dict(polar=True))



x=df3.Age

y1=df3.Women

y2=df3.Men



ax.bar(x,y1+y2, label='Women')

ax.bar(x,y2, label='Men')

ax.legend()

plt.show()
df5 = pd.read_csv('../input/visualization/sales.csv')

df5
fig,ax=plt.subplots(figsize=(12,4))

ax.bar('Date', 'Quantity', data=df5)

plt.show()
ax2=ax.twinx()

ax2.plot('Amount', data=df5, color='y')

fig
fig,ax=plt.subplots(figsize=(12,4))

df5.plot.bar(y='Quantity', label='Quantity', ax=ax)

df5.plot.line(y='Amount', label='Amount', ax=ax, color='y', secondary_y=True)

plt.show()
df4.head()
fig,ax=plt.subplots(figsize=(12,4))

df4.plot.kde(ax=ax)

plt.show()
df6 = pd.read_csv('../input/visualization/altcoins.csv')

df6.head()
df6.fillna(0, inplace=True)
fig,ax=plt.subplots(figsize=(12,4))

data=df6.tail(30)



ax.boxplot([data['ETH'],data['DASH']])

plt.show()
fig1,ax1=plt.subplots(figsize=(12,4))

bp = ax1.boxplot([data['ETH'],data['DASH']],

                vert=False,

                whis=1.5,

                widths=[0.3, 0.8],

                labels=['ETH','DASH'],

                showmeans=False,

                meanprops=dict(marker='.'),

                showcaps=True,

                showfliers=True,

                whiskerprops=dict(linestyle='dashed'),

                flierprops=dict(marker='*', markersize=10, markeredgecolor='r'),

                medianprops=dict(linestyle='dotted', linewidth=3))
df6 = pd.read_csv('../input/visualization/altcoins.csv')

df6.head()
df6.dropna(inplace=True)
corr_df6 = df6.corr()

corr_df6
fig,ax=plt.subplots()

cax = ax.imshow(corr_df6)
ax.set_xticklabels(df6.columns)

ax.set_yticklabels(df6.columns)

fig
fig.colorbar(cax)

fig
import numpy as np

for (i,j),t in np.ndenumerate(corr_df6):

    ax.annotate('{:.1f}'.format(t), xy=(i,j), va='center', ha='center')

fig
import seaborn as sns
df7 = pd.read_csv('../input/visualization/setosa-versicolor.csv')

df7.head()
fig,ax=plt.subplots(figsize=(12,4))

sns.regplot('Sepal length','Petal width', data=df7, ax=ax)

plt.show()
fig,ax=plt.subplots(figsize=(12,4))

sns.regplot('Sepal length','Petal width', data=df7, ax=ax,

           scatter=True,

           fit_reg=True,

           marker='o',

           scatter_kws=dict(edgecolor='red',lw=2,facecolor='white',s=80),

           line_kws=dict(color='green', linestyle='--'),

           ci=0.85)#confidence interval

plt.show()
sns.regplot('Sepal length','Class', data=df7, logistic=True)
titanic=sns.load_dataset('titanic')

titanic.head()
fig,ax=plt.subplots(3,figsize=(6,12))



sns.countplot(y='sex', data=titanic, ax=ax[0])

sns.countplot(y='sex', data=titanic, ax=ax[1], hue='alone')

sns.countplot(y='sex', data=titanic, ax=ax[2], hue='pclass')

plt.show()
#Feed file path to read_csv

tips = pd.read_csv('../input/visualization/tips.csv')



# view dataset

tips.head()
mi = tips.total_bill.min()

ma = tips.total_bill.max()



bins = np.linspace(mi, ma, 5)

tips['bill_bin'] = pd.cut(tips.total_bill, bins).astype('category')
tips.head()
# Instantiate a figure and allocate 4 axes, on a 2-by-2 grid

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))



# Create a barplot to

sns.barplot(x='sex', y='total_bill', data=tips, ax=axes[0,0], estimator=np.mean)

axes[0,0].set_title('Mean bill by gender')



sns.barplot(x='day', y='total_bill', data=tips, ax=axes[0, 1], hue='sex', estimator=np.median)

axes[0,1].set_title('Median bill by gender and weekday')



sns.barplot(x='size', y='total_bill', data=tips, ax=axes[1,0], estimator=np.sum)

axes[1,0].set_title('Total bill by group size')



sns.barplot(x='bill_bin', y='total_bill', data=tips, estimator=np.size, ax=axes[1,1])

axes[1,1].set_title('Most contributing bill ranges')

plt.show()
tips.head()
fig, axes = plt.subplots(figsize=(8, 4))

sns.boxplot(x='day',y='total_bill', data=tips, ax=axes, hue='smoker')

plt.show()
fig, axes = plt.subplots(figsize=(12, 4))

sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips)

plt.show()
fig, axes = plt.subplots(figsize=(12, 4))

sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, split=True)

plt.show()
fig, axes = plt.subplots(figsize=(12, 4))

sns.swarmplot(x='day', y='total_bill', hue='smoker', data=tips, dodge=True)

plt.show()
fig, axes = plt.subplots(figsize=(12, 4))

sns.violinplot(x='day', y='total_bill', hue='sex', data=tips)

sns.swarmplot(x='day', y='total_bill', hue='sex', data=tips, dodge=True, palette=dict(Female='pink', Male='yellow'))

plt.show()
fig, axes = plt.subplots(figsize=(12, 4))

sns.distplot(tips.total_bill)

plt.show()
sns.jointplot('total_bill', 'tip', data=tips)
sns.pairplot(tips)
tips.corr()
sns.heatmap(tips.corr(), annot=True)