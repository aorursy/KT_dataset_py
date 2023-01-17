import pandas as pd

import numpy as np

import seaborn as sns

import os

import matplotlib.pyplot as plt

plt.style.use('default')

from pandas_profiling import ProfileReport

from IPython.display import display
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/big-five-european-soccer-leagues/BIG FIVE 1995-2019.csv')



# data info

print(df.info())



df.head()
profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_widgets()
pd.crosstab(df['Year'],df['Country'])
df['GGD'].plot.hist(bins=np.linspace(-.25,9.25,20),edgecolor='w',title='Game Goals Difference')
pd.crosstab(df['GGD'],df['Country'],normalize='index').style.format("{:.2%}").background_gradient(cmap='viridis')
pd.crosstab(df.loc[df['Year']==2018,'GGD'],df.loc[df['Year']==2018,'Country'],normalize='index').style.format("{:.2%}").background_gradient(cmap='viridis')
pd.crosstab(df['GGD'],df['Country'],normalize='columns').plot.bar(figsize=(18,6),title='Game Goals Difference',rot=0)
plt.figure(figsize=(24,20))



for idx,year in enumerate(df['Year'].unique()[-12:],start=1):

    plt.subplot(4,3,idx)

    for country in df['Country'].unique():

        sns.distplot(df.loc[(df['Country']==country) & (df['Year']==year),'GGD'],label=country,hist=False)

    plt.title(year)

    plt.legend()

    

#plt.tight_layout()

plt.show()
df.columns
df_rank=df.groupby(['Country','Year','Team_1'],as_index=False)['Team_1_(pts)'].sum().merge(

df.groupby(['Country','Year','Team_2'],as_index=False)['Team_2_(pts)'].sum(),right_on=['Country','Year','Team_2'],left_on=['Country','Year','Team_1'])

#

df_rank.head()
df_rank['Pts']=df_rank[['Team_1_(pts)','Team_2_(pts)']].sum(axis=1)



#

df_rank=df_rank[['Country','Year','Team_1','Pts']]

df_rank.rename(columns={'Team_1':'Team'},inplace=True)



df_rank.head()
plt.figure(figsize=(18,7))

sns.violinplot(x="Year", y="Pts", hue="Country",

               inner="quart",

               data=df_rank.loc[df_rank.Year.isin(['1995','2000','2005','2010','2019'])])

sns.despine(left=True)

plt.show()


g = sns.catplot(x="Year", y="Pts", hue="Country", 

                capsize=.2, height=6, aspect=2.5,

                kind="point", data=df_rank)

g.despine(left=True)
plt.figure(figsize=(18,7))



sns.boxplot(x="Year", y="Pts",

            hue="Country",

            data=df_rank.loc[df_rank.Year.isin(['1995','2000','2005','2010','2019'])])

sns.despine(offset=10, trim=True)

plt.show()
# Initialize the figure

f, ax = plt.subplots()

sns.despine(bottom=True, left=True)



# Show each observation with a scatterplot

sns.stripplot(x="Year", y="Pts", hue="Country",

              data=df_rank.loc[df_rank.Year.isin(['1995','2000','2005','2010','2019'])], 

              dodge=True, alpha=.25, zorder=1)



# Show the conditional means

sns.pointplot(x="Year", y="Pts", hue="Country",

              data=df_rank.loc[df_rank.Year.isin(['1995','2000','2005','2010','2019'])], 

              dodge=.532, join=False, palette="dark",

              markers="d", scale=.75, ci=None)



# Improve the legend 

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles[3:], labels[3:], title="Country",

          handletextpad=0, columnspacing=1,

          loc="lower right", ncol=3, frameon=True)
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)



with sns.axes_style("dark"):

    g = sns.FacetGrid(df_rank, hue="Country", col="Country", height=3)

g.map(hexbin, "Year", "Pts");
df_rank['Pts'].plot.hist(bins=100,edgecolor='w',title='Pts')
grid = sns.FacetGrid(df_rank, col="Country", margin_titles=True)

grid.map(plt.hist,"Pts", bins=100,density=True);