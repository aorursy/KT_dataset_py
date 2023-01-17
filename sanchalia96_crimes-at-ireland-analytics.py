from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
Ireland_crime1 = pd.read_csv("../input/Ireland-crime1.csv")

Ireland_crime1.head()
Ireland_crime1.info()
Ireland_crime1.isnull().sum()

Ireland_crime1 = Ireland_crime1.drop("id",axis = 1)
Ireland_crime1.head()
Ireland_crime2 = Ireland_crime1[["Divisions","Year","Number.of.Crime.Record"]]

Ireland_crime2.head()
Ireland_crime1.iloc[0]
def standard_plot(title=None, x=None, y=None):

    

    fig, ax = plt.subplots(figsize=(20,4), dpi=80)

    

    ax.grid(False)

    ax.spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)

    ax.xaxis.label.set_fontsize(12)

    ax.yaxis.label.set_fontsize(14)

   

    if title != None:

        ax.set_title(title)

        ax.title.set_fontsize(15)

        

    if x != None:

        ax.set_xlabel(x)

        

    if y != None:

        ax.set_ylabel(y)

          

    return fig, ax
fig, ax = standard_plot('Ireland crime rate over the years')

sns.lineplot(data=Ireland_crime1, x='Year', y='Number.of.Crime.Record', color='green', ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of Crimes');
# Conclusion:

#     1. Crime rate in Ireland was lowest in 2016.

#     2. Crime rate was highest in 2008.

#     3. Rate was almost constant from 2008 to 2015.

#     4. Crime rate increased recently in 2018.  
fig, ax = standard_plot('Ireland crime rate over Stations')

sns.lineplot(data=Ireland_crime1, x='Station', y='Number.of.Crime.Record', color='blue', ax=ax)

ax.set_xlabel('Station')

ax.set_ylabel('Number of Crimes');
# Conclusion:

#     1. Crimes in D.M.R. station was highest.

#     2. Crimes in Tipperary station was lowest.

#     3. Cork and Carlow holds the second position of Crime rate in Ireland
# Let's plot yearly rate of some highest crime places in Ireland Divisions.



df = Ireland_crime2.set_index("Divisions")

df

df1_dmr = df.loc[["Total D.M.R. Division"]]

df2_carlow = df.loc[["Total Carlow Division"]]

df3_cork = df.loc[["Total Cork Division"]]
fig, ax = standard_plot('Total D.M.R. Division crime rate over the years')

sns.lineplot(data = df1_dmr, x='Year', y='Number.of.Crime.Record', color='gray', ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of Crimes')
fig, ax = standard_plot('Total Carlow Division crime rate over the years')

sns.lineplot(data = df2_carlow, x='Year', y='Number.of.Crime.Record', color='gray', ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of Crimes')
fig, ax = standard_plot('Total Cork Division crime rate over the years')

sns.lineplot(data = df3_cork, x='Year', y='Number.of.Crime.Record', color='gray', ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of Crimes')
#Now let's look at Station wise crime rates



Ireland_crime3 = Ireland_crime1[["Station","Year","Number.of.Crime.Record"]]

Ireland_crime3.head()
# Handling null values of Sations 



prev = ''

for index, row in Ireland_crime3.iterrows():

    i = row['Station']

    if isinstance(i, str):

        prev = i 

    else:  

        Ireland_crime3.Station[index] = prev
# Now making Station column an index



#Ireland_crime3.set_index('Station',inplace = True)

#Ireland_crime3 = Ireland_crime3.reset_index()
# Now taking mean of all the crimes over the year in some stations



Ireland_crime_mean = Ireland_crime3.groupby(['Station']).mean()

Ireland_crime_mean = Ireland_crime_mean.reset_index()

Ireland_crime_mean.head()
Ireland_crime_mean = Ireland_crime_mean.drop("Year",axis=1)
df_top5 = Ireland_crime_mean.nlargest(5, ['Number.of.Crime.Record']) 

df_top5
def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
fig, ax = standard_plot('Mean crime rates of top 5 Ireland stations over the last 10 years')

ax = sns.barplot(x="Station", y="Number.of.Crime.Record", data=df_top5)

show_values_on_bars(ax)