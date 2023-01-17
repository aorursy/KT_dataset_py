import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
print(apps.shape)
apps.info()
apps.describe()
apps.head()
apps.describe(include = 'O')
apps.dropna(subset = ['Rating'], inplace = True)
apps['App'].drop_duplicates(inplace = True)
apps.dropna(inplace = True)
apps.info()
def PlotDistribution(feature):

    fig, ax = plt.subplots(figsize = (18, 6))



    plt.subplot(121)

    sns.boxplot(x = feature, data = apps, palette = 'summer')

    plt.title('Boxplot - Distribution of ' + str(feature.strip("''")))



    plt.subplot(122)

    plt.hist(apps[feature], color = 'seagreen')

    plt.title('Histogram - Distribution of ' + str(feature.strip("''")))



    fig.suptitle('Univariate Analysis - ' + str(feature.strip("''")), fontsize = 18)

    plt.show()
def GroupByFeature(feature):

    gr_feature = apps.groupby(feature)['App'].count().reset_index()

    gr_feature['Percent'] = round(gr_feature['App'] / len(apps) * 100, 2).astype('str') + '%'

    return gr_feature
def PlotCount(feature):

    gr_data = GroupByFeature(feature)

    

    fig, ax = plt.subplots(figsize = (14, 6))

    ax = sns.barplot(x = feature, y = 'App', data = gr_data, palette = 'winter')

    

    for index, row in gr_data.iterrows():

        ax.text(row.name, row.App, row.Percent, color = 'k', ha = 'center', rotation = 90, va = 'bottom', fontsize = 10)

    

    plt.title('Applications by ' + str(feature.strip("''")), fontsize = 16)

    plt.xlabel(str(feature.strip("''")), fontsize = 13)

    plt.ylabel('#Applications', fontsize = 13)

    plt.xticks(rotation = 90)

    

    plt.show()
PlotDistribution('Rating')
PlotCount('Category')
apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors = 'coerce')

apps['Reviews'] = apps['Reviews'].astype('int64')
PlotDistribution('Reviews')
PlotCount('Installs')
PlotCount('Content Rating')
PlotCount('Android Ver')
gr_type = apps.groupby('Type')['App'].count().reset_index()

gr_type
fig, ax = plt.subplots(figsize = (12, 7))



labels = gr_type['Type']

size = gr_type['App']

colors = ['lightcoral', 'yellowgreen']



plt.pie(size, labels = labels, colors = colors, shadow = True, autopct = '%1.2f%%', explode = [0, 0.5])



plt.title('Applications by Type', fontsize = 16)

plt.show()
apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])
apps['Years Since Updated'] = 2020 - apps['Last Updated'].dt.year
PlotDistribution('Years Since Updated')
priced_apps = apps.loc[apps['Type'] == 'Paid']

priced_apps.head()
priced_apps['Price_Corrected'] = priced_apps['Price'].str.strip('$')



priced_apps['Price_Corrected'] = pd.to_numeric(priced_apps['Price_Corrected'], errors = 'coerce')

priced_apps['Price_Corrected'] = priced_apps['Price_Corrected'].astype('float64')



priced_apps.head()
fig, ax = plt.subplots(figsize = (18, 6))



plt.subplot(121)

sns.violinplot(x = 'Price_Corrected', data = priced_apps, palette = 'summer')

plt.title('Distribution of Price')



plt.subplot(122)

sns.kdeplot(priced_apps['Price_Corrected'], shade = True, color = 'k')

plt.title('KDE Plot - Distribution of Price')



fig.suptitle('Univariate Analysis - Price', fontsize = 18)

plt.show()