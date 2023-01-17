# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
G20 = ['Australia', 'Canada', 'Saudi Arabia', 'United States', 'India', 'Russian Federation',

      'South Africa', 'Turkey', 'Argentina', 'Brazil', 'Mexico', 'France', 'Germany', 'Italy',

      'United Kingdom', 'China', 'Indonesia', 'Japan', 'South Korea', 'Korea, Rep.']
df = pd.read_csv("/kaggle/input/world-development-indicators/Indicators.csv")

df = df[(df['Year'] >= 1995) & (df['CountryName'].isin(G20))]

df.drop(['CountryCode', 'IndicatorCode'], axis=1, inplace=True)
def plot_scatter_year(indicator1, indicator2, cut1, cut2, countries, font, s, palette):

    D1 = df[(df['IndicatorName'] == indicator1) & (df['CountryName'].isin(countries))]

    D2 = df[(df['IndicatorName'] == indicator2) & (df['CountryName'].isin(countries))]

    merge_df = pd.merge(D1, D2,  how='inner', left_on=['CountryName','Year'], right_on = ['CountryName','Year'])

    merge_df = merge_df[(merge_df['Value_x'] <= cut1) & (merge_df['Value_y'] <= cut2)]

    

    arr1 = merge_df['Value_x']

    arr2 = merge_df['Value_y']

    print('Correlation =' + str(arr1.corr(arr2)))

    colors = merge_df['Year']

    

    figsize = (20, 10)



    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    xax = ax.xaxis

    yax = ax.yaxis



    #plt.scatter(arr1, arr2, c=colors, s=60)

    sns.scatterplot(x=arr1, y=arr2, hue=colors, legend='full', palette=palette, s=s)

    plt.legend(fontsize=16, markerscale=1.8, loc='best')

    plt.xlabel(indicator1, fontsize=font)

    plt.ylabel(indicator2, fontsize=font)



    for label in ax.xaxis.get_ticklabels():

        label.set_fontsize(font)

    for label in ax.yaxis.get_ticklabels():

        label.set_fontsize(font)

    #plt.colorbar()

    plt.show()

def plot_scatter_country(indicator1, indicator2, countries, cut1, cut2, font, s, palette):

    #D1 = df[df['IndicatorName'] == indicator1]

    #D2 = df[df['IndicatorName'] == indicator2]

    D1 = df[(df['IndicatorName'] == indicator1) & (df['CountryName'].isin(countries))]

    D2 = df[(df['IndicatorName'] == indicator2) & (df['CountryName'].isin(countries))]

    merge_df = pd.merge(D1, D2,  how='inner', left_on=['CountryName','Year'], right_on = ['CountryName','Year'])

    merge_df = merge_df[(merge_df['Value_x'] <= cut1) & (merge_df['Value_y'] <= cut2)]

    

    arr1 = merge_df['Value_x']

    arr2 = merge_df['Value_y']

    print('Correlation =' + str(arr1.corr(arr2)))

    

    colors = merge_df['CountryName'].apply(lambda x: G20.index(x))

    #colors = merge_df['CountryName']

    

    figsize = (20, 10)



    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    xax = ax.xaxis

    yax = ax.yaxis

    



    #plt.scatter(arr1, arr2, c=colors, s=60)

    #plt.colorbar()

    sns.scatterplot(x=arr1, y=arr2, hue=colors, legend='full', palette=palette, s=s)

    plt.legend(fontsize=16, markerscale=1.8, loc='best')

    plt.xlabel(indicator1, fontsize=font)

    plt.ylabel(indicator2, fontsize=font)



    for label in ax.xaxis.get_ticklabels():

        label.set_fontsize(font)

    for label in ax.yaxis.get_ticklabels():

        label.set_fontsize(font)

    plt.show()

plot_scatter_year('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', 75, 50, G20, 18, 240, 'YlOrRd')
plot_scatter_year('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', 25, 25, G20, 18, 240, 'YlOrRd')
plot_scatter_year('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', 25, 25, ['Italy'], 18, 240, 'YlOrRd')

plot_scatter_year('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', 25, 25, ['Italy'], 18, 240, 'YlOrRd')
plot_scatter_year('Lending interest rate (%)', 'GDP growth (annual %)', 25, 25, ['Russian Federation'], 18, 240, 'YlOrRd')
plot_scatter_country('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', G20, 20, 20, 18, 120, 'Paired')

#Inflation, consumer prices (annual %)

#Inflation, GDP deflator (annual %)
def plot_line(indicator1, indicator2, country, cut1, cut2, font, s, palette, title):

    D1 = df[(df['IndicatorName'] == indicator1) & (df['CountryName'] == country)]

    D2 = df[(df['IndicatorName'] == indicator2) & (df['CountryName'] == country)]

    merge_df = pd.merge(D1, D2,  how='inner', left_on=['CountryName','Year'], right_on = ['CountryName','Year'])

    merge_df = merge_df[(merge_df['Value_x'] <= cut1) & (merge_df['Value_y'] <= cut2)]

    

    if merge_df.shape[0] <= 2:

        return 0

    

    arr1 = merge_df['Value_x']

    arr2 = merge_df['Value_y']

    year = merge_df['Year']

    

    figsize = (20, 4)



    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    

    xax = ax.xaxis

    yax = ax.yaxis

    



    #plt.scatter(arr1, arr2, c=colors, s=60)

    #plt.colorbar()

    sns.lineplot(x=year, y=arr1, label=indicator1)

    sns.lineplot(x=year, y=arr2, label=indicator2)

    #plt.legend(fontsize=16, markerscale=1.8, loc='best')

    plt.xlabel('Year', fontsize=font)

    plt.ylabel('', fontsize=font)



    for label in ax.xaxis.get_ticklabels():

        label.set_fontsize(font)

    for label in ax.yaxis.get_ticklabels():

        label.set_fontsize(font)

    plt.figtext(.5,.9,title, fontsize=48, ha='center')

    

    

    plt.legend()

    plt.show()
for i in G20:

    plot_line('Lending interest rate (%)', 'Inflation, GDP deflator (annual %)', i, 

          30, 35, 16, 120, 'Paired', i)
plot_scatter_year('Lending interest rate (%)', 'GDP growth (annual %)', 25, 35, ['Russian Federation'], 18, 240, 'YlOrRd')
for i in G20:

    plot_line('Lending interest rate (%)', 'GDP growth (annual %)', i, 

          30, 35, 16, 120, 'Paired', i)
plot_line('Inflation, GDP deflator (annual %)', 'Inflation, consumer prices (annual %)', 'Russian Federation', 

          100, 100, 16, 120, 'Paired', 'Russian Federation')
plot_line('Inflation, GDP deflator (annual %)', 'Inflation, consumer prices (annual %)', 'United States', 

          100, 100, 16, 120, 'Paired', 'United States')
def plot_barplot(indicator, country, font, s, palette):

    D = df[(df['IndicatorName'] == indicator) & (df['CountryName'] == country)]

    

    arr = D['Value']

    year = D['Year']

    

    figsize = (20, 10)



    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    

    xax = ax.xaxis

    yax = ax.yaxis

    



    #plt.scatter(arr1, arr2, c=colors, s=60)

    #plt.colorbar()

    #print(arr)

    sns.barplot(x=year, y=arr, color='cornflowerblue')

    #sns.barplot(x="Year", y="IndicatorName", data=tips, ci="sd")

    #plt.legend(fontsize=16, markerscale=1.8, loc='best')

    plt.xlabel('Year', fontsize=font)

    plt.ylabel(indicator, fontsize=font)



    for label in ax.xaxis.get_ticklabels():

        label.set_fontsize(font)

    for label in ax.yaxis.get_ticklabels():

        label.set_fontsize(font)

    plt.figtext(.5,.9,indicator + ' ' + country, fontsize=24, ha='center')

    

    

    #plt.legend()

    plt.show()
plot_barplot('GDP (constant LCU)', 'Russian Federation', 16, 120, 'Paired')
plot_barplot('GDP growth (annual %)', 'Russian Federation', 16, 120, 'Paired')
plot_barplot('Official exchange rate (LCU per US$, period average)', 'Russian Federation', 16, 120, 'Paired')
def plot_barplots(indicator, countries, font, s, palette):

    D = df[(df['IndicatorName'] == indicator) & (df['CountryName'].isin(countries))]

    

    arr = D['Value']

    year = D['Year']

    hue = D['CountryName']

    

    figsize = (20, 10)



    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    

    xax = ax.xaxis

    yax = ax.yaxis

    



    #plt.scatter(arr1, arr2, c=colors, s=60)

    #plt.colorbar()

    #print(arr)

    sns.barplot(x=year, y=arr, hue=hue)

    #sns.barplot(x="Year", y="IndicatorName", data=tips, ci="sd")

    plt.legend(fontsize=16, markerscale=1.8, loc='best')

    plt.xlabel('Year', fontsize=font)

    plt.ylabel(indicator, fontsize=font)



    for label in ax.xaxis.get_ticklabels():

        label.set_fontsize(font)

    for label in ax.yaxis.get_ticklabels():

        label.set_fontsize(font)

    #plt.figtext(.5,.9,indicator, fontsize=24, ha='center')

    

    

    #plt.legend()

    plt.show()
plot_barplots('Unemployment, total (% of total labor force)', ['Russian Federation', 'United States'], 16, 120, 'Paired')
plot_barplots('Central government debt, total (% of GDP)', ['Russian Federation', 'United States'], 16, 120, 'Paired')
plot_barplots('Trade (% of GDP)', ['Russian Federation', 'United States'], 16, 120, 'Paired')
plot_barplots('Current account balance (% of GDP)', ['Russian Federation', 'United States'], 16, 120, 'Paired')
plot_barplots('Foreign direct investment, net inflows (% of GDP)', ['Russian Federation', 'United States'], 16, 120, 'Paired')