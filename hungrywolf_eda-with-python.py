import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/gapminder-datacamp-2007/gapminder_full.csv')

df.head()
df.shape
df.isna().sum()
df.dtypes
df.describe()
df.country.nunique()
df.year.nunique()
df.year.unique()
df.continent.unique()
def life_exp_vs_gdp_cap(year):



    dff = df[df['year']==year]

    plt.figure(dpi=150)

    np_pop = np.array(dff.population)

    np_pop2 = np_pop*2

    plot = sns.scatterplot(dff['gdp_cap'], dff['life_exp'], hue=dff['continent'], 

                    size=np_pop2, sizes=(20,500), alpha=0.5)

    h, l = plot.get_legend_handles_labels()

    col_lgd = plt.legend(h[:6], l[:6])

    plt.gca().add_artist(col_lgd)

    plt.grid(True)

    

    plt.xscale('log')

    plt.xlabel('GDP per Capita [in USD]', fontsize=14)

    plt.ylabel('Life Expectancy [in years]', fontsize=14)

    plot_title = 'World Development in ' + str(year)

    plt.title(plot_title, fontsize=20)

    plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

    plt.show()

    



life_exp_vs_gdp_cap(1952)

life_exp_vs_gdp_cap(2007)
def plot_dendrogram(dataframe, feature, title):

    plt.figure(figsize=(70, 30))

    mergings = linkage(dataframe[[feature]], method='complete')

    dendrogram(mergings, leaf_rotation=90, leaf_font_size=16, labels=list(dataframe['country']))

    plt.title(title, fontsize=100)

    plt.show()



def hierarchical_cluster(year):

    dff = df[df['year']==year]

    plot_dendrogram(dff, 'population', 'Population  '+ str(year))

    plot_dendrogram(dff, 'life_exp', 'Life Expectancy  '+ str(year))

    plot_dendrogram(dff, 'gdp_cap', 'GDP per Capita  '+ str(year))

    



hierarchical_cluster(2007)
def population_most_and_least(year):



    figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    figs.subplots_adjust(wspace=0.8)

    temp = df[df['year']==year]

    pop_max = temp['population'].nlargest(5)

    pop_min = temp['population'].nsmallest(5)

    

    ax1.bar(temp['country'].loc[pop_max.index], pop_max, color=['c', 'm', 'k', 'chartreuse', 'burlywood'])

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    ax1.set_ylabel('Population [in Billion]', fontsize=16)

    ax1.set_yticks([200000000, 400000000, 600000000, 800000000,1000000000, 1200000000, 1400000000], 

                   ['0.2bn', '0.4bn', '0.6bn', '0.8bn', '1bn', '1.2bn', '1.4bn'])

    ax1.title.set_text('5 most populated countries')



    ax2.bar(temp['country'].loc[pop_min.index], pop_min, color=['salmon', 'olivedrab', 

                                                                'steelblue', 'orchid', 'lightseagreen'])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

    ax2.title.set_text('5 least populated countries')

    title = 'Population    ' + str(year)

    figs.suptitle(title, fontsize=30)

    plt.show()



population_most_and_least(1952)

population_most_and_least(2007)
def life_exp_most_and_least(year):



    figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    figs.subplots_adjust(wspace=0.8)

    temp = df[df['year']==year]

    life_max = temp['life_exp'].nlargest(5)

    life_min = temp['life_exp'].nsmallest(5)



    ax1.bar(temp['country'].loc[life_max.index], life_max, color=['orangered', 'm', 'olive', 'chartreuse', 'burlywood'])

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    ax1.set_ylim([68, 85])

    ax1.set_ylabel('Life Expectancy [in years]', fontsize=16)

    ax1.title.set_text('Top 5 countries')



    ax2.bar(temp['country'].loc[life_min.index], life_min, color=['salmon', 'cyan', 

                                                                'royalblue', 'maroon', 'lightseagreen'])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

    ax2.set_ylim([28, 45])

    ax2.set_ylabel('Life Expectancy [in years]', fontsize=16)

    ax2.title.set_text('Bottom 5 countries')

    title = 'Life Expectancy   ' + str(year)

    figs.suptitle(title, fontsize=30)

    plt.show()

life_exp_most_and_least(1952)

life_exp_most_and_least(2007)
def gdp_cap_most_and_least(year):

    figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    figs.subplots_adjust(wspace=0.8)

    temp = df[df['year']==year]

    gdp_max = temp['gdp_cap'].nlargest(5)

    gdp_min = temp['gdp_cap'].nsmallest(5)



    ax1.bar(temp['country'].loc[gdp_max.index], gdp_max, color=['darkorange', 'forestgreen', 'teal', 'silver', 'purple'])

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    ax1.set_ylabel('GDP per Capita [in USD]', fontsize=16)

    ax1.title.set_text('Top 5 countries')



    ax2.bar(temp['country'].loc[gdp_min.index], gdp_min, color=['peru', 'dodgerblue', 

                                                                'midnightblue', 'darkorchid', 'lightseagreen'])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)



    ax2.set_ylabel('GDP per Capita [in USD]', fontsize=16)

    ax2.title.set_text('Bottom 5 countries')

    title = 'GDP per Capita [in USD]   ' + str(year)

    figs.suptitle(title, fontsize=30)

    plt.show()



gdp_cap_most_and_least(1952) 

gdp_cap_most_and_least(2007)