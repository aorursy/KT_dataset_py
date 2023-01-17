import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid')

#plt.style.use('ggplot')
data2015 = pd.read_csv('../input/world-happiness-report/2015.csv')

data2015['Year'] = 2015 



data2016 = pd.read_csv('../input/world-happiness-report/2016.csv')

data2016['Year'] = 2016



data2017 = pd.read_csv('../input/world-happiness-report/2017.csv')

data2017 = data2017.rename(columns = {'Happiness.Rank':'Happiness Rank', 'Happiness.Score' : 'Happiness Score', 

                                      'Economy..GDP.per.Capita.' : 'Economy (GDP per Capita)', 'Health..Life.Expectancy.' : 'Health (Life Expectancy)',

                                      'Trust..Government.Corruption.' : 'Trust (Government Corruption)', 'Dystopia.Residual' : 'Dystopia Residual'})

data2017['Year'] = 2017 





data2018 = pd.read_csv('../input/world-happiness-report/2018.csv')

data2018 = data2018.rename(columns = {'Overall rank':'Happiness Rank', 'Country or region' : 'Country', 'Score' : 'Happiness Score',

                                      'GDP per capita' : 'Economy (GDP per Capita)', 'Social support' : 'Family',

                                      'Healthy life expectancy' : 'Health (Life Expectancy)','Freedom to make life choices' : 'Freedom',

                                      'Perceptions of corruption' : 'Trust (Government Corruption)'})

data2018['Year'] = 2018





data2019 = pd.read_csv('../input/world-happiness-report/2019.csv')

data2019 = data2019.rename(columns = {'Overall rank':'Happiness Rank', 'Country or region' : 'Country', 'Score' : 'Happiness Score',

                                      'GDP per capita' : 'Economy (GDP per Capita)', 'Social support' : 'Family',

                                      'Healthy life expectancy' : 'Health (Life Expectancy)','Freedom to make life choices' : 'Freedom'

                                     , 'Perceptions of corruption' : 'Trust (Government Corruption)'})

data2019['Year'] = 2019 



data2020 = pd.read_csv('../input/world-happiness-report/2020.csv')

data2020['Happiness Rank'] =  range(1, len(data2020.index)+1)

data2020 = data2020.rename(columns = {'Country name' : 'Country', 'Ladder score' : 'Happiness Score', 

                                      'Logged GDP per capita' : 'Economy (GDP per Capita)', 'Social support' : 'Family', 'Healthy life expectancy' : 'Health (Life Expectancy)',

                                      'Freedom to make life choices' : 'Freedom', 'Perceptions of corruption' : 'Trust (Government Corruption)'})

data2020['Year'] = 2020 



data2020
datarank = pd.DataFrame(columns = ['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)',

                                   'Family', 'Health (Life Expectancy)', 'Freedom','Trust (Government Corruption)',

                                   'Generosity', 'Year'])

n = [data2015, data2016, data2017, data2018, data2019, data2020]

for i in n:

    datarank = datarank.append(i[['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)',

                                   'Family', 'Health (Life Expectancy)', 'Freedom','Trust (Government Corruption)',

                                   'Generosity', 'Year']], ignore_index = 'True')

#pd.set_option('display.max_rows', 1000)

datarank
def color_rank(value):

    if value == 1:

        color = 'yellow'

    elif value == 2:

        color = 'silver'

    elif value == 3:

        color = 'moccasin'

    else:

        color = 'default'

    return 'background-color: %s' % color





def highlight_max(data, color='yellow'):

    '''

    highlight the maximum in a Series or DataFrame

    '''

    attr = 'background-color: {}'.format(color)

    data = data.astype(float)

    if data.ndim == 1: 

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''),

                            index=data.index, columns=data.columns)

    



datapivothappiness_rank = datarank.pivot(index='Country', columns='Year', values=['Happiness Rank'])





datapivothappiness_rank = datapivothappiness_rank.dropna(axis = 0)

datapivothappiness_rank['AVG'] = datapivothappiness_rank.mean(axis = 1)

datapivothappiness_rank = datapivothappiness_rank.sort_values(by = ['AVG'])

datapivothappiness_rank['Rank'] = range(1, len(datapivothappiness_rank)+1)

datapivothappiness_rank.style.applymap(color_rank, subset = ['Happiness Rank', 'Rank'])


datapivothappiness_score = datarank.pivot_table(index='Country', columns='Year', values=['Happiness Score'])



datapivothappiness_score = datapivothappiness_score.reset_index()  



datapivothappiness_score = datapivothappiness_score.dropna(axis = 0)

datapivothappiness_score['AVG'] = datapivothappiness_score.mean(axis = 1)

datapivothappiness_score = datapivothappiness_score.sort_values(by = ['AVG'], ascending = False)



datapivothappiness_score['Rank'] = range(1, len(datapivothappiness_score)+1)

datapivothappiness_score = datapivothappiness_score.set_index(['Country', 'Rank'])

datapivothappiness_score.style.apply(highlight_max)

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Happiness Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivothappiness_score[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 8, step = 1))

for index,data in enumerate(np.round(datapivothappiness_score[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.5 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Happiness Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Happiness Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivothappiness_score[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 8, step = 1))

for index,data in enumerate(np.round(datapivothappiness_score[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.5 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Happiness Score (AVG)', fontsize=14)



plt.show()
datapivot_economy = datarank.pivot_table(index='Country', columns='Year', values=['Economy (GDP per Capita)'])



datapivot_economy = datapivot_economy.reset_index()  



datapivot_economy = datapivot_economy.dropna(axis = 0)

datapivot_economy['AVG'] = datapivot_economy.mean(axis = 1)

datapivot_economy = datapivot_economy.sort_values(by = ['AVG'], ascending = False)



datapivot_economy['Rank'] = range(1, len(datapivot_economy)+1)

datapivot_economy = datapivot_economy.set_index(['Country', 'Rank'])

datapivot_economy.style.apply(highlight_max)

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Economy Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_economy[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 4, step = 1))

for index,data in enumerate(np.round(datapivot_economy[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.2 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Economy Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Economy Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_economy[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 4, step = 1))

for index,data in enumerate(np.round(datapivot_economy[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.2 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Economy Score (AVG)', fontsize=14)



plt.show()
datapivot_family = datarank.pivot_table(index='Country', columns='Year', values=['Family'])



datapivot_family = datapivot_family.reset_index()  



datapivot_family = datapivot_family.dropna(axis = 0)

datapivot_family['AVG'] = datapivot_family.mean(axis = 1)

datapivot_family = datapivot_family.sort_values(by = ['AVG'], ascending = False)



datapivot_family['Rank'] = range(1, len(datapivot_family)+1)

datapivot_family = datapivot_family.set_index(['Country', 'Rank'])

datapivot_family.style.apply(highlight_max)

plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Family Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_family[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 1.6, step = 0.2))

for index,data in enumerate(np.round(datapivot_family[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.1 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Family Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Family Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_family[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 1.6, step = 0.2))

for index,data in enumerate(np.round(datapivot_family[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.1 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Family Score (AVG)', fontsize=14)



plt.show()
datapivot_health = datarank.pivot_table(index='Country', columns='Year', values=['Health (Life Expectancy)'])



datapivot_health = datapivot_health.reset_index()  



datapivot_health = datapivot_health.dropna(axis = 0)

datapivot_health['AVG'] = datapivot_health.mean(axis = 1)

datapivot_health = datapivot_health.sort_values(by = ['AVG'], ascending = False)



datapivot_health['Rank'] = range(1, len(datapivot_health)+1)

datapivot_health = datapivot_health.set_index(['Country', 'Rank'])

datapivot_health.style.apply(highlight_max)
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Health Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_health[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 14, step = 1))

for index,data in enumerate(np.round(datapivot_health[0:5]['AVG'], 4)):

    plt.text(x=index-0.25 , y =data-1 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Health Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Health Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_health[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 14, step = 1))

for index,data in enumerate(np.round(datapivot_health[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-1 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Health Score (AVG)', fontsize=14)



plt.show()
datapivot_freedom = datarank.pivot_table(index='Country', columns='Year', values=['Freedom'])



datapivot_freedom = datapivot_freedom.reset_index()  



datapivot_freedom = datapivot_freedom.dropna(axis = 0)

datapivot_freedom['AVG'] = datapivot_freedom.mean(axis = 1)

datapivot_freedom = datapivot_freedom.sort_values(by = ['AVG'], ascending = False)



datapivot_freedom['Rank'] = range(1, len(datapivot_freedom)+1)

datapivot_freedom = datapivot_freedom.set_index(['Country', 'Rank'])

datapivot_freedom.style.apply(highlight_max)
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Freedom Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_freedom[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 0.8, step = 0.1))

for index,data in enumerate(np.round(datapivot_freedom[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.05 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Freedom Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Freedom Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_freedom[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 0.8, step = 0.1))

for index,data in enumerate(np.round(datapivot_freedom[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.05 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Freedom Score (AVG)', fontsize=14)



plt.show()
datapivot_trustgov = datarank.pivot_table(index='Country', columns='Year', values=['Trust (Government Corruption)'])



datapivot_trustgov = datapivot_trustgov.reset_index()  



datapivot_trustgov = datapivot_trustgov.dropna(axis = 0)

datapivot_trustgov['AVG'] = datapivot_trustgov.mean(axis = 1)

datapivot_trustgov = datapivot_trustgov.sort_values(by = ['AVG'], ascending = False)



datapivot_trustgov['Rank'] = range(1, len(datapivot_trustgov)+1)

datapivot_trustgov = datapivot_trustgov.set_index(['Country', 'Rank'])

datapivot_trustgov.style.apply(highlight_max)
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Trust Goverment Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_trustgov[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 0.5, step = 0.05))

for index,data in enumerate(np.round(datapivot_trustgov[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.03 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Trust Goverment Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Trust Goverment Freedom Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_trustgov[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 0.5, step = 0.05))

for index,data in enumerate(np.round(datapivot_trustgov[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.03 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Trust Goverment Score (AVG)', fontsize=14)



plt.show()
datapivot_generosity = datarank.pivot_table(index='Country', columns='Year', values=['Generosity'])



datapivot_generosity = datapivot_generosity.reset_index()  



datapivot_generosity = datapivot_generosity.dropna(axis = 0)

datapivot_generosity['AVG'] = datapivot_generosity.mean(axis = 1)

datapivot_generosity = datapivot_generosity.sort_values(by = ['AVG'], ascending = False)



datapivot_generosity['Rank'] = range(1, len(datapivot_generosity)+1)

datapivot_generosity = datapivot_generosity.set_index(['Country', 'Rank'])

datapivot_generosity.style.apply(highlight_max)
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

plt.title('Top 5 Countries Based on Average Generosity Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_generosity[0:5]['AVG'].plot(kind = 'bar')

plt.xlabel('Countries,Rank', fontsize=14)

plt.yticks(np.arange(0, 0.8, step = 0.1))

for index,data in enumerate(np.round(datapivot_generosity[0:5]['AVG'], 4)):

    plt.text(x=index-0.2 , y =data-0.05 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Freedom Score (AVG)', fontsize=14)



plt.subplot(1,2,2)    

plt.title('Bottom 5 Countries Based on Average Generosity Score From 2015 to 2020', fontsize = 14, y=1.05)

datapivot_generosity[-6:-1]['AVG'].plot(kind = 'bar', color = 'red')

plt.xlabel('Countries,Rank', fontsize=14)



for index,data in enumerate(np.round(datapivot_generosity[-6:-1]['AVG'], 4)):

    plt.text(x=index-0.2 , y =0.002 , s=f"{data}" , fontdict=dict(fontsize=12))

plt.ylabel('Freedom Score (AVG)', fontsize=14)



plt.show()