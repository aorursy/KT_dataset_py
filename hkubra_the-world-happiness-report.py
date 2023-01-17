import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import seaborn as sns



# Plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) # #do not miss this line

import plotly as py

import plotly.graph_objs as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_2015 = pd.read_csv('../input/world-happiness/2015.csv')

data_2016 = pd.read_csv('../input/world-happiness/2016.csv')

data_2017 = pd.read_csv('../input/world-happiness/2017.csv')

data_2018 = pd.read_csv('../input/world-happiness/2018.csv')

data_2019 = pd.read_csv('../input/world-happiness/2019.csv')
data_2015.head()
data_2016.head()
data_2017.head()
data_2018.head()
data_2019.head()
data_2015.info()
data_2016.info()
data_2017.info()
data_2018.info()
data_2019.info()
display(data_2015.describe())

print('2015 '+'*' * 40)

display(data_2016.describe())

print('2016 '+'*' * 40)

display(data_2017.describe())

print('2017 '+'*' * 40)

display(data_2018.describe())

print('2018 '+'*' * 40)

display(data_2019.describe())

print('2019 '+'*' * 40)
data_2015 = data_2015.rename(columns={"Happiness Rank":"Rank", "Happiness Score":"Score", "Economy (GDP per Capita)": "Economy", 

                                      "Health (Life Expectancy)": "Life_Expectancy", "Trust (Government Corruption)":"Corruption"})



data_2016 = data_2016.rename(columns={"Happiness Rank":"Rank","Happiness Score":"Score", "Economy (GDP per Capita)": "Economy", 

                                      "Health (Life Expectancy)": "Life_Expectancy", "Trust (Government Corruption)":"Corruption"})



data_2017 = data_2017.rename(columns={"Happiness.Rank":"Rank", "Happiness.Score":"Score", "Economy..GDP.per.Capita.": "Economy", 

                                      "Health..Life.Expectancy.": "Life_Expectancy", "Trust..Government.Corruption.":"Corruption"})



data_2018 = data_2018.rename(columns={"Overall rank":"Rank", "Country or region":"Country","GDP per capita":"Economy", 

                                      "Healthy life expectancy":"Life_Expectancy","Freedom to make life choices":"Freedom", 

                                      "Perceptions of corruption":"Corruption"})



data_2019 = data_2019.rename(columns={"Overall rank":"Rank", "Country or region":"Country", "GDP per capita": "Economy", 

                                      "Healthy life expectancy": "Life_Expectancy", "Freedom to make life choices":"Freedom", 

                                      "Perceptions of corruption":"Corruption"})
display(data_2015[data_2015.Country=='Turkey'])



display(data_2016[data_2016.Country=='Turkey'])



display(data_2017[data_2017.Country=='Turkey'])



display(data_2018[data_2018.Country=='Turkey'])



display(data_2019[data_2019.Country=='Turkey'])
# Türkiye'ye ait verilerin tek bir dataframe'de gösterilmesi

datas = [data_2015, data_2016, data_2017, data_2018, data_2019]

years = [2015,2016,2017,2018,2019]

rank_list = []

score_list = []

economy_list = []

life_list = []

freedom_list = []

generosity_list = []

corruption_list = []



for i in range(len(datas)):

    rank_list.append((datas[i][datas[i]['Country']=='Turkey']['Rank']).values[0])

    score_list.append((datas[i][datas[i]['Country']=='Turkey']['Score']).values[0])

    economy_list.append((datas[i][datas[i]['Country']=='Turkey']['Economy']).values[0])

    life_list.append((datas[i][datas[i]['Country']=='Turkey']['Life_Expectancy']).values[0])

    freedom_list.append((datas[i][datas[i]['Country']=='Turkey']['Freedom']).values[0])

    generosity_list.append((datas[i][datas[i]['Country']=='Turkey']['Generosity']).values[0])

    corruption_list.append((datas[i][datas[i]['Country']=='Turkey']['Corruption']).values[0])

    

    

turkey_data = pd.DataFrame({"years":years,"rank": rank_list, "score":score_list, "economy":economy_list,"life_expectancy":life_list,

                            "freedom":freedom_list, "generosity": generosity_list, "corruption":corruption_list})

turkey_data
tr_data = turkey_data.drop(['years'], axis=1)

f,ax = plt.subplots(figsize=(6, 6))

heat_map = sns.heatmap(tr_data.corr(), annot=True, linewidths=0.6, fmt= '.2f',ax=ax, cmap="coolwarm")

heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)

plt.show()
import plotly.graph_objs as go



trace1=go.Bar(

                x=years,

                y=turkey_data.economy,

                name="Economy",

                marker=dict(color = 'rgba(156, 30, 130, 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Economy')

trace2=go.Bar(

                x=years,

                y=turkey_data.life_expectancy,

                name="Life Expectancy",

                marker=dict(color = 'rgba(240,120,10 , 0.7)', 

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Life Expectancy')



trace3=go.Bar(

                x=years,

                y=turkey_data.freedom,

                name="Freedom",

                marker=dict(color = 'rgba( 50, 240,120 , 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Freedom')

trace4=go.Bar(

                x=years,

                y=turkey_data.generosity,

                name="Generosity",

                marker=dict(color = 'rgba(200, 250,20 , 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Generosity')

trace5=go.Bar(

                x=years,

                y=turkey_data.corruption,

                name="Corruption",

                marker=dict(color = 'rgba(200, 10,10 , 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Corruption')



edit_df=[trace1,trace2,trace3,trace4,trace5]

layout=go.Layout(barmode="group",title="2015-2019 arasındaki yıllara göre Türkiye'nin mutluluk raporu")

fig=dict(data=edit_df,layout=layout)

plt.savefig('graph.png')

iplot(fig)
# Yıllara göre değişim grafikleri

fig, ax =plt.subplots(nrows=2,ncols=3, figsize=(16,6))

sns.lineplot(x = "years", y = "score", data = turkey_data,color="coral", ax=ax[0][0])

sns.lineplot(x = "years", y = "economy", data = turkey_data,color="red", ax=ax[0][1])

sns.lineplot(x = "years", y = "life_expectancy", data = turkey_data, color="purple", ax=ax[0][2])

sns.lineplot(x = "years", y = "freedom", data = turkey_data, color="green", ax=ax[1][0])

sns.lineplot(x = "years", y = "generosity", data = turkey_data, color="blue", ax=ax[1][1])

sns.lineplot(x = "years", y = "corruption", data = turkey_data, color="black", ax=ax[1][2])

plt.show()
# 2019 yılının en mutlu ilk 10 ülkesi

data_2019.head(10)
f,ax = plt.subplots(figsize=(8, 8))

heat_map = sns.heatmap(data_2019.corr(), annot=True, linewidths=0.6, fmt= '.2f',ax=ax, cmap="coolwarm")

heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)

plt.show()