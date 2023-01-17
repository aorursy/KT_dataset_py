# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

from scipy import stats







# 3 option for print all columns in the dataframes

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import plotly.express as px

template = 'plotly_dark'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_complet = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates=['Date'])

df_complet.tail()
typ_case = ["Confirmed","Deaths","Recovered","Active"]

dataset = df_complet.copy()

dataset["Active"] = dataset["Confirmed"] - dataset["Deaths"] - dataset["Recovered"]

dataset[typ_case] = dataset[typ_case].fillna(0)

dataset_group = dataset.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active"].sum().reset_index()
temp_f = dataset_group.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
temp_flg = dataset_group[['Country/Region', 'Deaths']]

temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)

temp_flg = temp_flg.reset_index(drop=True)

temp_flg = temp_flg[temp_flg['Deaths']>0]

temp_flg.style.background_gradient(cmap='Reds')
temp = dataset_group[dataset_group['Confirmed']==dataset_group['Deaths']]

temp = temp[['Country/Region', 'Confirmed', 'Deaths']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Reds')
temp = temp_f[temp_f['Recovered']==0][['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

temp.reset_index(drop=True).style.background_gradient(cmap='Reds')
temp = dataset_group[dataset_group['Confirmed']==dataset_group['Recovered']]

temp = temp[['Country/Region', 'Confirmed', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
# html embedding

from IPython.display import Javascript

from IPython.core.display import display, HTML



HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script 

src="https://public.flourish.studio/resources/embed.js"></script></div>''')
df_br = df_complet[df_complet["Country/Region"] == "Brazil"].reset_index(drop=True)

df_br["serie"] = [x+1 for x in range(len(df_br))]

df_it = df_complet[df_complet["Country/Region"] == "Italy"].reset_index(drop=True)

df_it["serie"] = [x+1 for x in range(len(df_it))]

df_ch = df_complet[df_complet["Country/Region"] == "China"].reset_index(drop=True)

df_ch["serie"] = [x+1 for x in range(len(df_ch))]

df_us = df_complet[df_complet["Country/Region"] == "US"].reset_index(drop=True)

df_us["serie"] = [x+1 for x in range(len(df_us))]

df_ir = df_complet[df_complet["Country/Region"] == "Iran"].reset_index(drop=True)

df_ir["serie"] = [x+1 for x in range(len(df_ir))]

df_ge = df_complet[df_complet["Country/Region"] == "Germany"].reset_index(drop=True)

df_ge["serie"] = [x+1 for x in range(len(df_ge))]



plt.figure(figsize=(12,7))

"""

# gca stands for 'get current axis'

ax = plt.gca()



df_br.plot(x ='Date', y='Confirmed', color='blue', kind = 'line',ax=ax)

df_it.plot(x ='Date', y='Confirmed', color='green', kind = 'line',ax=ax)

#df_ch.plot(x ='Date', y='Confirmed', color='black', kind = 'line',ax=ax)

#df_us.plot(x ='Date', y='Confirmed', color='red', kind = 'line',ax=ax)

#df_ir.plot(x ='Date', y='Confirmed', color='purple', kind = 'line',ax=ax)

#df_ge.plot(x ='Date', y='Confirmed', color='brown', kind = 'line',ax=ax)



df_br.plot(x ='Date', y='Deaths', color='red', kind = 'line',ax=ax)

df_br.plot(x ='Date', y='Recovered', color='green', kind = 'line',ax=ax)



plt.legend(['confirmed', 'deaths','recovered'], loc='upper left')

plt.rcParams['figure.facecolor'] = 'xkcd:white'



dict_style_title = {'fontsize':30,

                    'fontweight' : 'bold',

                    'color' : 'black',

                    'verticalalignment': 'baseline'}



plt.title('Comp', fontdict = dict_style_title"""

#plt.plot(df_br["serie"], df_br["Confirmed"])

plt.plot(df_it["serie"], df_it["Confirmed"])

plt.show()
df_br.tail(1)
# Ploting cases on world map

import folium

world_curr = df_complet[df_complet['Date'] == df_complet['Date'].max()]

map = folium.Map(location=[30, 30], tiles = "CartoDB dark_matter", zoom_start=2.2)

for i in range(0,len(world_curr)):

    folium.Circle(location=[world_curr.iloc[i]['Lat'],

                            world_curr.iloc[i]['Long']],

                            radius=(math.sqrt(world_curr.iloc[i]['Confirmed'])*4000 ),

                            color='crimson',

                            fill=True,

                            fill_color='crimson').add_to(map)

map
# Exploring word cloud based on STATE value

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df_complet["Country/Region"]))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('country.png')

plt.show()
covid_oc =  df_complet[(df_complet['Country/Region'] == 'China')]

sotoc = covid_oc.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()



sotoc_d = sotoc.sort_values('Date', ascending=False)

sotoc_d["Tx Death/Confirmed"] = sotoc_d["Deaths"]/sotoc_d["Confirmed"]

sotoc_d["Tx Recovered/Confirmed"] = sotoc_d["Recovered"]/sotoc_d["Confirmed"]

sotoc_d["Tx Deaths/Recovered"] = sotoc_d["Deaths"]/sotoc_d["Recovered"]

sotoc_d.head(5).style.background_gradient(cmap='OrRd')
plt.plot(sotoc_d["Date"], sotoc_d["Tx Death/Confirmed"])

plt.xlabel("Date")

plt.ylabel("Rate")

plt.title("China")



plt.show()
covid_ita =  df_complet[(df_complet['Country/Region'] == 'Italy')]

sotoc_ita = covid_ita.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()



sotoc_ita = sotoc_ita.sort_values('Date', ascending=False)

sotoc_ita["Tx Death/Confirmed"] = (sotoc_ita["Deaths"]/sotoc_ita["Confirmed"])*100

sotoc_ita["Tx Recovered/Confirmed"] = (sotoc_ita["Recovered"]/sotoc_ita["Confirmed"])*100

sotoc_ita["Tx Deaths/Recovered"] = (sotoc_ita["Deaths"]/sotoc_ita["Recovered"])*100

sotoc_ita.head(1).style.background_gradient(cmap='OrRd')
plt.plot(sotoc_ita["Date"], sotoc_ita["Tx Death/Confirmed"])

plt.xlabel("Date")

plt.ylabel("Rate")

plt.title("Italy")

plt.show()
sotoc = sotoc[sotoc['Date'] > '2020-01-22']
# Producing daily data difference for Confirmed, Death, Recovered

#sum_ocd = covid.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()

sum_ocd = sotoc.sort_values('Date', ascending=True)

sum_ocd = pd.DataFrame(sum_ocd.set_index('Date').diff()).reset_index()

#sum_d = pd.DataFrame(round(sum_d.set_index('DATE').pct_change()*100)).reset_index()

sum_ocd = sum_ocd[sum_ocd['Date'] > '2020-01-23']

print(sum_ocd.tail())



plt.figure(figsize=(12,7))



# gca stands for 'get current axis'

ax = plt.gca()



sum_ocd.plot(x ='Date', y='Confirmed', color='blue', kind = 'line',ax=ax)

sum_ocd.plot(x ='Date', y='Deaths', color='red', kind = 'line',ax=ax)

sum_ocd.plot(x ='Date', y='Recovered', color='green', kind = 'line',ax=ax)



plt.legend(['confirmed', 'deaths','recovered'], loc='upper left')

plt.rcParams['figure.facecolor'] = 'xkcd:white'



dict_style_title = {'fontsize':30,

                    'fontweight' : 'bold',

                    'color' : 'black',

                    'verticalalignment': 'baseline'}



plt.title('Results Per Day', fontdict = dict_style_title)

plt.show()




covid_oc =  df_complet[(df_complet['Date'] == '2020-03-19 00:00:00')]

sotoc_all = covid_oc.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

#sotoc_d = sotoc.sort_values('Date', ascending=False)

sotoc_all["Tx Death/Confirmed"] = sotoc_all["Deaths"]/sotoc_all["Confirmed"]

sotoc_all["Tx Recovered/Confirmed"] = sotoc_all["Recovered"]/sotoc_all["Confirmed"]

#sotoc_all["Tx Deaths/Recovered"] = sotoc_all["Deaths"]/sotoc_all["Recovered"]

X = sotoc_all.sort_values("Tx Death/Confirmed", ascending = False)[sotoc_all["Tx Death/Confirmed"]>0]

#X = sotoc_all
#Alpha of Linear Regression



df_complet = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates=['Date'])



def round_up(n, decimals=0):

    multiplier = 10 ** decimals

    return math.ceil(n * multiplier) / multiplier



#To retrieve the intercept:

#print(regressor.intercept_)

#For retrieving the slope:

#print(regressor.coef_)







#sotoc_ita = covid_ita.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()



#sotoc_ita = sotoc_ita.sort_values('Date', ascending=False)

intercept_Conf = []

intercept_Death = []

intercept_Rec = []

for index, row in X.iterrows():

    input_data = df_complet[df_complet["Country/Region"]==row[0]].reset_index()

    list_variable = [x+1 for x in range(len(input_data))]

    y = input_data["Confirmed"]

    slope, intercept, r_value, p_value, std_err = stats.linregress(list_variable,y)

    intercept_Conf.append(round_up(slope,4))

    

    y = input_data["Deaths"]

    slope, intercept, r_value, p_value, std_err = stats.linregress(list_variable,y)

    intercept_Death.append(round_up(slope,4))

    

    y = input_data["Recovered"]

    slope, intercept, r_value, p_value, std_err = stats.linregress(list_variable,y)

    intercept_Rec.append(round_up(slope,4))

    

    

    

X["alpha_conf"] = intercept_Conf

X["alpha_death"] =  intercept_Death

X["alpha_rec"] = intercept_Rec

X.reset_index()

X_n = X.iloc[:,1:11].values
from sklearn.cluster import KMeans



km=[]

#kmeans = KMeans(n_clusters=2,max_iter=300 ,random_state=0).fit(X_n)



for i in range(1,15):

    kmeans = KMeans(n_clusters=i ,init="k-means++", max_iter=300, random_state=0).fit(X_n)

    km.append(kmeans.inertia_)    

plt.plot(range(1,15), km)

plt.show()



kmeans = KMeans(n_clusters=3,init="k-means++", max_iter=300 ,random_state=0).fit(X_n)

y_kmeans = kmeans.fit_predict(X_n)

X["Cluster"]=y_kmeans



#kmeans.labels_



#kmeans.predict([[0, 0], [12, 3]])



#kmeans.cluster_centers_
covid_ita =  df_complet[(df_complet['Country/Region'] == 'Italy')]

sotoc_ita = covid_ita.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()



sotoc_ita = sotoc_ita.sort_values('Date', ascending=False)

sotoc_ita["Tx Death/Confirmed"] = (sotoc_ita["Deaths"]/sotoc_ita["Confirmed"])*100

sotoc_ita["Tx Recovered/Confirmed"] = (sotoc_ita["Recovered"]/sotoc_ita["Confirmed"])*100

sotoc_ita["Tx Deaths/Recovered"] = (sotoc_ita["Deaths"]/sotoc_ita["Recovered"])*100

sotoc_ita.head(10).style.background_gradient(cmap='OrRd')
ita = sotoc_ita.sort_values('Date', ascending=True)

ita = pd.DataFrame(ita.set_index('Date').diff()).reset_index()

ita = ita.sort_values('Date', ascending = False) 

ita.head(9).style.background_gradient(cmap='OrRd')
ita["Deaths"].sum()
plt.plot(ita["Date"], ita["Confirmed"])

plt.show()