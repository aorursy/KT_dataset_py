import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns 

import random





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, glob

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename)) 
#We first import the data (copy-paste of the input files' paths)

df_data =pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df_ts_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_ts_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_ts_confirmed_us = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv")

df_ts_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_ts_death_us = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv")

df_ll = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

df_oll = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
#We will first look at the Covid 'data'

df_data.head(3)
df_data[['Confirmed','Deaths','Recovered']]=df_data[['Confirmed','Deaths','Recovered']].astype(int) #To convert a float into an int



#Next we want to plot the number of deaths/recovered/confirmed cases according to the date 

#So we sum all the confirmed/deaths/recovered cases grouped by the date

df_plot = df_data.groupby(["ObservationDate"])['Confirmed','Deaths','Recovered'].sum().reset_index().sort_values("ObservationDate",ascending=True)

df_plot.head()
#We keep just some dates to have a better looking graph (else there is too much x labels to see something) :

L=np.linspace(0,219,150).astype(int)

df_plot_reduced=df_plot

for i in L :

    df_plot_reduced = df_plot_reduced.drop(i) 

df_plot_reduced.head()
#Graph that shows the number of confirmed cases according to the date

plt.figure(figsize=(20,5))

ax = sns.barplot(x=df_plot_reduced['ObservationDate'],

                 y=df_plot_reduced['Confirmed'],

                 data = df_plot_reduced, 

                 palette = sns.cubehelix_palette(86,start = 2.5)) #Palette of color purple/blue



ax.set(xlabel='Date',

       ylabel='Number of confirmed cases',

      title='Number of covid-19 confirmed cases')



#To be able to read the x-axis, we rotate the labels

ax.set_xticklabels(ax.get_xticklabels(),

                   rotation = 90, 

                   horizontalalignment='right');
#Number of recovery cases :

plt.figure(figsize=(20,5))

ax = sns.barplot(x=df_plot_reduced['ObservationDate'],y=df_plot_reduced['Recovered'],data = df_plot_reduced, palette = sns.cubehelix_palette(86,start =1.6,rot=0.1))



ax.set(xlabel='Date',

       ylabel='Number of recovered cases',

      title='Number of covid-19 recovered cases')



#To be able to read the x-axis, we rotate the labels

ax.set_xticklabels(ax.get_xticklabels(),

                   rotation = 90, 

                   horizontalalignment='right');
#Number of deaths :

plt.figure(figsize=(20,5))

ax = sns.barplot(x=df_plot_reduced['ObservationDate'],y=df_plot_reduced['Deaths'],data = df_plot_reduced, palette = sns.cubehelix_palette(86,start=1.1,rot=0.1))



ax.set(xlabel='Date',

       ylabel='Number of deaths',

      title='Number of covid-19 deaths')



#To be able to read the x-axis, we rotate the labels

ax.set_xticklabels(ax.get_xticklabels(),

                   rotation = 90, 

                   horizontalalignment='right');
#Dataframe with just the last day of the month, in order to find later the number of confirmed/deaths/recovered per month (and not the cumulate number)

labels=['01/31/2020','02/29/2020','03/31/2020','04/30/2020','05/31/2020','06/30/2020','07/31/2020','08/31/2020']



df_month = pd.DataFrame() #New DataFrame

for date in labels: #We just want the last day of the different months, and we concatenate the dataframe with its previous version for each month

    df_month = pd.concat(

        [df_plot.loc[df_plot['ObservationDate']==date] , df_month], 

        ignore_index=True)

    

df_month = df_month.sort_values(by='ObservationDate',ascending=False) #The frist line is for August, not necessary

df_month.head(10) #Number of total confirmed/deaths/recovered cases at the end pf each month
#Number of confirmed cases/recovered/deaths per month

#For example : Number in August = (number on 8/29/20) - (number on 7/31/20)



df_month.loc['August'] = df_month.loc[0][['Confirmed','Deaths','Recovered']] - df_month.loc[1][['Confirmed','Deaths','Recovered']]

df_month.loc['July'] = df_month.loc[1][['Confirmed','Deaths','Recovered']] - df_month.loc[2][['Confirmed','Deaths','Recovered']]

df_month.loc['June'] = df_month.loc[2][['Confirmed','Deaths','Recovered']] - df_month.loc[3][['Confirmed','Deaths','Recovered']]

df_month.loc['May'] = df_month.loc[3][['Confirmed','Deaths','Recovered']] - df_month.loc[4][['Confirmed','Deaths','Recovered']]

df_month.loc['April'] = df_month.loc[4][['Confirmed','Deaths','Recovered']] - df_month.loc[5][['Confirmed','Deaths','Recovered']]

df_month.loc['March'] = df_month.loc[5][['Confirmed','Deaths','Recovered']] - df_month.loc[6][['Confirmed','Deaths','Recovered']]

df_month.loc['February'] = df_month.loc[6][['Confirmed','Deaths','Recovered']] - df_month.loc[7][['Confirmed','Deaths','Recovered']]

df_month.loc['January'] = df_month.loc[7][['Confirmed','Deaths','Recovered']]



#We keep just the monthly cases and use iloc to sort it the right way

df_month = df_month.drop([i for i in range(0,8)]).drop('ObservationDate',axis=1).iloc[::-1]

df_month.head(8)
labels=['January','February','March','April','May','June','July','August'] #x label as well as the name of the df_month lines



fig, axs= plt.subplots(ncols=3,figsize=(21,5)) #Several charts on one figure

ax1=sns.barplot(x=labels,

                y=df_month['Confirmed'],

                data=df_month,

                ax=axs[0], #First chart

                palette = 'Blues')

ax1.set(xlabel='Month',ylabel='Number of confirmed cases',title='Number of covid-19 confirmed cases per month');



ax2=sns.barplot(x=labels,

                y=df_month['Deaths'],

                data=df_month,

                ax=axs[1], #Second chart

                color='Grey')

ax2.set(xlabel='Month',ylabel='Number of deaths cases',title='Number of covid-19 deaths per month');



ax3=sns.barplot(x=labels,

                y=df_month['Recovered'],

                data=df_month,

                ax=axs[2], #Third chart

                palette = 'Greens')

ax3.set(xlabel='Month',ylabel='Number of recovered cases',title='Number of covid-19 recovered cases per month');
df_ts_confirmed.head() #To see this dataframe
df_confirmed_top20 = df_ts_confirmed.sort_values(by='8/31/20',ascending=False).head(20) #because we want the 20 countries where there is the most recovered cases



plt.figure(figsize=(20,5)) #To have a bigger figure

ax=sns.barplot(

    x='Country/Region',

    y='8/31/20',

    data=df_confirmed_top20, 

    palette = "Blues_d"

)



ax.set(xlabel='Country',

       ylabel='Number of cases',

       title='Top 20 countries with the most confirmed cases on 8/31/2020');

#Scatter graph using plotly

fig = go.Figure(data=[go.Scatter(

    x=df_confirmed_top20['Country/Region'],

    y=df_confirmed_top20['8/31/20'],

    mode='markers',

    marker=dict(size=(df_confirmed_top20['8/31/20']/30000))

)])



fig.update_layout(title='Top 20 countries with the most confirmed cases on 8/31/2020',

    xaxis_title="Country",

    yaxis_title="Confirmed Cases"

)

fig.show()
df_recovered_top20 = df_ts_recovered.sort_values(by='8/31/20',ascending=False).head(20) #because we want the 20 countries where there is the most recovered cases



plt.figure(figsize=(20,5)) #To have a bigger figure

ax=sns.barplot(

    x='Country/Region',

    y='8/31/20',

    data=df_recovered_top20,

    palette = "Greens_d"

)



ax.set(xlabel='Country',

       ylabel='Number of cases',

       title='Top 20 countries with the most recovered cases on 8/31/2020');
df_deaths_top20 = df_ts_deaths.sort_values(by='8/31/20',ascending=False).head(20) #because we want the 20 countries where there is the most deaths cases



plt.figure(figsize=(20,5)) #To have a bigger figure

ax=sns.barplot(

    x='Country/Region',

    y='8/31/20',

    data=df_deaths_top20,

    palette = 'Greys_d'

)



ax.set(xlabel='Country',

       ylabel='Number of cases',

       title='Top 20 countries with the most recovered cases on 8/31/2020');
#We first create a new Dataframe

df_ratio = pd.DataFrame() 



df_ratio['Country/Region'] = df_ts_confirmed['Country/Region'] #To have the countries

df_ratio['Ratio deaths/confirmed'] = (df_ts_deaths.loc[df_ts_deaths['Province/State'].isnull()]['8/31/20']/df_ts_confirmed.loc[df_ts_confirmed['Province/State'].isnull()]['8/31/20'] ) * 100 # Deaths/confirmed percentage ratio

df_ratio = df_ratio.sort_values(by='Ratio deaths/confirmed',ascending = False).head(20)

df_ratio.head()
plt.figure(figsize=(20,5)) #To have a bigger figure

ax=sns.barplot(

    x='Country/Region',

    y='Ratio deaths/confirmed',

    data=df_ratio,

    palette='Reds_d'

)



ax.set(xlabel='Country',

       ylabel='Percentage',

       title='Top 20 countries with the highest deaths/confirmed cases on 8/31/2020')



ax.set_xticklabels(ax.get_xticklabels(),

                   rotation = 90, 

                   horizontalalignment='right');
#We will create a new dataframe which contains the number of confirmed cases per month

df_new_confirmed = pd.DataFrame() #Cretaion of the dataframe



df_new_confirmed['Country/Region'] = df_ts_confirmed['Country/Region'] #We copy the country column



#We calculate the number of confirmed cases for each month

df_new_confirmed['Number of new confirmed cases in August'] = df_ts_confirmed['8/31/20']-df_ts_confirmed['8/1/20']

df_new_confirmed['Number of new confirmed cases in July'] = df_ts_confirmed['7/31/20']-df_ts_confirmed['7/1/20']

df_new_confirmed['Number of new confirmed cases in June'] = df_ts_confirmed['6/30/20']-df_ts_confirmed['6/1/20']

df_new_confirmed['Number of new confirmed cases in May'] =df_ts_confirmed['5/31/20']-df_ts_confirmed['5/1/20']

df_new_confirmed['Number of new confirmed cases in April'] = df_ts_confirmed['4/30/20']-df_ts_confirmed['4/1/20']

df_new_confirmed['Number of new confirmed cases in March'] = df_ts_confirmed['3/31/20']-df_ts_confirmed['3/1/20']

df_new_confirmed['Number of new confirmed cases in February'] = df_ts_confirmed['2/29/20']-df_ts_confirmed['2/1/20']

df_new_confirmed['Number of new confirmed cases in January'] = df_ts_confirmed['1/31/20']-df_ts_confirmed['1/22/20']

df_new_confirmed.head()
#Graph that shows the 5 counties with the most number of confirmed cases due of covid-19 per month

fig, axs=plt.subplots(nrows=2,ncols=4,figsize = (25,15))



ax1=sns.barplot(x='Country/Region',

                y='Number of new confirmed cases in January',

                data=df_new_confirmed.sort_values(by='Number of new confirmed cases in January',ascending=False).head(5), #Top 5 countries

                ax=axs[0][0],

                palette="Blues_d")

ax1.set(xlabel='Country',

        ylabel='Number of confirmed cases',

        title='The number of confirmed cases in January')



ax2=sns.barplot(x='Country/Region',y='Number of new confirmed cases in February',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in February',ascending=False).head(5),ax=axs[0][1],palette="Blues_d")

ax2.set(xlabel='Country',ylabel='',title='The number of confirmed cases in February')



ax3=sns.barplot(x='Country/Region',y='Number of new confirmed cases in March',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in March',ascending=False).head(5),ax=axs[0][2],palette="Blues_d")

ax3.set(xlabel='Country',ylabel='',title='The number of confirmed cases in March')



ax4=sns.barplot(x='Country/Region',y='Number of new confirmed cases in April',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in April',ascending=False).head(5),ax=axs[0][3],palette="Blues_d")

ax4.set(xlabel='Country',ylabel='',title='The number of confirmed cases in April')



ax5=sns.barplot(x='Country/Region',y='Number of new confirmed cases in May',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in May',ascending=False).head(5),ax=axs[1][0],palette="Blues_d")

ax5.set(xlabel='Country',ylabel='Number of confirmed cases',title='The number of confirmed cases in May')



ax6=sns.barplot(x='Country/Region',y='Number of new confirmed cases in June',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in June',ascending=False).head(5),ax=axs[1][1],palette="Blues_d")

ax6.set(xlabel='Country',ylabel='',title='The number of confirmed cases in June')



ax7=sns.barplot(x='Country/Region',y='Number of new confirmed cases in July',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in July',ascending=False).head(5),ax=axs[1][2],palette="Blues_d")

ax7.set(xlabel='Country',ylabel='',title='The number of confirmed cases in July')



ax8=sns.barplot(x='Country/Region',y='Number of new confirmed cases in August',data=df_new_confirmed.sort_values(by='Number of new confirmed cases in August',ascending=False).head(5),ax=axs[1][3],palette="Blues_d")

ax8.set(xlabel='Country',ylabel='',title='The number of confirmed cases in August')



plt.show()

#Again we create a new dataframe (because the 'Country/Region' may not be in the same order between the confirmed and the deaths database )

df_new_deaths = pd.DataFrame()

df_new_deaths['Country/Region'] = df_ts_deaths['Country/Region']



df_new_deaths['Number of deaths in August'] = df_ts_deaths['8/31/20']-df_ts_deaths['8/1/20']

df_new_deaths['Number of deaths in July'] = df_ts_deaths['7/31/20']-df_ts_deaths['7/1/20']

df_new_deaths['Number of deaths in June'] = df_ts_deaths['6/30/20']-df_ts_deaths['6/1/20']

df_new_deaths['Number of deaths in May'] = df_ts_deaths['5/31/20']-df_ts_deaths['5/1/20']

df_new_deaths['Number of deaths in April'] = df_ts_deaths['4/30/20']-df_ts_deaths['4/1/20']

df_new_deaths['Number of deaths in March'] = df_ts_deaths['3/31/20']-df_ts_deaths['3/1/20']

df_new_deaths['Number of deaths in February'] = df_ts_deaths['2/29/20']-df_ts_deaths['2/1/20']

df_new_deaths['Number of deaths in January'] = df_ts_deaths['1/31/20']-df_ts_deaths['1/22/20']

df_new_deaths.head()
#Graph that shows the 5 countries with the most number of deaths due to the covid-19 per month

fig, axs=plt.subplots(nrows=2,ncols=4,figsize = (25,15))



ax1=sns.barplot(x='Country/Region',y='Number of deaths in January',data=df_new_deaths.sort_values(by='Number of deaths in January',ascending=False).head(5),ax=axs[0][0],palette=sns.dark_palette('white'))

ax1.set(xlabel='Country',ylabel='Number of deaths',title='The number of deaths in January')



ax2=sns.barplot(x='Country/Region',y='Number of deaths in February',data=df_new_deaths.sort_values(by='Number of deaths in February',ascending=False).head(5),ax=axs[0][1],palette=sns.dark_palette('white'))

ax2.set(xlabel='Country',ylabel='',title='The number of deaths in February')



ax3=sns.barplot(x='Country/Region',y='Number of deaths in March',data=df_new_deaths.sort_values(by='Number of deaths in March',ascending=False).head(5),ax=axs[0][2],palette=sns.dark_palette('white'))

ax3.set(xlabel='Country',ylabel='',title='The number of deaths in March')



ax4=sns.barplot(x='Country/Region',y='Number of deaths in April',data=df_new_deaths.sort_values(by='Number of deaths in April',ascending=False).head(5),ax=axs[0][3],palette=sns.dark_palette('white'))

ax4.set(xlabel='Country',ylabel='',title='The number of deaths in April')



ax5=sns.barplot(x='Country/Region',y='Number of deaths in May',data=df_new_deaths.sort_values(by='Number of deaths in May',ascending=False).head(5),ax=axs[1][0],palette=sns.dark_palette('white'))

ax5.set(xlabel='Country',ylabel='Number of deaths',title='The number of deaths in May')



ax6=sns.barplot(x='Country/Region',y='Number of deaths in June',data=df_new_deaths.sort_values(by='Number of deaths in June',ascending=False).head(5),ax=axs[1][1],palette=sns.dark_palette('white'))

ax6.set(xlabel='Country',ylabel='',title='The number of deaths in June')



ax7=sns.barplot(x='Country/Region',y='Number of deaths in July',data=df_new_deaths.sort_values(by='Number of deaths in July',ascending=False).head(5),ax=axs[1][2],palette=sns.dark_palette('white'))

ax7.set(xlabel='Country',ylabel='',title='The number of deaths in July')



ax8=sns.barplot(x='Country/Region',y='Number of deaths in August',data=df_new_deaths.sort_values(by='Number of deaths in August',ascending=False).head(5),ax=axs[1][3],palette=sns.dark_palette('white'))

ax8.set(xlabel='Country',ylabel='',title='The number of deaths in August')



plt.show()
#Same thing as before but with the recovered database

df_new_recovered = pd.DataFrame()

df_new_recovered['Country/Region'] = df_ts_recovered['Country/Region']

df_new_recovered['Number of recovery cases in August']= df_ts_recovered['8/31/20']-df_ts_recovered['8/1/20']

df_new_recovered['Number of recovery cases in July']= df_ts_recovered['7/31/20']-df_ts_recovered['7/1/20']

df_new_recovered['Number of recovery cases in June']= df_ts_recovered['6/30/20']-df_ts_recovered['6/1/20']

df_new_recovered['Number of recovery cases in May']= df_ts_recovered['5/31/20']-df_ts_recovered['5/1/20']

df_new_recovered['Number of recovery cases in April']= df_ts_recovered['4/30/20']-df_ts_recovered['4/1/20']

df_new_recovered['Number of recovery cases in March']= df_ts_recovered['3/31/20']-df_ts_recovered['3/1/20']

df_new_recovered['Number of recovery cases in February']= df_ts_recovered['2/29/20']-df_ts_recovered['2/1/20']

df_new_recovered['Number of recovery cases in January']= df_ts_recovered['1/31/20']-df_ts_recovered['1/22/20']

df_new_recovered.head()

#Graph that shows the 5 countries with the most number of recovery cases of covid-19 per month

fig, axs=plt.subplots(nrows=2,ncols=4,figsize = (25,15))

ax1=sns.barplot(x='Country/Region',y='Number of recovery cases in January',data=df_new_recovered.sort_values(by='Number of recovery cases in January',ascending=False).head(5),ax=axs[0][0],palette=sns.light_palette('green'))

ax1.set(xlabel='Country',ylabel='Number of recovery cases',title='The number of recovery cases in January')



ax2=sns.barplot(x='Country/Region',y='Number of recovery cases in February',data=df_new_recovered.sort_values(by='Number of recovery cases in February',ascending=False).head(5),ax=axs[0][1],palette=sns.light_palette('green'))

ax2.set(xlabel='Country',ylabel='',title='The number of recovery cases in February')



ax3=sns.barplot(x='Country/Region',y='Number of recovery cases in March',data=df_new_recovered.sort_values(by='Number of recovery cases in March',ascending=False).head(5),ax=axs[0][2],palette=sns.light_palette('green'))

ax3.set(xlabel='Country',ylabel='',title='The number of recovery cases in March')



ax4=sns.barplot(x='Country/Region',y='Number of recovery cases in April',data=df_new_recovered.sort_values(by='Number of recovery cases in April',ascending=False).head(5),ax=axs[0][3],palette=sns.light_palette('green'))

ax4.set(xlabel='Country',ylabel='',title='The number of recovery cases in April')



ax5=sns.barplot(x='Country/Region',y='Number of recovery cases in May',data=df_new_recovered.sort_values(by='Number of recovery cases in May',ascending=False).head(5),ax=axs[1][0],palette=sns.light_palette('green'))

ax5.set(xlabel='Country',ylabel='Number of recovery cases',title='The number of recovery cases in May')



ax6=sns.barplot(x='Country/Region',y='Number of recovery cases in June',data=df_new_recovered.sort_values(by='Number of recovery cases in June',ascending=False).head(5),ax=axs[1][1],palette=sns.light_palette('green'))

ax6.set(xlabel='Country',ylabel='',title='The number of recovery cases in June')



ax7=sns.barplot(x='Country/Region',y='Number of recovery cases in July',data=df_new_recovered.sort_values(by='Number of recovery cases in July',ascending=False).head(5),ax=axs[1][2],palette=sns.light_palette('green'))

ax7.set(xlabel='Country',ylabel='',title='The number of recovery cases in July')



ax8=sns.barplot(x='Country/Region',y='Number of recovery cases in August',data=df_new_recovered.sort_values(by='Number of recovery cases in August',ascending=False).head(5),ax=axs[1][3],palette=sns.light_palette('green'))

ax8.set(xlabel='Country',ylabel='',title='The number of recovery cases in August')

plt.show()
#In this section we will try to re-create a DataFrame with the lines being the months and the columns being confirmed, deaths and recovered cases



#First we find the 'France' line for the data base 'df_data'

df_fr_date = df_data.loc[df_data['Country/Region']=='France'].sort_values(by='ObservationDate',ascending=False).loc[df_data['Province/State'].isnull()]

df_fr_date = df_fr_date.drop(df_fr_date.columns[[0,2,3,4]],axis=1)

df_fr_date.head()
#Dataframe with just the last day of the month, in order to find later the number of confirmed/deaths/recovered per month (and not the cumulate number)

labels=['01/31/2020','02/29/2020','03/31/2020','04/30/2020','05/31/2020','06/30/2020','07/31/2020','08/31/2020']



df_fr= pd.DataFrame() #New DataFrame

for date in labels: #We just want the last day of the different months, and we concatenate the dataframe with its previous version for each month

    df_fr = pd.concat(

        [df_fr_date.loc[df_fr_date['ObservationDate']==date] , df_fr], 

        ignore_index=True)

    

df_fr = df_fr.sort_values(by='ObservationDate',ascending=False) #The frist line is for August, not necessary

df_fr.head(10) #Number of total confirmed/deaths/recovered cases at the end pf each month
#Number of confirmed cases/recovered/deaths per month

#For example : Number in August = (number on 8/29/20) - (number on 7/31/20)



df_fr.loc['August'] = df_fr.loc[0][['Confirmed','Deaths','Recovered']] - df_fr.loc[1][['Confirmed','Deaths','Recovered']]

df_fr.loc['July'] = df_fr.loc[1][['Confirmed','Deaths','Recovered']] - df_fr.loc[2][['Confirmed','Deaths','Recovered']]

df_fr.loc['June'] = df_fr.loc[2][['Confirmed','Deaths','Recovered']] - df_fr.loc[3][['Confirmed','Deaths','Recovered']]

df_fr.loc['May'] = df_fr.loc[3][['Confirmed','Deaths','Recovered']] - df_fr.loc[4][['Confirmed','Deaths','Recovered']]

df_fr.loc['April'] = df_fr.loc[4][['Confirmed','Deaths','Recovered']] - df_fr.loc[5][['Confirmed','Deaths','Recovered']]

df_fr.loc['March'] = df_fr.loc[5][['Confirmed','Deaths','Recovered']] - df_fr.loc[6][['Confirmed','Deaths','Recovered']]

df_fr.loc['February'] = df_fr.loc[6][['Confirmed','Deaths','Recovered']] - df_fr.loc[7][['Confirmed','Deaths','Recovered']]

df_fr.loc['January'] = df_fr.loc[7][['Confirmed','Deaths','Recovered']]



#We keep just the monthly cases and use iloc to sort it the right way

df_fr = df_fr.drop([i for i in range(0,8)]).drop('ObservationDate',axis=1).iloc[::-1]

df_fr.head(8) #There is some errors in the database fot the recovered cases
labels=['January','February','March','April','May','June','July','August'] #x label as well as the name of the df_month lines



fig, axs= plt.subplots(ncols=3,figsize=(21,5)) #Several charts on one figure

ax1=sns.barplot(x=labels,

                y=df_fr['Confirmed'],

                data=df_fr,

                ax=axs[0], #First chart

                palette = 'Blues')

ax1.set(xlabel='Month',ylabel='Number of confirmed cases',title='Number of covid-19 confirmed cases per month in France');



ax2=sns.barplot(x=labels,

                y=df_fr['Deaths'],

                data=df_fr,

                ax=axs[1], #Second chart

                color='Grey')

ax2.set(xlabel='Month',ylabel='Number of deaths cases',title='Number of covid-19 deaths per month in France');



ax3=sns.barplot(x=labels,

                y=df_fr['Recovered'],

                data=df_fr,

                ax=axs[2], #Third chart

                palette = 'Greens')

ax3.set(xlabel='Month',ylabel='Number of recovered cases',title='Number of covid-19 recovered cases per month in France');