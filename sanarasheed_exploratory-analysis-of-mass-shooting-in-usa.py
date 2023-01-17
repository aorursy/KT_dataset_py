# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib as mpl

import seaborn as sns

from geopy.geocoders import Nominatim

color = sns.color_palette()



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



ms = pd.read_csv("../input/Mass Shootings Dataset.csv", encoding = "ISO-8859-1", parse_dates=["Date"])

print("Data Dimensions are: ", ms.shape)
ms.columns
ms.head()
ms = ms.sort_values('Date')
plt.figure(figsize=(8,6))

plt.scatter(np.sort(ms['Date']), np.sort(ms['Total victims'].values))

plt.xlabel('Date', fontsize=12)

plt.ylabel('Total Victoms', fontsize=12)

plt.show()
# mass shooting by date

ms_perdate = np.asarray(ms.groupby('Date')['Fatalities'].sum())



# thirty day moving average of ms fatalites by date

ms_average = pd.Series(ms_perdate).rolling(window=30).mean()

ms_average = np.asarray(ms_average.drop(ms_average.index[:397]))

ms_average = np.round(ms_average, 0)



ms_dates = np.arange('2016-01', '2017-01', dtype='datetime64[D]')

ms_range = ms_dates[15:351]



trace_date = go.Scatter(

             x = ms_dates,

             y = ms_perdate,

             mode = 'lines',

             name = 'Fatalities',

             line = dict(

                 color = 'rgb(215, 0, 0)',

                 width = 3)

             )



trace_mean = go.Scatter(

             x = ms_range,

             y = ms_average,

             mode = 'lines',

             name = 'Average',

             line = dict(

                 color = 'rgb(215, 0, 0)',

                 width = 5),

             opacity = 0.33

             )



layout = go.Layout(

         title = 'Mass Shooting Fatalities by Date in United States <br>'

                 '<sub>Hover & Rescale Plot to Desired Dates</sub>',

         showlegend = False,

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             type = 'date',

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             range = [0, 65],

             autotick = False,

             tick0 = 10,

             dtick = 10,

             showline = True,

             showgrid = False)

         )



data = [trace_date, trace_mean]

figure = dict(data = data, layout = layout)

iplot(figure)
ms['Year'] = ms['Date'].dt.year



cnt_srs = ms['Year'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Year of Shooting', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('No. of Attacks per Year', fontsize=18)

plt.show()
ms['Month'] = ms['Date'].dt.month



cnt_srs = ms['Month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of Shooting', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Month', fontsize=18)

plt.show()
ms['Quarter'] = ms['Date'].dt.quarter



cnt_srs = ms['Quarter'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Quarter of Shooting', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Quarter', fontsize=18)

plt.show()
ms['WoY'] = ms['Date'].dt.weekofyear



cnt_srs = ms['WoY'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Shooting by Week of the Year', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Week of the Year', fontsize=18)

plt.show()

ms['DoW'] = ms['Date'].dt.dayofweek



cnt_srs = ms['DoW'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Shooting by Weekdays', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Weekdays', fontsize=18)

plt.show()
ms['DoM'] = ms['Date'].dt.day



cnt_srs = ms['DoM'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Day of the Month', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks by Day of the Month', fontsize=18)

plt.show()
ms['DoY'] = ms['Date'].dt.dayofyear



cnt_srs = ms['DoY'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Day of the year', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Quarter', fontsize=18)

plt.show()
ms['weekdayflg'] =( ms['DoW'] // 5 != 1).astype(float)



cnt_srs = ms['weekdayflg'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Shooting by Weekdays', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Overall No. of Attacks per Weekdays', fontsize=18)

plt.show()

plt.figure(figsize=(12,8))

sns.boxplot(x="weekdayflg", y="Total victims", data=ms)

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Is Weekday', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("No. of Shooting Attacks on Weekdays and Weekends", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="weekdayflg", y="Total victims", data=ms[ms['Total victims'] < 500])

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Is Weekday', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("No. of Shooting Attacks on Weekdays and Weekends", fontsize=15)

plt.show()   
ms['City'] = ms['Location'].str.rpartition(',')[0]#.str.replace(",", " ")

ms['State'] = ms['Location'].str.rpartition(',')[2]
ms[ms[['City']].apply(lambda x: x[0].isdigit(), axis=1)].head(10)
cnt_srs = ms['State'].value_counts()

cnt_srs = cnt_srs.head(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('State', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Top 10 States of Mass Shooting', fontsize=18)

plt.show()
cnt_srs = ms['City'].value_counts()

cnt_srs = cnt_srs.head(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('City', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Top 10 Cities of Mass Shooting', fontsize=18)

plt.show()
ms['text'] = ms['Date'].dt.strftime('%B %-d'

                          ) + ', ' + ms['Fatalities'].astype(str) + ' Fatalities'



data = [dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = ms[ms.Longitude < 0]['Longitude'],

        lat = ms[ms.Longitude < 0]['Latitude'],

        text = ms[ms.Longitude < 0]['text'],

        mode = 'markers',

        marker = dict( 

            size = ms[ms.Longitude < 0]['Fatalities'] ** 0.5 * 5,

            opacity = 0.75,

            color = 'rgb(215, 0, 0)')

        )]



layout = dict(

         title = 'Shooting Fatalities by Latitude/Longitude in United States <br>'

                 '<sub>Hover to View Date and Fatalitiess</sub>',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             showland = True,

             landcolor = 'rgb(250, 250, 250)',

             subunitwidth = 1,

             subunitcolor = 'rgb(217, 217, 217)',

             countrywidth = 1,

             countrycolor = 'rgb(217, 217, 217)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)
cnt_srs = ms['Gender'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Shooting by Gender', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Attacks by Gender', fontsize=18)

plt.show()
ms.Gender.replace(['M', 'M/F'], ['Male', 'Male/Female'], inplace=True)
plt.figure(figsize=(12,8))

sns.boxplot(x="Gender", y="Total victims", data=ms)

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Gender', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Victims targetted by Genders", fontsize=15)

plt.show()
ms[ms['Gender'] == "Male/Female"][['Race', 'Mental Health Issues', 'State','City', 'Total victims']]
ms.Race.replace(['white', 'black', 'Some other race', 'unclear'], ['White', 'Black', 'Other','Unknown'], inplace=True)



cnt_srs = ms['Race'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Races', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('No. of Attacks by Races', fontsize=18)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="Race", y="Total victims", data=ms)

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Race', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Victims Targetted by Races", fontsize=15)

plt.show()
ms['Mental Health Issues'].replace(['unknown', 'unclear', 'Unclear'], ['Unknown','Unclear', 'Unclear'], inplace=True)
cnt_srs = ms['Mental Health Issues'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Shooting by Mental Disorder', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('No. of Attacks by Mental Cases', fontsize=18)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="Mental Health Issues", y="Total victims", data=ms)

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Mental Health Issues', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Victims targetted by Mental Persons", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="Mental Health Issues", y="Total victims", data=ms[ms['Mental Health Issues'] == 'Yes'] )

plt.ylabel('Total Victims', fontsize=12)

plt.xlabel('Mental Health Issues', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Victims targetted by Mental Persons", fontsize=15)

plt.show()
ms1 = ms

plt.figure(figsize=(14,8))



N = ms1.State.value_counts().size

# Choose some random colors

colors=cm.rainbow(np.random.rand(N))



# Use those colors as the color argument

plt.scatter(ms1.Fatalities, ms1.Injured, s=ms['Total victims'],color=colors)

for i in range(N):

    plt.annotate(ms1.State[i],xy=(ms1.Fatalities[i],ms1.Injured[i]))

plt.xlabel('Deaths')

plt.ylabel('Injuries')

# Move title up with the "y" option

plt.title('Deaths, Injuries and Total Victims in all States',y=1.05)

plt.show()

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(ms['Summary']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(ms[ms['Gender'] == "Male/Female"]['Summary']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(ms[ms['Race'] == "White American or European American"]['Summary']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(ms[ms['Mental Health Issues'] == 'Yes']['Summary']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()