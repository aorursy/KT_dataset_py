# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns











from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import plotly as py



import plotly.graph_objs as go



init_notebook_mode(connected=True)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cases = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

cases.info()


#cases.confirmed = cases.confirmed.astype(float)



province_list = list(cases['province'].unique())

number_of_cases = []

for j in province_list:

    x = cases[cases['province'] == j]

    total = sum(x.confirmed)

    number_of_cases.append(total)

data = pd.DataFrame({'province_list': province_list, 'number_of_cases' : number_of_cases})

new_index = (data['number_of_cases'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)



#visualization

plt.figure(figsize = (10,5))

sns.barplot(x = sorted_data['province_list'], y = sorted_data['number_of_cases'])

plt.xticks(rotation = 80)

plt.xlabel('Provinces')

plt.ylabel('Number of Cases')

plt.title('Number of Cases for each province')







labels = cases.province.value_counts().index









#figure

fig = {

    "data" : [

        {

            "values" : number_of_cases,

            "labels" : labels,

            "domain" : {"x":[0, .5]},

            "name" : "Distrubition of Cases over cities",

            "hoverinfo" : "label+percent+name",

            "hole" : .3,

            "type" : "pie"

        },

    ],

    "layout" : {

        "title" : "Distrubiton of Cases Over Cities",

        "annotations" : [

            {

                "font" : {"size" : 20},

                "showarrow" : False,

                "text" : "Cases",

                "x" : 0.20,

                "y" : 1

            },

        ]

    }

}

iplot(fig)



cases.group.dropna(inplace = True)

labels = ['Group Infection','Not Group']

colors = ['red', 'green']

explode = [0,0]

sizes = cases.group.value_counts().values



#visualization

plt.figure(figsize = (8,8))

plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%.1f%%')

plt.title('Group Infection Rates', color = 'black',fontsize = 12)

plt.show()
patients = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')

patients.head()
patients.info()
patients.age.dropna(inplace = True)

assert 1 == 1

ages = []

for x in patients.age:

    ages.append(x)



ages.sort()

age_counts = Counter(ages) 

p = age_counts.most_common(10)



x,y = zip(*p)

x,y = list(x),list(y)



plt.figure(figsize = (10,7))

ax = sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(17))

plt.xlabel('Ages')

plt.ylabel('Counts')

plt.title('Number of cases by age')

plt.show()
patients.sex.value_counts()

sns.countplot(patients.sex)

plt.title('Distribution of cases over genders', color = 'red', fontsize = 13)

plt.show()

patients.state.dropna(inplace = True)

labels = ['Isolated','Released','Deceased']

colors = ['purple','green','red', ]

explode = [0,0,0]

sizes = patients.state.value_counts()



#visualization

plt.figure(figsize = (7,7))

plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%.2f%%')

plt.title('States of People who exposed virus', color = 'black',fontsize = 14)

plt.show()
time_data = pd.read_csv('../input/coronavirusdataset/Time.csv')

time_data.info()
first_21_days = time_data.head(21)



trace1 = go.Scatter(

                   x = first_21_days.test,

                   y = first_21_days.negative,

                   mode = "lines",

                   name = "Negatives",

                   marker = dict(color = 'rgba(16,112,2,0.8)'),

                   text = first_21_days.date

                   )



trace2 = go.Scatter(

                   x = first_21_days.test,

                   y = first_21_days.confirmed,

                   mode = "lines + markers",

                   name = "Positives",

                   marker = dict(color = 'rgba(80,10,22,0.8)'),

                   text = first_21_days.date

                   )

data = [trace1, trace2]

layout = dict(title = 'Negative - positive comparison in the first 21 days', xaxis = dict(title = '# of negative or positive', ticklen = 5, zeroline = False), yaxis = dict(title = 'Number of tests'))

fig = dict(data = data, layout = layout)

iplot(fig)







f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=first_21_days.confirmed,y=first_21_days.date,color='blue',alpha = 0.5,label='Confirmed' )

sns.barplot(x=first_21_days.released,y=first_21_days.date,color='green',alpha = 0.7,label='Released')

sns.barplot(x=first_21_days.deceased,y=first_21_days.date,color='red',alpha = 0.6,label='Deceased')



ax.legend( loc = 'upper right', frameon = True)

ax.set(xlabel = 'States', ylabel = 'Days', title = 'States of Confirmed People')

plt.show()
last_21_days = time_data.tail(21) 



trace1 = go.Scatter(

                   x = last_21_days.test,

                   y = last_21_days.negative,

                   mode = "lines",

                   name = "Negatives",

                   marker = dict(color = 'rgba(16,112,2,0.8)'),

                   text = last_21_days.date

                   )



trace2 = go.Scatter(

                   x = last_21_days.test,

                   y = last_21_days.confirmed,

                   mode = "lines + markers",

                   name = "Positives",

                   marker = dict(color = 'rgba(80,10,22,0.8)'),

                   text = last_21_days.date

                   )

data = [trace1, trace2]

layout = dict(title = 'Negative - positive comparison in the last 21 days', xaxis = dict(title = '# of negative or positive', ticklen = 5, zeroline = False), yaxis = dict(title = 'Number of tests'))

fig = dict(data = data, layout = layout)

iplot(fig)

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=last_21_days.confirmed,y=last_21_days.date,color='blue',alpha = 0.3,label='Confirmed' )

sns.barplot(x=last_21_days.released,y=last_21_days.date,color='green',alpha = 0.7,label='Released')

sns.barplot(x=last_21_days.deceased,y=last_21_days.date,color='red',alpha = 0.9,label='Deceased')



ax.legend( loc = 'upper right', frameon = True)

ax.set(xlabel = 'States', ylabel = 'Days', title = 'States of Confirmed People')

plt.show()