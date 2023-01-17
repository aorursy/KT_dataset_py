# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")

# plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# students math course

student_mat_data = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')

# students portuguese language

student_por_data = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')
student_mat_data.columns
student_por_data.columns
student_mat_data.head()
student_por_data.head()
student_mat_data.describe()
student_mat_data.info()
def bar_plot(variable):

    """

   input: variable ex : "School"

   output: bar plot & value count

    

    """

    # get feature

    var = student_mat_data[variable]

    # count number of categorical variable(Value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}/{}".format(variable,varValue))
category1 = ["address","school","sex","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","schoolsup",

             "famsup","paid","activities","higher","internet","romantic","famrel","goout","Dalc","Walc","famrel","failures",

             "health","G1","G2","G3","absences","freetime","studytime","traveltime"]

for c in category1:

    bar_plot(c)
age_data = student_mat_data.age

plt.figure(figsize=(9,3))

plt.hist(age_data, bins = 50)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title( "Age distribition with hist")

plt.show()
# sex vs goout

student_mat_data[["sex","goout"]].groupby(["sex"], as_index = False).mean().sort_values(by = "goout", ascending = False)
# age vs goout

student_mat_data[["age","goout"]].groupby(["age"], as_index = False).mean().sort_values(by = "goout", ascending = False)
# school vs failures

student_mat_data[["school","failures"]].groupby(["school"], as_index = False).mean().sort_values(by = "failures",ascending = False)
# sex vs Dalc

student_mat_data[["sex","Dalc"]].groupby(["sex"], as_index = False).mean().sort_values(by = "Dalc", ascending = False)
# age vs Walc

student_mat_data[["age","Walc"]].groupby(["age"], as_index = False).mean().sort_values(by = "Walc", ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indexes

        outlier_list_col = df[(df[c]< Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indices

        outlier_indices.extend(outlier_list_col) 

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
student_mat_data.loc[detect_outliers(student_mat_data,["age","Dalc","Walc","absences"])]
student_mat_data.boxplot(column="Dalc", by = "Walc")

plt.show()
# correlation map

f,ax = plt.subplots(figsize = (10,10))

sns.heatmap(student_mat_data.corr(), annot = True, linewidths=.5, fmt = '.1f', ax=ax)
# father's job

fjob_list = student_mat_data['Fjob']

labels,values = zip(*Counter(fjob_list).items())

labels,values = list(labels),list(values)

#Visualization

plt.figure(figsize=(15,10))

sns.barplot(x = labels, y = values)

plt.xticks(rotation = 45)

plt.xlabel = ('Fathers'' jobs')

plt.ylabel = ('Frequency')

plt.title = ('father''s job')
# mother's job

mjob_list = student_mat_data['Mjob']

labels,values = zip(*Counter(mjob_list).items())

labels,values = list(labels), list(values)

# visualization

plt.figure(figsize = (15,10))

ax = sns.barplot(x= labels, y = values, palette = sns.cubehelix_palette(len(labels)))

plt.xlabel = 'mother''s job'

plt.ylabel = 'Frequency'

plt.title = 'mother''s job'
# workday alcohol consumption vs weekend alcohol consumption

age_list = list(student_mat_data.age.unique())

dalc_ratio = []

walc_ratio = []

for i in age_list:

    x = student_mat_data[student_mat_data.age == i]

    dalc_rate = sum(x.Dalc) / len(x)

    walc_rate = sum(x.Walc) / len(x)

    dalc_ratio.append(dalc_rate)

    walc_ratio.append(walc_rate)

# sorting

data = pd.DataFrame({'age_list':age_list, 'dalc_ratio':dalc_ratio})

data1 = pd.DataFrame({'age_list':age_list,'walc_ratio':walc_ratio})

new_index = (data['dalc_ratio'].sort_values(ascending = False)).index.values

new_index1 =(data1['walc_ratio'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)

sorted_data1 = data1.reindex(new_index1)



sorted_data['dalc_ratio'] = sorted_data['dalc_ratio']/ max(sorted_data['dalc_ratio'])

sorted_data1['walc_ratio'] = sorted_data1['walc_ratio']/ max(sorted_data1['walc_ratio'])

data = pd.concat([sorted_data,sorted_data1['walc_ratio']], axis = 1)



#visualization

f,ax = plt.subplots(figsize= (20,10))

sns.pointplot(x = 'age_list', y = 'dalc_ratio', data= data, color = 'lime', alpha = 0.8)

sns.pointplot(x = 'age_list', y = 'walc_ratio', data= data, color = 'red', alpha = 0.8)

plt.text(7,0.6,'workday alcohol consumption ratio',color='red',fontsize = 17,style = 'italic')

plt.text(7,0.55,'weekend alcohol consumption ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel = ('ages')

plt.ylabel = ('Values')

plt.title = 'workday alcohol consumption vs weekend alcohol consumption'

plt.grid()
# goout

# sex

# age

sns.swarmplot(x = 'sex', y = 'age', hue = "goout", data= student_mat_data)

plt.show()
df_mat = student_mat_data.iloc[:350,:]

pie1 = df_mat.age

labels = df_mat.age

fig = {

    "data":[

        {      

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "ages",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"   

        },],

     "layout": {

        "title":"students ages",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "ages of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
# prepare data frame

df_mat = student_mat_data.iloc[:395,:]

df_por = student_por_data.iloc[:395,:]

# creating trace1

trace1 = go.Scatter(

                        x = df_mat.age,

                        y = df_mat.G1,

                        mode = "markers",

                        name = "Math course",

                        marker = dict(color = 'rgba(255, 120, 255, 0.8)'),

                        text = df_mat.age

)

trace2 = go.Scatter(

                        x = df_por.age,

                        y = df_por.G1,

                        mode = "markers",

                        name = "Portuguese language course",

                        marker = dict(color = 'rgba(15, 150, 6, 0.8)'),

                        text = df_por.age

            

)



data = [trace1,trace2]

layout = dict(title = ' mat course and portuguese language course student workday alcohol consumption by age',

                 xaxis = dict(title = 'age',ticklen = 5, zeroline = False),

              yaxis = dict(title = 'workday alcohol consumption',ticklen = 5, zeroline = False)

             )

fig = dict(data = data , layout = layout)

iplot(fig)
trace1 = go.Histogram(

            x = df_mat.G1,

            opacity = 0.75,

            name = "G1",

            marker = dict(color = 'rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

            x = df_mat.G2,

            opacity = 0.75,

            name = "G2",

            marker = dict(color = 'rgba(50,250,10,0.6)'))

trace3 = go.Histogram(

                x = df_mat.G3,

                opacity = 0.75,

                name = "G3",

                marker = dict(color = 'rgba(50,40,255, 0.6)'))

data = [trace1,trace2,trace3]

layout = go.Layout(barmode = 'overlay',

                   title = 'first period grade vs second period grade vs first period grade',

                   xaxis = dict(title = 'G1 - G2 - G3'),

                   yaxis = dict(title = 'count'),

                  )

fig = go.Figure(data = data , layout = layout)

iplot(fig)
df_por = student_por_data.Mjob

plt.subplots(figsize = (8,8))

wordcloud = WordCloud(

                        background_color = 'white',

                        width = 512,

                        height = 384).generate(" ".join(df_por))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()