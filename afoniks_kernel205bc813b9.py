#загрузка библитек

import pandas as pd #для работы с набором данных

import numpy as np #мат библиотека

import seaborn as sns #Библиотека графиков, которые используют matplot в фоновом режиме

import matplotlib.pyplot as plt #импорт



import os

df_credit = pd.read_csv("../input/german_credit_data.csv", index_col=0)
#Поиск пропусков, тип данных, а также известных форма данных. Проще говоря отобращем наши данные

print(df_credit.info())
#поиск уникальных значений

print(df_credit.nunique())

print(df_credit.head())
# эта основная библиотека для работы

import plotly.offline as py 

py.init_notebook_mode(connected=True) # позволяем работать offline 

import plotly.graph_objs as go # это как "PLT" из библиотеки Matplot

import plotly.tools as tls # Это полезно, чтобы мы получили некоторые инструменты библиотеки plotly

import warnings # Эта библиотека будет использоваться для игнорирования некоторых предупреждений

from collections import Counter # Чтобы сделать счетчик некоторых функций



trace0 = go.Bar(

            x = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().index.values,

            y = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().values,

            name='Good credit'

    )



trace1 = go.Bar(

            x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values,

            y = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().values,

            name='Bad credit'

    )



data = [trace0, trace1]



layout = go.Layout(

    

)



layout = go.Layout(

    yaxis=dict(

        title='Count'

    ),

    xaxis=dict(

        title='Risk Variable'

    ),

    title='Target variable distribution'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='grouped-bar')
df_good = df_credit.loc[df_credit["Risk"] == 'good']['Age'].values.tolist()

df_bad = df_credit.loc[df_credit["Risk"] == 'bad']['Age'].values.tolist()

df_age = df_credit['Age'].values.tolist()



#первый график

trace0 = go.Histogram(

    x=df_good,

    histnorm='probability',

    name="Good Credit"

)

#второй график

trace1 = go.Histogram(

    x=df_bad,

    histnorm='probability',

    name="Bad Credit"

)

#третий график

trace2 = go.Histogram(

    x=df_age,

    histnorm='probability',

    name="Overall Age"

)



#создаем сетку

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Good','Bad', 'General Distribuition'))



#настройки таблицы

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
df_good = df_credit[df_credit["Risk"] == 'good']

df_bad = df_credit[df_credit["Risk"] == 'bad']



fig, ax = plt.subplots(nrows=2, figsize=(12,8))

plt.subplots_adjust(hspace = 0.4, top = 0.8)



g1 = sns.distplot(df_good["Age"], ax=ax[0], 

             color="g")

g1 = sns.distplot(df_bad["Age"], ax=ax[0], 

             color='r')

g1.set_title("Age Distribuition", fontsize=15)

g1.set_xlabel("Age")

g1.set_xlabel("Frequency")



g2 = sns.countplot(x="Age",data=df_credit, 

              palette="hls", ax=ax[1], 

              hue = "Risk")

g2.set_title("Age Counting by Risk", fontsize=15)

g2.set_xlabel("Age")

g2.set_xlabel("Count")

plt.show()
#просмотрим столбец сумма кредита

interval = (18, 25, 35, 60, 120)



cats = ['Student', 'Young', 'Adult', 'Senior']

df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)





df_good = df_credit[df_credit["Risk"] == 'good']

df_bad = df_credit[df_credit["Risk"] == 'bad']
trace0 = go.Box(

    y=df_good["Credit amount"],

    x=df_good["Age_cat"],

    name='Good credit',

    marker=dict(

        color='#3D9970'

    )

)



trace1 = go.Box(

    y=df_bad['Credit amount'],

    x=df_bad['Age_cat'],

    name='Bad credit',

    marker=dict(

        color='#FF4136'

    )

)

    

data = [trace0, trace1]



layout = go.Layout(

    yaxis=dict(

        title='Credit Amount (US Dollar)',

        zeroline=False

    ),

    xaxis=dict(

        title='Age Categorical'

    ),

    boxmode='group'

)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='box-age-cat')
#первый график

trace0 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'good']["Housing"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'good']["Housing"].value_counts().values,

    name='Good credit'

)



#второй график

trace1 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'bad']["Housing"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'bad']["Housing"].value_counts().values,

    name="Bad Credit"

)



data = [trace0, trace1]



layout = go.Layout(

    title='Housing Distribuition'

)





fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='Housing-Grouped')
fig = {

    "data": [

        {

            "type": 'violin',

            "x": df_good['Housing'],

            "y": df_good['Credit amount'],

            "legendgroup": 'Good Credit',

            "scalegroup": 'No',

            "name": 'Good Credit',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'blue'

            }

        },

        {

            "type": 'violin',

            "x": df_bad['Housing'],

            "y": df_bad['Credit amount'],

            "legendgroup": 'Bad Credit',

            "scalegroup": 'No',

            "name": 'Bad Credit',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'green'

            }

        }

    ],

    "layout" : {

        "yaxis": {

            "zeroline": False,

        },

        "violingap": 0,

        "violinmode": "overlay"

    }

}





py.iplot(fig, filename = 'violin/split', validate = False)
#первый график

trace0 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'good']["Sex"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'good']["Sex"].value_counts().values,

    name='Good credit'

)



#первый график 2

trace1 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'bad']["Sex"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'bad']["Sex"].value_counts().values,

    name="Bad Credit"

)



#второй график 

trace2 = go.Box(

    x = df_credit[df_credit["Risk"]== 'good']["Sex"],

    y = df_credit[df_credit["Risk"]== 'good']["Credit amount"],

    name=trace0.name

)



#второй график 2

trace3 = go.Box(

    x = df_credit[df_credit["Risk"]== 'bad']["Sex"],

    y = df_credit[df_credit["Risk"]== 'bad']["Credit amount"],

    name=trace1.name

)



data = [trace0, trace1, trace2,trace3]





fig = tls.make_subplots(rows=1, cols=2, 

                        subplot_titles=('Sex Count', 'Credit Amount by Sex'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)



fig['layout'].update(height=400, width=800, title='Sex Distribuition', boxmode='group')

py.iplot(fig, filename='sex-subplot')


trace0 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'good']["Job"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'good']["Job"].value_counts().values,

    name='Good credit Distribuition'

)





trace1 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'bad']["Job"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'bad']["Job"].value_counts().values,

    name="Bad Credit Distribuition"

)



data = [trace0, trace1]



layout = go.Layout(

    title='Job Distribuition'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='grouped-bar')
trace0 = go.Box(

    x=df_good["Job"],

    y=df_good["Credit amount"],

    name='Good credit'

)



trace1 = go.Box(

    x=df_bad['Job'],

    y=df_bad['Credit amount'],

    name='Bad credit'

)

    

data = [trace0, trace1]



layout = go.Layout(

    yaxis=dict(

        title='Credit Amount distribuition by Job'

    ),

    boxmode='group'

)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='box-age-cat')
fig = {

    "data": [

        {

            "type": 'violin',

            "x": df_good['Job'],

            "y": df_good['Age'],

            "legendgroup": 'Good Credit',

            "scalegroup": 'No',

            "name": 'Good Credit',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'blue'

            }

        },

        {

            "type": 'violin',

            "x": df_bad['Job'],

            "y": df_bad['Age'],

            "legendgroup": 'Bad Credit',

            "scalegroup": 'No',

            "name": 'Bad Credit',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'green'

            }

        }

    ],

    "layout" : {

        "yaxis": {

            "zeroline": False,

        },

        "violingap": 0,

        "violinmode": "overlay"

    }

}





py.iplot(fig, filename = 'Age-Housing', validate = False)
fig, ax = plt.subplots(figsize=(12,12), nrows=2)



g1 = sns.boxplot(x="Job", y="Credit amount", data=df_credit, 

            palette="hls", ax=ax[0], hue="Risk")

g1.set_title("Credit Amount by Job", fontsize=15)

g1.set_xlabel("Job Reference", fontsize=12)

g1.set_ylabel("Credit Amount", fontsize=12)



g2 = sns.violinplot(x="Job", y="Age", data=df_credit, ax=ax[1],  

               hue="Risk", split=True, palette="hls")

g2.set_title("Job Type reference x Age", fontsize=15)

g2.set_xlabel("Job Reference", fontsize=12)

g2.set_ylabel("Age", fontsize=12)



plt.subplots_adjust(hspace = 0.4,top = 0.9)



plt.show()
import plotly.figure_factory as ff



import numpy as np



# Add histogram data

x1 = np.log(df_good['Credit amount']) 

x2 = np.log(df_bad["Credit amount"])



# Group data together

hist_data = [x1, x2]



group_labels = ['Good Credit', 'Bad Credit']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)



# Plot!

py.iplot(fig, filename='Distplot with Multiple Datasets')
#построение прохих и хороших исходов в dataframe 

plt.figure(figsize = (8,5))



g= sns.distplot(df_good['Credit amount'], color='r')

g = sns.distplot(df_bad["Credit amount"], color='g')

g.set_title("Credit Amount Frequency distribuition", fontsize=15)

plt.show()
from plotly import tools

import numpy as np

import plotly.graph_objs as go



count_good = go.Bar(

    x = df_good["Saving accounts"].value_counts().index.values,

    y = df_good["Saving accounts"].value_counts().values,

    name='Good credit'

)

count_bad = go.Bar(

    x = df_bad["Saving accounts"].value_counts().index.values,

    y = df_bad["Saving accounts"].value_counts().values,

    name='Bad credit'

)





box_1 = go.Box(

    x=df_good["Saving accounts"],

    y=df_good["Credit amount"],

    name='Good credit'

)

box_2 = go.Box(

    x=df_bad["Saving accounts"],

    y=df_bad["Credit amount"],

    name='Bad credit'

)



scat_1 = go.Box(

    x=df_good["Saving accounts"],

    y=df_good["Age"],

    name='Good credit'

)

scat_2 = go.Box(

    x=df_bad["Saving accounts"],

    y=df_bad["Age"],

    name='Bad credit'

)



data = [scat_1, scat_2, box_1, box_2, count_good, count_bad]



fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Count Saving Accounts','Credit Amount by Savings Acc', 

                                          'Age by Saving accounts'))



fig.append_trace(count_good, 1, 1)

fig.append_trace(count_bad, 1, 1)



fig.append_trace(box_2, 1, 2)

fig.append_trace(box_1, 1, 2)



fig.append_trace(scat_1, 2, 1)

fig.append_trace(scat_2, 2, 1)







fig['layout'].update(height=700, width=800, title='Saving Accounts Exploration', boxmode='group')



py.iplot(fig, filename='combined-savings')
print("Description of Distribuition Saving accounts by Risk:  ")

print(pd.crosstab(df_credit["Saving accounts"],df_credit.Risk))



fig, ax = plt.subplots(3,1, figsize=(12,12))

g = sns.countplot(x="Saving accounts", data=df_credit, palette="hls", 

              ax=ax[0],hue="Risk")

g.set_title("Saving Accounts Count", fontsize=15)

g.set_xlabel("Saving Accounts type", fontsize=12)

g.set_ylabel("Count", fontsize=12)



g1 = sns.violinplot(x="Saving accounts", y="Job", data=df_credit, palette="hls", 

               hue = "Risk", ax=ax[1],split=True)

g1.set_title("Saving Accounts by Job", fontsize=15)

g1.set_xlabel("Savings Accounts type", fontsize=12)

g1.set_ylabel("Job", fontsize=12)



g = sns.boxplot(x="Saving accounts", y="Credit amount", data=df_credit, ax=ax[2],

            hue = "Risk",palette="hls")

g2.set_title("Saving Accounts by Credit Amount", fontsize=15)

g2.set_xlabel("Savings Accounts type", fontsize=12)

g2.set_ylabel("Credit Amount(US)", fontsize=12)



plt.subplots_adjust(hspace = 0.4,top = 0.9)



plt.show()
print("Values describe: ")

print(pd.crosstab(df_credit.Purpose, df_credit.Risk))



plt.figure(figsize = (14,12))



plt.subplot(221)

g = sns.countplot(x="Purpose", data=df_credit, 

              palette="hls", hue = "Risk")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=12)

g.set_title("Purposes Count", fontsize=20)



plt.subplot(222)

g1 = sns.violinplot(x="Purpose", y="Age", data=df_credit, 

                    palette="hls", hue = "Risk",split=True)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

g1.set_xlabel("", fontsize=12)

g1.set_ylabel("Count", fontsize=12)

g1.set_title("Purposes by Age", fontsize=20)



plt.subplot(212)

g2 = sns.boxplot(x="Purpose", y="Credit amount", data=df_credit, 

               palette="hls", hue = "Risk")

g2.set_xlabel("Purposes", fontsize=12)

g2.set_ylabel("Credit Amount", fontsize=12)

g2.set_title("Credit Amount distribuition by Purposes", fontsize=20)



plt.subplots_adjust(hspace = 0.6, top = 0.8)



plt.show()
print("Values describe: ")

print(pd.crosstab(df_credit.Purpose, df_credit.Risk))



plt.figure(figsize = (14,12))



plt.subplot(221)

g = sns.countplot(x="Purpose", data=df_credit, 

              palette="hls", hue = "Risk")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=12)

g.set_title("Purposes Count", fontsize=20)



plt.subplot(222)

g1 = sns.violinplot(x="Purpose", y="Age", data=df_credit, 

                    palette="hls", hue = "Risk",split=True)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

g1.set_xlabel("", fontsize=12)

g1.set_ylabel("Count", fontsize=12)

g1.set_title("Purposes by Age", fontsize=20)



plt.subplot(212)

g2 = sns.boxplot(x="Purpose", y="Credit amount", data=df_credit, 

               palette="hls", hue = "Risk")

g2.set_xlabel("Purposes", fontsize=12)

g2.set_ylabel("Credit Amount", fontsize=12)

g2.set_title("Credit Amount distribuition by Purposes", fontsize=20)



plt.subplots_adjust(hspace = 0.6, top = 0.8)



plt.show()
plt.figure(figsize = (12,14))



g= plt.subplot(311)

g = sns.countplot(x="Duration", data=df_credit, 

              palette="hls",  hue = "Risk")

g.set_xlabel("Duration Distribuition", fontsize=12)

g.set_ylabel("Count", fontsize=12)

g.set_title("Duration Count", fontsize=20)



g1 = plt.subplot(312)

g1 = sns.pointplot(x="Duration", y ="Credit amount",data=df_credit,

                   hue="Risk", palette="hls")

g1.set_xlabel("Duration", fontsize=12)

g1.set_ylabel("Credit Amount(US)", fontsize=12)

g1.set_title("Credit Amount distribuition by Duration", fontsize=20)



g2 = plt.subplot(313)

g2 = sns.distplot(df_good["Duration"], color='g')

g2 = sns.distplot(df_bad["Duration"], color='r')

g2.set_xlabel("Duration", fontsize=12)

g2.set_ylabel("Frequency", fontsize=12)

g2.set_title("Duration Frequency x good and bad Credit", fontsize=20)



plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)



plt.show()
#First plot

trace0 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'good']["Checking account"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'good']["Checking account"].value_counts().values,

    name='Good credit Distribuition' 

    

)



#Second plot

trace1 = go.Bar(

    x = df_credit[df_credit["Risk"]== 'bad']["Checking account"].value_counts().index.values,

    y = df_credit[df_credit["Risk"]== 'bad']["Checking account"].value_counts().values,

    name="Bad Credit Distribuition"

)



data = [trace0, trace1]



layout = go.Layout(

    title='Checking accounts Distribuition',

    xaxis=dict(title='Checking accounts name'),

    yaxis=dict(title='Count'),

    barmode='group'

)





fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename = 'Age-ba', validate = False)
df_good = df_credit[df_credit["Risk"] == 'good']

df_bad = df_credit[df_credit["Risk"] == 'bad']



trace0 = go.Box(

    y=df_good["Credit amount"],

    x=df_good["Checking account"],

    name='Good credit',

    marker=dict(

        color='#3D9970'

    )

)



trace1 = go.Box(

    y=df_bad['Credit amount'],

    x=df_bad['Checking account'],

    name='Bad credit',

    marker=dict(

        color='#FF4136'

    )

)

    

data = [trace0, trace1]



layout = go.Layout(

    yaxis=dict(

        title='Cheking distribuition'

    ),

    boxmode='group'

)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='box-age-cat')
print("Total values of the most missing variable: ")

print(df_credit.groupby("Checking account")["Checking account"].count())



plt.figure(figsize = (12,10))



g = plt.subplot(221)

g = sns.countplot(x="Checking account", data=df_credit, 

              palette="hls", hue="Risk")

g.set_xlabel("Checking Account", fontsize=12)

g.set_ylabel("Count", fontsize=12)

g.set_title("Checking Account Counting by Risk", fontsize=20)



g1 = plt.subplot(222)

g1 = sns.violinplot(x="Checking account", y="Age", data=df_credit, palette="hls", hue = "Risk",split=True)

g1.set_xlabel("Checking Account", fontsize=12)

g1.set_ylabel("Age", fontsize=12)

g1.set_title("Age by Checking Account", fontsize=20)



g2 = plt.subplot(212)

g2 = sns.boxplot(x="Checking account",y="Credit amount", data=df_credit,hue='Risk',palette="hls")

g2.set_xlabel("Checking Account", fontsize=12)

g2.set_ylabel("Credit Amount(US)", fontsize=12)

g2.set_title("Credit Amount by Cheking Account", fontsize=20)



plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top = 0.9)



plt.show()

plt.show()
plt.figure(figsize = (12,14))



g= plt.subplot(311)

g = sns.countplot(x="Duration", data=df_credit, 

              palette="hls",  hue = "Risk")

g.set_xlabel("Duration Distribuition", fontsize=12)

g.set_ylabel("Count", fontsize=12)

g.set_title("Duration Count", fontsize=20)



g1 = plt.subplot(312)

g1 = sns.pointplot(x="Duration", y ="Credit amount",data=df_credit,

                   hue="Risk", palette="hls")

g1.set_xlabel("Duration", fontsize=12)

g1.set_ylabel("Credit Amount(US)", fontsize=12)

g1.set_title("Credit Amount distribuition by Duration", fontsize=20)



g2 = plt.subplot(313)

g2 = sns.distplot(df_good["Duration"], color='g')

g2 = sns.distplot(df_bad["Duration"], color='r')

g2.set_xlabel("Duration", fontsize=12)

g2.set_ylabel("Frequency", fontsize=12)

g2.set_title("Duration Frequency x good and bad Credit", fontsize=20)



plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)



plt.show()
print(pd.crosstab(df_credit.Sex, df_credit.Job))
plt.figure(figsize = (10,6))



g = sns.violinplot(x="Housing",y="Job",data=df_credit,

                   hue="Risk", palette="hls",split=True)

g.set_xlabel("Housing", fontsize=12)

g.set_ylabel("Job", fontsize=12)

g.set_title("Housing x Job - Dist", fontsize=20)



plt.show()
print(pd.crosstab(df_credit["Checking account"],df_credit.Sex))
date_int = ["Purpose", 'Sex']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_credit[date_int[0]], df_credit[date_int[1]]).style.background_gradient(cmap = cm)
date_int = ["Purpose", 'Sex']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_credit[date_int[0]], df_credit[date_int[1]]).style.background_gradient(cmap = cm)
print("Purpose : ",df_credit.Purpose.unique())

print("Sex : ",df_credit.Sex.unique())

print("Housing : ",df_credit.Housing.unique())

print("Saving accounts : ",df_credit['Saving accounts'].unique())

print("Risk : ",df_credit['Risk'].unique())

print("Checking account : ",df_credit['Checking account'].unique())

print("Aget_cat : ",df_credit['Age_cat'].unique())
def one_hot_encoder(df, nan_as_category = False):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns
df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')

df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')



#назначение макетной переменной

df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)

#переменная пол в макете

df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)

# переменная жилье в макете

df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)

#жилье в сохраненных аккаунтах

df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)

# жилье подверженно риску

df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)

df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)

# жилье в категории "год"

df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)
#исключаем пропущенные столбцы

del df_credit["Saving accounts"]

del df_credit["Checking account"]

del df_credit["Purpose"]

del df_credit["Sex"]

del df_credit["Housing"]

del df_credit["Age_cat"]

del df_credit["Risk"]

del df_credit['Risk_good']
plt.figure(figsize=(14,12))

sns.heatmap(df_credit.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True,  linecolor='white', annot=True)

plt.show()
from sklearn.model_selection import train_test_split, KFold, cross_val_score # разделим даныне

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #чтобы оценить нашу модель



from sklearn.model_selection import GridSearchCV



# Алгоритмы моделей для сравнения

from sklearn.ensemble import RandomForestClassifier #рандом_форест

from sklearn.linear_model import LogisticRegression #логистическая регрессия 

from sklearn.tree import DecisionTreeClassifier #классификатор дерева решений

from sklearn.neighbors import KNeighborsClassifier #метод ближайших соседей

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #линейниый дискриминантный анализ

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC #Метод опорных векторов

from xgboost import XGBClassifier
df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
#создадим переменные х и у

X = df_credit.drop('Risk_bad', 1).values

y = df_credit["Risk_bad"].values



# Разделить X и Y на обучающую и тестовую версию

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
# заполнять рандомом

seed = 7



# подготовим модели

models = []

models.append(('Логическая регрессия', LogisticRegression()))

models.append(('Линейный дискриминантный анализ', LinearDiscriminantAnalysis()))

models.append(('Метод ближайших соседей', KNeighborsClassifier()))

models.append(('Классификатор дерева решений', DecisionTreeClassifier()))

models.append(('Гауссан ', GaussianNB()))

models.append(('Рандом форест', RandomForestClassifier()))

models.append(('Метод опорных векторов', SVC(gamma='auto')))

models.append(('Повышение градиента', XGBClassifier()))



# оценивать модели по очереди

results = []

names = []

scoring = 'recall'



for name, model in models:

        kfold = KFold(n_splits=10, random_state=seed)

        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

# сравнение алгоритма boxplot

fig = plt.figure(figsize=(11,6))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#Установка гиперпараметров

param_grid = {"max_depth": [3,5, 7, 10,None],

              "n_estimators":[3,5,10,25,50,150],

              "max_features": [4,7,15,20]}



#Создание классификатора

model = RandomForestClassifier(random_state=2)

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)

grid_search.fit(X_train, y_train) 
print(grid_search.best_score_)

print(grid_search.best_params_)
rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)



#обучение с лучшими параметрами

rf.fit(X_train, y_train)
#Тестирование модели

# Прогнозирование с использованием нашей модели

y_pred = rf.predict(X_test)



print(accuracy_score(y_test,y_pred))

print("\n")

print(confusion_matrix(y_test, y_pred))

print("\n")

print(fbeta_score(y_test, y_pred, beta=2))
from sklearn.utils import resample

from sklearn.metrics import roc_curve
GNB = GaussianNB()

model = GNB.fit(X_train, y_train)
# Показать тренировочные результаты

print("Training score data: ")

print(model.score(X_train, y_train))
y_pred = model.predict(X_test)



print(accuracy_score(y_test,y_pred))

print("\n")

print(confusion_matrix(y_test, y_pred))

print("\n")

print(classification_report(y_test, y_pred))
#проба предсказания

y_pred_prob = model.predict_proba(X_test)[:,1]



# создание значений ROC-кривой: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# нарисовать ROC-кривую

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest
features = []

features.append(('pca', PCA(n_components=2)))

features.append(('select_best', SelectKBest(k=6)))

feature_union = FeatureUnion(features)

# create pipeline

estimators = []

estimators.append(('feature_union', feature_union))

estimators.append(('logistic', GaussianNB()))

model = Pipeline(estimators)

# evaluate pipeline

seed = 7

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(model, X_train, y_train, cv=kfold)

print(results.mean())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(accuracy_score(y_test,y_pred))

print("\n")

print(confusion_matrix(y_test, y_pred))

print("\n")

print(fbeta_score(y_test, y_pred, beta=2))