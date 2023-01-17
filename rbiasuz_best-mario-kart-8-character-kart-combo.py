#Importing our packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Interactive visualizations

import plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks as cf

import plotly.figure_factory as ff

from plotly import tools



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



#Reading our datasets

characters = pd.read_csv('/kaggle/input/characters.csv')

tires = pd.read_csv('/kaggle/input/tires.csv')

bodies = pd.read_csv('/kaggle/input/bodies.csv')

gliders = pd.read_csv('/kaggle/input/gliders.csv')



#Functions to unificate speed columns, handling columns and name columns:

def new_speed(row):

    return row['Speed'] * 0.75 + row['Speed (Water)'] * 0.1 + row['Speed (Air)'] * 0.05 + row['Speed (Ground)'] * 0.1

def new_handling (row):

    return row['Handling'] * 0.75 + row['Handling (Water)'] * 0.1 + row['Handling (Air)'] * 0.05 + row['Handling(Ground)'] * 0.1

def name (row):

    return row['Character'] + ' + ' + row['tires'] + ' + ' + row['Vehicle'] + ' + ' + row['gliders']
#Before, let's think in a name for these runners

names =['Babies', 'Toad & Friends', 'Peach/Daisy/Yoshi', "Marios", 'DK/Rosa/Waluig', 'Metal/Gold', 'Heavy Heavies']



characters = characters.drop_duplicates(['Class', 'Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'])

characters['Character'] = names



gliders.drop_duplicates(['Type', 'Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'], inplace=True)

gliders.drop('Body', axis=1, inplace=True)



tires.drop_duplicates(['Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'], inplace=True)



bodies.drop_duplicates(['Speed', 'Acceleration', 'Weight', 'Handling',

       'Traction', 'Mini Turbo'], inplace=True)
cols = ['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)',

       'Acceleration', 'Weight', 'Handling', 'Handling (Water)',

       'Handling (Air)', 'Handling(Ground)', 'Traction', 'Mini Turbo']



df_fim = pd.DataFrame()

for index, row in gliders.iterrows():

    df_temp = characters.copy()

    df_temp['gliders'] = row['Type']

    for col in cols:

        df_temp[col] = df_temp[col] + row[col]

    df_fim = df_fim.append(df_temp)    

    

aux = df_fim.copy()

df_fim = pd.DataFrame()

for index, row in tires.iterrows():

    df_temp = aux.copy()

    df_temp['tires'] = row['Body']

    for col in cols:

        df_temp[col] = df_temp[col] + row[col]

    df_fim = df_fim.append(df_temp)   

    

cols = ['Speed', 'Acceleration', 'Weight', 'Handling', 'Traction', 'Mini Turbo']

aux = df_fim.copy()

df_fim = pd.DataFrame()

for index, row in bodies.iterrows():

    df_temp = aux.copy()

    df_temp['Vehicle'] = row['Vehicle']

    for col in cols:

        df_temp[col] = df_temp[col] + row[col]

    df_fim = df_fim.append(df_temp)   

    

df_fim['new_speed'] = df_fim.apply(new_speed, axis=1)

df_fim.drop(['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)'], axis=1, inplace=True)



df_fim['new_handling'] = df_fim.apply(new_handling, axis=1)

df_fim.drop(['Handling', 'Handling (Water)', 'Handling (Air)', 'Handling(Ground)'], axis=1, inplace=True)



df_fim['name'] = df_fim.apply(name, axis=1)
# Normal (same value for everithing)

def normal(row):

    return row['new_speed'] * 0.2 + row['new_handling'] * 0.2 + row['Mini Turbo'] * 0.2 + row['Traction'] * 0.2 + row['Acceleration'] * 0.2



# Speed (Priorize speed / acceleration and turbo)

def speed(row):

    return row['new_speed'] * 0.4 + row['new_handling'] * 0.1 + row['Mini Turbo'] * 0.2 + row['Traction'] * 0.1 + row['Acceleration'] * 0.2



# Acceleration (Priorize acceleration and mini turbo)

def acceleration(row):

    return row['new_speed'] * 0.2 + row['new_handling'] * 0.1 + row['Mini Turbo'] * 0.2 + row['Traction'] * 0.1 + row['Acceleration'] * 0.4



# handling (Priorize handling)

def handling(row):

    return row['new_speed'] * 0.1 + row['new_handling'] * 0.4 + row['Mini Turbo'] * 0.2 + row['Traction'] * 0.2 + row['Acceleration'] * 0.1



# turbo (Priorize turbo)

def turbo(row):

    return row['new_speed'] * 0.2 + row['new_handling'] * 0.1 + row['Mini Turbo'] * 0.4 + row['Traction'] * 0.1 + row['Acceleration'] * 0.2



#Applying

df_fim['normal'] = df_fim.apply(normal, axis=1)

df_fim['speed'] = df_fim.apply(speed, axis=1)

df_fim['acceleration'] = df_fim.apply(acceleration, axis=1)

df_fim['handling'] = df_fim.apply(handling, axis=1)

df_fim['turbo'] = df_fim.apply(turbo, axis=1)



#Normalization 

df_fim["normal"] = (df_fim["normal"] - df_fim["normal"].mean()) / df_fim["normal"].std()

df_fim["speed"] = (df_fim["speed"] - df_fim["speed"].mean()) / df_fim["speed"].std()

df_fim["acceleration"] = (df_fim["acceleration"] - df_fim["acceleration"].mean()) / df_fim["acceleration"].std()

df_fim["handling"] = (df_fim["handling"] - df_fim["handling"].mean()) / df_fim["handling"].std()

df_fim["turbo"] = (df_fim["turbo"] - df_fim["turbo"].mean()) / df_fim["turbo"].std()
trace0 = go.Bar(

    x=df_fim.sort_values(by=['normal'], ascending=False).head(100)['name'],

    y=df_fim.sort_values(by=['normal'], ascending=False).head(100)['normal'], 

    showlegend=False, visible=True

)



trace1 = go.Bar(

    x=df_fim.sort_values(by=['speed'], ascending=False).head(100)['name'],

    y=df_fim.sort_values(by=['speed'], ascending=False).head(100)['speed'], 

    marker_color='crimson',

    showlegend=False, visible=False

)



trace2 = go.Bar(

    x=df_fim.sort_values(by=['acceleration'], ascending=False).head(100)['name'],

    y=df_fim.sort_values(by=['acceleration'], ascending=False).head(100)['acceleration'], 

    showlegend=False, visible=False

)



trace3 = go.Bar(

    x=df_fim.sort_values(by=['handling'], ascending=False).head(100)['name'],

    y=df_fim.sort_values(by=['handling'], ascending=False).head(100)['handling'], 

    marker_color='crimson',

    showlegend=False, visible=False

)



trace4 = go.Bar(

    x=df_fim.sort_values(by=['turbo'], ascending=False).head(100)['name'],

    y=df_fim.sort_values(by=['turbo'], ascending=False).head(100)['turbo'], 

    showlegend=False, visible=False

)





data = [trace0, trace1, trace2, trace3, trace4]





updatemenus = list([

    dict(active=0,

         x=-0.15,

         buttons=list([  

            dict(

                label = 'Normal',

                 method = 'update',

                 args = [{'visible': [True, False, False, False, False]}, 

                     {'title': 'Normal'}]),

             

             dict(

                  label = 'Speed',

                 method = 'update',

                 args = [{'visible': [False, True, False, False, False]},

                     {'title': 'speed'}]),



            dict(

                 label = 'Acceleration',

                 method = 'update',

                 args = [{'visible': [False, False, True, False, False]},

                     {'title': 'acceleration'}]),



            dict(

                 label =  'Handling',

                 method = 'update',

                 args = [{'visible': [False, False, False, True, False]},

                     {'title': 'Handling'}]),

             

            dict(

                 label =  'Turbo',

                 method = 'update',

                 args = [{'visible': [False, False, False, False, True]},

                     {'title': 'Turbo'}])             

        ]),

    )

])



layout = dict(title='Bar chart by (Select from Dropdown)', 

              showlegend=False,

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)



iplot(fig)
parameters_set = ['normal', 'speed', 'acceleration', 'handling', 'turbo']

lista = []

for par in parameters_set:

    for i in list(df_fim.sort_values(by=[par], ascending=False).head(10)['name']):

        lista.append(i)

from collections import Counter

count = Counter(lista) # Cria uma contagem de cada valor na lista

commons = count.most_common(15) # Retorna os 'n' mais frequentes valores

labels, values = zip(*commons)



trace0 = go.Bar(

    x=labels,

    y=values, 

    showlegend=False, visible=True

)



layout = dict(title='Most Common Combinations in the Top 10', showlegend=False)



fig = dict(data=trace0, layout=layout)



iplot(fig)
parameters_set = ['normal', 'speed', 'acceleration', 'handling', 'turbo']

lista = []

for par in parameters_set:

    for i in list(df_fim.sort_values(by=[par], ascending=False).tail(10)['name']):

        lista.append(i)

from collections import Counter

count = Counter(lista) # Cria uma contagem de cada valor na lista

commons = count.most_common(15) # Retorna os 'n' mais frequentes valores

labels, values = zip(*commons)



trace0 = go.Bar(

    x=labels,

    y=values, 

    showlegend=False, visible=True,

    marker_color='crimson'

)



layout = dict(title='Most Common Combinations in the bottom 10', showlegend=False)



fig = dict(data=trace0, layout=layout)



iplot(fig)
to_drop = ['Class', 'Acceleration', 'Weight', 'Traction','Mini Turbo','new_speed','new_handling', 'name']

df_sorted = df_fim.sort_values(by=['normal'], ascending=False).drop(to_drop, axis=1)



plt.figure(figsize=(17,7))

sns.set(font_scale=1.4)

sns.heatmap(df_sorted.groupby('Character').mean(),

           linewidths=1,

           annot=True,

           fmt=".1f",

           cmap='plasma')

plt.title('Heatmap by Characters')
plt.figure(figsize=(17,7))

sns.set(font_scale=1.4)

sns.heatmap(df_sorted.groupby('tires').mean(),

           linewidths=1,

           annot=True,

           fmt=".1f",

           cmap='plasma')

plt.title('Heatmap by Tires')
plt.figure(figsize=(17,12))

sns.set(font_scale=1.4)

sns.heatmap(df_sorted.groupby('Vehicle').mean(),

           linewidths=1,

           annot=True,

           fmt=".1f",

           cmap='plasma')

plt.title('Heatmap by Vehicle')
df_line = df_fim.sort_values(by=['normal'], ascending=False).head(15)

plt.figure(figsize=(17,12))

plt.plot( 'name', 'speed', data=df_line, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( 'name', 'acceleration', data=df_line, marker='', color='olive', linewidth=2)

plt.plot( 'name', 'handling', data=df_line, marker='', color='olive', linewidth=2, linestyle='dashed',)

plt.legend()

plt.xticks(rotation='45')

plt.title('Beginning of the data')

plt.show()

df_line = df_fim.sort_values(by=['normal'], ascending=False).iloc[740:755]

plt.figure(figsize=(17,12))

plt.plot( 'name', 'speed', data=df_line, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( 'name', 'acceleration', data=df_line, marker='', color='olive', linewidth=2)

plt.plot( 'name', 'handling', data=df_line, marker='', color='olive', linewidth=2, linestyle='dashed',)

plt.legend()

plt.xticks(rotation='45')

plt.title('Middle of the data')

plt.show()
df_line = df_fim.sort_values(by=['normal'], ascending=False).tail(15)

plt.figure(figsize=(17,12))

plt.plot( 'name', 'speed', data=df_line, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( 'name', 'acceleration', data=df_line, marker='', color='olive', linewidth=2)

plt.plot( 'name', 'handling', data=df_line, marker='', color='olive', linewidth=2, linestyle='dashed',)

plt.legend()

plt.title('End of the data')

plt.xticks(rotation='45')

plt.show()