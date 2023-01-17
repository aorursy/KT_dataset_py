#Importing our packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading our datasets



characters = pd.read_csv('/kaggle/input/characters.csv')

tires = pd.read_csv('/kaggle/input/tires.csv')

bodies = pd.read_csv('/kaggle/input/bodies.csv')

gliders = pd.read_csv('/kaggle/input/gliders.csv')



#Inspecting

characters.head()
tires.head()
bodies.head()
gliders.head()
characters[characters['Speed'] != characters['Speed (Air)']]

#characters[characters['Speed'] != characters['Speed (Water)']]   ----

#characters[characters['Speed'] != characters['Speed (Ground)']]  ---- These lines generated the same results 
tires[tires['Speed'] != tires['Speed (Air)']]
tires[tires['Speed'] != tires['Speed (Water)']]
tires[tires['Speed'] != tires['Speed (Ground)']]
def new_speed(row):

    return row['Speed'] * 0.75 + row['Speed (Water)'] * 0.1 + row['Speed (Air)'] * 0.05 + row['Speed (Ground)'] * 0.1



characters['new_speed'] = characters.apply(new_speed, axis=1)

characters.drop(['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)'], axis=1, inplace=True)



characters.head()
#Now let's do similar to the Handling factor:

def new_handling (row):

    return row['Handling'] * 0.75 + row['Handling (Water)'] * 0.1 + row['Handling (Air)'] * 0.05 + row['Handling(Ground)'] * 0.1



characters['new_handling'] = characters.apply(new_handling, axis=1)

characters.drop(['Handling', 'Handling (Water)', 'Handling (Air)', 'Handling(Ground)'], axis=1, inplace=True)



characters.head()
#fig = go.Figure(data=go.Heatmap(

#                   z=characters.groupby('Class').mean().values,

#                   x=characters.groupby('Class').mean().columns,

#                   y=characters.groupby('Class').mean().index,

#                   colorscale='Viridis'

#))

#

#fig.show()



# --- Just left this scrap here because it could be useful another time
# Evaluating the characters by class:



plt.figure(figsize=(17,7))

sns.set(font_scale=1.4)

sns.heatmap(characters.groupby('Class').mean(),

           linewidths=1,

           annot=True,

           fmt=".1f",

           cmap='viridis')

plt.title('Heatmap of Classes')
#how speed, acceleration and the class are distribuited?

fig = px.scatter(characters, x="new_speed", y="Acceleration", color="Class",hover_data=['Character'], size='Mini Turbo')

fig.show()
# Who are the five best of each factor?



factors = ['Acceleration', 'Weight', 'Traction', 'Mini Turbo', 'new_speed', 'new_handling']

for factor in factors:

    print('')

    print('Factor: {}'.format(factor))

    print('')

    print(characters.sort_values(by=[factor], ascending=False).head()['Character'])

def effectiveness(row):

    return row['new_speed'] * 0.3 + row['new_handling'] * 0.1 + row['Mini Turbo'] * 0.3 + row['Traction'] * 0.1 + row['Acceleration'] * 0.2



characters['effectiveness'] = characters.apply(effectiveness, axis=1)



characters.head()
characters.sort_values(by=['effectiveness'], ascending=False)
characters = pd.read_csv('/kaggle/input/characters.csv')



characters.drop_duplicates(['Class', 'Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'])
#Okay, let's think in a name for these runners

names =['Babies', 'Toad & Friends', 'Peach/Daisy/Yoshi', "Marios", 'DK/Rosa/Waluig', 'Metal/Gold', 'Heavy Heavies']



characters = characters.drop_duplicates(['Class', 'Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'])

characters['Character'] = names

characters
#Now let's do the same for the components



gliders.drop_duplicates(['Type', 'Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'], inplace=True)

gliders.drop('Body', axis=1, inplace=True)

gliders 

#Only two actual changes here...
tires.drop_duplicates(['Speed', 'Speed (Water)', 'Speed (Air)',

       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',

       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',

       'Mini Turbo'], inplace=True)

tires

#7 here...
bodies.drop_duplicates(['Speed', 'Acceleration', 'Weight', 'Handling',

       'Traction', 'Mini Turbo'], inplace=True)

bodies

#18 here...
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
df_fim.head(10)
len(df_fim)
df_fim['new_speed'] = df_fim.apply(new_speed, axis=1)

df_fim.drop(['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)'], axis=1, inplace=True)



df_fim['new_handling'] = df_fim.apply(new_handling, axis=1)

df_fim.drop(['Handling', 'Handling (Water)', 'Handling (Air)', 'Handling(Ground)'], axis=1, inplace=True)



df_fim['effectiveness'] = df_fim.apply(effectiveness, axis=1)
df_fim.describe()
plt.figure(figsize=(17,7))

sns.boxplot(x="Character", y="effectiveness", data=df_fim, palette="Set3")

plt.title('Effectiveness by Characters')
sns.pairplot(df_fim, hue="Character")
plt.figure(figsize=(17,7))

sns.violinplot(x="tires", y="new_handling", data=df_fim)

plt.title('Tires by handling')
plt.figure(figsize=(17,7))

sns.boxplot(x="Character", y="new_speed", hue="gliders", data=df_fim)

plt.title('new_speed and gliders by Character')
plt.figure(figsize=(17,7))

sns.violinplot(x="Character", y="new_handling", hue="gliders", data=df_fim)

plt.title('new_handling and gliders by Character')
plt.figure(figsize=(17,7))

sns.set(font_scale=1.4)

sns.heatmap(df_fim.corr(),

           linewidths=1,

           annot=True,

           fmt=".1f")

plt.title('Correlation heatmap')
#TOP 10

df_fim.sort_values(by=['effectiveness'], ascending=False).head(10)
# BOT 10

df_fim.sort_values(by=['effectiveness'], ascending=False).tail(10)
fig = px.scatter(df_fim, x="new_speed", y="Acceleration", color="Class",hover_data=['Character', 'tires', 'Vehicle'], size='new_handling')

fig.show()
fig = px.scatter(df_fim, x="effectiveness", y="Mini Turbo", color="Class",hover_data=['Character', 'tires', 'Vehicle'], size='new_handling')

fig.show()