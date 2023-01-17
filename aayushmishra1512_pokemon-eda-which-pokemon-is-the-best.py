import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import tools

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot
df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv') #importing our dataset
df.head() #checking the head of our data
df.info()
df['Type 1'].value_counts() #checking the value count of type 1 of pokemons
df['Type 2'].value_counts() #checking value count of type 2 of pokemons
df['Legendary'].value_counts() #checking the number of legendary pokemons in our data
df.isnull().sum() #checking for null values in our data
df.describe().T #calling the describe method on our data
def filltype(): #method to fill the null values of type 2 with type 1 as the pokemon might only have one type instead of ddropping the null values we can just replace them with the primary type of the pokemon

    df['Type 2'].fillna(df['Type 1'],inplace = True)
df.isnull().sum() #no null values in our data set now
plt.style.use('dark_background')

plt.figure(figsize=(15,8))

corr=sns.heatmap(df[[ 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',

       'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr(), annot = True, linewidths=1, cmap='inferno')

corr.set_title(label='Correlation between all the columns of our data', fontsize=20)

plt.show()
plt.figure(figsize=(15,8)) #count of type 1 pokemons

sns.countplot(df['Type 1'],palette='Set1')

plt.xlabel('Types')

plt.title('Pokemon types')

plt.show()
plt.figure(figsize=(15,8)) #count of type 2 pokemons

sns.countplot(df['Type 2'],palette='Set1')

plt.xlabel('Types')

plt.title('Pokemon types')

plt.show()
plt.title("Type 1 Pokemon distribbution")

df['Type 1'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(15,8)) #distribution according to the rating

plt.show()
plt.title("Type 2 Pokemon distribbution")

df['Type 2'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(15,8)) #distribution according to the rating

plt.show()
plt.figure(figsize=(15,8))

sns.jointplot(x="Attack",y="Defense",data=df,kind="hex",color="red");
plt.figure(figsize=(15,8))

sns.jointplot(x="Sp. Atk", y="Sp. Def", data=df,kind ="hex",color = 'red'); #comparing special attack and speed
plt.figure(figsize=(15,8))

sns.jointplot(x="Sp. Atk",y="Speed",data=df,color="red");
plt.figure(figsize=(15,8))

sns.jointplot(x="Sp. Def", y="Speed", data=df,color = 'red'); #comparing special Defense and speed
plt.figure(figsize=(15,8))

sns.violinplot(x="Generation",y="Attack",data = df)
plt.figure(figsize=(15,8))

sns.violinplot(x="Generation",y="Defense",data = df)
plt.figure(figsize=(15,8)) #checking the number of pokemons from each generation

sns.countplot(x='Generation',data=df)

plt.show()
df[df['HP']>150][['Name','Type 1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Total']].sort_values(by = 'HP',ascending = False).head(5) #top 5 pokemons with highest HP
df[df['Attack']>150][['Name','Type 1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Total']].sort_values(by = 'Attack',ascending = False).head(5) #top 5 pokemons with highest Attack
df[df['Defense']>180][['Name','Type 1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Total']].sort_values(by = 'Defense',ascending = False).head(5) #top 5 pokemons with highest Defense
df[df['Speed']>130][['Name','Type 1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Total']].sort_values(by = 'Speed',ascending = False).head(5) #top 5 pokemons with highest Speed
plt.figure(figsize=(15,8)) #comparing total scores of different types in Type 1

sns.boxplot(x = df['Type 1'],y = df['Total'])
plt.figure(figsize=(15,8)) #comparing total scores of different types in Type 2

sns.boxplot(x = df['Type 2'],y = df['Total'])
legend = df[df["Legendary"]] #comparing attributes of different legendary pokemons

legend = legend[["HP","Attack","Defense","Sp. Atk","Sp. Def"]]

fig = plt.figure(figsize= (15,8))

sns.boxplot(data=legend)

plt.ylabel("Power Points",color="grey")

plt.show()
plt.figure(figsize=(15,8)) #distribution of legends according to their generations

plot = sns.countplot(x='Generation',data=df,hue='Legendary',palette="Set1")
def compare(p1,p2,c): #method to compare any attributes of two pokemons

    comp = df[(df.Name == p1) | (df.Name ==p2)]

    sns.barplot(x='Name',y=c,data=comp,palette="Set1")
compare('Blaziken','Empoleon','Attack')
compare('Blaziken','Empoleon','Defense')
compare('Blaziken','Empoleon','Total')
compare('Blaziken','Empoleon','Speed')
compare('Blaziken','Empoleon','Sp. Atk')
def PokeFight(p1,p2,p3,p4,p5): #method to compare overall power of any 5 pokemons of your choice

    x = df[df["Name"] == p1]

    trace1 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],

      fill = 'toself',

      name = p1

    )

    x = df[df["Name"] == p2]

    trace2 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],

      fill = 'toself',

      name = p2

    )

    x = df[df["Name"] == p3]

    trace3 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],

      fill = 'toself',

      name = p3

    )

    x = df[df["Name"] == p4]

    trace4 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],

      fill = 'toself',

      name = p4

    )

    x = df[df["Name"] == p5]

    trace5 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],

      fill = 'toself',

      name = p5

    )

    

    layout = go.Layout(

      xaxis=dict(

            domain=[0, 0.45]

        ),

        yaxis=dict(

            domain=[0, 0.45]

        ),

        xaxis2=dict(

            domain=[0.55, 1]

        ),

        xaxis3=dict(

            domain=[0, 0.45],

            anchor='y3'

        ),

        xaxis4=dict(

            domain=[0.55, 1],

            anchor='y4'

        ),

        yaxis2=dict(

            domain=[0, 0.45],

            anchor='x2'

        ),

        yaxis4=dict(

            domain=[0.55, 1],

            anchor='x4'

        ),

        

      showlegend = True,

      title = "Pokémons' Performance"

    )



    data = [trace1, trace2, trace3,trace4,trace5]

    fig = go.Figure(data=data, layout=layout)



    iplot(fig, filename = "Pokemon stats")

    
PokeFight("Raichu","Charizard","Venusaur","Blastoise","Blaziken") #comparing the overall attributes of the pokemon
from wordcloud import WordCloud

plt.subplots(figsize=(12,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from wordcloud import WordCloud

plt.subplots(figsize=(12,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df['Type 1']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()