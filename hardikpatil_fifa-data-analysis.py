import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import re

sns.set_style("darkgrid")

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected =True)

import plotly.graph_objs as go

from plotly import tools

from wordcloud import WordCloud     #word cloud library

import matplotlib.pyplot as plt     #matplotlib

#dataset = pd.read_csv('../input/data.csv', delimiter=',', nrows = nRowsRead)

dataset = pd.read_csv("../input/data.csv", low_memory=False)

dataset.columns

dataset.head()

# Prepare Data

fifa_data = pd.read_csv("../input/data.csv",low_memory = False)

#fifa_data = pd.read_csv('../input/data.csv', delimiter=',', nrows = nRowsRead)

plt.show()

fifa_data.head()

# In order to be able to start our ranking from 1, we have assigned the 'array' variable.

fifa_data['array'] = fifa_data['Unnamed: 0']+1

# Let's change the names of some columns

fifa_data.rename(columns={'Ball control':'ball_control','Free kick accuracy':'free_kick',

                     'Shot power':'shot_power'}, inplace=True)

# Just take the columns we'll use

fifa_data= fifa_data[['array','Name','Age','Nationality','Overall','Potential','Club','Value',

                      'Wage','Special','Acceleration','ball_control','Dribbling','free_kick',

                      'Penalties','shot_power']]

# Prepare Data Frame

d_frame = fifa_data.iloc[:100,:]



#Creating trace1

trace1 = go.Scatter(

    x= d_frame.array,

    y= d_frame.Overall,

    mode='lines',

    name='Overall',

    marker= dict(color='rgba(12, 255, 250,0.9)'),

    text= d_frame.Name

)

#Creating trace2

trace2 = go.Scatter(

    x= d_frame.array,

    y= d_frame.Potential,

    mode='lines+markers',

    name='Potential',

    

    text= d_frame.Name

)

data = [trace1,trace2]

layout= dict(title="Comparing the 'Overall' and 'Potential' of players",

            xaxis= dict(title='Player Rank', ticklen=5, zeroline=False)

            )

fig= dict(data=data, layout=layout)

#iplot(fig)



def conversion(money_str):

    notes = ''

    # Find the numbers and append

    for letter in money_str:

        if letter in '1234567890.':

            notes = notes + letter

        else:

            pass

    # Divide by 1000 to convert K to M for value

    if 'K' in money_str:

        return (float(notes)/1000)    

    else:

        return float(notes)

def wage_conversion(money_str):

    notes = ''

    # Find the numbers and append

    for letter in money_str:

        if letter in '1234567890.':

            notes = notes + letter

        else:

            pass

    

    return float(notes)

def convert_attributes(number_str):

    if type(number_str) == str:

        if '+' in number_str:

            return float(number_str.split('+')[0])

        elif '-' in number_str:

            return float(number_str.split('-')[0])

        else:

            return float(number_str)

dataset['Wage'] = dataset['Wage'].apply(wage_conversion) # Units = K

#print(dataset['Wage'][-10:].dtype)

dataset['Value'] = dataset['Value'].apply(conversion) # Units = M

#print(dataset['Value'][-10:].dtype)



dataset['Remaining Potential'] = dataset['Potential'] - dataset['Overall']



dataset['Preferred Position'] = dataset['Preferred Positions'].str.split().str[0]

###Best 11 based on overall rating in fifa data set

def formation_best_squad(position):

    dataset_copy = dataset.copy()

    store = []

    for i in position:

        store.append([i,dataset_copy.loc[[dataset_copy[dataset_copy["Preferred Position"] == i]["Overall"].idxmax()]]['Name'].to_string(index = False), dataset_copy[dataset_copy['Preferred Position'] == i]['Overall'].max()])

        dataset_copy.drop(dataset_copy[dataset_copy['Preferred Position'] == i]['Overall'].idxmax(), inplace = True)

    #return store

    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)

# 4-3-3

formation433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

#print ('4-3-3')

#print (formation_best_squad(formation433))

#3-5-2

formation352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

#print ('3-5-2')

#print (formation_best_squad(formation352))

##4-2-3-1

formation4231=['GK','LB','CB','CB','RB','CDM','CDM','LM','CAM','RM','ST']

#print('4-2-3-1')

#print(formation_best_squad(formation4231))

##potential against overall rating based on age parameter

##basic visualization





dataset_potential = dataset.groupby(['Age'])['Potential'].mean()

dataset_overall = dataset.groupby(['Age'])['Overall'].mean()



dataset_summary = pd.concat([dataset_potential, dataset_overall], axis=1)







###Top potential low rated players 

dataframe=dataset

dataframe['growth']=dataframe['Potential']-dataframe['Overall']

high_potential=dataframe[['Name','Overall','growth','Club','Preferred Positions']]

Top_Growths=high_potential.sort_values(by=['growth','Overall'],ascending=False)

#print(Top_Growths[:10])



####Top 20 players

Top=dataset[['Name','Age','Preferred Positions','Overall']]

Top_20=Top.sort_values(by=['Overall'],ascending=False)

#print(Top_20[:20])



# Prepare Data Frame

d_frame = fifa_data.iloc[90:100,:]



#create trace1

trace1 = go.Bar(

    x= d_frame.array,

    y= d_frame.Overall,

    name= 'Overall',

    marker= dict(color= 'rgba(255,106,0,0.9)',

                line= dict(color= 'rgb(0,0,0)', width=1)),

    text= d_frame.Name

)

#Create trace2

trace2 = go.Bar(

    x= d_frame.array,

    y= d_frame.Potential,

    name= 'Potential',

    marker= dict(color= 'rgba(148, 255, 130,0.9)',

                line= dict(color='rgb(0,0,0)', width=1)),

)

data= [trace1, trace2]

layout= go.Layout(barmode= "group")

fig1= go.Figure(data=data, layout=layout)

#iplot(fig1)





#Prepare Data Frame

d_frame = fifa_data.iloc[:5,:]



#Create trace1

trace1= {

    'x': d_frame.array,

    'y': d_frame.free_kick,

    'name': 'Free Kick',

    'type': 'bar',

    'text': d_frame.Name

};

#Create trace2

trace2= {

    'x': d_frame.array,

    'y': d_frame.Penalties,

    'name': 'Penalty',

    'type': 'bar',

    'text': d_frame.Name,

    'marker': dict(color= 'rgba(148, 255, 130,0.9)'),

};

data= [trace1, trace2]

layout= {

    'xaxis': {'title':'First 5 Player'},

    'barmode': 'relative',

    'title': 'Top 5 Players freekick and penalty strokes comparison'

};

fig2 = go.Figure(data=data, layout=layout)

#iplot(fig2)



#Prepare Data Frame

d_frame = fifa_data.iloc[:10,:]



#Create trace

trace = go.Scatter3d(

    x= d_frame.ball_control,

    y= d_frame.Dribbling,

    z= d_frame.shot_power,

    text= d_frame.Name,

    mode='markers',

    marker=dict(

        size=12,

        #color= z,          #set color to an array/list of desired value (plotly.ly)

#When we enters 'Fork Notebook' he describes 'z'. But why doesn't he recognize this right now? 

        colorscale='Viridis',   #Choose a colorscale

        opacity=0.8

    )

)

data = [trace]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig3 = go.Figure(data=data, layout=layout)

#iplot(fig3)



#Prepare Data Frame

d_frame = fifa_data.iloc[:100,:]

donut= d_frame.Nationality.value_counts()

labels = d_frame.Nationality.value_counts().index #Country names of the top 100 players



#Creat Figure

fig4 = {

    "data":

    [

        {

            "values": donut,

            "labels": labels,

            "domain": {"x": [0, 1]},

            "name": "Clubs Rate",

            "hoverinfo": "label+percent+name",

            "hole": .4,

            "type": "pie"

        }, 

    ],

    "layout":

    {

        "title":"Nationality rates of the top 100 players",

        "annotations":

        [

            { 

                "font":{"size":20},

                "showarrow":False,

                "text": "",

                "x": 0,

                "y": 1

            },

        ]

    }

}

#iplot(fig4)



#Prepare Data Frame

d_frame = fifa_data.Name[:50]



#plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                   background_color='White',

                        width = 700,

                        height = 400

    ).generate(" ".join(d_frame))



#plt.imshow(wordcloud)

#plt.axis('off')

#plt.savefig('graph.png')



#We'll edit 'Wage' to compare our values                                       

#removes € -> it will be like that: '£100K' -> '100K'

wage = fifa_data['Wage'].map(lambda x: x.replace("€", ""))

wage.head()                                                                

                                                                                        

#removes K -> it will be like that: '100K' -> '100'

wage = wage.map(lambda x: x.replace("K", ""))

wage.head()



 

#Prepare Data Frame

d_frame = fifa_data.iloc[:100,:]

donut= d_frame.Club.value_counts()

labels = d_frame.Club.value_counts().index



#Creat figure

fig5 = {

    "data":

    [

        {

            "values": donut,

            "labels": labels,

            "domain": {"x": [0, 1]},

            "name": "Clubs Rate",

            "hoverinfo": "label+percent+name",

            "hole": .4,

            "type": "pie"

        }, 

    ],

    "layout":

    {

        "title":"Club rates of the top 100 players",

        "annotations":

        [

            { 

                "font":{"size":20},

                "showarrow":False,

                "text": "",

                "x": 0,

                "y": 1

            },

        ]

    }

}

#iplot(fig5)

#We'll edit 'Value' to compare our values                                                 

#removes € -> it will be like that: '£100M or £100K' -> '100M or 100K'

value = fifa_data['Value'].map(lambda x: x.replace("€", ""))

                                                                       

#removes M -> it will be like that: '100M' -> '100 or 100K'

value = value.map(lambda x: x.replace("M", ""))

value.head()

#removes K -> it will be like that: '100K' -> '100'

value = value.map(lambda x: x.replace("K", ""))

value.head()                 

#We multiply by '1000' because we remove 'K'

wage = wage.astype("int")*1000

wage.head()

#We multiply by '1000000' because we remove 'M'

value = value.astype("float")*1000000

value.head()

#Let's create new columns now

fifa_data['wage']=wage

fifa_data['value']=value

fifa_data.head()

#Now let's see 20 players with the highest 'Wage'

#fifa_data.sort_values("wage", ascending=False).head(20)

#print(fifa_data.sort_values("wage", ascending=False).head(20))



#Prepare Data Frame

d_frame = fifa_data.sort_values("wage", ascending=False).head(20)



#Create trace1

trace1 = go.Bar(

    x= d_frame.wage,

    opacity = 0.75,

    name= 'Wage',

    text= d_frame.Name,

    marker = dict(color='rgba(0, 250, 0,0.6)'))

#Create trace2

trace2 = go.Bar(

    x=d_frame.value,

    opacity = 0.75,

    name= "Value",

    text= d_frame.Name,

    marker= dict(color='rgba(26, 26, 26,0.6)'))



data = [trace1,trace2]

layout = go.Layout(barmode='stack',

                  title="Comparison of 'Wage' and 'Value' Among 20 Players with the Highest Wage",

                  xaxis= dict(title= 'Wage - Value'),

                  yaxis= dict(title='Array'),)



#fig6 = go.Figure(data=data, layout=layout)

#iplot(fig6)

#Prepare Data Frame

d_frame = fifa_data.iloc[:100,:]



#Create trace1

trace1 = go.Scatter(

    x=d_frame.array,

    y=d_frame.value,

    text= d_frame.Name,

    name='value',

    marker= dict(color='rgba(26, 163, 242,0.9)'),

    mode= 'lines+markers'

)

#Create trace2

trace2 = go.Scatter(

    x=d_frame.array,

    y=d_frame.wage,

    xaxis='x2',

    yaxis='y2',

    text= d_frame.Name,

    name='wage',

    marker= dict(color='rgba(70, 71, 71,0.9)'),

    mode= 'markers'

)

data = [trace1, trace2]

layout = go.Layout(

    title = "Top 100 players' wage and values",

    xaxis2=dict(

        domain=[0.3, 1],

        anchor='y2'

    ),

    yaxis2=dict(

        domain=[0.6, 1],

        anchor='x2'

    )

)

fig7 = go.Figure(data=data, layout=layout)

#iplot(fig7)

fifa_data.groupby("Club")["wage"].mean().sort_values(ascending=False).head(10)



#Prepare Data Frame

d_frame = fifa_data.groupby("Club")["wage"].mean().sort_values(ascending=False).head(10)



#Create trace

trace = go.Bar(

    x= fifa_data.array,

    y= d_frame,

    opacity = 0.75,

    text= fifa_data.Club,

    marker = dict(color='rgba(0, 250, 100,0.6)'))



data = [trace]

layout = go.Layout(barmode='group',

                  title='Top 10 teams paying maximum wage',

                  xaxis= dict(title= 'Array'),

                  yaxis= dict(title='Total Wage'),)



fig8 = go.Figure(data=data, layout=layout)

#iplot(fig8)



#Now let's see the top 10 countries with the have more players

fifa_data.groupby("Nationality").Name.count().sort_values(ascending=False).head(10)



print('SAM')

import ipywidgets as widgets

from ipywidgets import HBox, VBox

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display

%matplotlib inline

print("Fifa Data Analysis 2018")

print("Dataset Contents:")

@widgets.interact_manual(

    x = ['Dataset Head','Top potential low rated players','Top 20 Players','Players with the highest Wage','Top 10 Clubs','Nationality Based Player Count'])  

def fine(x=0):

    if(x == 'Dataset Head'):

        print("DATASET HEAD IS")

        print(dataset.head())

    if(x == 'Top potential low rated players'):

        print(Top_Growths[:10])

    if(x == 'Top 20 Players'):

        print(Top_20[:20])

    if(x == 'Players with the highest Wage'):

        print(fifa_data.sort_values("wage", ascending=False).head(20))

    if(x == 'Top 10 Clubs'):

        print(fifa_data.groupby("Club")["wage"].mean().sort_values(ascending=False).head(10))

    if(x == 'Nationality Based Player Count'):

        print(fifa_data.groupby("Nationality").Name.count().sort_values(ascending=False).head(10))



print("Squad Analysis:")



@widgets.interact_manual(

    y = ['4-3-3 Formation','3-5-2 Formation','4-2-3-1 Formation'])  

def fine(y=0):

    if(y == '4-3-3 Formation'):

        print(" Best 4-3-3 Formation")

        print (formation_best_squad(formation433))

    if(y == '3-5-2 Formation'):

        print(" Best 3-5-2 Formation")

        print (formation_best_squad(formation352))

    if(y == "4-2-3-1 Formation"):

        print(" Best 4-2-3-1 Formation")

        print (formation_best_squad(formation4231))

        

print("Visualisation Graphs: ")



@widgets.interact_manual(

    z = ['Overall VS Potential','Player based on Overall & Potential','Players Free Kick and Penalties','Comparision of Player Skills','Club rates of Top 100 players','Wage Vs Value of Player'])  

def fine(z=0):

    if(z == 'Overall VS Potential'):

        iplot(fig1)

    if(z == 'Player based on Overall & Potential'):

        iplot(fig)

    if(z == 'Players Free Kick and Penalties'):

        iplot(fig2)

    if(z == 'Comparision of Player Skills'):

        iplot(fig3)

    if(z == 'Club rates of Top 100 players'):

        iplot(fig5)

    if(z == 'Wage Vs Value of Player'):

        iplot(fig6)

        

print("Type Based Classification:")

@widgets.interact_manual(

    a = ['Nationality rates of players','Word Cloud','Top 20 Players','Wage-Value Analysis','Maximum Wage Clubs','Player Nationality'])  

def fine(a=0):

    if(a == 'Nationality rates of players'):

        iplot(fig4)

    if(a == 'Word Cloud'):

        plt.imshow(wordcloud)

        plt.axis('off')

        plt.savefig('graph.png')

    if(a == 'Top 20 Players'):

        print(Top_20[:20])

    if(a == 'Wage-Value Analysis'):

        iplot(fig7)

    if(a == 'Maximum Wage Clubs'):

        iplot(fig8)

    if(a == 'Player Nationality'):

        #How many players are from the same country?

        df = fifa_data['Nationality'].value_counts()



        iplot([

                go.Choropleth(

                locationmode='country names',

                locations=df.index.values,

                text= df.index,

                z=df.values,

                colorscale= 'Jet'

                )

                ])
