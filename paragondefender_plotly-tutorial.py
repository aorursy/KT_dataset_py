# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud



import matplotlib.pyplot as plt



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.io as pio

pio.templates
pio.templates.default = "plotly_dark"
data=pd.read_csv('../input/fifa19/data.csv')
data.info()
data.head()
data.isnull().sum()
data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Weight'].fillna('200lbs', inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['Wage'].fillna('€200K', inplace = True)
data.fillna(0, inplace = True)
def convert(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)



#prepare data frame

data['Value'] = data['Value'].apply(lambda x : convert(x)/100000)



nationlist=list(data['Nationality'].unique())

overall_ratio=[]

age_ratio=[]

value_ratio=[]

count=[]

for i in nationlist:

    x=data[data['Nationality']==i]

    overall_rate=sum(x.Overall)/len(x)

    age_rate=sum(x.Age)/len(x)

    value_rate=sum(x.Value)/len(x)

    counter=len(x)

    

    overall_ratio.append(overall_rate)

    age_ratio.append(age_rate)

    value_ratio.append(value_rate)

    count.append(counter/10)



newdata=pd.DataFrame({'Nation': nationlist, 'overallRatio':overall_ratio,

                      'ageRatio':age_ratio, 'valueRatio':value_ratio, 'PlayerCounts':count })

newdata.head()





# import graph objects as "go"

import plotly.graph_objs as go



#creating traces

trace0 = go.Scatter(x=newdata.Nation.head(15), y=newdata.overallRatio, name='Overall Ratio')

trace1 = go.Scatter(x=newdata.Nation.head(15), y=newdata.ageRatio, name='Age Ratio')

trace2 = go.Scatter(x=newdata.Nation.head(15), y=newdata.valueRatio, name='Value Ratio x100000')

trace3 = go.Scatter(x=newdata.Nation.head(15), y=newdata.PlayerCounts, name='Number of Player x10')

dataV = [trace0, trace1, trace2, trace3]

iplot(dataV)
# import graph objects as "go"

import plotly.graph_objs as go



#creating traces

trace0=go.Scatter(

                    x = data.Name.head(15),

                    y = data.GKDiving,

                    mode = "markers",

                    name = "GKdiving",

                    marker = dict(symbol = 'diamond',

                                  sizemode = 'diameter',

                                  size = 20,

                                  color = 'rgba(66,133,244,0.8)'),

                    text=data.Overall

)



trace1=go.Scatter(

                    x = data.Name.head(15),

                    y = data.Finishing,

                    mode = "markers",

                    name = "Finishing",

                    marker = dict(

                                  symbol = 'star-square',

                                  sizemode = 'diameter',

                                  size = 20,

                                  color = 'rgba(219,68,55,0.8)'),

                    text=data.Overall)



trace2=go.Scatter(

                    x = data.Name.head(15),

                    y = data.Strength,

                    mode = "markers",

                    name = "Strength",

                    marker = dict(

                                  symbol = 'hexagon2-dot',

                                  sizemode = 'diameter',

                                  size = 20,

                                  color = 'rgba(244,160,0,0.8)'),

                    text=data.Overall)



trace3=go.Scatter(

                    x = data.Name.head(15),

                    y = data.Vision,

                    mode = "markers",

                    name = "Vision",

                    marker = dict(

                                  symbol = 'bowtie',

                                  sizemode = 'diameter',

                                  size = 20,

                                  color = 'rgba(15,157,88,0.8)'),

                    text=data.Overall)



dataV=[trace0,trace1,trace2,trace3]

layout=dict(title='GKDiving, Finishing, Strength, Vision',

            xaxis=dict(title='Highest Overall Score', ticklen=5, zeroline=False),

            yaxis=dict(title='Rating on Scale of 100', ticklen=5, zeroline=False)

)



fig=dict(data=dataV,layout=layout)

iplot(fig)

#prepare data frame

nationlist=list(data['Nationality'].unique())

left=[]

right=[]

for i in nationlist:

    x=data[data['Nationality']==i]

    leftf=len(x[x['Preferred Foot']=="Left"])

    rightf=len(x[x['Preferred Foot']=="Right"])

    

    left.append(leftf)

    right.append(rightf)

    

newdata0=pd.DataFrame({'Nation': nationlist, 'LeftFoot': left, 'RightFoot': right})

# import graph objects as "go"

import plotly.graph_objs as go



#creating traces

trace0=go.Bar(

                    x = newdata0.Nation.head(4),

                    y = newdata0.LeftFoot,

                    name = "Left Foot",

                    marker = dict(color = 'rgba(60,30,180,0.9)',

                                  line=dict(color='rgb(205,141,45)',width=1.5)),

                    )

trace1=go.Bar(

                    x = newdata0.Nation.head(4),

                    y = newdata0.RightFoot,

                    name = "Right Foot",

                    marker = dict(color = 'rgba(255,30,18,0.9)',

                                  line=dict(color='rgb(111,191,54)',width=1.5)),

                    )



dataV=[trace0,trace1]

layout=go.Layout(barmode="group")

fig=go.Figure(data=dataV,layout=layout)

iplot(fig)
# import graph objects as "go"

import plotly.graph_objs as go



#creating traces

trace0={

    'x': newdata0.Nation.head(4),

    'y': newdata0.LeftFoot,

    'name' : 'Left Foot',

    'type' : 'bar'

};

trace1={

    'x' : newdata0.Nation.head(4),

    'y' : newdata0.RightFoot,

    'name' : 'Right Foot',

    'type' : 'bar'

    

};

dataV=[trace0,trace1];

layout={

    'xaxis':{'title': 'Nations'},

    'barmode':'relative',

    'title':'Preferred Foot'

    };

fig=go.Figure(data=dataV,layout=layout)

iplot(fig)
newdata.PlayerCounts=newdata.apply(lambda x: newdata.PlayerCounts*10, axis=0)
#prepare data frame

sorted_data=newdata.sort_values(by='PlayerCounts', ascending=False)

sorted_data.head()



size=sorted_data.PlayerCounts.head(10)

label=sorted_data.Nation



# import graph objects as "go"

import plotly.graph_objs as go



#creating trace

trace=go.Pie(labels=label,

             values=size,

             )

dataV=[trace]

iplot(dataV)
# import graph objects as "go"

import plotly.graph_objs as go



newcolors = ['#355BC5', '#CB4A3E', '#D2D25A', '#487247', '#AD7C1B', '#6A1467', '#9F9C99', 

          '#755139','#88B04B','#00A591']



#creating trace

trace = go.Pie(labels=label, 

               values=size, 

               marker=dict(colors=newcolors),

               textposition = "outside",

               hole = .3)



layout=go.Layout(

                 title="Top 10 Country with the Most Football Players ",

                 legend=dict(orientation="h")

                 )

fig = go.Figure(data=trace, layout=layout)

iplot(fig)
trace0 = go.Histogram(

    x=data.Age,

    opacity=0.7,

    name="Age",

    marker=dict(color='rgba(85,172,238,0.8)',

                line=dict(color='rgb(0,0,0)',width=1.5)

               )

)



dataV=[trace0]

layout=go.Layout(barmode='overlay', title='Histogram of Players Age',

                 xaxis=dict(title='Age'),

                 yaxis=dict(title='count')

                )

fig=go.Figure(data=dataV, layout=layout)

iplot(fig)

trace0 = go.Histogram(

    x=data.Overall,

    opacity=0.7,

    name="Overall",

    marker = dict(color = 'rgba(255,187,0,0.9)',

                           line=dict(color='rgb(0,0,0)',width=1.5)

                 )

)



dataV=[trace0]

layout=go.Layout(barmode='overlay', title='Histogram of Players Age',

                 xaxis=dict(title='Age'),

                 yaxis=dict(title='count')

                )

fig=go.Figure(data=dataV, layout=layout)

iplot(fig)
import plotly.figure_factory as ff
hist_data = [data.Special]

labels = ['Speciality Scores']



fig = ff.create_distplot(hist_data, labels)

fig.update_layout(title_text='Distribution of Speciality Scores')

iplot(fig)
Nations = data.Nationality

plt.subplots(figsize=(15,15))

wordcloud = WordCloud(

            background_color = 'black',

            #width=800,

            #height=600,

            #max_font_size=34,

            min_font_size=10,

            colormap="jet").generate(" ".join(Nations))



plt.imshow(wordcloud)

plt.axis('off')

plt.show()
sorted_data.Nation[11]="United Kingdom" # replace England to United Kingdom



trace =go.Choropleth(

            locationmode = 'country names',

            locations = sorted_data['Nation'],

            text = sorted_data['Nation'],

            z = sorted_data['PlayerCounts'],

            colorscale = 'inferno',

            marker_line_color='darkgray',

            marker_line_width=0.5,

            colorbar_title='Number of<br>Player'

         

)



layout = go.Layout(title = 'Country vs Number of Players')



fig = go.Figure(data = trace, layout = layout)

iplot(fig)
dataV= [

    {

        'y' : data.Overall,

        'x' : data.Name.head(20),

        'mode' : 'markers',

        'marker' : {

            'color': data.Value,

            'size' : data.Special,

            'showscale': True,

            'sizeref': 2.*max(size)/(9.**2),

            'sizemin': 1,

            'colorbar_title':'Value €<br>(x100000)'

        },

        "text" : data.Position

    }

]



layout=go.Layout(title='',

                 xaxis=dict(title='Top 20 Players'),

                 yaxis=dict(title='Overall Score')

                )



fig=go.Figure(data=dataV, layout=layout)

iplot(fig)


trace0= go.Box(

    y=data.Overall,

    x=data.Position,

    name='Overall vs Position',

    marker=dict(color='#FF851B',outliercolor='#000000'),

    

    text=data.Name

)





layout=go.Layout(title='Overall Score vs Positions',

                 xaxis=dict(title='Positions',linecolor='#87BDD8', color='#FFCC5C'),

                 yaxis=dict(title='Overall Score', linecolor='#87BDD8', color='#87BDD8'),

                )





fig=go.Figure(data=trace0,layout=layout)

iplot(fig)
clubs = ('Juventus', 'Paris Saint-Germain', 'Manchester United', 

                  'Manchester City', 'Atlético Madrid', 'FC Barcelona', 

                  'FC Bayern München', 'Chelsea', 'Real Madrid')



selected_clubs = data.loc[data['Club'].isin(clubs) & data['Overall']]

leagues=('Serie A', 'League A', 'La Liga', 'Premiere', 'Bundesliga')
juv=selected_clubs.loc[selected_clubs['Club']=='Juventus']

trace0= go.Box(

    y=juv.Overall,   

    name='Juventus',

    marker=dict(color='#DCDCDC'),

    boxmean='sd',

    text=juv.Name

)



paris=selected_clubs.loc[selected_clubs['Club']=='Paris Saint-Germain']

trace1= go.Box(

    y=paris.Overall,    

    name='Paris Saint-Germain',

    marker=dict(color='#DA291C'),

    boxmean='sd',

    text=paris.Name

)



munited=selected_clubs.loc[selected_clubs['Club']=='Manchester United']

trace2= go.Box(

    y=munited.Overall,    

    name='Manchester United',

    marker=dict(color='#FBE122'),

    boxmean='sd',

    text=munited.Name

)



mcity=selected_clubs.loc[selected_clubs['Club']=='Manchester City']

trace3= go.Box(

    y=mcity.Overall,    

    name='Manchester City',

    marker=dict(color='#6CABDD'),

    boxmean='sd',

    text=mcity.Name

)



aMadrid=selected_clubs.loc[selected_clubs['Club']=='Atlético Madrid']

trace4= go.Box(

    y=aMadrid.Overall,    

    name='Atlético Madrid',

    marker=dict(color='#CB3524'),

    boxmean='sd',

    text=aMadrid.Name

)



barca=selected_clubs.loc[selected_clubs['Club']=='FC Barcelona']

trace5= go.Box(

    y=barca.Overall,    

    name='FC Barcelona',

    marker=dict(color='#A50044'),

    boxmean='sd',

    text=barca.Name

)



bayern=selected_clubs.loc[selected_clubs['Club']=='FC Bayern München']

trace6= go.Box(

    y=bayern.Overall,    

    name='FC Bayern München',

    marker=dict(color='#0066B2'),

    boxmean='sd',

    text=bayern.Name

)



chelsea=selected_clubs.loc[selected_clubs['Club']=='Chelsea']

trace7= go.Box(

    y=chelsea.Overall,    

    name='Chelsea',

    marker=dict(color='#034694'),

    boxmean='sd',

    text=chelsea.Name

)



rMadrid=selected_clubs.loc[selected_clubs['Club']=='Real Madrid']

trace8= go.Box(

    y=rMadrid.Overall,    

    name='Real Madrid',

    marker=dict(color='#FEBE10'),

    boxmean='sd',

    text=rMadrid.Name

)



layout=go.Layout(title='Overall Score vs Clubs',

                 xaxis=dict(tickangle=90,linecolor='#87BDD8', color='#87BDD8'),

                 yaxis=dict(title='Overall Score', linecolor='#87BDD8', color='#87BDD8')

                )



dataV=[trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]

fig=go.Figure(data=dataV,layout=layout)

iplot(fig)
juv=selected_clubs.loc[selected_clubs['Club']=='Juventus']

trace0= go.Box(

    y=juv.Value,   

    name='Juventus',

    marker=dict(color='#DCDCDC'),

    boxmean='sd',

    text=juv.Name

)



paris=selected_clubs.loc[selected_clubs['Club']=='Paris Saint-Germain']

trace1= go.Box(

    y=paris.Value,    

    name='Paris Saint-Germain',

    marker=dict(color='#DA291C'),

    boxmean='sd',

)



munited=selected_clubs.loc[selected_clubs['Club']=='Manchester United']

trace2= go.Box(

    y=munited.Value,    

    name='Manchester United',

    marker=dict(color='#FBE122'),

    boxmean='sd',

    text=munited.Name

)



mcity=selected_clubs.loc[selected_clubs['Club']=='Manchester City']

trace3= go.Box(

    y=mcity.Value,    

    name='Manchester City',

    marker=dict(color='#6CABDD'),

    boxmean='sd',

    text=mcity.Name

)



aMadrid=selected_clubs.loc[selected_clubs['Club']=='Atlético Madrid']

trace4= go.Box(

    y=aMadrid.Value,    

    name='Atlético Madrid',

    marker=dict(color='#CB3524'),

    boxmean='sd',

)



barca=selected_clubs.loc[selected_clubs['Club']=='FC Barcelona']

trace5= go.Box(

    y=barca.Value,    

    name='FC Barcelona',

    marker=dict(color='#A50044'),

    boxmean='sd',

)



bayern=selected_clubs.loc[selected_clubs['Club']=='FC Bayern München']

trace6= go.Box(

    y=bayern.Value,    

    name='FC Bayern München',

    marker=dict(color='#0066B2'),

    boxmean='sd',

)



chelsea=selected_clubs.loc[selected_clubs['Club']=='Chelsea']

trace7= go.Box(

    y=chelsea.Value,    

    name='Chelsea',

    marker=dict(color='#034694'),

    boxmean='sd',

    text=chelsea.Name

)



rMadrid=selected_clubs.loc[selected_clubs['Club']=='Real Madrid']

trace8= go.Box(

    y=rMadrid.Value,    

    name='Real Madrid',

    marker=dict(color='#FEBE10'),

    boxmean='sd',

)



layout=go.Layout(title='Value vs Clubs',

                 xaxis=dict(tickangle=90,linecolor='#87BDD8', color='#87BDD8'),

                 yaxis=dict(title='Value € (x100000)', linecolor='#87BDD8', color='#87BDD8')

                )



dataV=[trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]

fig=go.Figure(data=dataV,layout=layout)

iplot(fig)
#import figure factory

import plotly.figure_factory as ff



#prepare data frame

dataV=data.loc[:2500]

dataV1 = dataV.loc[:,["Overall","Age","Value"]]

dataV1["index"]=np.arange(1,len(dataV1)+1)



#scatter matrix

fig=ff.create_scatterplotmatrix(dataV1, diag="box", index="index", colormap='Portland',

                                colormap_type='cat',

                                height=700,width=700)



iplot(fig)
dataV=data.loc[:250]



trace1=go.Scatter3d(

    x=dataV.Overall,

    y=dataV.BallControl,

    z=dataV.ShotPower,

    mode='markers',

    marker=dict(

        size=dataV.Age,

        color=dataV.Value,

        colorscale='inferno',

        showscale=True

        

    )

)



dataV1=[trace1]



layout=go.Layout(

    margin=dict(l=0,r=0,b=0,t=0),

    

    scene = dict(

    xaxis = dict(

        title='Overall'),

    yaxis = dict(

        title='Ball Control'),

    zaxis = dict(

        title='Shot Power'),),





)



fig=go.Figure(data=dataV1, layout=layout)

iplot(fig)
selected = data.loc[:,["Potential","Overall","Age","Value","Crossing", 

                        "Finishing","HeadingAccuracy","ShortPassing","Volleys", 

                        "Dribbling","Curve","FKAccuracy","LongPassing","BallControl", 

                        "Acceleration","SprintSpeed","Agility","Reactions","Balance", 

                        "ShotPower","Jumping","Stamina","Strength","Positioning", 

                        "Vision","Penalties","Marking","GKDiving"]]

fltr=selected.loc[:2500]

corr=fltr.corr()



trace0 = go.Heatmap(z=corr,

                    x=fltr.columns,

                    y=fltr.columns, 

                    colorscale='thermal')



layout = go.Layout(dict(title = "Correlation Matrix",

                        autosize = False,

                        height  = 600,

                        width   = 800,

                        margin  = dict(l = 200),

                        yaxis   = dict(tickfont = dict(size = 8)),

                        xaxis   = dict(tickfont = dict(size = 8))

                       )

                  )

fig = go.Figure(data=trace0, layout=layout)

iplot(fig)
sorted_data.head()
pio.templates.default = "plotly"

trace0 = go.Sunburst(

    ids=[

    "United Kingdom", "Germany", "Spain", "Argentina", "France",

    "United Kingdom - Players", "Germany - Players", "Spain - Players", "Argentina - Players", "France - Players",

    "United Kingdom - Overall", "Germany - Overall", "Spain - Overall", "Argentina - Overall", "France - Overall",

    "United Kingdom - Age", "Germany - Age", "Spain - Age", "Argentina - Age", "France - Age",

    "United Kingdom - Value", "Germany - Value", "Spain - Value", "Argentina - Value", "France - Value"

    ],

    labels=["United Kingdom", "Germany", "Spain", "Argentina", "France", 

            "Number of<br>Players", "Number of<br>Players", "Number of<br>Players","Number of<br>Players","Number of<br>Players",

            "Age Ratio", "Value<br>Ratio", "Overall",

            "Age Ratio", "Value<br>Ratio", "Overall", 

            "Age Ratio", "Value<br>Ratio", "Overall",

            "Age Ratio", "Value<br>Ratio", "Overall",

            "Age Ratio", "Value<br>Ratio", "Overall"],

    parents=["", "", "", "", "",  

             "United Kingdom", "Germany", "Spain", "Argentina", "France",

             "United Kingdom - Players", "United Kingdom - Players", "United Kingdom - Players", 

             "Germany - Players", "Germany - Players", "Germany - Players",

             "Spain - Players", "Spain - Players", "Spain - Players",

             "Argentina - Players", "Argentina - Players", "Argentina - Players",

             "France - Players", "France - Players", "France - Players",

             

             ],

    values=[0, 0, 0, 0, 0, 

            sorted_data.PlayerCounts[0], sorted_data.PlayerCounts[1], sorted_data.PlayerCounts[2], sorted_data.PlayerCounts[3], sorted_data.PlayerCounts[4], 

            sorted_data.ageRatio[0], sorted_data.valueRatio[0]*10, sorted_data.overallRatio[0],

            sorted_data.ageRatio[1], sorted_data.valueRatio[1]*10, sorted_data.overallRatio[1],

            sorted_data.ageRatio[2], sorted_data.valueRatio[2]*10, sorted_data.overallRatio[2],

            sorted_data.ageRatio[3], sorted_data.valueRatio[3]*10, sorted_data.overallRatio[3], 

            sorted_data.ageRatio[4], sorted_data.valueRatio[4]*10, sorted_data.overallRatio[4]

           ],

    outsidetextfont = {"size": 20, "color": "#377eb8"},

    marker = {"line": {"width": 2}},

)



layout = go.Layout(

    margin = go.layout.Margin(t=0, l=0, r=0, b=0)

)



fig=go.Figure(data=trace0, layout=layout)

iplot(fig)
dataV = dict(

    type='sankey',

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(

        color = "black",

        width = 0.5

      ),

      label=["United Kingdom", "Germany", "Spain", "Argentina", "France", 

             "Number of<br>Players", "Number of<br>Players", "Number of<br>Players", "Number of<br>Players","Number of<br>Players", 

             "Overall Ratio", "Age Ratio", "Value Ratio",

             "Overall Ratio", "Age Ratio", "Value Ratio",

             "Overall Ratio", "Age Ratio", "Value Ratio",

             "Overall Ratio", "Age Ratio", "Value Ratio",

             "Overall Ratio", "Age Ratio", "Value Ratio",],

      color = ["yellow", "blue", "red", "orange", "green",

               "yellow", "blue", "red", "orange", "green",

               "brown","purple","cyan",

               "brown","purple","cyan",

               "brown","purple","cyan",

               "brown","purple","cyan",

               "brown","purple","cyan"]

    ),

    link = dict(

      source = [0,1,2,3,4,

                5,5,5,

                6,6,6,

                7,7,7,

                8,8,8,

                9,9,9],

      target = [5,6,7,8,9,

                10,11,12,

                13,14,15,

                16,17,18,

                19,20,21,

                22,23,24],

      value = [sorted_data.PlayerCounts[0], sorted_data.PlayerCounts[1], sorted_data.PlayerCounts[2],sorted_data.PlayerCounts[3],sorted_data.PlayerCounts[4], 

               sorted_data.overallRatio[0], sorted_data.ageRatio[0], sorted_data.valueRatio[0],

               sorted_data.overallRatio[1], sorted_data.ageRatio[1], sorted_data.valueRatio[1],

               sorted_data.overallRatio[2], sorted_data.ageRatio[2], sorted_data.valueRatio[2],

               sorted_data.overallRatio[3], sorted_data.ageRatio[3], sorted_data.valueRatio[3], 

               sorted_data.overallRatio[4], sorted_data.ageRatio[4], sorted_data.valueRatio[4], 

              ]

  ))









fig = dict(data=[dataV])

iplot(fig, validate=False)