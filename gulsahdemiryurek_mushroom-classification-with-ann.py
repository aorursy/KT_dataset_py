# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data=pd.read_csv("../input/mushrooms.csv")
data.info()
data.head()
values={"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"}

data["cap-shape"]=data["cap-shape"].replace(values)

values2={"f": "fibrous", "g": "grooves","y":"scaly","s": "smooth"}

data["cap-surface"]=data["cap-surface"].replace(values2)

values3={"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}

data["cap-color"]=data["cap-color"].replace(values3)

values4={"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}

data["odor"]=data["odor"].replace(values4)

values5={"a":"attached","f":"free"}

data["gill-attachment"]=data["gill-attachment"].replace(values5)

values6={"c":"close","w":"crowded"}

data["gill-spacing"]=data["gill-spacing"].replace(values6)

values7={"b":"broad","n":"narrow"}

data["gill-size"]=data["gill-size"].replace(values7)

values8={"k":"black","b":"buff","n":"brown","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}

data["gill-color"]=data["gill-color"].replace(values8)

values9={"t":"tapering","e":"enlarging"}

data["stalk-shape"]=data["stalk-shape"].replace(values9)

values10={"b":"bulbous","c":"club","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"}

data["stalk-root"]=data["stalk-root"].replace(values10)

values11={"s":"smooth","k":"silky","f":"fibrous","y":"scaly"}

data["stalk-surface-above-ring"]=data["stalk-surface-above-ring"].replace(values11)

data["stalk-surface-below-ring"]=data["stalk-surface-below-ring"].replace(values11)

values12={"n":"brown","b":"buff","c":"cinnamon","g":"gray","p":"pink","e":"red","w":"white","y":"yellow","o":"orange"}

data["stalk-color-above-ring"]=data["stalk-color-above-ring"].replace(values12)

data["stalk-color-below-ring"]=data["stalk-color-below-ring"].replace(values12)

veil_type={"p":"partial","u":"universal"} 

data["veil-type"]=data["veil-type"].replace(veil_type)

veil_color={"n":"brown","o":"orange","w":"white","y":"yellow"} 

data["veil-color"]=data["veil-color"].replace(veil_color)

ring_number= {"n":"none","o":"one","t":"two"}

data["ring-number"]=data["ring-number"].replace(ring_number)

ring_type={"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}

data["ring-type"]=data["ring-type"].replace(ring_type)

spore_print_color= {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"}

data["spore-print-color"]=data["spore-print-color"].replace(spore_print_color)

population={"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"}

data["population"]=data["population"].replace(population)

habitat={"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"}

data["habitat"]=data["habitat"].replace(habitat)

bruises={"t":"bruises","f":"no"}

data["bruises"]=data["bruises"].replace(bruises)
data.head()
edible=data[data["class"]=="e"]

poisonous=data[data["class"]=="p"]
class_dim = go.parcats.Dimension(

    values=data["class"].values,

    label="Mushroom Types",

    categoryarray=["e", "p"],

    ticktext=['edible', 'poisonous']

)



cap_shape_dim = go.parcats.Dimension(

    values=data["cap-shape"].values,

    label="Cap Shape"

)



cap_surface_dim = go.parcats.Dimension(

  values=data["cap-surface"].values,

  label="Cap Surface"

)

cap_color_dim = go.parcats.Dimension(

  values=data["cap-color"].values,

  label="Cap Color"

)



# Create parcats trace

color = [1 if i=="e" else 0 for i in data["class"]]

colorscale = [[0, 'lightcoral'], [1, 'mediumseagreen']];

data1 = [

    go.Parcats(

        dimensions=[class_dim,cap_surface_dim,cap_shape_dim,cap_color_dim],

        line={'color': color,

              'colorscale': colorscale},

        hoveron='dimension',

        hoverinfo='count+probability',

        labelfont={'size': 18, 'family': 'Times'},

        tickfont={'size': 16, 'family': 'Times'},

        arrangement='fixed',

    )



]





iplot(data1)
data4 = [

  go.Histogram(

    histfunc = "count",

    x = edible["bruises"], 

    name = "edible",

    marker=dict(color="lightgreen",line=dict(color='darkgreen', width=5))

  ),

  go.Histogram(

    histfunc = "count",

    x = poisonous["bruises"],

    name = "poisonous",

    marker=dict(color="mistyrose",line=dict(color='maroon', width=5)),

    opacity=0.75

  )

]



layout = go.Layout(

    title='Bruises Counts with Mushroom Type',

    xaxis=dict(

        title=''

    ),

    yaxis=dict(

        title='Count'

    ),

    bargap=0.2,

    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor="rgb(243, 243, 243)")

fig = go.Figure(data=data4, layout=layout)

iplot(fig)

odor_edible=pd.DataFrame(edible["odor"].value_counts())

odor_poisonous=pd.DataFrame(poisonous["odor"].value_counts())



layout = go.Layout(yaxis=go.layout.YAxis(title='Color'),

                   xaxis=go.layout.XAxis(

                       range=[-1200, 1200],

                     tickvals=[-1000, -700, -300, 0, 300, 700, 1000],

                       ticktext=[1000, 700, 300, 0, 300, 700, 1000],

                       title='Count'),

                   barmode='overlay',

                   bargap=0.1,

                   paper_bgcolor='rgb(243, 243, 243)',

                   plot_bgcolor='cornsilk')



data3 = [go.Bar(y=odor_edible.index,

               x=odor_edible["odor"],

               orientation='h',

               name='Edible',

               hoverinfo='x',

               marker=dict(color='lightslategrey')

               ),

        go.Bar(y=odor_poisonous.index,

               x=-1*odor_poisonous["odor"],

               orientation='h',

               name='Poisonous',

               hoverinfo='text',

               text=odor_poisonous["odor"].astype('int'),

               marker=dict(color='darksalmon')

               )]

iplot(dict(data=data3, layout=layout))
gill_attachment_edible=pd.DataFrame(edible["gill-attachment"].value_counts())

gill_attachment_poisonous=pd.DataFrame(poisonous["gill-attachment"].value_counts())

gill_spacing_edible=pd.DataFrame(edible["gill-spacing"].value_counts())

gill_spacing_poisonous=pd.DataFrame(poisonous["gill-spacing"].value_counts())

gill_size_edible=pd.DataFrame(edible["gill-size"].value_counts())

gill_size_poisonous=pd.DataFrame(poisonous["gill-size"].value_counts())

gill_color_edible=pd.DataFrame(edible["gill-color"].value_counts())

gill_color_poisonous=pd.DataFrame(poisonous["gill-color"].value_counts())
trace0 = go.Scatter(

    x = gill_attachment_edible.index,

    y = gill_attachment_edible["gill-attachment"],

    mode = 'markers',

    name = 'Edible',

    marker= dict(size= 14,

                    line= dict(width=1),

                    color= "cadetblue",

                    opacity= 0.7

                   )

)

trace1 = go.Scatter(

    x = gill_attachment_poisonous.index,

    y = gill_attachment_poisonous["gill-attachment"],

    mode = 'markers',

    name = 'Poisonous',

       marker= dict(size= 14,

                    line= dict(width=1),

                    color= "firebrick",

                    opacity= 0.7,

                   symbol=220

                   )

)

trace2 = go.Scatter(

    x = gill_spacing_edible.index,

    y = gill_spacing_edible["gill-spacing"],

    mode = 'markers',

    name = 'Edible',

    marker= dict(size= 14,

                    line= dict(width=1),

                    color= "cadetblue",

                    opacity= 0.7

                   )

)

trace3 = go.Scatter(

    x = gill_spacing_poisonous.index,

    y = gill_spacing_poisonous["gill-spacing"],

    mode = 'markers',

    name = 'Poisonous',

       marker= dict(size= 14,

                    line= dict(width=1),

                    color= "firebrick",

                    opacity= 0.7,

                    symbol=220

                   )

)

trace4 = go.Scatter(

    x = gill_size_edible.index,

    y = gill_size_edible["gill-size"],

    mode = 'markers',

    name = 'Edible',

    marker= dict(size= 14,

                    line= dict(width=1),

                    color= "cadetblue",

                    opacity= 0.7

                   )

)

trace5 = go.Scatter(

    x = gill_size_poisonous.index,

    y = gill_size_poisonous["gill-size"],

    mode = 'markers',

    name = 'Poisonous',

       marker= dict(size= 14,

                    line= dict(width=1),

                    color= "firebrick",

                    opacity= 0.7,

                   symbol=220

                   )

)

trace6 = go.Scatter(

    x = gill_color_edible.index,

    y = gill_color_edible["gill-color"],

    mode = 'markers',

    name = 'Edible',

    marker= dict(size= 14,

                    line= dict(width=1),

                    color= "cadetblue",

                    opacity= 0.7

                   )

)

trace7 = go.Scatter(

    x = gill_color_poisonous.index,

    y = gill_color_poisonous["gill-color"],

    mode = 'markers',

    name = 'Poisonous',

     marker= dict(size= 14,

                    line= dict(width=1),

                    color= "firebrick",

                    opacity= 0.7,

                   symbol=220

                   )

)



fig = tools.make_subplots(rows=2, cols=2, 

                          subplot_titles=('Gill Attachment','Gill Size', 'Gill Spacing',"Gill Color"))





fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 1, 2)

fig.append_trace(trace5, 1, 2)

fig.append_trace(trace6, 2, 2)

fig.append_trace(trace7, 2, 2)



fig['layout'].update(showlegend=False,height=800, width=800, title='Gill Properties' ,paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor="moccasin")

iplot(fig)
stalk_shape_edible=pd.DataFrame(edible["stalk-shape"].value_counts())

stalk_shape_poisonous=pd.DataFrame(poisonous["stalk-shape"].value_counts())

stalk_root_edible=pd.DataFrame(edible["stalk-root"].value_counts())

stalk_root_poisonous=pd.DataFrame(poisonous["stalk-root"].value_counts())

stalk_surface_above_ring_edible=pd.DataFrame(edible["stalk-surface-above-ring"].value_counts())

stalk_surface_above_ring_poisonous=pd.DataFrame(poisonous["stalk-surface-above-ring"].value_counts())

stalk_surface_below_ring_edible=pd.DataFrame(edible["stalk-surface-below-ring"].value_counts())

stalk_surface_below_ring_poisonous=pd.DataFrame(poisonous["stalk-surface-below-ring"].value_counts())

stalk_color_above_ring_edible=pd.DataFrame(edible["stalk-color-above-ring"].value_counts())

stalk_color_above_ring_poisonous=pd.DataFrame(poisonous["stalk-color-above-ring"].value_counts())

stalk_color_below_ring_edible=pd.DataFrame(edible["stalk-color-below-ring"].value_counts())

stalk_color_below_ring_poisonous=pd.DataFrame(poisonous["stalk-color-below-ring"].value_counts())
data8=[go.Scatterpolar(

      r = list(stalk_shape_edible["stalk-shape"].values),

      theta = stalk_shape_edible.index,

      fill = 'toself',

      name = "Edible",

    thetaunit = "radians",

    ),

 go.Scatterpolar(

      r = list(stalk_shape_poisonous["stalk-shape"].values),

      theta = stalk_shape_poisonous.index,

      fill = 'toself',

      name = 'Poisonous',

    thetaunit = "radians"

    ),

go.Scatterpolar(

      r = stalk_root_edible["stalk-root"].values,

      theta = stalk_root_edible.index,

      fill = 'toself',

      name = "Edible",

    thetaunit = "radians",

     subplot = "polar2"

    ),

go.Scatterpolar(

      r = stalk_root_poisonous["stalk-root"].values,

      theta =stalk_root_poisonous.index,

      fill = 'toself',

      name = 'Poisonous',

    subplot = "polar2",

    thetaunit = "radians"

    ),

go.Scatterpolar(

      r = stalk_surface_above_ring_edible["stalk-surface-above-ring"].values,

      theta = stalk_surface_above_ring_edible.index,

      fill = 'toself',

      name = "Edible",

    subplot = "polar3",

     thetaunit = "radians"

    ),

go.Scatterpolar(

      r = stalk_surface_above_ring_poisonous["stalk-surface-above-ring"].values,

      theta = stalk_surface_above_ring_poisonous.index,

      fill = 'toself',

      name = 'Poisonous',

    subplot = "polar3",

     thetaunit = "radians"

    ),

go.Scatterpolar(

        r = stalk_surface_below_ring_edible["stalk-surface-below-ring"].values,

      theta = stalk_surface_below_ring_edible.index,

      fill = 'toself',

      name = "Edible",

    subplot = "polar4"

    ),

go.Scatterpolar(

      r = stalk_surface_below_ring_poisonous["stalk-surface-below-ring"].values,

      theta = stalk_surface_below_ring_poisonous.index,

      fill = 'toself',

      name = "Poisonoıs",

    subplot = "polar4",

    

    ),]

layout = go.Layout(

    showlegend=False,

    paper_bgcolor='moccasin',

    title="STALK PROPERTIES",

    font=dict(family='Gravitas One',size=20,color='darkred'),

     

    

    polar = dict(

      bgcolor="linen",

      domain = dict(

        y = [0.60, 0.90],

        x = [0, 0.48]

      ),

      radialaxis = dict(

             visible = False,

        angle = 45

      ),

      angularaxis = dict(

        direction = "clockwise",

        period = 6,

          gridwidth=3,

          tickfont=dict(size=11,color="black"),

      )

    ),

    polar2 = dict(

        bgcolor="linen",

      domain = dict(

        y = [0.60, 0.90],

        x = [0.52, 1]

      ),

      radialaxis = dict(

             visible = False,

        angle = 45

      ),

      angularaxis = dict(

        direction = "clockwise",

        period = 5,

           gridwidth=3,

          tickfont=dict(size=11,color="black"),

      )),

     polar3 = dict(

         bgcolor="linen",

      domain = dict(

        x = [0, 0.48],

        y = [0, 0.30]

      ),

      

    radialaxis = dict(

             visible = False,

        

        angle = 45

      ),

      angularaxis = dict(

        direction = "clockwise",

        period = 6,

           gridwidth=3,

          tickfont=dict(size=11,color="black"),

      )

     

    ),

    polar4 = dict(

         bgcolor="linen",

      domain = dict(

        y = [0, 0.30],

        x = [0.52, 1]

      ),

   radialaxis = dict(

             visible = False,

        angle = 45,

       

      ),

      angularaxis = dict(

        direction = "clockwise",

        period = 4,

           gridwidth=3,

          tickfont=dict(size=11,color="black")

      )

    ),

     annotations=[dict(showarrow=False,text="Stalk Shape",x=0.18,y=1.05,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5),

                                  dict(showarrow=False,text="Stalk Root",x=0.83,y=1.05,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5),

                 dict(showarrow=False,text="Stalk Surface Above Ring",x=0.13,y=0.40,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5),

                 dict(showarrow=False,text="Stalk Surface Below Ring",x=0.88,y=0.40,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5)]

)



fig = go.Figure(data=data8,layout=layout)

iplot(fig)
value1=stalk_color_above_ring_edible["stalk-color-above-ring"].values

label1=stalk_color_above_ring_edible.index

value2=stalk_color_above_ring_poisonous["stalk-color-above-ring"].values

label2=stalk_color_above_ring_poisonous.index

value3=stalk_color_below_ring_edible["stalk-color-below-ring"].values

label3=stalk_color_below_ring_edible.index

value4=stalk_color_below_ring_poisonous["stalk-color-below-ring"].values

label4=stalk_color_below_ring_poisonous.index



trace1=go.Bar(

      x = value1,

      y =label1 ,

      name='Edible- Stalk Color Above Ring',

    orientation = 'h',

    marker = dict(

        color = "darksalmon",

        line = dict(

            color = 'rgba(58, 71, 80, 1.0)',

            width = 3),

        opacity=0.8,

    ))

    

trace2=go.Bar(

     x = value2 ,

     y = label2,

      name='Poisonous-Stalk Color Above Ring',

    orientation = 'h',

    marker = dict(

        color = "plum",

        line = dict(

            color = 'rgba(58, 71, 80, 1.0)',

            width = 3), opacity=0.8))

    

trace3=go.Bar(

     x = value3,

     y = label3,

            name='Edible-Stalk Color Below Ring',

    orientation = 'h',

    marker = dict(

        color = "palegreen",

        line = dict(

            color = 'rgba(58, 71, 80, 1.0)',

            width = 3), opacity=0.8))

   

trace4=go.Bar(

      x =value4 ,

      y =label4 ,

             name='Poisonous- Stalk Color Below Ring',

    orientation = 'h',

    marker = dict(

        color = "sienna",

        line = dict(

            color = 'rgba(58, 71, 80, 1.0)',

            width = 3), opacity=0.8))

   



fig= tools.make_subplots(rows=1, cols=2,subplot_titles=('Stalk Color Counts Above Ring','Stalk Color Counts Below Rings'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)



fig['layout'].update(showlegend=True,height=600, width=800, barmode='stack',legend=dict(x=.58, y=-0.1,orientation="h",font=dict(size=11,color='#000')),

                     title='Stalk Colors Above and Below Ring')

iplot(fig)
edible_veil_color=pd.DataFrame(edible["veil-color"].value_counts())

poisonous_veil_color=pd.DataFrame(poisonous["veil-color"].value_counts())
trace1 = go.Bar(

    x=edible_veil_color.index,

    y=edible_veil_color["veil-color"].values,

    text=edible_veil_color["veil-color"].values,

    textposition = 'auto',

    name="Edible",

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.6

)



trace2 = go.Bar(

    x=poisonous_veil_color.index,

    y=poisonous_veil_color["veil-color"],

    text=poisonous_veil_color["veil-color"],

    name="Poisonous",

    textposition = 'auto',

    marker=dict(

        color='rgb(58,200,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.6

    

)

Layout=go.Layout(

    title='Veil Colors',

    barmode='stack',

    paper_bgcolor='rgba(245, 246, 249, 1)',

    plot_bgcolor='rgba(245, 246, 249, 1)'

   

)



data65 = [trace1,trace2]

fig = go.Figure(data=data65, layout=Layout)

iplot(fig)
import plotly.figure_factory as ff



z=[[0, 3680, 528], [36, 3808, 72]]



x=['None', 'One', 'Two']

y = ['Edible', 'Poisonous']



z_text = [["0", '3680', '528'],  

          ['36', '3808', '72']]



fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blackbody')

iplot(fig)
sns.catplot(y="ring-type", hue="class", kind="count",height=5,

            palette="pastel", edgecolor=".6",

            data=data)
fig = {

  "data": [

    {

      "values": [1744,1648,576,48,48,48,48,48],

      "labels": ['Brown','Black','White','Orange',"Purple","Chocolate","Yellow","Buff"],

      "domain": {"column": 0},

      "name": "Edible Mushrooms",

      "hoverinfo":"label+percent+name",

      "type": "pie",

         "hole": .4,

        'marker': {'colors': ['brown', 'black', 'white', 'orange',"purple","sienna","yellow","peru"],

                  "line":{"color":'#000000',"width":2}}

    },

    {

      "values": [1812,1584,224,224,72],

      "labels": ["White","Chocolate","Brown","Black","Green"],   

      "domain": {"column": 1},

      "name": "Poisonous Mushrooms",

      "hoverinfo":"label+percent+name",

         "hole": .4,

      "type": "pie",

        "marker": {"colors":["white","sienna","brown","black","green"],

                  "line":{"color":'#000000',"width":2}}



    }],

  "layout": {

      

        "title":"Edible and Poisonous Mushrooms Spore Print Color Percentages",

        "grid": {"rows": 1, "columns": 2},

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Edible",

                "x": 0.20,

                "y": 1.05

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Poisonous",

                "x": 0.85,

                "y": 1.05

            }

        ]

    }

}

iplot(fig)
class_dim = go.parcats.Dimension(

    values=data["class"].values,

    label="Mushroom Types",

    categoryarray=["e", "p"],

    ticktext=['edible', 'poisonous']

)



population_dim = go.parcats.Dimension(

    values=data["population"].values,

    label="Population"

)



habitat_dim = go.parcats.Dimension(

  values=data["habitat"].values,

  label="Habitat"

)





# Create parcats trace

color = [1 if i=="e" else 0 for i in data["class"]]

colorscale = [[0, 'coral'], [1, 'gray']];

data19 = [

    go.Parcats(

        dimensions=[class_dim,population_dim,habitat_dim],

        line={'color': color,"showscale":True,

              'colorscale': colorscale},

        hoveron='dimension',

        hoverinfo='count+probability',

        labelfont={'size': 18, 'family': 'Arial'},

        tickfont={'size': 16, 'family': 'Arial'},

        arrangement='freeform',

    )



]





iplot(data19)
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for column in data.columns:

    data[column] = labelencoder.fit_transform(data[column])
y = data["class"].values

x = data.drop(["class"],axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 70,batch_size=10)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
history = classifier.fit(x_test, y_test, validation_split=0.20, epochs=70, batch_size=10, verbose=1)



# Plot training & validation accuracy values



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()