import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#Some Styling
import plotly.io as pio
pio.templates.default = "plotly_dark"
sns.set_style("darkgrid")


#displaying markdown
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))
    

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')
#We'll be using the training dataset.
forest = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
forest.head()
forest.shape
forest.info()
forest.iloc[:,1:17].describe()
forest.iloc[:,17:37].describe()
forest.iloc[:,37:].describe()
forest.skew()
forest.isna().sum()
forest['Cover_Type'].replace({1:'Spruce/Fir', 2:'Lodgepole Pine', 3:'Ponderosa Pine', 4:'Cottonwood/Willow', 5:'Aspen', 6:'Douglas-fir', 7:'Krummholz'}, inplace=True)

forest = forest.rename(columns={"Wilderness_Area1":"Rawah_WA","Wilderness_Area2":"Neota_WA",
"Wilderness_Area3":"Comanche_Peak_WA","Wilderness_Area4":"Cache_la_Poudre_WA","Horizontal_Distance_To_Hydrology":"HD_Hydrology",
"Vertical_Distance_To_Hydrology":"VD_Hydrology","Horizontal_Distance_To_Roadways":"HD_Roadways",
                               "Horizontal_Distance_To_Fire_Points":"HD_Fire_Points"})
#We can see the new column names......

forest.columns
# Here I have converted the encoded values for columns Wilderness_Areas 
#and Soil_types back to a single column for better analysis.

forest['Wild Areas'] = (forest.iloc[:,11:15] == 1).idxmax(1)
forest['Soil types'] = (forest.iloc[:,15:55] == 1).idxmax(1)

#Drop the columns which are not required now
forest = forest.drop(columns=["Id",'Rawah_WA', 'Neota_WA', 'Comanche_Peak_WA',
       'Cache_la_Poudre_WA', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])
#I don't like this big Soil_Type name
#Let's replace it with some short name
#Which will help us in visualizations
lst = []
for value in forest['Soil types']:
    value = value.replace('Soil_Type',"ST")
    lst.append(value)
    
forest['Soil types'] = lst
fig = px.histogram(forest,x="Cover_Type",color="Cover_Type",height=400,width=800)
fig.show()
fig = px.pie(forest,names="Wild Areas",height=300,width=800)
fig.show()

fig = px.histogram(forest,x="Wild Areas",color="Cover_Type",barmode="group",
                   height=400,width=800)
fig.show()
fig = px.histogram(forest,x="Soil types",color="Cover_Type",height=400,width=850)
fig.show()

fig = px.pie(forest,names="Soil types",height=400,width=850)
fig.update_traces(textposition='inside')
fig.show()
temp =forest[forest['Wild Areas']=="Rawah_WA"][['Wild Areas','Soil types',"Cover_Type"]]
fig = px.histogram(temp,x="Soil types",color="Cover_Type",height=500,width=1000,
                  title="Rawah Wild Area",barmode="group")
fig.show()

temp =forest[forest['Wild Areas']=="Comanche_Peak_WA"][['Wild Areas','Soil types',"Cover_Type"]]
fig = px.histogram(temp,x="Soil types",color="Cover_Type",height=500,width=1000,
                  title="Comanche Peak Area",barmode="group")
fig.show()

temp =forest[forest['Wild Areas']=="Cache_la_Poudre_WA"][['Wild Areas','Soil types',"Cover_Type"]]
fig = px.histogram(temp,x="Soil types",color="Cover_Type",height=500,width=1000,
                  title="Cache la Poudre Wild Area",barmode="group")
fig.show()

temp =forest[forest['Wild Areas']=="Neota_WA"][['Wild Areas','Soil types',"Cover_Type"]]
fig = px.histogram(temp,x="Soil types",color="Cover_Type",height=500,width=1000,
                  title="Neota Wild Area",barmode="group")
fig.show()
fig = px.histogram(forest,x="Elevation",color="Cover_Type",marginal='rug',title="Elevation Histogram",
                  height=500,width=800)
fig.show()
fig = px.box(forest,x="Cover_Type",y="Elevation",color="Cover_Type",height=400,width=900)
fig.update_layout(title={'text':"Elevation Box Plot"})
fig.show()
temp = forest.groupby(['Cover_Type'],as_index=False)[["Elevation"]].median()
temp.sort_values(by="Elevation",ascending=False).style.background_gradient(cmap="Reds")
#Let's look have a look at the wild areas 
#and how are the forest covers distributed in these areas along with these features.

temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Elevation']].median()

#Both barplot and treemap help in better understanding the features.
fig = px.bar(temp, x="Wild Areas", y="Elevation", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

fig = px.treemap(temp, path=['Wild Areas','Cover_Type'],values='Elevation',height=400,width=900)
fig.show()

temp.style.background_gradient(cmap='plasma')
fig = px.histogram(forest,x="Aspect",color="Cover_Type",marginal='rug',title="Aspect Histogram",
                  height=500,width=900)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Aspect']].median()

fig = px.bar(temp, x="Wild Areas", y="Aspect", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

fig = px.treemap(temp, path=['Wild Areas','Cover_Type'],values='Aspect',height=400,width=900)
fig.show()

temp.style.background_gradient(cmap='YlGnBu')
fig = px.histogram(forest,x="Slope",color="Cover_Type",marginal='box',title="Slope Histogram",
                  height=500,width=800)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Slope']].median()

fig = px.bar(temp, x="Wild Areas", y="Slope", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()
fig = px.histogram(forest,x="HD_Hydrology",color="Cover_Type",marginal='rug',title="HD_Hydrology Histogram",
                  height=500,width=800)
fig.show()
fig = px.box(forest,x="Cover_Type",y="HD_Hydrology",color="Cover_Type",height=500,width=800)
fig.update_layout(title={'text':"Horizontal Dis to Hydrology Box Plot"})
fig.show()
#Let's look at their relation with wild areas.....
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['HD_Hydrology']].median()

fig = px.bar(temp, x="Wild Areas", y="HD_Hydrology", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

temp.style.background_gradient(cmap="Blues")

fig = px.histogram(forest,x="VD_Hydrology",color="Cover_Type",marginal='rug',title="VD_Hydrology Histogram",
                  height=500,width=800)
fig.show()
#Let's look at their relation with wild areas.....
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['VD_Hydrology']].median()

fig = px.bar(temp, x="Wild Areas", y="VD_Hydrology", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()
#We can also use treeplot for better visualization
fig = px.treemap(temp, path=['Wild Areas','Cover_Type'],values='VD_Hydrology',height=400,width=800)
fig.show()

temp.style.background_gradient(cmap="BuPu")


fig = px.histogram(forest,x="HD_Roadways",color="Cover_Type",marginal='rug',title="HD_Roadways Histogram",
                  height=500,width=800)
fig.show()
#This plot shows us on average distance to roadways for each forest covers.
temp = forest.groupby(['Cover_Type'],as_index=False)[['HD_Roadways']].median()

fig = px.bar(temp.sort_values(by="HD_Roadways",ascending=False), x="HD_Roadways", y="Cover_Type", color='Cover_Type',orientation='h',
             height=300,width=900)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['HD_Roadways']].median()

fig = px.bar(temp, x="Wild Areas", y="HD_Roadways", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

fig = px.treemap(temp, path=['Wild Areas','Cover_Type'],values='HD_Roadways',height=400,width=800)
fig.show()

temp.style.background_gradient(cmap="Greys")
fig = px.histogram(forest,x="HD_Fire_Points",color="Cover_Type",marginal='rug',title="HD Fire Points Histogram",
                  height=500,width=800)
fig.show()
fig = px.box(forest,x="Cover_Type",y="HD_Fire_Points",color="Cover_Type",height=500,width=800)
fig.update_layout(title={'text':"Horizontal Dis to Fire points Box Plot"})
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['HD_Fire_Points']].median()

fig = px.bar(temp, x="Wild Areas", y="HD_Fire_Points", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

fig = px.treemap(temp, path=['Wild Areas','Cover_Type'],values='HD_Fire_Points',height=400,width=800)
fig.show()

temp.style.background_gradient(cmap='YlOrRd')

fig = px.histogram(forest,x="Hillshade_9am",color="Cover_Type",marginal='box',title="Hillshade at 9am Histogram",
                  height=500,width=800)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Hillshade_9am']].median()

fig = px.bar(temp, x="Wild Areas", y="Hillshade_9am", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()
#No use of treemap as we don't see any difference in bar plots.
fig = px.histogram(forest,x="Hillshade_Noon",color="Cover_Type",marginal='box',title="Hillshade at Noon Histogram",
                  height=500,width=800)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Hillshade_Noon']].median()

fig = px.bar(temp, x="Wild Areas", y="Hillshade_Noon", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

fig = px.histogram(forest,x="Hillshade_3pm",color="Cover_Type",marginal='box',title="Hillshade at 3pm Histogram",
                  height=500,width=800)
fig.show()
temp = forest.groupby(['Wild Areas','Cover_Type'],as_index=False)[['Hillshade_3pm']].median()

fig = px.bar(temp, x="Wild Areas", y="Hillshade_3pm", color='Cover_Type', barmode='group',
             height=400,width=900)
fig.show()

temp.style.background_gradient(cmap="cividis")
forest_corr = forest.corr()
forest_corr.style.background_gradient(cmap="cool")
fig=plt.figure(figsize=(12,10))
sns.heatmap(forest_corr,annot=True,linewidths=.3,cmap='YlOrBr')
fig = px.scatter(forest,x='Elevation',y= 'HD_Roadways',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='Aspect',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='HD_Hydrology',y= 'VD_Hydrology',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='Hillshade_Noon',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='Aspect',y= 'Hillshade_9am',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='Hillshade_9am',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)
fig.show()
fig = px.scatter(forest,x='Slope',y= 'Hillshade_Noon',color='Cover_Type',width=800,height=400)
fig.show()