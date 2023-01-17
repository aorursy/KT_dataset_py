# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

#Plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import seaborn as sns



from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from collections import Counter



from warnings import filterwarnings

filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/kc_house_data.csv')

data.head()
data.info()
plt.subplots(figsize=(17,14))

sns.heatmap(data.corr(),annot=True,linewidths=0.5,linecolor="Black",fmt="1.1f")

plt.title("Data Correlation",fontsize=50)

plt.show()
#Null Values

data.isnull().sum()
data.grade.value_counts().head(10)
data.yr_built.value_counts().head(10)
hist1 = [go.Histogram(x=data.grade,marker=dict(color='rgb(102, 0, 102)'))]



histlayout1 = go.Layout(title="Grade Counts",xaxis=dict(title="Grades"),yaxis=dict(title="Counts"))



histfig1 = go.Figure(data=hist1,layout=histlayout1)



iplot(histfig1)
hist2 = [go.Histogram(x=data.yr_built,xbins=dict(start=np.min(data.yr_built),size=1,end=np.max(data.yr_built)),marker=dict(color='rgb(0,102,0)'))]



histlayout2 = go.Layout(title="Built Year Counts",xaxis=dict(title="Years"),yaxis=dict(title="Built Counts"))



histfig2 = go.Figure(data=hist2,layout=histlayout2)



iplot(histfig2)
v1 = [go.Box(y=data.price,name="Price",marker=dict(color="rgba(64,64,64,0.9)"),hoverinfo="name+y")]



layout1 = go.Layout(title="Price")



fig1 = go.Figure(data=v1,layout=layout1)

iplot(fig1)
v21 = [go.Box(y=data.bedrooms,name="Bedrooms",marker=dict(color="rgba(51,0,0,0.9)"),hoverinfo="name+y")]

v22 = [go.Box(y=data.bathrooms,name="Bathrooms",marker=dict(color="rgba(0,102,102,0.9)"),hoverinfo="name+y")]

v23 = [go.Box(y=data.floors,name="Floors",marker=dict(color="rgba(204,0,102,0.9)"),hoverinfo="name+y")]



layout2 = go.Layout(title="Bedrooms,Bathrooms and Floors",yaxis=dict(range=[0,13])) #I hate 33 bedroom



fig2 = go.Figure(data=v21+v22+v23,layout=layout2)

iplot(fig2)
bdata2014 = data[data.yr_built == 2014]



bubble1 = go.Scatter(x=bdata2014.grade,y=bdata2014.bedrooms,name="Bedroom 2014",mode="markers",marker=dict(size=bdata2014.floors*5,color="rgba(0,0,0,1)"))

bubble2 = go.Scatter(x=bdata2014.grade,y=bdata2014.bathrooms,name="Bathroom 2014",mode="markers",marker=dict(size=bdata2014.floors*5,color="rgba(255,0,255,1)"))



bubbledata = [bubble1,bubble2]



layoutbubble = go.Layout(title="Bedroom vs Bathroom with Floors(Size) 2014",xaxis=dict(title="Grades"),yaxis=dict(title="Bedrooms & Bathrooms"))



bubblefig = go.Figure(data=bubbledata,layout=layoutbubble)



iplot(bubblefig)
bdata2015 = data[data.yr_built == 2015]



bubble1 = go.Scatter(x=bdata2015.grade,y=bdata2015.bedrooms,name="Bedroom 2015",mode="markers",marker=dict(size=bdata2015.floors*5,color="rgba(0,0,0,1)"))

bubble2 = go.Scatter(x=bdata2015.grade,y=bdata2015.bathrooms,name="Bathroom 2015",mode="markers",marker=dict(size=bdata2015.floors*5,color="rgba(255,0,255,1)"))



bubbledata = [bubble1,bubble2]



layoutbubble = go.Layout(title="Bedroom vs Bathroom with Floors(Size) 2015",xaxis=dict(title="Grades"),yaxis=dict(title="Bedrooms & Bathrooms"))



bubblefig = go.Figure(data=bubbledata,layout=layoutbubble)



iplot(bubblefig)
data2015 = data[data.yr_built == 2015]

data2015.head()
#normalize 2015 data prices

from sklearn.preprocessing import MinMaxScaler

normalizeprice2015 = (MinMaxScaler().fit_transform(data2015.iloc[:,2:3]))*50

normalizeprice2015
s1 = [go.Scatter3d(x=data2015.bedrooms,y=data2015.bathrooms,z=data2015.floors,mode="markers",marker=dict(size=normalizeprice2015,color="rgba(255,0,0,0.8)"),hoverinfo ="text",text=" Bedroom(s):"+data.bedrooms.apply(str)+" Bathroom(s):"+data.bathrooms.apply(str)+" Floor(s):"+data.floors.apply(str)+" ID: "+data.id.apply(str))]



layout3 = go.Layout(title="Bedrooms vs Bathrooms vs Floors with Price 2015",margin=dict(l=0,b=0,t=0,r=0))



fig3 = go.Figure(data=s1,layout=layout3)



iplot(fig3)
#Create Grade Frame

gradeframe = pd.DataFrame({"Grades":data.grade.value_counts().index,"House_Grade":data.grade.value_counts().values})

gradeframe["Grades"] = gradeframe["Grades"].apply(lambda x : "Grade " + str(x))

gradeframe.set_index("Grades",inplace=True)

gradeframe
p1 = [go.Pie(labels = gradeframe.index,values = gradeframe.House_Grade,hoverinfo="percent+label+value",hole=0.1,marker=dict(line=dict(color="#000000",width=2)))]



layout4 = go.Layout(title="Grade Pie Chart")



fig4 = go.Figure(data=p1,layout=layout4)



iplot(fig4)

builtyear = pd.DataFrame({"Years":data.yr_built})

builtyear["Years"] = builtyear["Years"].apply(lambda x: "y" + str(x)) #I can't use wordcloud with integers so I put y on head

builtyear["Years"].head()
plt.subplots(figsize=(10,10))

wcloud  = WordCloud(background_color="white",width=500,height=500).generate(",".join(builtyear["Years"]))

plt.imshow(wcloud)

plt.title("Years for Most Built Homes",fontsize=40)

plt.axis("off")

plt.show()
#set colors

data["color"] = ""

data.color[data.grade == 1] = "rgb(255,255,255)"

data.color[data.grade == 2] = "rgb(220,220,220)"

data.color[data.grade == 3] = "rgb(242, 177, 172)"

data.color[data.grade == 4] = "rgb(255,133,27)"

data.color[data.grade == 5] = "rgb(255,255,204)"

data.color[data.grade == 6] = "rgb(255,65,54)"

data.color[data.grade == 7] = "rgb(178,37,188)"

data.color[data.grade == 8] = "rgb(51,51,0)"

data.color[data.grade == 9] = "rgb(37,188,127)"

data.color[data.grade == 10] = "rgb(26,51,176)"

data.color[data.grade == 11] = "rgb(132,10,10)"

data.color[data.grade == 12] = "rgb(82,80,80)"

data.color[data.grade == 13] = "rgb(0,0,0)"
#slice +7 grade

dataplus = data[np.logical_and(data.grade >= 7,data.yr_built >= 2000)] 

#list lat and long

lats = list(dataplus.lat.values)

longs = list(dataplus.long.values)
mapbox_access_token = 'pk.eyJ1IjoiZGFya2NvcmUiLCJhIjoiY2pscGFheHA1MXdqdjNwbmR3c290MTZ6dCJ9.K1FMv_q3ZVlKP13RrjFkjg'



mapp = [go.Scattermapbox(lat=lats,lon=longs,mode="markers",marker=dict(size=4.5,color=dataplus["color"]) ,hoverinfo="text",text="Grade:"+dataplus.grade.apply(str)+" Built Year:"+dataplus.yr_built.apply(str)+" Price:"+dataplus.price.apply(str))]



layout5 = dict(title="Grade(+7) - Built Year(+2000) Map",width=800,height=750,hovermode="closest",mapbox=dict(bearing=0,pitch=0,zoom=9,center=dict(lat=47.5,lon=-122.161),accesstoken=mapbox_access_token))



fig5 = go.Figure(data=mapp,layout=layout5)



iplot(fig5)