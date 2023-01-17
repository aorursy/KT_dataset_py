from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/proje.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/children.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#import plotly_express as px



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt





from plotly.offline import download_plotlyjs, init_notebook_mode, iplot



from plotly.graph_objs import *

init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#reading the files from datasets by using read_csv

years_database=pd.read_csv("../input/yearspopulation/yillar.csv",encoding='latin-1')

#when we use .colums with any database it gives the column names in order 

years_database.drop(columns=['Unnamed: 4'],inplace=True)

# .info method gives us the column names and their data types

print(years_database.info())

# .head() method gives the first five rows of the dataset

years_database.head()



#in every dataset we'll use the same methods to understand about the datasets in detail first, and then go on to 

#the visualization tools,so if you forget turn back to this point it won't be reviewed.

#we'll use plotly library to create barplots we'll visualize the dataset 

#which is about proportion of child population in total population using bars.

import plotly.graph_objs as go



x=list(years_database.Year)

#x is the list of years in dataset

trace1={ 

        'x':x,

        'y':years_database['Total population'],

        'name':'Total Population',

        'type':'bar' }

#first trace will indicate the values of total population in the yaxis.

trace2={

        'x':x,

        'y':years_database['Total child population'],

        'name':'Total Child Population',

        'type':'bar'

                     }

#second trace will indicate the values of total child population in the yaxis.

data=[trace1,trace2]

#and we combine the trace1 and trace2 in data list

layout={

         'xaxis':{'title':'Total Population-Total Child Population'},

         'barmode':'relative',

         #relative puts values in a stack

         'title':'Total Population vs. Child Population',

      

}



fig=go.Figure( data=data,layout=layout )

iplot( fig )

#we'll use the seaborn library to visualize Proportion Of child Population in total population in our dataset.

#but don't forget to sort your dataset according to years if it wasn't sorted.

#and reindex your values otherwise it won't be printed in the correct order when you want to use them with their indexes.

new_index = (years_database['Year'].sort_values(ascending=False)).index.values

sorted_data = years_database.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['Year'], y=sorted_data['Proportion Of child Population in total population'])

plt.xticks(rotation= 45)

plt.xlabel('Years',size=15)

plt.ylabel('Percent Values',size=15)

plt.title('Proportion Of child Population in total population',size=15)

#reading another dataset

earth=pd.read_csv("../input/earths/world-child.csv",encoding='latin-1')

earth.columns

most_child_population = earth.sort_values(['childpopulation'],ascending=False)

most_child_population.head()
import folium 

import plotly.graph_objs as go

import folium.plugins as plugins







world=folium.Map(location=[28.644800,77.216721],zoom_start=2.7)



dataUsa=np.array([[28.644800,77.216721,earth['childpopulation'][0]],[39.913818,116.363625,earth['childpopulation'][1]],[6.465422, 3.406448,earth['childpopulation'][2]],[-8.409518, 115.188919,earth['childpopulation'][3]],[33.738045,73.084488,earth['childpopulation'][4]]])



plugins.HeatMap(dataUsa,name='Highest Number Of Childs',radius = 20, min_opacity = 0.6, max_val = 50,gradient={.6: 'red', .98: 'yellow', 1: 'black'}).add_to(world)



world

most_child_proportion = earth.sort_values(['ProportionofChildpopulation'],ascending=False)

most_child_proportion.head(3)
import folium 

import plotly.graph_objs as go

import folium.plugins as plugins



lat=16.4499982

lang= 14.5166646



world=folium.Map(location=[lat,lang],zoom_start=4)



dataUsa=np.array([[9.077751, 8.6774567,earth['childpopulation'][0]],[17.5739347, -3.9861092,earth['childpopulation'][1]],[1.3707295, 32.3032414,earth['childpopulation'][2]]])



plugins.HeatMap(dataUsa,name='Highest Number Of Childs',radius=15).add_to(world)



world
from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/uganda-child.jpg")
#we import other files to use in forwarding visualization tools.



causes_of_punishment=pd.read_csv("../input/punish1/regions2.csv",encoding='latin-1')

obtaining_info=pd.read_csv("../input/obtaininfo/obtaining-info.csv",encoding='latin-1')

#provinces=pd.read_csv("../input/provincest/iller.csv",encoding='latin-1')

#provincest=pd.read_csv("../input/turkeyprovinces/turkey-provinces.csv",encoding='latin-1')

punishments=pd.read_csv("../input/punishes/punishment.csv",encoding='latin-1')

behaviours=pd.read_csv("../input/behaves2/behaviours.csv",encoding='latin-1')

world_childs=pd.read_csv("../input/earths/world-child.csv",encoding='latin-1')

age_groups=pd.read_csv("../input/mosthapp/mosthappy.csv",encoding='latin-1')



#we are reading the rest of csv files from datasets.



age_groups.sort_values(by='Proportion Of Child Population in total population', ascending=True,inplace=True)

#we've sorted the age_groups dataset according to the Proportion Of Child Population in total population.



from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/birthj.jpg")
age_groups.head()
age_groups.tail()
f, axes = plt.subplots(1, 2)

#we'll use subplots to use two graph in one code

f.set_figheight(8)

f.set_figwidth(15)



pie1_list =age_groups['Proportion Of Child Population in total population'] # str(2,4) => str(2.4) = > float(2.4) = 2.4

labels_list =age_groups.Province



#plt.figure( figsize=(8,8))

sns.barplot( x=labels_list[-5:] ,y=pie1_list[-5:] ,palette=sns.cubehelix_palette(),orient='v' , ax=axes[0] )

sns.barplot( x=labels_list[:5] ,y=pie1_list[:5] ,palette=sns.cubehelix_palette(rot=-.15),orient='v' , ax=axes[1] )



plt.xlabel('Provinces')

plt.ylabel('Proportion Of Child Population in total population in %')

birth_data=pd.read_csv("../input/births/cesarean.csv",encoding='latin-1')

birth_data.head()

 #The 'hoverinfo' property is a flaglist and may be specified

 #   as a string containing:

  #    - Any combination of ['x', 'y', 'z', 'text', 'name'] joined with '+' characters

   #     (e.g. 'x+y')



list_r=list(birth_data['Regions'])

trace1 = go.Scatter3d(

    x=birth_data['year-2002'],

    y=birth_data['year-2014'],

    z=birth_data['year-2015'],

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,128,16)',  

        

        # set color to an array/list of desired values      

    ),

    text = ["t: {}".format(x) for x in birth_data['Regions'] ], 

    hoverinfo='x+y+z+text+name'

)



data = [trace1]

layout = go.Layout(

    margin=dict(

      

        l=0,

        r=0,

        b=0,

        t=0  

    ),

     scene = dict(

    xaxis = dict(

        title='Year-2002'),

    yaxis = dict(

        title='Year-2014'),

    zaxis = dict(

        title='Year-2015'),

     ),

      

  

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)




birth_data=pd.read_csv("../input/births/cesarean.csv",encoding='latin-1')

birth_data.head()

 #The 'hoverinfo' property is a flaglist and may be specified

 #   as a string containing:

  #    - Any combination of ['x', 'y', 'z', 'text', 'name'] joined with '+' characters

   #     (e.g. 'x+y')



import folium 

import plotly.graph_objs as go



lat=38.9637451

lang=35.2433205



Turkey=folium.Map(location=[lat,lang],zoom_start=6.4)

col1=birth_data[ birth_data['Regions']=='TR1 Istanbul' ]

col2=birth_data[ birth_data['Regions']=='TR2 Bati Marmara-West Marmara' ]

col3=birth_data[ birth_data['Regions']=='TR3 Ege-Aegean' ]

col4=birth_data[ birth_data['Regions']=='TR4 Dogu Marmara-East Marmara' ]

col5=birth_data[ birth_data['Regions']=='TR5 Bati Anadolu-West Anatolia' ]

col6=birth_data[ birth_data['Regions']=='TR6 Akdeniz-Mediterranian' ]

col7=birth_data[ birth_data['Regions']=='TR7 Orta Anadolu-Central Anatolia' ]

col8=birth_data[ birth_data['Regions']=='TR8 Bati Karadeniz-West Black Sea' ]

col9=birth_data[ birth_data['Regions']=='TR9 Dogu Karadeniz-East Black Sea' ]

col10=birth_data[ birth_data['Regions']=='TRA Kuzeydogu Anadolu-Northeast Anatolia' ]

col11=birth_data[ birth_data['Regions']=='TRB Ortadogu Anadolu-Centraleast Anatolia' ]

col12=birth_data[ birth_data['Regions']=='TRC Guneydogu Anadolu-Southeast Anatolia' ]

pop_infoIst='<b>Istanbul</b><br>'+'<b>Year 2002: </b>'+'%'+str(col1['year-2002'][0])+'<br/><b>Year 2014:</b>'+' %'+str(col1['year-2014'][0])+'<br/><b>Year 2015</b>: ' +'%'+str(col1['year-2015'][0])

pop_infoSouthEast='<b>Southeast Anatolia</b><br>'+'<b>Year 2002: </b>'+'%'+str(col12['year-2002'][11])+'<br/><b>Year 2014:</b>'+' %'+str(col12['year-2014'][11])+'<br/><b>Year 2015</b>: ' +'%'+str(col12['year-2015'][11])

pop_infoWestMar='<b>West Marmara</b><br>'+'<b>Year 2002: </b>'+'%'+str(col2['year-2002'][1])+'<br/><b>Year 2014:</b>'+' %'+str(col2['year-2014'][1])+'<br/><b>Year 2015</b>: ' +'%'+str(col2['year-2015'][1])

pop_infoEge='<b>Aegean</b><br>'+'<b>Year 2002: </b>'+'%'+str(col3['year-2002'][2])+'<br/><b>Year 2014:</b>'+' %'+str(col3['year-2014'][2])+'<br/><b>Year 2015</b>: ' +'%'+str(col3['year-2015'][2])

pop_infoEastMar='<b>East Marmara</b><br>'+'<b>Year 2002: </b>'+'%'+str(col4['year-2002'][3])+'<br/><b>Year 2014:</b>'+' %'+str(col4['year-2014'][3])+'<br/><b>Year 2015</b>: ' +'%'+str(col4['year-2015'][3])

pop_infoWestAnat='<b>West Anatolia</b><br>'+'<b>Year 2002: </b>'+'%'+str(col5['year-2002'][4])+'<br/><b>Year 2014:</b>'+' %'+str(col5['year-2014'][4])+'<br/><b>Year 2015</b>: ' +'%'+str(col5['year-2015'][4])

pop_infoMedi='<b>Mediterranian</b><br>'+'<b>Year 2002: </b>'+'%'+str(col6['year-2002'][5])+'<br/><b>Year 2014:</b>'+' %'+str(col6['year-2014'][5])+'<br/><b>Year 2015</b>: ' +'%'+str(col6['year-2015'][5])

pop_infoCentralAnat='<b>Central Anatolia</b><br>'+'<b>Year 2002: </b>'+'%'+str(col7['year-2002'][6])+'<br/><b>Year 2014:</b>'+' %'+str(col7['year-2014'][6])+'<br/><b>Year 2015</b>: ' +'%'+str(col7['year-2015'][6])

pop_infoWestBlackSea='<b>West Black Sea</b><br>'+'<b>Year 2002: </b>'+'%'+str(col8['year-2002'][7])+'<br/><b>Year 2014:</b>'+' %'+str(col8['year-2014'][7])+'<br/><b>Year 2015</b>: ' +'%'+str(col8['year-2015'][7])

pop_infoNortheastAnat='<b>Northeast Anatolia</b><br>'+'<b>Year 2002: </b>'+'%'+str(col10['year-2002'][9])+'<br/><b>Year 2014:</b>'+' %'+str(col10['year-2014'][9])+'<br/><b>Year 2015</b>: ' +'%'+str(col10['year-2015'][9])

pop_infoEastBlackSea='<b>East Black Sea</b><br>'+'<b>Year 2002: </b>'+'%'+str(col9['year-2002'][8])+'<br/><b>Year 2014:</b>'+' %'+str(col9['year-2014'][8])+'<br/><b>Year 2015</b>: ' +'%'+str(col9['year-2015'][8])

pop_infoCentralEast='<b>Centraleast Anatolia</b><br>'+'<b>Year 2002: </b>'+'%'+str(col11['year-2002'][10])+'<br/><b>Year 2014:</b>'+' %'+str(col11['year-2014'][10])+'<br/><b>Year 2015</b>: ' +'%'+str(col11['year-2015'][10])

folium.Marker(location=[41.015137, 28.979530], popup=folium.Popup( pop_infoIst,max_width=200,height=200),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[41.674965, 26.583481], popup=folium.Popup( pop_infoWestMar,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[38.6140, 27.4296], popup=folium.Popup( pop_infoEge,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[40.193298,29.074202], popup=folium.Popup( pop_infoEastMar,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[ 38.756886,30.538704], popup=folium.Popup( pop_infoWestAnat,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[ 36.549362, 31.996994], popup=folium.Popup( pop_infoMedi,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[ 39.925533,32.866287], popup=folium.Popup( pop_infoCentralAnat,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[39.9,41.27], popup=folium.Popup( pop_infoNortheastAnat,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[42.02314,35.153069], popup=folium.Popup( pop_infoWestBlackSea,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[41.02005,40.523449], popup=folium.Popup( pop_infoEastBlackSea,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[39.747662,37.017879], popup=folium.Popup( pop_infoCentralEast,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)

folium.Marker(location=[37.91441,40.230629], popup=folium.Popup( pop_infoSouthEast,max_width=100,height=100),icon=folium.Icon(color='green')).add_to(Turkey)



print(col11)

#folium.CircleMarker([41.015137, 28.979530],

                   # radius=30,

                   # popup='East London',

                   # color='red',

                   # ).add_to(Turkey)

Turkey

import plotly.graph_objs as go



labels =list(birth_data['Regions'])

values2015 =list( birth_data['year-2015'])



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values2015, hole=.3)])

fig.show()
from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/offended.png")
print( causes_of_punishment.info())

causes_of_punishment.head()

#Selected reasons of punishments given to children by their parents, SR Level 1 and three major provinces, 2016

#General behaviours of children tht caues their parents  to punish them according to regions of turkey

from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/hw.png")
area_list=list(causes_of_punishment.Regions)

print(causes_of_punishment.columns)

plt.figure( figsize=(9,15))

a1=sns.barplot( x=list(causes_of_punishment['Neglecting his/her education']) ,y=area_list,color='green',alpha=0.5,label='White')

sns.barplot( x=list(causes_of_punishment['Spending too much time playing on Internet/computer']) ,y=area_list,color='blue',alpha=0.5,label='African American')

sns.barplot( x=list(causes_of_punishment['Failing to perform dutles such as personal care']) ,y=area_list,color='cyan',alpha=0.5,label='Native American')

sns.barplot( x=list(causes_of_punishment['Lying']) ,y=area_list,color='yellow',alpha=0.5,label='Asian')

sns.barplot( x=list(causes_of_punishment['Disrespectfulattitude towards  the elders']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Being violent towards his/her siblings and friends']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Making friends to wrong people']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Not assisting to household chores']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Coming home late']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Excessive spending habits']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

sns.barplot( x=list(causes_of_punishment['Clothing style']) ,y=area_list,color='red',alpha=0.5,label='Hispanic')

    

    

#ax.legend(  loc='upper right',frameon = True)     # legendlarin gorunurlugu

#ax.set(xlabel='Percentage of Rates', ylabel='Regions',title = "Percentage of Punishment Types ")

from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/punishmentj.jpg")
print( behaviours.info())

behaviours.head()

#Perceptions regarding children by sex, SR Level 1 and three major provinces, 2016 Turkey

#Classification according to the regions of Turkey

behs=behaviours.columns[1:]

#we can create new data frames from the previously existing ones to get better results

#for example I've calculated the average values of each perception in Turkey

avgs=[]

for each in behs:

    x=sum(behaviours[each])/len(behaviours[each])

    avgs.append(x)

dictionary={'Perception':behs,'Average Value in Turkey':avgs}



#we sort the average values and print the table to see most common perception types

dfbehs=pd.DataFrame(data=dictionary)

res=dfbehs.sort_values(by=['Average Value in Turkey'],ascending=False)

res
import plotly.graph_objs as go

x = list(behaviours.Regions)



print(behaviours.columns)

fig = go.Figure()

fig.add_trace(go.Bar(x=x, y=behaviours['Only a son can assure the continuation of the bloodline'],name='Only a son can assure the continuation of the bloodline'))

fig.add_trace(go.Bar(x=x, y=behaviours['Each family should have kids depending on their economic standing'],name='Each family should have kids depending on their economic standing'))

fig.add_trace(go.Bar(x=x, y=behaviours['A kid has a negative impact on mothers social/educational and professional life'],name='A kid has a negative impact on mothers social/educational and professional life'))

fig.add_trace(go.Bar(x=x, y=behaviours['A kid has a negative impact on fathers social/educational and professional life '],name='A kid has a negative impact on fathers social/educational and professional life '))

fig.add_trace(go.Bar(x=x, y=behaviours['A woman who has a kid is more respectable than a woman who does not  have a kid'],name='A woman who has a kid is more respectable than a woman who does not  have a kid'))

fig.add_trace(go.Bar(x=x, y=behaviours['A son makes mother more respectable'],name='A son makes mother more respectable'))

fig.add_trace(go.Bar(x=x, y=behaviours['A kid makes the couple closer'],name='A kid makes the couple closer'))

fig.add_trace(go.Bar(x=x, y=behaviours['Once having grown up the kid should financially support the parent'],name='Once having grown up the kid should financially support the parent'))

fig.add_trace(go.Bar(x=x, y=behaviours['The kid should take care of the parent once they get old'],name='The kid should take care of the parent once they get old'))

#fig.add_trace(go.Bar(x=x, y=[-1, 3, -3, -4]))





fig.show()
punishment=pd.read_csv("../input/punishes/punishment.csv",encoding='latin-1')

punishment['Banning mobile phone'].replace(to_replace ='-', value ='0.0',inplace=True)

punishment['Other'].replace(to_replace ='-', value ='0.0',inplace=True)

punishment.replace(to_replace = np.nan, value ='0.0',inplace=True)

print( punishment.info() )

#cleaning the data,there were some '-' and NaN values we've changed them

#and then we've changed the object types to float

punishment['Banning mobile phone']=punishment['Banning mobile phone'].astype('float') #we changed the types and assign to itself

punishment['Other']=punishment['Other'].astype('float',inplace=True) #we changed the types and assign to itself

#Punishments given to children by their parents, SR Level 1 and three major provinces, 2016


import plotly.graph_objs as go



puns=list(punishment.columns[1:])



avg=[]

for each in puns:

    x=sum(punishment[each])/len(punishment[each])

    avg.append(x)

avg



dictionary={'Punishment Type':puns,'Average Value In Turkey':avg }

df=pd.DataFrame(data=dictionary)



df1=df.sort_values(by=['Average Value In Turkey'],ascending=False)

df1
import plotly.graph_objs as go

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']



fig = go.Figure(data=[go.Pie(labels=df['Punishment Type'], 

                             values=df['Average Value In Turkey'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.show()
from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/childmarr.jpg")
marriages=pd.read_csv("../input/marrchild2/mar.csv",encoding='latin-1')

#sorted_df=marriages.sort_values(by='Proportion of child marriages for girls in total marriages', ascending=False)

new_index = (marriages['Proportion of child marriages for girls in total marriages'].sort_values(ascending=False)).index.values

sorted_df = marriages.reindex(new_index)

sorted_df2=sorted_df

sorted_df=sorted_df.iloc[:5,:]

sorted_df




f,ax = plt.subplots(figsize =(20,10))



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +

              ax.get_xticklabels() + ax.get_yticklabels()):

    item.set_fontsize(15)

    

sns.pointplot(x='Province',y='Total Marriages',data=sorted_df,color='pink',alpha=1)

sns.pointplot(x='Province',y='Total Child Marriages For Girls',data=sorted_df,color='purple',alpha=1)



plt.text(3,3500,'Total Marriages',color='pink',fontsize = 17,style = 'italic')

plt.text(3,3000,'Total Child Marriages For Girls',color='purple',fontsize = 18,style = 'italic')





plt.xlabel('Province',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Total Marr. vs Child Marr. In Turkey',fontsize = 20,color='black')

plt.grid()
import plotly_express as px



fig = px.scatter(sorted_df2, y="Total Child Marriages For Girls", color="Proportion of child marriages for girls in total marriages",

                 size='Total Marriages', hover_data=['Province'])

fig.show()
depend_df=pd.read_csv("../input/dependency/depend.csv",encoding='latin-1')

depend_df.head()
import plotly_express as px



px.scatter(depend_df, y="Youth dependency ratio (0-14 age)", x="Total age dependency ratio", animation_frame="Year",

           size="Total age dependency ratio", hover_name="Year",

           log_x=False, size_max=55, range_x=[0,100], range_y=[30,150])
from IPython.display import Image

import os

!ls ../input/



Image("../input/turkishimg/worker.jpg")
labour_df=pd.read_csv("../input/labour/labour.csv",encoding='latin-1')

labour_df


import plotly.graph_objs as go



lb_2014=labour_df[ labour_df['Year']==2014 ]

lb_2015=labour_df[ labour_df['Year']==2015 ]

lb_2016=labour_df[ labour_df['Year']==2016 ]









trace1={ 

        'x':lb_2014.Year,

        'y':lb_2014['Labour force participation rate'],

        'name':'Labour force participation rate',

        'type':'bar' }

trace2={

        'x':lb_2014.Year,

        'y':lb_2014['Employment rate'],

        'name':'Employment rate',

        'type':'bar'

                     }

trace3={

        'x':lb_2014.Year,

        'y':lb_2014['Unemployment rate'],

        'name':'Unemployment rate',

        'type':'bar'

                     }

trace4={

        'x':lb_2015.Year,

        'y':lb_2015['Labour force participation rate'],

        'name':'Labour force participation rate',

        'type':'bar'

                     }

trace5={

        'x':lb_2015.Year,

        'y':lb_2015['Employment rate'],

        'name':'Employment rate',

        'type':'bar'

                     }

trace6={

        'x':lb_2015.Year,

        'y':lb_2015['Unemployment rate'],

        'name':'Unemployment rate',

        'type':'bar'

                     }

trace7={

        'x':lb_2016.Year,

        'y':lb_2016['Labour force participation rate'],

        'name':'Labour force participation rate',

        'type':'bar'

                     }

trace8={

        'x':lb_2016.Year,

        'y':lb_2016['Employment rate'],

        'name':'Employment rate',

        'type':'bar'

                     }

                     

trace9={

        'x':lb_2016.Year,

        'y':lb_2016['Unemployment rate'],

        'name':'Unemployment rate',

        'type':'bar'

                     }



data=[trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9]

layout={

         'xaxis':{'title':'--years--'},

         'yaxis':{'title':'Value Percents'},

         'barmode':'group',

         'title':'Labour force participation-Employment-Unemployment Rates'

}



fig=go.Figure( data=data,layout=layout )

iplot( fig )

import plotly_express as px



fig = px.scatter(labour_df, x='Year',y="Labour force", color="Labour force participation rate",

                 size='Employment rate', hover_data=['Gender'])

fig.show()
import plotly_express as px



labour_df['Labour force participation rate']=sum(labour_df['Labour force participation rate'])/min(labour_df['Labour force participation rate'])



px.scatter(labour_df, x="Labour force", y="Labour force participation rate", animation_frame="Year", animation_group="Gender",

           size="Employment rate", color="Gender", hover_name="Labour force participation rate",

           log_x=False, size_max=55, range_x=[200,700], range_y=[0,60])
killed_df=pd.read_csv("../input/killed/killed.csv",encoding='latin-1')

killed_df.head()



import plotly.graph_objs as go



labels = ['Number Of Traffic Accidents','Number Traffic accidents involving death or persoal injury']

values = [ sum(killed_df['Number Of Traffic Accidents']),sum(killed_df['Number Traffic accidents involving death or persoal injury'])]



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()


import plotly.graph_objs as go





totalkilled=sum(killed_df['Total Killed'])

totalinjured=sum(killed_df['Total Injured'])

totalkilledchild=sum(killed_df['Total Killed Child'])

totalinjuredchild=sum(killed_df['Total Injured Child'])









trace1={ 

        'x':killed_df['Year'],

        'y':killed_df['Total Killed'],

        'name':'Total Killed People',

        'type':'bar' }

trace2={

        'x':killed_df['Year'],

        'y':killed_df['Total Killed Child'],

        'name':'Total Killed Children',

        'type':'bar'

                     }

trace3={ 

        'x':killed_df['Year'],

        'y':killed_df['Total Injured'],

        'name':'Total Injured People',

        'type':'bar' }

trace4={

        'x':killed_df['Year'],

        'y':killed_df['Total Injured Child'],

        'name':'Total Injured Children',

        'type':'bar'

                     }



data=[trace1,trace2,trace3,trace4]

layout={

         'xaxis':{'title':'--years--'},

         'yaxis':{'title':'Values'},

         'barmode':'group',

         'title':'Killed or injured People vs Children',

         'width':1200

}



fig=go.Figure( data=data,layout=layout )

iplot( fig )
import plotly_express as px



fig = px.scatter(killed_df, x="Year", y="Number Traffic accidents involving death or persoal injury", color="Total Injured Child",

                 size='Total Killed Child', hover_data=["Total Injured Child",'Total Killed Child'])

fig.show()
killed_df.columns

import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(15,15),nrows=2, ncols=2)

# Data to plot , 



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +

              ax.get_xticklabels() + ax.get_yticklabels()):

    item.set_fontsize(15)

   

labels1 = 'Age group 0-9 killed', 'Age Group 0-9 Injured'

sizes1 = [sum(killed_df['Age group 0-9 killed']),sum(killed_df['Age Group 0-9 Injured'])]

colors1 = ['gold', 'yellowgreen']

explode = (0.2,0)  # explode 1st slice

explode2 = (0.2,0,0) 

labels2 = 'Age group 10-14 killed', 'Age Group 10-14 Injured'

sizes2 = [sum(killed_df['Age group 10-14 killed']),sum(killed_df['Age Group 10-14 Injured'])]

colors2 = ['lightcoral', 'lightskyblue']



labels3 = 'Age group 15-17 killed', 'Age Group 15-17 Injured'

sizes3 = [sum(killed_df['Age group 15-17 killed']),sum(killed_df['Age group 15-17 Injured'])]

colors3 = ['pink', 'purple']



labels4 = 'Age group 10-14 killed', 'Age group 0-9 killed','Age group 15-17 killed'

sizes4 = [sum(killed_df['Total Killed Child']),sum(killed_df['Age group 0-9 killed']),sum(killed_df['Age group 15-17 killed'])]

colors4 = ['red', 'green','yellow']



# Plot

axes[0,0].pie(sizes1, explode=explode, labels=labels1, colors=colors1,

autopct='%1.2f%%', shadow=True, startangle=140, textprops={'fontsize': 14})



axes[0,1].pie(sizes2, explode=explode, labels=labels2, colors=colors2,

autopct='%1.2f%%', shadow=True, startangle=140, textprops={'fontsize': 14})



axes[1,0].pie(sizes3, explode=explode, labels=labels3, colors=colors3,

autopct='%1.2f%%', shadow=True, startangle=140, textprops={'fontsize': 14})



axes[1,1].pie(sizes4, explode=explode2, labels=labels4, colors=colors4,

autopct='%1.2f%%', shadow=True, startangle=140, textprops={'fontsize': 14})







plt.axis('equal')

plt.show()
