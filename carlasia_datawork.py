# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import json

import glob

import os

from folium.plugins import MarkerCluster

from folium import plugins

from pandas.io.json import json_normalize

from folium import FeatureGroup, LayerControl, Map, Marker

# import calmap

import plotly.graph_objects as go

import folium
full_table = pd.read_csv('../input/trajec/train.csv' )

full_table
# Create two lists for the loop results to be placed

lat = []

lon = []



for row in full_table['space']:

    try:

        lat.append(row.split(' ')[0])

        lon.append(row.split(' ')[1])

    except:

        # append a missing value to lat

        lat.append(np.NaN)

        # append a missing value to lon

        lon.append(np.NaN)

       # y_data4 = ((2, 3, 4), (1, 2, "hi"))

    

f = [float(lat[i]) for i in range (0,len(lat))]

type(f[0])

g = [float(lon[i]) for i in range (0,len(lon))]

type(f[0])

full_table['latitude'] = f

full_table['longitude'] = g



full_table 



    
# Function Gather the lat and lon in a list into this form : [(lat,lon)]    

def merge(list1, list2): 

      

    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 

    return merged_list



merged = merge(f,g)

merged

full_table['newspace']= merged



full_table



one = full_table[full_table['tid'] == 127]

two = full_table[full_table['tid'] == 129]

three = full_table[full_table['tid'] == 130]

four = full_table[full_table['tid'] == 131]

five = full_table[full_table['tid'] == 132]

six = full_table[full_table['tid'] == 133]

seven = full_table[full_table['tid'] == 135]

eight = full_table[full_table['tid'] == 137]

nine = full_table[full_table['tid'] == 139]





# Function to give a position number, an order 

def position(lis):

    bar = [i for i in range(1,len(lis)+1)]

    lis['position'] = bar

    return lis

# Affecting every trajectories a position to give an order

onep = position(one)

twop = position(two)

threep = position(three)

fourp = position(four)

fivep = position(five)

sixp = position(six)

sevenp = position(seven)

eightp = position(eight)

ninep = position(nine)



# Adding in the dataframe the position order

df_new = pd.concat([onep['position'] , twop['position'] , threep['position'] ,fourp['position'] , fivep['position'] , sixp['position'] , sevenp['position'] , eightp['position'] , ninep['position']])

full_table['position']= df_new

full_table



m = folium.Map(location=[40.8, -74], tiles='cartodbpositron',

                   min_zoom=1, max_zoom=25, zoom_start=10)



#marker_cluster = MarkerCluster().add_to(m)

#mcg = folium.plugins.MarkerCluster(control=False)



mcg = folium.FeatureGroup(name='groups')

m.add_child(mcg)



g1 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 127')

m.add_child(g1)

g2 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 129')

m.add_child(g2)

g3 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 130')

m.add_child(g3)

g4 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 131')

m.add_child(g4)

g5 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 132')

m.add_child(g5)

g6 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 133')

m.add_child(g6)

g7 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 135')

m.add_child(g7)

g8 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 137')

m.add_child(g8)

g9 = folium.plugins.FeatureGroupSubGroup(mcg, 'trajectory 139')

m.add_child(g9)



#CircleMarker

for i in range (0, len(one)):

    folium.CircleMarker(

            location=[one.iloc[i]['latitude'], one.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(one.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(one.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(one.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(one.iloc[i]['poi']),radius= int(one.iloc[i]['position'])/2).add_to(g1)

    

    

for i in range (0, len(two)):

    folium.Marker(

            location=[two.iloc[i]['latitude'], two.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(two.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(two.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(two.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(two.iloc[i]['poi']),radius=0.1).add_to(g2)

    

for i in range (0, len(three)):

    folium.Marker(

            location=[three.iloc[i]['latitude'], three.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(three.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(three.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(three.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(three.iloc[i]['poi']),radius=0.1).add_to(g3)

for i in range (0, len(four)):

    folium.Marker(

            location=[four.iloc[i]['latitude'], four.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(four.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(four.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(four.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(four.iloc[i]['poi']),radius=0.1).add_to(g4)

for i in range (0, len(five)):

    folium.Marker(

            location=[five.iloc[i]['latitude'], five.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(five.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(five.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(five.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(five.iloc[i]['poi']),radius=0.1).add_to(g5)

for i in range (0, len(six)):

    folium.Marker(

            location=[six.iloc[i]['latitude'], six.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(six.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(six.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(six.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(six.iloc[i]['poi']),radius=0.1).add_to(g6)

for i in range (0, len(seven)):

    folium.Marker(

            location=[seven.iloc[i]['latitude'], seven.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(seven.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(seven.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(seven.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(seven.iloc[i]['poi']),radius=0.1).add_to(g7)

for i in range (0, len(eight)):

    folium.Marker(

            location=[eight.iloc[i]['latitude'], eight.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(eight.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(eight.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(eight.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(eight.iloc[i]['poi']),radius=0.1).add_to(g8)

for i in range (0, len(nine)):

    folium.Marker(

            location=[nine.iloc[i]['latitude'], nine.iloc[i]['longitude']],

            tooltip = '<li><bold>Position : '+str(nine.iloc[i]['position'])

                        +'<li><bold>Trajectory : '+str(nine.iloc[i]['tid'])

                        +'<li><bold>Date : '+str(nine.iloc[i]['datetime'])

                        +'<li><bold>Activity : '+str(nine.iloc[i]['poi']),radius=0.1).add_to(g9)



folium.LayerControl(collapsed=False).add_to(m)



plugins.Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(m)



m
#import sys

#sys.setrecursionlimit(100)

# Add line to map

folium.PolyLine(one['newspace'],color="grey", weight=2, opacity=0.5).add_to(g1)

folium.PolyLine(two['newspace'],color="grey", weight=2, opacity=0.5).add_to(g2)

folium.PolyLine(three['newspace'],color="grey", weight=2, opacity=0.5).add_to(g3)

folium.PolyLine(four['newspace'],color="grey", weight=2, opacity=0.5).add_to(g4)

folium.PolyLine(five['newspace'],color="grey", weight=2, opacity=0.5).add_to(g5)

folium.PolyLine(six['newspace'],color="grey", weight=2, opacity=0.5).add_to(g6)

folium.PolyLine(seven['newspace'],color="grey", weight=2, opacity=0.5).add_to(g7)

folium.PolyLine(eight['newspace'],color="grey", weight=2, opacity=0.5).add_to(g8)

folium.PolyLine(nine['newspace'],color="grey", weight=2, opacity=0.5).add_to(g9)



m
full_table.to_csv('newtrain.csv', index = False)





traj = full_table

#Change of the index

traj.index.name = 'Position'

#Transpos of full_table

traj= full_table.T

traj
#load json object

with open('../input/newtraj/moveletsOnTrain.json') as f:

    d = json.load(f)



nycphil = pd.json_normalize(d['shapelets'])

datas =nycphil['Data'][0]



df = pd.DataFrame()

st = pd.DataFrame()

count = 1



for x in range(0, 27):

    new = []

    posi = d['shapelets'][x]['points_with_only_the_used_features']

    new = pd.json_normalize(posi)

    new['tid'] = d['shapelets'][x]['trajectory']

    new['start'] = d['shapelets'][x]['start']

    new['end'] = d['shapelets'][x]['end']

    new['label'] = d['shapelets'][x]['label']

    new['size'] = int(d['shapelets'][x]['quality']['size'])

    new['quality'] = int(d['shapelets'][x]['quality']['quality'] * 100)

    new['movelet_id'] = count

    st = st.append(new)

    count += 1



st = st.sort_values(by = ['tid', 'movelet_id'])



st.reset_index(drop = True, inplace = True)



#st.set_index(['tid','Position'], inplace=True)



st
mov13 = st[st['movelet_id'] == 13]

mov15 = st[st['movelet_id'] == 15]

mov15 = st[st['movelet_id'] == 15]

mov16 = st[st['movelet_id'] == 16]

mov17 = st[st['movelet_id'] == 17]

mov18 = st[st['movelet_id'] == 18]

mov19 = st[st['movelet_id'] == 19]

mov20 = st[st['movelet_id'] == 20]

    

traj127 =full_table[full_table['tid'] == 127]



def mergee(list1, list2):

    h = pd.DataFrame()

    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]

    print(merged_list)

    y = [(i) for i in range(merged_list[0][0],merged_list[0][-1]+1)]

    h =h.append(yo)

    return y



pl13 =mergee(mov13['start'],mov13['end'])





mov13['place'] =pl13



v13 = mov13['place']

b13 = mov13['movelet_id']





plt.plot(v13,b13)



plt.legend()

 

# Add title and axis names

my_range=range(1,len(traj127)+1)

plt.xticks(my_range)

plt.title("movelets in trajectories", loc='left')

plt.xlabel('Trajectory points')

plt.ylabel('movelets')

e = pd.DataFrame()

g =pd.DataFrame()

e =mov13['start']

g =mov13['end']

merged_list = [(mov13['start'], mov13['end']) for i in range(0, len(mov13['start']))] 

merged_list 
import pandas as pd

import matplotlib.pyplot as plt

 

# Create a dataframe

value1=np.random.uniform(size=20)

value2=value1+np.random.uniform(size=20)/4

df = pd.DataFrame({'group':list(map(chr, range(65, 85))), 'value1':value1 , 'value2':value2 })

 

# Reorder it following the values of the first value:

ordered_df = df.sort_values(by='value1')

my_range=range(1,len(st.index)+1)

 

# The vertical plot is made using the hline function

# I load the seaborn library only to benefit the nice looking feature

import seaborn as sns

plt.hlines(y=my_range, xmin=st['start'], xmax=st['end'], color='grey', alpha=0.4)

plt.scatter(st['start'], my_range, color='skyblue', alpha=1, label='value1')

plt.scatter(st['end'], my_range, color='green', alpha=0.4 , label='value2')

plt.legend()

 

# Add title and axis names

plt.yticks(my_range, ordered_df['group'])

plt.title("Comparison of the value 1 and the value 2", loc='left')

plt.xlabel('Trajectory points')

plt.ylabel('movelets')

x=[2,17]

for i in range(x):

    h=h.append(i)
# Function Gather the lat and lon in a list into this form : [(lat,lon)]    

def merge(list1, list2): 

      

    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 

    return merged_list

y= st['features.space.y']

x = st['features.space.x']

love =merge(x,y)

love



st['space']= love

st





a = st[st['tid'] == 127]

b = st[st['tid'] == 129]

c = st[st['tid'] == 130]

d = st[st['tid'] == 131]

e = st[st['tid'] == 132]

f = st[st['tid'] == 133]

g = st[st['tid'] == 135]

h = st[st['tid'] == 137]

i = st[st['tid'] == 139]

i





folium.PolyLine(a['space'],color="cyan", weight=2, opacity=0.5).add_to(g1)

folium.PolyLine(b['space'],color="red", weight=2, opacity=0.5).add_to(g2)

folium.PolyLine(c['space'],color="green", weight=2, opacity=0.5).add_to(g3)

folium.PolyLine(d['space'],color="purple", weight=2, opacity=0.5).add_to(g4)

folium.PolyLine(e['space'],color="magenta", weight=2, opacity=0.5).add_to(g5)

folium.PolyLine(f['space'],color="yellow", weight=2, opacity=0.5).add_to(g6)

folium.PolyLine(g['space'],color="blue", weight=2, opacity=0.5).add_to(g7)

folium.PolyLine(h['space'],color="orange", weight=2, opacity=0.5).add_to(g8)

folium.PolyLine(i['space'],color="black", weight=2, opacity=0.5).add_to(g9)



m