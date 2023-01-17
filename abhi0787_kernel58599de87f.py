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
import pandas as pd

df=pd.read_csv('../input/caribou-location-tracking/individuals.csv')
df.info()

import matplotlib.pyplot as plt

temp=list(df['study_site'].value_counts().keys())

tempPrime=list(df['study_site'].value_counts())

plt.tick_params(axis='x',rotation=70)

plt.title('Distribution of the study site')

plt.ylabel('#Frequency')

plt.xlabel('study sites')

plt.bar(temp,tempPrime)
df['life_stage']=df['life_stage'].dropna()
df['life_stage'].isna().sum()
df=df.dropna(subset=['life_stage'])
temp=list(df['life_stage'].value_counts().keys())

tempPrime=list(df['life_stage'].value_counts())

plt.tick_params(axis='x',rotation=70)

plt.title('Distribution of life stage of the animals')

plt.ylabel('#Frequency count')

plt.xlabel('Age groups of animals')

plt.bar(temp,tempPrime)
df['pregnant'].isnull().sum()
df=df.dropna(subset=['with_calf'])
temp=['False','True']

tempPrime=list(df['with_calf'].value_counts())

labels = temp

sizes = tempPrime

#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Percentage of the herd with Calfs')

plt.show()
#Age groups of animals With Calf:



temp=df[df['with_calf']==1]['life_stage'].value_counts().keys()

tempPrime=df[df['with_calf']==1]['life_stage'].value_counts()



plt.bar(temp,tempPrime)

plt.title('Distribution of age-groups of animals with calfs')

plt.ylabel('#Count')

plt.xlabel('Age groups')

plt.show()
#Age groups of animals Without Calf:



temp=df[df['with_calf']==0]['life_stage'].value_counts().keys()

tempPrime=df[df['with_calf']==0]['life_stage'].value_counts()



plt.bar(temp,tempPrime)

plt.title('Distribution of age-groups of animals without calfs')

plt.ylabel('#Count')

plt.xlabel('Age groups')

plt.show()
tempSum={'Moberly':0, 'Burnt Pine':0, 'Quintette':0, 'Kennedy':0, 'Narraway':0,

       'Scott':0}

for i in df['study_site'].unique():

    

    temp=df[df['study_site']==i]['life_stage'].value_counts().keys()

    tempPrime=df[df['study_site']==i]['life_stage'].value_counts()

    #print(sum(tempPrime))

    tempSum[i]=sum(tempPrime)

    plt.bar(temp,tempPrime)

    plt.title(f'frequency distribution of age groups of animals at \'{i}\'')

    plt.xlabel('Age groups')

    plt.ylabel('Count')

    plt.show()

print(tempSum)
tempName=[]

temp=[]

tempPrime=[]

studySitesLat={'Moberly':[] ,'Burnt Pine':[], 'Quintette':[],'Kennedy':[],'Narraway':[],'Scott':[]}

studySitesLon={'Moberly':[] ,'Burnt Pine':[], 'Quintette':[],'Kennedy':[],'Narraway':[],'Scott':[]}



for i in df['study_site'].unique():

    

    tempLst=df[df['study_site']==i]['deploy_on_latitude']

    studySitesLat[i]=tempLst

    tempLst=[]

    tempLst=df[df['study_site']==i]['deploy_on_longitude']

    studySitesLon[i]=tempLst
studySitesLat
import plotly.graph_objects as go



limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]

colors = ["royalblue","crimson","lightseagreen","orange","lightgrey",'red']

cities = []



fig = go.Figure()

def namesPlot(tempLon,tempLat,tempText,tempCount,tempSum):

    fig.add_trace(go.Scattergeo(

        locationmode = 'country names',

        lon = tempLon,

        lat = tempLat,

        text = [tempText],

        marker = dict(

            size = tempSum[tempText],

            color = colors[count],

            line_color='rgb(128,128,128)',

            line_width=0.5,

            sizemode = 'diameter'

        ),

        name = f'Total count for {tempText}'

        ))

    fig.update_layout(

            title_text = 'Caribou Location tracking',

            showlegend = True,

            geo = dict(

                scope = 'north america',

                landcolor = 'rgb(120, 120, 120)',

            )

        )

    

    if tempCount==4:

        fig.show()
for count,i in enumerate(studySitesLat.keys()):

    tempLst=list(zip(studySitesLat[i],studySitesLon[i]))

    tempLat=[]

    tempLon=[]

    [tempLat.append(j[0]) for j in tempLst]

    [tempLon.append(j[1]) for j in tempLst]

    namesPlot(tempLon,tempLat,i,count,tempSum)