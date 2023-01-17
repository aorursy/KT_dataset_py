# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
credits = pd.read_csv("/kaggle/input/the-movies-dataset/credits.csv")

meta = pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv",usecols=['revenue','release_date','id','budget'])

meta = meta[~meta.id.str.contains("-")]

meta['id'] = meta['id'].astype(int)

meta.drop_duplicates(keep='first', inplace=True)
from ast import literal_eval

credits['crew'].fillna('[]', inplace=True)

credits['cast'].fillna('[]', inplace=True)

credits['crew']=credits['crew'].apply(literal_eval)

credits['cast']=credits['cast'].apply(literal_eval)
def get_actor(namedict):

    result={}

    try:

        result['actor_1_name']=[x['name'] for x in namedict if x['order']==0][0]

        result['actor_2_name']=[x['name'] for x in namedict if x['order']==1][0]

        result['actor_3_name']=[x['name'] for x in namedict if x['order']==2][0]

    except:

        name=np.nan

    return result

        

def get_director(namedict):

    try:

        name=[x['name'] for x in namedict if x['job']=='Director'][0]

    except:

        name=np.nan

    return name



def get_gender(namedict):

    result={}

    try:

        result['actor_1_gender']=[x['gender'] for x in namedict if x['order']==0][0]

        result['actor_2_gender']=[x['gender'] for x in namedict if x['order']==1][0]

        result['actor_3_gender']=[x['gender'] for x in namedict if x['order']==2][0]

    except:

        gender=np.nan

    return result







#Creating two additional  columns for director and actor    

credits['director']=credits['crew'].apply(get_director)



#creating the 3 main actors for the movie

credits['actor_1_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_1_name',''))

credits['actor_2_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_2_name',''))

credits['actor_3_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_3_name',''))

credits['actor_1_gender']   =credits['cast'].apply(get_gender).apply(lambda x :x.get('actor_1_gender',''))

credits['actor_2_gender']   =credits['cast'].apply(get_gender).apply(lambda x :x.get('actor_2_gender',''))

credits['actor_3_gender']   =credits['cast'].apply(get_gender).apply(lambda x :x.get('actor_3_gender',''))





#Dropping cast and cast and crew columns as no longer needed

credits.drop(['cast','crew'], inplace=True, axis=1)

credits.drop_duplicates(keep='first', inplace=True)
credits.head()
credits.set_index('id',inplace=True)

meta.set_index('id',inplace=True)

df_1 = credits.merge(meta,on='id',how='left')
df_1 = df_1[df_1.actor_1_gender.isin([1,2])]
df_1.describe(include='all')
gmap = {0:'Unknown',1:'Female',2:'Male'}

df_1['Gender_STR'] = df_1['actor_1_gender'].map(gmap)

df_1['release_date_f'] = pd.to_datetime(df_1['release_date'])

df_1['Year'] = df_1['release_date_f'].apply(lambda x: x.year)

recent = df_1.loc[df_1['Year'] >= 1980]
temp = recent.groupby("Gender_STR")['actor_1_gender'].count()

temp_sum=temp.sum()



size_pct =round(temp.apply(lambda x: 100*x/temp_sum),2)

size_pct
# create data

names=['Female','Male']

size=size_pct

 

# Create a circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

# Give color names

plt.pie(size, labels=names, colors=['red','green','blue','skyblue'])

p=plt.gcf()

p.gca().add_artist(my_circle)

 

from palettable.colorbrewer.qualitative import Pastel1_7

plt.pie(size, labels=names, colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(20,6))

sns.set_context('notebook')

sns.countplot(data=recent,x='Year',hue='Gender_STR')

plt.xticks(rotation=30)
pivot_1 = recent.pivot_table(values='director', index='Year', columns='Gender_STR', aggfunc='count', fill_value=0,dropna=False)

pivot_1['Perc_Female'] = pivot_1['Female']/(pivot_1['Female']+pivot_1['Male'])

pivot_1['Perc_Male'] = pivot_1['Male']/(pivot_1['Female']+pivot_1['Male'])



pivot_1 = pivot_1.reset_index()
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,PercentFormatter)





fig, ax = plt.subplots(figsize=(20,6)) 

p1 = plt.bar(pivot_1['Year'], pivot_1['Perc_Male'])

p2 = plt.bar(pivot_1['Year'], pivot_1['Perc_Female'],bottom=pivot_1['Perc_Male'])



plt.ylabel('Percentage Of Total Split')

plt.title('Gender of lead roles in movies over time')

plt.legend((p1[0], p2[0]), ('Male', 'Female'), loc='lower center',bbox_to_anchor=(0.5, -0.25))



ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%'))
fig, ax = plt.subplots(figsize=(20,8)) 



df_2 = pivot_1.loc[(pivot_1['Year'] >= 2000)&(pivot_1['Year'] <= 2017),:]



# Plot the observed data

plt.plot(df_2['Year'], df_2['Perc_Female'], marker="x", label='Observed',linewidth=0.1)

# Fit a model and predict future dates

predict_dates = range(2000,2025,1)



model = np.polyfit(df_2['Year'], df_2['Perc_Female'], 1)

model_exp = np.polyfit(df_2['Year'], df_2['Perc_Female'], 2)

predicted = np.polyval(model, predict_dates)

predicted_exp = np.polyval(model_exp, predict_dates)



# Plot the model

plt.plot(predict_dates, predicted, lw=2,ls='--',label='Predicted Linear')

plt.plot(predict_dates, predicted_exp, lw=2,ls='--',label='Predicted Expoential')

#plt.plot(model)







plt.ylabel('Percentage Lead Females')

plt.title('Gender of female lead roles in movies over time')

ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%'))

plt.legend()

plt.ylim(bottom=0,top=0.5)