# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/who-dataset/WHO.csv')
df.head()
df['Population']=df['Population'].apply(lambda x : x*1000)
df.info()
df.describe()
import matplotlib.pyplot as plt 

import seaborn as sns
for i in df:

    if df[i].dtypes !=object:

        print('\n')

        print('top '+i+' ++++++++++++++++')

        print(df.nlargest(10, i)[['Country',i]])
plt.figure(figsize=(20,20))

p=df.sort_values('Population')

plt.pie(p['Population'].values,labels=p['Country'].values,autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
from sklearn.preprocessing import minmax_scale
d=pd.DataFrame()

for i in df :

    if i not in ['Region','Country']:

        d[i]=minmax_scale(df[i].values)

    else:

        d[i]=df['Country'].values

d.drop('Region',inplace=True,axis=1)

d.set_index('Country',inplace=True)

d.fillna(0,inplace=True)    

plt.figure(figsize=(20,20))

sns.heatmap(d)
corr=df.corr()

plt.figure(figsize=(12,9))

sns.heatmap(corr,cmap="YlGnBu")
import pycountry
for i in df['Country'].values:

    if pycountry.countries.get(name=i) == None:

        print(i)
not_con='''

Bolivia (Plurinational State of)

BOL

Cape Verde

CPV

Ivory Coast

CIV

Czech Republic

CZE

Democratic People's Republic of Korea

PRK

Democratic Republic of the Congo

COD

Iran (Islamic Republic of)

IRN

Micronesia (Federated States of)

FSM

Republic of Korea

KOR

Republic of Moldova

MDA 

Swaziland

SWZ

The former Yugoslav Republic of Macedonia

MKD

United Republic of Tanzania

TZA

United States of America

USA

Venezuela (Bolivarian Republic of)

VEN

'''
def co_code(x,not_con):

    co = pycountry.countries.get(name=x)

    if co != None:

         return co.alpha_3

    else:

        for c in range(len(not_con)):

            if not_con[c]==x :

                return not_con[c+1]

    print(x)

    return np.nan

    

df['code']=df['Country'].apply(lambda x:co_code(x,not_con.split('\n')))

#dd=df[pd.notnull(df['code'])]

df.head()
import geopandas
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.head()
len(world['name'])
world.plot('gdp_md_est')
w=world
w=w.merge(df,left_on='iso_a3', right_on='code')
w.head()
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 1,figsize=(20,15))



divider = make_axes_locatable(ax)

cax = divider.append_axes("bottom", size="5%", pad=0.1)



ax.axes.get_xaxis().set_visible(False)

ax.axes.get_yaxis().set_visible(False)



w.plot(column='FertilityRate', ax=ax, legend=True, cax=cax,

      legend_kwds={'label': "FertilityRate",'orientation': "horizontal" })

plt.title('', fontsize=16)
def create_map(w,conti,col,color,shape=(25,23)):

    if type(conti)== list:

        data = w.query("continent  in {}".format(str(conti)))

    elif conti=='all':

        data=w

        conti='The world'

    else:

        data=w[w['continent']==conti]

    #data.dropna(col,inplace=True)

    #data[pd.notnull(data[col])]

    fig, ax = plt.subplots(1, 1,figsize=shape)



    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="5%", pad=0.1)



    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    

    data.plot(color="grey",ax=ax,cax=cax)

    data[np.isfinite(data[col])].plot(column=col, ax=ax, legend=True, cax=cax,cmap=color,

          legend_kwds={'label': col ,'orientation': "horizontal" })

   



    tt='This figure show {} for each contrey in {}'.format(col,conti)

    plt.title(tt, fontsize=16,pad=20)

    plt.show()
create_map(w,'Asia','LifeExpectancy','Blues')

create_map(w,'Asia','Over60','Blues')
create_map(w,'all','GNI','BuGn')
#LiteracyRate

create_map(w,'all','LiteracyRate','YlOrRd')

#PrimarySchoolEnrollmentFemale

create_map(w,'Africa','PrimarySchoolEnrollmentFemale','viridis')
#europe

create_map(w,'Europe','Population','BuGn',(25,20))
#CellularSubscribers

create_map(w,['Asia','Africa'],'CellularSubscribers','plasma')