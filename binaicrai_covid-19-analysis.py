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



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline
geotab_borderwaittime = pd.read_csv('/kaggle/input/uncover/UNCOVER/geotab/border-wait-times-at-us-canada-border.csv')

geotab_airporttraffic = pd.read_csv('/kaggle/input/uncover/UNCOVER/geotab/airport-traffic-analysis.csv')

print('Geotab Border Wait Time - geotab_borderwaittime')

display(geotab_borderwaittime.head())

print('Geotab Airport Traffic - geotab_airporttraffic')

display(geotab_airporttraffic.head())
for i in geotab_borderwaittime.columns:

    if geotab_borderwaittime[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if geotab_borderwaittime[i].nunique() == geotab_borderwaittime.shape[0]:

        print('With all unique value: ', i)
# Dropping 'version' and 'borderid' as port detail retained

geotab_borderwaittime = geotab_borderwaittime.drop(['borderid', 'version'], axis=1)
for i in geotab_airporttraffic.columns:

    if geotab_airporttraffic[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if geotab_airporttraffic[i].nunique() == geotab_airporttraffic.shape[0]:

        print('With all unique value: ', i)
# Dropping 'aggregationmethod', 'version'

geotab_airporttraffic = geotab_airporttraffic.drop(['aggregationmethod', 'version'], axis=1)
sns.catplot('tripdirection', data= geotab_borderwaittime, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_borderwaittime['tripdirection'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of tripdirection', fontsize = 20, color = 'black')

plt.show()
sns.set(font_scale = 2)

p1 = sns.catplot('canadaport', data= geotab_borderwaittime, kind='count',

                 order = geotab_borderwaittime['canadaport'].value_counts().index, alpha=1, height=6, aspect= 9)

p1.set_xticklabels(rotation=90)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_borderwaittime['canadaport'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height()/1.5,'%d' % int(p.get_height()),

            fontsize=20, color='blue', ha='center', va='bottom')





plt.title('Frequency plot of canadaport', fontsize = 50, color = 'black')

plt.show()
sns.set(font_scale = 2)

p1 = sns.catplot('americaport', data= geotab_borderwaittime, kind='count',

                 order = geotab_borderwaittime['americaport'].value_counts().index, alpha=1, height=6, aspect= 9)

p1.set_xticklabels(rotation=90)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_borderwaittime['americaport'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=20, color='blue', ha='center', va='bottom')





plt.title('Frequency plot of americaport', fontsize = 50, color = 'black')

plt.show()
l = geotab_borderwaittime.canadaport.unique()

c = []

for i in l:

    c.append(geotab_borderwaittime[geotab_borderwaittime.canadaport == i].americaport.unique())
m = geotab_borderwaittime.americaport.unique()

a = []

for i in m:

    a.append(geotab_borderwaittime[geotab_borderwaittime.americaport == i].canadaport.unique())
pd.set_option('display.max_rows', 65)

df = pd.DataFrame(c, columns = ['canadaport'])

df['americaport'] = pd.DataFrame(a)

df
geotab_borderwaittime['tripdirection'] = geotab_borderwaittime['tripdirection'].replace({'US to Canada': 'U->C','Canada to US':'C->U'})
geotab_borderwaittime['Trip_Details'] = geotab_borderwaittime['tripdirection'] + ' '+geotab_borderwaittime['canadaport'] +' '+ geotab_borderwaittime['americaport'] 
sns.set(font_scale = 2)

fig, ax = plt.subplots(figsize=(30,10))

geotab_borderwaittime.groupby(['averageduration']).count()['Trip_Details'].plot(ax=ax)
tripdetails_array = geotab_borderwaittime.Trip_Details.unique()

tripdetails_array
sns.set(font_scale = 2)

fig, ax = plt.subplots(figsize=(30,10))

geotab_borderwaittime.groupby(['Trip_Details']).sum()['averageduration'].plot(ax=ax)

ax.tick_params(axis='x', rotation=90, labelsize = 10)

ax.tick_params(axis='y', labelsize = 10)



ax.set_xlabel('Trip_Details',fontsize=20, fontweight='bold')

ax.set_ylabel('Sum', fontsize=20, fontweight='bold')



N = geotab_borderwaittime.Trip_Details.nunique()

ind = np.arange(N)

plt.xticks(ind, (tripdetails_array))



plt.title('Sum of averageduration by Trip_Details', fontsize = 20, color = 'black')

plt.show
ld = geotab_borderwaittime[['localdate']]

ld.sort_values(by=['localdate'], inplace=True)



sns.set(font_scale = 4)

p1 = sns.catplot('localdate', data= ld, kind='count', 

                  alpha=1, height=35, aspect= 9)

p1.set_xticklabels(rotation=90)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = ld['localdate'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=60, color='blue', ha='center', va='bottom')



ax.set_xlabel('State',fontsize=100, fontweight='bold')

ax.set_ylabel('Count', fontsize=100, fontweight='bold')



ax.tick_params(axis='x', rotation = 90, labelsize = 100)

ax.tick_params(axis='y', labelsize = 100) 



plt.title('Frequency plot of localdate', fontsize = 200, color = 'black')

plt.show()
sns.catplot('daytype', data= geotab_borderwaittime, kind='count', alpha=0.7, height=8, aspect= 2)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_borderwaittime['daytype'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/6., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')



ax.tick_params(axis='x', labelsize = 12)

ax.tick_params(axis='y', labelsize = 12)   





ax.set_xlabel('State',fontsize=15, fontweight='bold')

ax.set_ylabel('Count', fontsize=15, fontweight='bold')





plt.title('Frequency plot of daytype', fontsize = 20, color = 'black')

plt.show()
lat_list = geotab_borderwaittime.borderlatitude.tolist()

lon_list = geotab_borderwaittime.borderlongitude.tolist()
from mpl_toolkits.basemap import Basemap



fig = plt.figure(figsize=(20,15))



m = Basemap(projection = 'mill', llcrnrlat = -90, urcrnrlat = 90, llcrnrlon = -180, urcrnrlon = 180, resolution  = 'c')



m.drawcoastlines()



m.drawparallels(np.arange(-90,90,10), labels = [True,False,False,False], fontsize = 12)

m.drawmeridians(np.arange(-180,180,30), labels = [0,0,1], fontsize = 12)



m.scatter(lon_list, lat_list, latlon = True, s = 100, c = 'red', marker = "P", alpha = 0.5, edgecolor = 'k', linewidth = 2)



plt.title('Basemap', fontsize = 20)

plt.show()
geotab_airporttraffic.head()
sns.catplot('airportname', data= geotab_airporttraffic, kind='count', alpha=0.7, height=10, aspect= 8)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_airporttraffic['airportname'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/6., p.get_height()/2.,'%d' % int(p.get_height()),

            fontsize=40, color='blue', ha='center', va='bottom')



ax.tick_params(axis='x', rotation = 90, labelsize = 40)

ax.tick_params(axis='y', labelsize = 40) 



ax.set_xlabel('airportname',fontsize=40, fontweight='bold')

ax.set_ylabel('count', fontsize=40, fontweight='bold')



plt.title('Frequency plot of airportname', fontsize = 60, color = 'black')

plt.show()
sns.catplot('percentofbaseline', data= geotab_airporttraffic, kind='count', 

            order = geotab_airporttraffic['percentofbaseline'].value_counts().index, alpha=0.7, height=15, aspect= 7)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_airporttraffic['percentofbaseline'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/6., p.get_height()/2.,'%d' % int(p.get_height()),

            fontsize=50, color='blue', ha='center', va='bottom')



ax.tick_params(axis='x', rotation = 90, labelsize = 45)

ax.tick_params(axis='y', labelsize = 45) 



ax.set_xlabel('percentofbaseline',fontsize=45, fontweight='bold')

ax.set_ylabel('count', fontsize=45, fontweight='bold')



plt.title('Frequency plot of percentofbaseline', fontsize = 100, color = 'black')

plt.show()
airport_array = geotab_airporttraffic.airportname.unique()
sns.set(font_scale = 2)

fig, ax = plt.subplots(figsize=(20,5))

geotab_airporttraffic.groupby(['airportname']).mean()['percentofbaseline'].plot(ax=ax)

ax.tick_params(axis='x', rotation=90, labelsize = 10)

ax.tick_params(axis='y', labelsize = 10)



ax.set_xlabel('airportname',fontsize=20, fontweight='bold')

ax.set_ylabel('mean', fontsize=20, fontweight='bold')



N = geotab_airporttraffic.airportname.nunique()

ind = np.arange(N)

plt.xticks(ind, (airport_array))



plt.title('Mean percentofbaseline by airportname', fontsize = 20, color = 'black')

plt.show
sns.catplot('state', data= geotab_airporttraffic, kind='count', 

            order = geotab_airporttraffic['state'].value_counts().index, alpha=0.7, height=18, aspect= 4)





# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = geotab_airporttraffic['state'].value_counts().max() 



# Iterate through the list of axes' patches



for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/6., p.get_height()/1.2,'%d' % int(p.get_height()),

            fontsize=60, color='blue', ha='center', va='bottom')



ax.set_xlabel('State',fontsize=40, fontweight='bold')

ax.set_ylabel('Count', fontsize=40, fontweight='bold')



ax.tick_params(axis='x', rotation = 90, labelsize = 40)

ax.tick_params(axis='y', labelsize = 40)    

plt.title('Frequency plot of State', fontsize = 100, color = 'black')

plt.show()