# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mg = pd.read_csv('../input/missing-migrants-project/MissingMigrants-Global-2019-12-31_correct.csv')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime

mg.head(3)
mg.shape
mg.info()
mg['Region of Incident'].value_counts().plot.bar()
pd.pivot_table(mg,values='Total Dead and Missing',index='Reported Date',aggfunc=np.sum).sort_index().head()

# Cannot sort as the date is a string and do not like to see in alphabetical order

# Will Create a date field
# Found a good article here https://www.datacamp.com/community/tutorials/converting-strings-datetime-objects
#Get one scalar to learn the new function

date_s = mg.iloc[1,2]

date_s
date_test = datetime.strptime(date_s,'%B %d, %Y')

print(date_test)



#now create a new proper datetime column

mg['Reported Date dt']= mg['Reported Date'].apply(lambda x: datetime.strptime(x,'%B %d, %Y'))
pd.pivot_table(mg,values='Total Dead and Missing',index='Reported Date dt',aggfunc=np.sum).sort_index().plot()
pd.pivot_table(mg,values='Total Dead and Missing',index='Reported Year',aggfunc=np.sum).sort_index().plot(kind='bar')
# Visualise a table per year with number of dead

pd.pivot_table(mg,values='Total Dead and Missing',index='Reported Year',aggfunc=np.sum).sort_index()
print('Maximum date is : {}'.format(mg['Reported Date dt'].max()))

# last reported date was end of 1st Quarted 2019

print('Minimum date is : {}'.format(mg['Reported Date dt'].min()))


pd.pivot_table(mg,values=['Number of Males','Number of Females','Number of Children'],

               index='Reported Year',

               aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum}).plot(kind='bar')



# Create a column that sum Total Male and Female

mg['Total MFC']= mg['Number of Males']+mg['Number of Females']+mg['Number of Children']
# filter the data frame that meet the criteria and check how many records has with correctly reporting Male and Female

mg[mg['Total MFC']==mg['Total Dead and Missing']].shape[0]

# Create a dataframe with only the observations with correct report of Male,Female and Children

mg_reportMFC = mg[mg['Total MFC']==mg['Total Dead and Missing']]

datasets=[mg,mg_reportMFC]



for data in datasets:

    pd.pivot_table(data,values=['Number of Males','Number of Females','Number of Children'],

                   index='Reported Year',

                   aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum}).plot(kind='bar')

  
pd.pivot_table(mg,values=['Number of Males','Number of Females','Number of Children'],

               index='Reported Year',

               aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum})





# transform into tidy format using melt



tidy =pd.pivot_table(mg_reportMFC,values=['Number of Males','Number of Females','Number of Children'],

               index='Reported Year',

               aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum})



tidy['year']=tidy.index

pd.melt(tidy,id_vars=['year'])
# split on ',' coma and expand then rename columns

lat_lon = mg['Location Coordinates'].str.split(',',expand=True).rename(index=int, columns={0: "lat", 1: "lon"})
#Concat expanded columns

mg =pd.concat([mg,lat_lon],axis=1)
mg.head()
# Will attemp use folim to project this in a map

# here is a good page https://alysivji.github.io/getting-started-with-folium.html

# Folium package looks very good



import folium



#wow it is in Kaggle



mg.info()

# cast lat lon as float

mg['lat'] = mg['lat'].astype(float)

mg['lon'] = mg['lon'].astype(float)

max_lat =mg['lat'].max()

max_lon =mg['lon'].max()
# Getting a random sample of 3000 records to get an idea where in the world migrants are missing



m = folium.Map([max_lat, max_lon], zoom_start=3)



for point in mg.loc[:,['lat','lon']].dropna().sample(frac=1).values.tolist()[:3000]:

    folium.Circle(

        radius=100,

        location=point,

        color='blue',

        fill=True,

    ).add_to(m)

    

m        