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
import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

import folium
dataset = pd.read_csv("../input/crimeanalysis/crime_by_state_rt.csv")
dataset['STATE/UT'] = dataset['STATE/UT'].str.title()

dataset["STATE/UT"].unique()
# numerical cols

cols = ['Murder', 'Assault on women', 'Kidnapping and Abduction', 'Dacoity', 

        'Robbery', 'Arson', 'Hurt', 'Prevention of atrocities (POA) Act', 

        'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']
# import district level shape files

dist_gdataset = gpd.read_file('../input/india-district-wise-shape-files/output.shp')



# group by state

states_gdataset = dist_gdataset.dissolve(by='statename').reset_index() 



# just select statename and geometry column

states_gdataset = states_gdataset[['statename', 'geometry']]
# replace state's name

states_gdataset['statename'] = states_gdataset['statename'].replace('Ladakh', 'Jammu & Kashmir')

states_gdataset['statename'] = states_gdataset['statename'].replace('Telangana', 'Andhra Pradesh')

states_gdataset['statename'] = states_gdataset['statename'].replace('Andaman & Nicobar Islands', 'A & N Islands')

states_gdataset['statename'] = states_gdataset['statename'].replace('Chhatisgarh', 'Chhattisgarh')

states_gdataset['statename'] = states_gdataset['statename'].replace('Dadra & Nagar Haveli', 'D & N Haveli')

states_gdataset['statename'] = states_gdataset['statename'].replace('Orissa', 'Odisha')

states_gdataset['statename'] = states_gdataset['statename'].replace('Pondicherry', 'Puducherry')

states_gdataset['statename'] = states_gdataset['statename'].replace('NCT of Delhi', 'Delhi')



# group 10 years of data

states_dataset = dataset.groupby('STATE/UT')[cols].sum().reset_index()

states_dataset.head()
# merge shape file with count file

states_full = pd.merge(states_gdataset, states_dataset, left_on='statename', right_on='STATE/UT', how='left')

states_full.head()
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

fig.suptitle('No. of Hate crimes from 2001-2012', fontsize=16)

cmap = 'bwr'



states_full.plot(column='Murder', ax=axes[0,0], cmap=cmap)   

axes[0,0].set_title('Murder')

axes[0,0].set_axis_off()                                       



states_full.plot(column='Assault on women', ax=axes[0,1], cmap=cmap)   

axes[0,1].set_title('Assault on women')

axes[0,1].set_axis_off()          



states_full.plot(column='Kidnapping and Abduction', ax=axes[0,2], cmap=cmap)   

axes[0,2].set_title('Kidnapping and Abduction')

axes[0,2].set_axis_off()          



states_full.plot(column='Dacoity', ax=axes[0, 3], cmap=cmap)   

axes[0, 3].set_title('Dacoity')

axes[0, 3].set_axis_off()          



states_full.plot(column='Robbery', ax=axes[1,0], cmap=cmap)   

axes[1,0].set_title('Robbery')

axes[1,0].set_axis_off()          



states_full.plot(column='Arson', ax=axes[1,1], cmap=cmap)   

axes[1,1].set_title('Arson')

axes[1,1].set_axis_off()    



states_full.plot(column='Hurt', ax=axes[1,2], cmap=cmap)   

axes[1,2].set_title('Hurt')

axes[1,2].set_axis_off()          



states_full.plot(column='Prevention of atrocities (POA) Act', ax=axes[1,3], cmap=cmap)   

axes[1,3].set_title('Prevention of atrocities (POA) Act')

axes[1,3].set_axis_off()          



states_full.plot(column='Protection of Civil Rights (PCR) Act', ax=axes[2,0], cmap=cmap)   

axes[2,0].set_title('Protection of Civil Rights (PCR) Act')

axes[2,0].set_axis_off()  



states_full.plot(column='Other Crimes Against SCs', ax=axes[2,1], cmap=cmap)   

axes[2,1].set_title('Other Crimes Against SCs')

axes[2,1].set_axis_off()  



axes[2,2].set_axis_off()  



axes[2,3].set_axis_off()  



plt.show()