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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')
data=pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

data.head()
#removeing unwanted columns

data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)

data.head()
#converting to datetime obj

data['DateTime']=pd.to_datetime(data['Datum'])

#Dropping the datum column

data.drop(['Datum'],axis=1,inplace=True)

data.head(5)
data['Year']=data['DateTime'].apply(lambda x: x.year)

data.head()
data.drop(['DateTime'],axis=1,inplace=True)
data.head()
data['Country']=data['Location'].apply(lambda x: x.split(', ')[-1])

data.head()
data.drop(['Location'],axis=1,inplace=True)
data.head()
sns.heatmap(data.isnull(),yticklabels=False)   
data.drop([' Rocket','Detail'],axis='columns',inplace=True)

data.head()
data.head()
def show_values_on_bars(axs, h_v="v", space=0.4):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center") 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height()

                value = int(p.get_width())

                ax.text(_x, _y, value, ha="left")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
fig=plt.figure(figsize=(9,16))

axes=fig.add_axes([0,0,1,1])

sns.countplot(y='Company Name',data=data,order=data['Company Name'].value_counts().index)

show_values_on_bars(axes, "h", 0.3)

axes.set_title('Company vs Launches',fontsize=20)

axes.set_xlabel('Launches',fontsize=16)

axes.set_ylabel('Company Name',fontsize=16)

fig=plt.figure(figsize=(9,16))

axes=fig.add_axes([0,0,1,1])

sns.countplot(y='Country',data=data,order=data['Country'].value_counts().index)

show_values_on_bars(axes, "h", 0.3)

axes.set_title('Country v/s Launches',fontsize=20)

axes.set_xlabel('Launches',fontsize=16)

axes.set_ylabel('Country',fontsize=16)

fig=plt.figure(figsize=(5,5))

axes=fig.add_axes([0,0,1,1])

sns.countplot(x='Status Rocket',data=data,order=data['Status Rocket'].value_counts().index)

axes.set_title('Rocket Status',fontsize=20)

axes.set_xlabel('Status',fontsize=16)

axes.set_ylabel('Count',fontsize=16)

show_values_on_bars(axes, "v", 0.3)

fig=plt.figure(figsize=(5,5))

axes=fig.add_axes([0,0,1,1])

sns.countplot(x='Status Mission',data=data,order=data['Status Mission'].value_counts().index)

axes.set_title('Mission Status',fontsize=20)

axes.set_xlabel('Status',fontsize=16)

axes.set_ylabel('Count',fontsize=16)

show_values_on_bars(axes, "v", 0.3)

fig=plt.figure(figsize=(20,20))

axes=fig.add_axes([0,0,1,1])

sns.countplot(y='Year',data=data)

axes.set_title('Launch by Year',fontsize=20)

axes.set_xlabel('Launches',fontsize=16)

axes.set_ylabel('Year',fontsize=16)

show_values_on_bars(axes, "h", 0.3)