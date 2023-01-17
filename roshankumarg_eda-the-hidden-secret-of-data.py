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

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv")

data.shape 
data.head(3).append(data.tail(3))

data['ID'].nunique() 

import missingno as msno

print(data.isnull().sum())



p = msno.bar(data, figsize = (9,6))
data.info()

data['Number_Weeks_Used'].fillna(method = 'ffill', inplace = True)

data['Number_Weeks_Used'] = data['Number_Weeks_Used'].astype('int64')



col = data.columns.tolist()

col.remove('ID')

data[col].describe(percentiles = [.25,.5,.75,.95,.97,.99])  
data[(data['Season'] == 1) & (data['Crop_Damage'] == 1) & (data['Soil_Type'] == 0)].head() 

pd.DataFrame(data.groupby(['Crop_Damage','Crop_Type'])['Pesticide_Use_Category'].count())

pd.DataFrame(data.groupby(['Crop_Damage','Season','Crop_Type'])['Estimated_Insects_Count'].count())



df = pd.DataFrame( data[data['Crop_Damage'] == 1 ].mean(), columns = ['Values'])

df[ 'Variance'] = pd.DataFrame( data[data['Crop_Damage'] == 1 ].var())

df[ 'Standard deviation'] = pd.DataFrame( data[data['Crop_Damage'] == 1 ].std())

df[ 'Median'] = pd.DataFrame( data[data['Crop_Damage'] == 1 ].median())

df
plt.subplot(1,2,1)

sns.countplot(x = 'Crop_Damage' , palette= 'cool', data= data) 

plt.title("Count plot of Crop damage (target variable)")



plt.subplot(1,2,2)

count = data['Crop_Damage'].value_counts()

count.plot.pie(autopct = '%1.1f%%',colors=['green','orange','blue'], figsize = (10,7),explode = [0,0.1,0.1],title = "Pie chart of Percentage of Crop_Damage")
plt.figure(figsize = (10,6))

plt.subplot(1,2,1)

sns.countplot(x = 'Crop_Type' , palette= 'cool', data= data) 

plt.title("Count plot of Crop_Type")



plt.subplot(1,2,2)

sns.countplot(data['Crop_Type'], hue = data['Crop_Damage'],palette="rocket_r")

plt.title("Plot of crop damage Vs Crop type")
data[col].hist(figsize=(10,15),color = 'green')

sns.distplot(data['Estimated_Insects_Count'], kde = True, hist = True, bins= 30)

plt.title("Density plot of Estimated_Insects_Count")


plt.figure(figsize = (15,5))

sns.countplot(data['Number_Weeks_Used'], palette = 'hsv')

plt.title('Count of Number_Weeks_Used')

plt.show() 

sns.countplot(data['Number_Doses_Week'], palette = 'hsv')

plt.title('Count of Number_Doses_Week')

plt.show() 


sns.countplot(data['Pesticide_Use_Category'], palette = 'dark')

plt.title("Count plot of Pesticide_Use_Category")

plt.show()

sns.catplot(x = 'Pesticide_Use_Category', y = 'Estimated_Insects_Count', kind = 'box', data = data, hue = 'Crop_Damage', palette= 'rocket_r')

plt.title("Box plot of Pesticide_Use_Category")



plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

sns.countplot(data['Season'], palette = 'hsv')

plt.title('Count plot of Season')

plt.subplot(1,2,2)

sns.countplot(data['Season'], hue = data['Crop_Damage'], palette = 'hsv')

plt.title('Count plot of Crop_Damage in Seasons')

plt.show() 
import plotly.express as px



fig = px.sunburst(data, path=[ 'Season','Crop_Type'], title="Crop type in various seasons")



fig.show()


sns.countplot(data['Season'], hue = data['Crop_Type'])

plt.title('Count plot of Crop_type in Seasons')