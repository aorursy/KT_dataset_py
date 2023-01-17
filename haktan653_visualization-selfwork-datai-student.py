# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data = data.rename(columns={'Preferred Foot': 'Preferred_Foot'})
data['Preferred_Foot']
overallRating = data['Overall']

potentialRating = data['Potential']

nation = data['Nationality']

age = data['Age']

marketingValue = data['Value']

currentWage=data['Wage']

whichFoot = data['Preferred_Foot']
whichFoot.head()
data.Age.value_counts()

#testing for unusable datas to replacing later.
nation_list = list(data['Nationality'].unique())

data.Overall = data.Overall.astype(float)

area_poverty_ratio = []

for i in nation_list:

    x = data[data['Nationality']==i]

    nation_rate = sum(x.Overall)

    area_poverty_ratio.append(nation_rate)

newData = pd.DataFrame({'nation_list' : nation_list, 'area_poverty_ratio':area_poverty_ratio})

new_index=(newData['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = newData.reindex(new_index)



#Visualization Part

plt.figure(figsize=(45,25))

sns.barplot(x=sorted_data['nation_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Nation Rate')

plt.title('Nation Rate Given States')

    

