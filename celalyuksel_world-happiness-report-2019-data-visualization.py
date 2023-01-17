import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data2019 = pd.read_csv('../input/world-happiness/2019.csv')
plt.figure(figsize= (15,10))

sns.barplot(x= data2019['Country or region'].head(10), y= data2019['Score'].head(10))

plt.show()
sns.lmplot(x="GDP per capita", y="Score", data=data2019, size= 5)

plt.show()
sns.jointplot("Social support", "Healthy life expectancy", data=data2019, size = 5 , ratio=3, color="r")

plt.show()
f,ax1 = plt.subplots(figsize=(30,10))



sns.pointplot(x= 'Country or region', y= 'Freedom to make life choices', data=data2019, color='red',alpha=0.8)

sns.pointplot(x= 'Country or region', y= 'Perceptions of corruption',data=data2019, color='blue', alpha= 0.8)



plt.text(110,0.64,'Freedom to make life choices', color= 'red', fontsize= 18, style='italic')

plt.text(110,0.61, 'Perceptions of corruption', color='blue', fontsize=18, style= 'italic')

plt.xlabel('Regions',fontsize=20,color='lime')

plt.xticks(rotation=90,size=10)

plt.xticks

plt.ylabel('Values',fontsize=20, color='lime')

plt.yticks(rotation=90)

plt.title('Freedom to make life choices VS Perceptions of corruption', fontsize=25, color='lime')

plt.grid()
sns.kdeplot(data2019.Generosity, data2019['GDP per capita'], shade= True, cut=1, color= 'purple')

plt.show()
region_list = list(data2019['Country or region'])

Healthy_life_expectancy_list = list(data2019['Healthy life expectancy'])

Social_support_list = list(data2019['Social support'])

Freedom_to_make_life_choices_list = list(data2019['Freedom to make life choices'])
data2019_new_dataframe = pd.DataFrame({'region_list': region_list,'Social_support_list': Social_support_list, 'Freedom_to_make_life_choices_list': Freedom_to_make_life_choices_list, 'Healthy_life_expectancy_list': Healthy_life_expectancy_list})
f,ax = plt.subplots(figsize= (10,10))

sns.heatmap(data2019_new_dataframe.corr(), annot=True,linewidths=0.5, linecolor='red', fmt= '.1f',ax=ax)

plt.show()