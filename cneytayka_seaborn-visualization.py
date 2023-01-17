# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

import warnings

warnings.filterwarnings('ignore') 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
shot=pd.read_csv("/kaggle/input/police-shootings/database.csv")
shot.head()
shot.info()
shot.describe()
shot.threat_level.unique()
shot.head()
#shot.state.value_counts()
shot.info()
#shot.city.value_counts()
#state(eyalet) göre ölen insan sayısı





area_list=list(shot["state"].unique())



area_state=[]



for i in area_list:

    

    x=shot[shot["state"]==i]

    area_state_rate=sum(x.age)/len(x)

    area_state.append(area_state_rate)

    

#sorting

data=pd.DataFrame({"area_list":area_list,"area_state_ratio":area_state})

new_index=(data["area_state_ratio"].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)



#visualization



plt.figure(figsize=(20,15))



sns.barplot(x=sorted_data["area_list"],y=sorted_data["area_state_ratio"])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
#En tehlikeli 10 Şehir.

city=shot.city.value_counts()

plt.figure(figsize=(10,5))



sns.barplot(x=city[:10].index,y=city[:10].values)

plt.xticks(rotation=45)

plt.title("Most dangerous cities",color="red",fontsize=10)
shot.head()
race = shot.race.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=race[:20].values,y=race[:20].index)

plt.title('Most dangerous race',color = 'blue',fontsize=15)
sns.boxplot(x="race",y="age",data=shot)


#ırklara göre ölüm oranları



shot.race.dropna(inplace=True)



labels=shot.race.value_counts().index



colors = ['purple','blue','red','yellow','green','brown']



explode = [0,0,0,0,0,0]



sizes = shot.race.value_counts().values



plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
sns.countplot(data=shot,x="manner_of_death")

plt.title("manner_of_death",color = 'blue',fontsize=20)
shot.head()
# age of killed people

a =['Old' if i >= 45 else 'Young' for i in shot.age]

df = pd.DataFrame({'age':a})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)