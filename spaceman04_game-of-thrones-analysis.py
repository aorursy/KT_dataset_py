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
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing datasets 
battles = pd.read_csv("../input/battles.csv")
deaths = pd.read_csv("../input/character-deaths.csv")
battles.describe()
battles = battles.drop(['defender_3','defender_4'],axis = 1)
deaths.describe()

#Checking the How many battles caused in which year
plt.subplot(2,1,1)
sns.countplot(battles['year'])
plt.title('How many battles caused in whuch year')
plt.show()

#Checking the maximum death caused in whuch year
plt.subplot(2,1,2)
sns.countplot(deaths['Death Year'])
plt.title('How many death caused in whuch year')
plt.show()

sns.countplot(deaths['Allegiances'])
plt.xticks(Rotation = 90)
plt.title('Which allegiances people died mostly')
plt.show()

sns.countplot(deaths['Gender'])
plt.title('Which gender died most')
plt.xticks(np.arange(2),('Female','Male'))
plt.show()
sns.countplot(x='attacker_outcome',data = battles)
plt.title('counting of attacker\'s outcome')
plt.show()
#Lets see which king attacked the most
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.countplot(x='attacker_king',data = battles)
plt.show()

#Lets see which king defended the most
plt.figure(figsize=(15,10))
plt.subplot(2,1,2)
sns.countplot(x='defender_king',data = battles)
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(4,1,1)
sns.countplot(x= battles['attacker_1'],hue=battles['attacker_outcome'])
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(4,1,2)
sns.countplot(x= battles['attacker_2'],hue=battles['attacker_outcome'])
plt.show()

plt.figure(figsize=(5,10))
plt.subplot(4,1,3)
sns.countplot(x= battles['attacker_3'],hue=battles['attacker_outcome'])
plt.show()

plt.figure(figsize=(5,10))
plt.subplot(4,1,4)
sns.countplot(x= battles['attacker_4'],hue=battles['attacker_outcome'])

plt.show()
plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
sns.countplot(x= battles['defender_1'],hue=battles['attacker_outcome'])
plt.xticks(rotation  = 90)
plt.show()

plt.figure(figsize = (3,5))
plt.subplot(2,1,2)
sns.countplot(x= battles['defender_2'],hue=battles['attacker_outcome'])
plt.show()


 
sns.countplot(x= battles['battle_type'])
plt.title('Which battle type is done most?')
plt.show()

sns.countplot(x= battles['battle_type'],hue=battles['attacker_outcome'])
plt.title('Type of battle VS Attacker_Outcome')
plt.show()

sns.countplot(hue= battles['battle_type'],x=battles['attacker_king'],palette = 'Set3')
plt.title('Which type of battle used by attacker king ')
plt.legend(loc = 'upper right')
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x='attacker_outcome',y='attacker_size',data=battles)
plt.title('Attacker_outcome VS Attacker_size')
plt.show()
sns.barplot(x='attacker_outcome',y='defender_size',data=battles)
plt.title('Attacker_outcome VS defender_size')
plt.show()
sns.countplot(x='summer',data = battles)
plt.title('Counting of battles in summers')
plt.xticks(np.arange(2),('winters','summers'))
plt.show()
sns.countplot(x='region', data = battles)
plt.title('In which region there are mostly battles done')
plt.xticks(rotation = 90)
plt.show()
sns.countplot(x='region',hue='attacker_king', data = battles)
plt.title('Which king attacked which region')
plt.xticks(rotation = 90)
plt.legend(loc = 'upper right')
plt.show()