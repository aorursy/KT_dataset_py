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
#For visualising and decorating
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

#Looking at the structure of the data
preds = pd.read_csv('../input/character-predictions.csv')
preds.head()
#Checking for NaNs or missing values
preds.info()
#Drop useless features
to_drop = ['actual','pred','alive','plod','book1','book2','book3','book4','book5']
predictionsData = preds.drop(columns = to_drop)
predictionsData.describe()
predictionsData['male'].plot.hist(color = ['#99cc00'], hatch = '.')
plt.title('Male vs Female Ratio', fontsize = 25, color = '#660066', backgroundcolor = '#9999ff')
sns.despine()
plt.xlabel('Gender')
plt.xticks(np.array([0,1]) ,['Female','Male'], rotation = 90, fontsize = 13)
plt.figure(figsize = (10,10))
predictionsData['popularity'].sort_values().plot.hist(bins = 10, color = 'r')
plt.xlabel('Popularity')
plt.title('Popularity',fontsize = 25, color = '#000000', backgroundcolor = '#66ff99')
print('Most relevant characters are:\n {}'.format(predictionsData[predictionsData['popularity']==1].name))
sns.despine()
groups = predictionsData[predictionsData['popularity']>0.7].groupby('house')
fig, ax = plt.subplots(figsize = (12,8))
ax.margins(0.05)
NUM_COLORS = 20
sns.reset_orig()  # get default matplotlib styles back
clrs = sns.color_palette('Paired', n_colors=NUM_COLORS)
i = 0
for name, group in groups:
    t = ax.plot(group.age, group.numDeadRelations, marker='o', linestyle='', ms=20, label=name, alpha = 0.6)
    t[0].set_color(clrs[i])
    t[0].set_linestyle('solid')
    i += 1
plt.legend(loc = 1, fontsize = 15, title = 'Houses')
plt.xlabel('Age',rotation = 45, fontsize = 15)
plt.title('Houses with maximum death',fontsize = 25, color = '#ffccff', backgroundcolor = '#800000')
plt.ylabel('Number of Dead Relatives', fontsize = 15)
sns.despine()
plt.title('Age at time of death',fontsize = 20, color = '#ffccff', backgroundcolor = '#9900ff')
plt.xlabel('Age', fontsize = 15, rotation = 45)
sns.despine()
predictionsData[predictionsData['isAlive']==0][abs(predictionsData[predictionsData['isAlive']==0]['age'])<150]['age'].plot.hist(color = '#800000')
plt.figure(figsize = (6,4))
plt.xlabel('Salutation')
plt.ylabel('Frequency')
predictionsData.title.value_counts().sort_values(ascending = False)[:10].plot.bar()

unique = predictionsData.title.unique()
p_rates = {}
predictionsDataPop = sum(predictionsData.popularity)
for sal in unique:
    if(type(sal)==str):
        p_rates[sal] = sum(predictionsData.loc[predictionsData['title'] == sal,'popularity'])/(sum(predictionsData.title == sal))

plot_rates = []       
for key, value in zip(p_rates.keys(), p_rates.values()):
    plot_rates.append([key,round(value,2)])
pops = sorted(plot_rates, key = lambda x: x[1], reverse = False)
pops = np.array(pops)
pops = pops[len(pops)-20:len(pops),:]
plt.figure(figsize = (10,5))
plt.xlabel('Title')
plt.ylabel('Popularity')
plt.xticks(rotation = 90, fontsize = 12)
plt.title('Title vs Average Popularity', fontsize = 15)
plt.bar(pops[:,0],pops[:,1] , color = sns.color_palette('Paired', n_colors=15), hatch = '//', linestyle = '--')
sns.despine()
ages = predictionsData.loc[abs(predictionsData.age)<150,:]
x = np.array(ages.loc[ages.isAlive ==0, :].age)
y = np.array(ages.loc[ages.isAlive ==1, :].popularity)
z = np.array(ages.loc[ages.isAlive>=0, :].numDeadRelations)
fig, ax = plt.subplots(figsize = (8,5))
plt.ylabel('Popularity', fontsize = 12)
plt.title('Age vs Popularity and Survived', fontsize = 15)
plt.xlabel('Age', fontsize = 12, rotation = 45)
for xp, yp, sp in zip(x, y, z):
    ax.scatter([xp],[yp], marker='x' if yp == 0 else 'o', color = 'r' if yp == 0 else 'g', alpha = 1 if yp == 0 else 0.4, s=(sp+0.8)*50)

print("Size of marker : Number of Dead Relations\n 'X': isAlive = 0, 'O': isAlive = 1")