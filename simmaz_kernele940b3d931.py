# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#discover income diversification methids for farm owners
#System of farming proportions with respect to land usage
agri_data = pd.read_csv("../input/data.csv", low_memory=False)
agri_data['fsystem1'] #1. Shifting Cultivating 2. Continous Cropping 3. Continous Cropping with multiple rotations 4. Livestock grazing land 5.Other 6. Combinationation of above
util_cat = agri_data['fsystem1'].fillna(0)
Cat_List = [1,2,3,4,5,6,0]
CatCount = [sum(map(lambda x: x==Cat,util_cat)) for Cat in Cat_List]
CatPerc = [x*100.0/sum(CatCount) for x in CatCount]
plt.pie(CatPerc,None,Cat_List)
plt.show
#plt.pie(util_cat)
#plt.show
#farm type proportions
agri_data = pd.read_csv("../input/data.csv", low_memory=False)
agri_data['farmtype'] # 1. Small-scale 2. medium scale 3. large-scale
ftype_cat = agri_data['farmtype'].fillna(0)
fCat_List = [1,2,3]
fCatCount = [sum(map(lambda x: x==Cat,ftype_cat)) for Cat in fCat_List]
fCatPerc = [x*100.0/sum(fCatCount) for x in fCatCount]
plt.pie(fCatPerc,None,fCat_List,autopct='%1.1f%%')
plt.show()
#Farm system compared to type of farm, part of utilisation(two ring donut plot)
# Make data: I have 3 groups and 7 subgroups
group_names= fCat_List
group_size=fCatPerc
subgroup_names=Cat_List
subgroup_size=CatPerc
#subgroup_size = [(x*100/sum(y) for x in fCatCount)for y in fCatCount]
#[x*100.0/sum(fCatCount) for x in fCatCount]
 
# Create colors
A1, A2, A3=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[A1(0.6), A2(0.6), A3(0.6)] )
plt.setp( mypie, width=0.3, edgecolor='white')
 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.7, colors=[A1(0.5), A1(0.4), A1(0.3), A2(0.5), A2(0.4), A3(0.6), A3(0.5), A3(0.4), A3(0.3), A3(0.2)])
plt.setp( mypie2, width=0.4, edgecolor='white')
plt.margins(0,0)
 
# show it
plt.show()
sns.violinplot(agri_data['gender1'], agri_data['age1'], palette = "Blues") #Variable Plot
sns.despine()
sns.violinplot(agri_data['gender2'], agri_data['age2'], palette = "muted") #Variable Plot
sns.despine()
sns.violinplot(agri_data['gender3'], agri_data['age3']) #Variable Plot
sns.despine()
sns.violinplot(agri_data['gender10'], agri_data['age10']) #Variable Plot
sns.despine()
sns.violinplot(agri_data['gender30'], agri_data['age30'], palette = 'Blues') #Variable Plot
sns.despine()
var = agri_data.groupby('gender1').incnfarm_n.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Income')
ax1.set_title("Gender wise Sum of Income")
var.plot(kind='bar')

var = agri_data.groupby('gender2').incnfarm_n.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Income')
ax1.set_title("Gender wise Sum of Income")
var.plot(kind='bar')

var = agri_data.groupby('gender3').incnfarm_n.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Income')
ax1.set_title("Gender wise Sum of Income")
var.plot(kind='bar')

var = agri_data.groupby('gender30').incnfarm_n.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Income')
ax1.set_title("Gender wise Sum of Income")
var.plot(kind='bar')
#Income from farming activities
inc_piv = pd.pivot_table(agri_data, values = 'incfarm', index=['adm0'], aggfunc=np.sum)
Inc_type = agri_data['adm0'].fillna(0)
Inc_list = []
for x in agri_data.adm0:
    if x not in Inc_list:
        Inc_list.append(x)
Inc_Count = [sum(map(lambda x: x==Cat,Inc_type)) for Cat in Inc_list]
Inc_Perc = [x*100.0/sum(Inc_Count) for x in Inc_Count]
plt.pie(Inc_Perc,None,Inc_list,autopct='%1.1f%%')

