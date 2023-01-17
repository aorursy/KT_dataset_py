# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
data2015 = pd.read_csv("../input/world-happiness/2015.csv")

data2016 = pd.read_csv("../input/world-happiness/2016.csv")

data2017 = pd.read_csv("../input/world-happiness/2017.csv")
data2015.columns
data2015 = data2015.rename(columns={'Health (Life Expectancy)':'Health','Dystopia Residual':'Dystopia_Residual','Economy (GDP per Capita)':'Economy','Trust (Government Corruption)':'Trust','Happiness Rank':'Happiness_Rank','Happiness Score':'Happiness',"Standard Error":"Standard_Error"})

data2016 = data2016.rename(columns={'Health (Life Expectancy)':'Health','Dystopia Residual':'Dystopia_Residual','Economy (GDP per Capita)':'Economy','Trust (Government Corruption)':'Trust','Happiness Rank':'Happiness_Rank','Happiness Score':'Happiness',"Standard Error":"Standard_Error"})

data2017 = data2017.rename(columns={'Health..Life.Expectancy.':'Health','Dystopia.Residual':'Dystopia_Residual','Economy..GDP.per.Capita.':'Economy','Trust..Government.Corruption.':'Trust','Happiness.Rank':'Happiness_Rank','Happiness.Score':'Happiness',"Standard Error":"Standard_Error"})
data2015 = data2015.set_index(["Happiness_Rank"])

data2016 = data2016.set_index(["Happiness_Rank"])

data2017 = data2017.set_index(["Happiness_Rank"])
data2015.head()
data2015.corr()
data2015.Region.unique()
data2015.loc[:,"Economy":"Family"].apply(np.argmax,axis=1)
data2015region = data2015.groupby("Region").mean()

data2015region
region_list = list(data2015['Region'].unique())

area_happiness_ratio = []

for i in region_list:

    x = data2015[data2015['Region']==i]

    Region_Happiness_rate = sum(x.Happiness)/len(x)

    area_happiness_ratio.append(Region_Happiness_rate)

data = pd.DataFrame({'region_list': region_list,'area_happiness_ratio':area_happiness_ratio})

new_index = (data['area_happiness_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['region_list'], y=sorted_data['area_happiness_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Regions')

plt.ylabel('Happiness Score')

plt.title('Hapiness Score of Regions')

plt.show()
list2015 = data2015.groupby("Region").mean().Happiness

new_index = (list2015.sort_values(ascending=False).index)

new = (list2015.sort_values(ascending=False))

plt.figure(figsize=(15,10))

sns.barplot(x = new_index, y = new)

plt.xticks(rotation= 45)

plt.xlabel('Regions')

plt.ylabel('Happiness Score')

plt.title('Hapiness Score of Regions')

plt.show()
Happiness_Score_ratio = data2015["Happiness"]/data2015.Happiness.max()

Family_ratio = data2015["Family"]/data2015.Family.max()

plt.subplots(figsize =(30,15))

sns.pointplot(y=Happiness_Score_ratio, x="Country", data= data2015,color='lime')

sns.pointplot(y=Family_ratio, x="Country", data= data2015,color='blue')

plt.xticks(rotation= 90)

plt.xlabel('Country',fontsize = 20,color='black')

plt.ylabel('Family Score',fontsize = 20,color='black')

plt.title('Happiness Score VS Family Score',fontsize = 25,color='black')

plt.text(20,0.45,'Family Score',color='blue',fontsize = 30,style = 'italic')

plt.text(20,0.35,'Happiness Score',color='lime',fontsize = 30,style = 'italic')

plt.grid()

plt.show()
sns.jointplot(data2015.Health, data2015.Happiness, kind="kde",shade = True, height= 10)

plt.show()
sns.jointplot(data2015.Freedom, data2015.Economy, height= 10,ratio=3, color="g")

plt.show()
sns.lmplot("Economy","Generosity", data = data2015, height=10)

plt.show()
plt.figure(figsize=(15,10))

sns.kdeplot(data2015.Health,data2015.Family, shade= True, cut=3, color="r")

plt.show()
plt.subplots(figsize =(15,10))

list1 = data2015.Trust

list2 = data2015.Family

data_new = pd.DataFrame({'Trust': list1,'Family':list2})

sns.violinplot(data = data_new, inner="points")

sns.despine(left=True)

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(data2015.corr(),annot = True, linewidths=0.5, fmt = ".1f")

plt.show()
ListEco = ["Below" if i<data2015.Economy.mean() else "Above" if i>data2015.Economy.mean() else "Equal" for i in data2015.Economy]

data2015["AVGoECO"] = ListEco

plt.figure(figsize=(15,10))

sns.boxplot(x="AVGoECO",y="Happiness", hue ="Region", data = data2015)

plt.xticks(rotation= 90)

plt.show()



plt.figure(figsize=(15,10))

sns.swarmplot(x="AVGoECO",y="Happiness",hue="Region",data = data2015)

plt.xticks(rotation= 90)

plt.show()

sns.countplot(data2015.AVGoECO)

plt.show()
sns.pairplot(data2015.loc[0:,"Happiness":"Family"],height=5)

plt.show()
sns.pairplot(data2015.loc[0:,"Happiness":"Economy"], kind="reg")

plt.show()

plt.subplots(figsize = (15,30))

data2015.Happiness.sort_values(ascending = True)

sns.barplot(x="Economy",y="Country",data = data2015, color='black',alpha = 1,label='Economy' )

sns.barplot(x="Family",y="Country",data = data2015, color='brown',alpha = 0.9,label='Family')

sns.barplot(x="Health",y="Country",data = data2015, color='blue',alpha = 0.8,label='Health')

sns.barplot(x="Freedom",y="Country",data = data2015, color='red',alpha = 0.7,label='Freedom')

sns.barplot(x="Generosity",y="Country",data = data2015, color='purple',alpha = 0.6,label='Generosity')

sns.barplot(x="Trust",y="Country",data = data2015, color='yellow',alpha = 0.7,label='Trust')

sns.barplot(x="Dystopia_Residual",y="Country",data = data2015, color='cyan',alpha = 0.6,label='Dystopia_Residual')

sns.barplot(x="Happiness",y="Country",data = data2015, color='pink',alpha = 0.4,label='Happiness_Score')

plt.legend(loc='lower right',frameon = True)

plt.show()

data2015.columns.unique()
data2015.index = range(1,159,1)

data2016.index = range(1,158,1)

data2017.index = range(1,156,1)

from matplotlib.lines import Line2D

dict_of_df={}

list_columns = ['Happiness','Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']

for i in list_columns:

    fig, ax = plt.subplots(figsize = (20,12))

    dict_of_df["{}".format(i)] = pd.DataFrame({'2015': data2015[i],'2016':data2016[i],'2017':data2017[i]})

    sns.pointplot(x= dict_of_df[i].index,y="2015", data= dict_of_df[i],color="red")

    sns.pointplot(x= dict_of_df[i].index,y="2016", data= dict_of_df[i],color="black")

    sns.pointplot(x= dict_of_df[i].index,y="2017", data= dict_of_df[i],color="blue")

    custom_lines = [Line2D([0], [0], color="red", lw=4),Line2D([0], [0], color="black", lw=4),Line2D([0], [0], color="blue", lw=4)]

    ax.legend(custom_lines, ['2015', '2016', '2017'])

    plt.xticks(rotation= 90)

    plt.xlabel('Country Number',fontsize = 20,color='black')

    plt.ylabel("{} Score".format(i),fontsize = 20,color='black')

    plt.title("{}".format(i),fontsize = 25,color='black')

plt.show()