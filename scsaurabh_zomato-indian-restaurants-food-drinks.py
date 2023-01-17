import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")
data.head()

country = pd.read_excel("../input/Country-Code.xlsx")

country.head()
data1 = pd.merge(data, country, on='Country Code')

data1.head(3)
fig,ax = plt.subplots(1,1,figsize = (15,4))

ax = sns.countplot(data1[data1.Country != 'India']['Country'])

plt.show()
res_India = data1[data1.Country == 'India']

res_India.head(3)
top5 = res_India.City.value_counts().head()

top5
f , ax = plt.subplots(1,1,figsize = (14,4))

ax = sns.barplot(top5.index,top5,palette ='Set1')

plt.show()
NCR = ['New Delhi','Gurgaon','Noida','Faridabad']

res_NCR = res_India[(res_India.City == NCR[0])|(res_India.City == NCR[1])|(res_India.City == NCR[2])|

                    (res_India.City == NCR[3])]

res_NCR.head(3)
f,ax = plt.subplots(1,1,figsize = (14,4))

sns.countplot(res_NCR.City,palette ='cubehelix')

plt.show()
print(res_NCR['Has Table booking'].value_counts())

fig,ax = plt.subplots(1,1,figsize=(10,4))

ax = sns.countplot(res_NCR['Has Table booking'],palette= 'Set1')

plt.show()
print(res_NCR['Has Online delivery'].value_counts())

fig,ax = plt.subplots(1,1,figsize=(10,4))

ax = sns.countplot(res_NCR['Has Online delivery'],hue = res_NCR['City'],palette ='Set1')

plt.show()
f, ax = plt.subplots(1,1, figsize = (14, 5))

ax = sns.countplot(res_NCR['Rating text'],palette ='Set1')

plt.show()
f, ax = plt.subplots(1,1, figsize = (14, 5))

ax = sns.countplot(res_NCR['Price range'],hue = res_NCR['City'])

plt.show()
agg_rat = res_NCR[res_NCR['Aggregate rating'] > 0]

f, ax = plt.subplots(1,1, figsize = (14, 4))

ax = sns.countplot(agg_rat['Aggregate rating'])

plt.show()
res_NCR[(res_NCR.City == 'New Delhi') & (res_NCR['Aggregate rating'] >=4 )]['Locality'].value_counts().head()
res_NCR[(res_NCR['City']=='Gurgaon') & (res_NCR['Aggregate rating']> 4) & (res_NCR['Votes'] > 1000) 

& (res_NCR['Rating text'] =='Excellent')]
res_NCR.reset_index(inplace=True)

res_NCR.head(3)
cuisines = {"North Indian":0,'Chinese':0,'Fast Food':0,'Mughlai':0,'Bakery':0,'Continental':0,'Italian':0,

           "South Indian":0,'Cafe':0,'Desserts':0,'Street Food':0,'Mithai':0,'Pizza':0,'American':0,'Ice Cream':0}



for i in range(len(res_NCR.Cuisines)):

    for j in res_NCR.loc[i,'Cuisines'].split(','):

        if  j in cuisines.keys():

            cuisines[j] +=1

print(cuisines)
f, ax = plt.subplots(1,1, figsize = (15, 4))

ax = sns.barplot(x = list(cuisines.keys()),y=list(cuisines.values()),palette='cubehelix')

plt.show()
fig,ax = plt.subplots(1,1,figsize=(10,6))

ax  = sns.boxplot(x='Price range',y = 'Aggregate rating',data=res_NCR,palette='cubehelix')

plt.show()