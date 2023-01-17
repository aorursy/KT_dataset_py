import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

data = pd.read_csv("../input/pokemon.csv")
data.info()
data.head()
top_type1 = data['Type 1'].value_counts()

f,ax = plt.subplots(figsize=(9,15))
sns.barplot(y=top_type1.index,x=top_type1.values,alpha=0.6,label='Type 1 Counts')
plt.xlabel('Values')
plt.ylabel('Type 1')
plt.title('Type 1 Counts')
plt.show()
labels = data['Type 1'].value_counts().index
sizes = data['Type 1'].value_counts().values
explode = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.2f%%')
plt.title('Count of Type 1',color='red',fontsize=18)
plt.show()
top_type2 = data['Type 2'].value_counts()

f,ax = plt.subplots(figsize=(9,15))
sns.barplot(y=top_type2.index,x=top_type2.values,alpha=0.6,label='Type 2 Counts')
plt.yticks(rotation=15)
plt.xlabel('Values')
plt.ylabel('Type 2')
plt.title('Type 2 Counts')
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidths=.7,linecolor='b',fmt='.2f',ax=ax)
plt.title('Correlation Map',fontsize=18,color='red')
plt.show()

type1_list = list(data['Type 1'].unique()) 
type1_hp_ratio = []
for i in type1_list:
    x = data[data['Type 1']==i]
    type1_hp_rate = sum(x.HP)/len(x)
    type1_hp_ratio.append(type1_hp_rate)
df = pd.DataFrame({'Type1':type1_list,'Hp_Ratio':type1_hp_ratio})
new_index = (df.Hp_Ratio.sort_values(ascending=False)).index.values
sorted_data = df.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x="Type1",y="Hp_Ratio",data=sorted_data)
plt.xticks(rotation=90)
plt.xlabel('Types')
plt.ylabel('Hp')
plt.title('Hp rate given Type 1',fontsize=18,color='red')
plt.show()
type1_speed_ratio = []
for j in type1_list:
    y = data[data['Type 1']==j]
    type1_speed_rate = sum(y.Speed)/len(y)
    type1_speed_ratio.append(type1_speed_rate)
df1 = pd.DataFrame({'Type 1':type1_list,'Speed_Ratio':type1_speed_ratio})
new_index1 = (df1.Speed_Ratio.sort_values(ascending=False)).index.values
sorted_data1 = df1.reindex(new_index1)

plt.figure(figsize=(15,10))
sns.barplot(x="Type 1",y="Speed_Ratio",data=sorted_data1)
plt.xlabel('Types')
plt.ylabel('Speed')
plt.title('Speed rate given Type1')
plt.show()
type1_attack_ratio = []
for k in type1_list:
    z = data[data['Type 1']==k]
    type1_attack_rate = sum(z.Attack)/len(z)
    type1_attack_ratio.append(type1_attack_rate)
df2 = pd.DataFrame({'Type 1':type1_list,'Attack_Ratio':type1_attack_ratio})
new_index2 = (df2.Attack_Ratio.sort_values(ascending=False)).index.values
sorted_data2 = df2.reindex(new_index2)

plt.figure(figsize=(15,10))
sns.barplot(x="Type 1",y="Attack_Ratio",data=sorted_data2)
plt.xlabel('Types')
plt.ylabel('Attack')
plt.title('Attack rate given Type1')
plt.show()
type1_defense_ratio = []
for m in type1_list:
    n = data[data['Type 1']==m]
    type1_defense_rate = sum(n.Defense)/len(n)
    type1_defense_ratio.append(type1_defense_rate)
df3 = pd.DataFrame({'Type 1':type1_list,'Defense_Ratio':type1_defense_ratio})
new_index3 = (df3.Defense_Ratio.sort_values(ascending=False)).index.values
sorted_data3 = df3.reindex(new_index3)

plt.figure(figsize=(15,10))
sns.barplot(x="Type 1",y="Defense_Ratio",data=sorted_data3)
plt.xlabel('Types')
plt.ylabel('Defense')
plt.title('Defense rate given Type1')
plt.show()
sorted_data2['Attack_Ratio'] = sorted_data2['Attack_Ratio']/max(sorted_data2['Attack_Ratio'])
sorted_data3['Defense_Ratio'] = sorted_data3['Defense_Ratio']/max(sorted_data3['Defense_Ratio'])

concat = pd.concat([sorted_data2,sorted_data3['Defense_Ratio']],axis=1)
concat.sort_values('Attack_Ratio',inplace=True)

f,ax1=plt.subplots(figsize=(20,10))
sns.pointplot(x='Type 1',y='Attack_Ratio',data=concat,color='blue',alpha=0.7)
sns.pointplot(x='Type 1',y='Defense_Ratio',data=concat,color='red',alpha=0.6)
plt.text(0,1,'Defense',color='red',fontsize=12,style='italic')
plt.text(0,0.95,'Attack',color='blue',fontsize=12,style='italic')
plt.xlabel('Type')
plt.ylabel('Values')
plt.title('Attack Rate vs Defense Rate')
plt.grid()
plt.show()
sns.jointplot(data=concat,x='Attack_Ratio',y='Defense_Ratio',kind='kde',size=7,color='g')
plt.show()
sns.jointplot(data=concat,x='Attack_Ratio',y='Defense_Ratio',size=7,color='b')
plt.show()
sorted_data1['Speed_Ratio'] = sorted_data1['Speed_Ratio']/max(sorted_data1['Speed_Ratio'])

concat1 = pd.concat([sorted_data2,sorted_data1['Speed_Ratio']],axis=1)
concat1.sort_values('Speed_Ratio',inplace=True)

sns.lmplot(data=concat1,x='Attack_Ratio',y='Speed_Ratio',size=8)
plt.xlabel('Attack')
plt.ylabel('Speed')
plt.grid()
plt.show()
concat2=pd.concat([sorted_data1,sorted_data3.Defense_Ratio],axis=1)
concat2.sort_values('Defense_Ratio',inplace=True)


sns.lmplot(data=concat2,x='Defense_Ratio',y='Speed_Ratio',size=8)
plt.xlabel('Defense')
plt.ylabel('Speed')
plt.grid()
plt.show()

f,ax=plt.subplots(figsize=(8,7))
pal=sns.color_palette("RdBu_r", 7)
sns.violinplot(data=concat,palette=pal,inner="stick",ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(10,8))
sns.countplot(data.Legendary,ax=ax)
plt.show()
sns.lmplot(x='Defense',y='Speed',col='Legendary',data=data,palette="Set1")
plt.show()
sns.lmplot(x='Attack',y='Speed',col='Legendary',data=data)
plt.show()
f,ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="Type 1",y="HP",hue="Legendary",data=data,ax=ax)
plt.xticks(rotation=90)
plt.show()
sns.pairplot(concat,size=4.5)
plt.show()
f,ax = plt.subplots(figsize=(10,8))
sns.countplot(data['Generation'],ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(10,7))
sns.boxplot(x='Generation',y='Sp. Atk',data=data,hue='Legendary',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(7,7))
sns.heatmap(concat.corr(),annot=True,linewidths=0.7,linecolor='b',fmt='.2f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(7,7))
sns.heatmap(concat1.corr(),annot=True,linewidths=0.7,linecolor='r',fmt='.2f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(7,7))
sns.heatmap(concat2.corr(),annot=True,linewidths=0.7,linecolor='g',fmt='.2f',ax=ax)
plt.show()

