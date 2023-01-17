import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from subprocess import check_output
import plotly.plotly as py
from wordcloud import WordCloud
import os
print(check_output(["ls", "../input"]).decode("utf8"))
terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
terror2 = terror.copy()
terrorTr2 = terror2[terror2.country_txt == "Turkey"]
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror['casualities']=terror['Killed']+terror['Wounded']
terror['Killed'] = terror['Killed'].fillna(0)
terror['Wounded'] = terror['Wounded'].fillna(0)
x = terror.Country == 'Turkey'
terrorTr = terror[x]
terrorTr.head(5)
terrorTr.tail(5)
terrorTr.describe()
print(type(terrorTr))
terrorTr.columns
terror.isnull().sum()
print('En fazla terör saldırısı gerçekleşen şehir:',terrorTr['city'].value_counts().index[0])
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terrorTr,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.yticks(rotation=30)
plt.title('Yıllara Göre Terörist Eylemler Grafiği')
plt.show()
terrorTr.city.value_counts().drop('Unknown').head(10).plot.bar(figsize=[16,9])
plt.yticks(fontsize=14,rotation=30)
plt.xticks(fontsize=14)
plt.xlabel("Cities", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("En Çok Hedef Alınan Şehirler", fontsize=16)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terrorTr,palette='inferno',order=terrorTr['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.yticks(rotation=30)
plt.title('Teröristlerin Saldırı Metodları')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(terror['Target_type'],palette='inferno',order=terror['Target_type'].value_counts().index)
plt.xticks(rotation=90)
plt.yticks(rotation=30)
plt.title('Hedefler')
plt.show()
terorism_actors = terrorTr2.gname.unique()
print(terorism_actors)
cities = terrorTr2.provstate.dropna(False)
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.imsave(arr = wordcloud, fname = 'wordcloud.png')
plt.show()
attacks_of_groups = []
for name in terorism_actors:
    temp = terrorTr2.gname[terrorTr2.gname == name].count()
    attacks_of_groups.append(temp)
    
dataframe_temp = pd.DataFrame({'actor':terorism_actors, 'attack_num':attacks_of_groups})
dataframe_temp = dataframe_temp[dataframe_temp.attack_num >= 6]
dataframe_temp
terrorTr.Group.value_counts().head(10).plot.bar(figsize=[18,9])
plt.yticks(fontsize=14,rotation=30)
plt.xticks(fontsize=14)
plt.ylabel("Saldırı Sayıları", fontsize=15)
plt.show()
labels = ['PKK','Unknown','Dev Sol']
explode = (0, 0.1, 0)
colrs = ['cyan','tan','wheat']
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.pie(terrorTr.Group.value_counts().head(3), explode= explode, labels=labels, autopct = '%1.1f%%',startangle=270, colors=colrs)
ax.axis('equal')
fig.suptitle('Terör Örgütleri')
fig.savefig('filename.png', dpi=125)