# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/googleplaystore.csv')
new_df = ['App','Category','Rating','Reviews','Size','Installs','Type','Price','ContentRating','Genres','LastUpdated','CurrentVer','AndroidVer']
df.columns = new_df
df.head()
df.tail()
df.info()
df.Category.replace(['1.9'],0.0,inplace=True)
df['Installs'].replace(['Free'],0.0,inplace=True)
df['Reviews'] = df['Reviews'].apply(lambda x: x.replace('M', '') if 'M' in str(x) else x)
df['Reviews'] = df['Reviews'].apply(lambda x: float(x))
#--------------------------------------------------------------------------------------------
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int(x))
#df.Installs.replace(['+'],'',inplace=True)
#df.Installs.replace([','],'',inplace=True)
#df.Installs=df.Installs.astype(int)
#--------------------------------------------------------------------------------------------
df.Rating=df.Rating.astype(str)
df['Rating'] = df['Rating'].apply(lambda x: x.replace('nan', '0.0') if 'nan' in str(x) else x)
df['Rating'] = df['Rating'].apply(lambda x: float(x))
#--------------------------------------------------------------------------------------------
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: x.replace('Everyone', '') if 'Everyone' in str(x) else x)

df.Type=df.Type.astype(str)
df.Type.replace(['nan'],'null',inplace=True)
# What is the download rate by categories?
app_list=list(df['Category'].unique())
app_rate=[]
for i in app_list:
    j=df[df['Category']==i]
    app_rate1=sum(j.Installs)/len(j)
    app_rate.append(app_rate1)
data=pd.DataFrame({'app_list':app_list,'app_rate':app_rate})
new_index=(data['app_rate'].sort_values(ascending=False)).index.values
sortedData=data.reindex(new_index)


plt.figure(figsize=(10,5))
sns.barplot(x=sortedData['app_list'], y=sortedData['app_rate'])
plt.xticks(rotation= 90)
plt.xlabel('Category')
plt.ylabel('app_rate')
plt.title('drop-down categories')
plt.show()
sortedData.head()#dowlads by cateogory(kategorilere göre indirilme oranı)
#df.App.value_counts()
data_m=df
new_index_m=(data_m['Installs'].sort_values(ascending=False)).index.values
data_m=data_m.reindex(new_index_m)
x_m=data_m.head(30)

plt.figure(figsize=(15,8))
sns.barplot(x=x_m['App'], y=x_m['Installs'])
plt.xticks(rotation= 90)
plt.xlabel('App')
plt.ylabel('Installs')
plt.title('most downloaded applications')
plt.show()

data_m.head()#most downloaded applications (en çok indirilen uygulamalar)
data_m.Type.value_counts()
cate_list=list(df['Type'].unique())
type_rate=[]
for i in cate_list:
    k=df[df['Type']==i]
    type_rate.append(sum(k.Installs)/len(k))
df_m=pd.DataFrame({'cate_list':cate_list,'type_rate':type_rate})
v_index=(df_m['type_rate'].sort_values(ascending=False)).index.values
sota=df_m.reindex(v_index)


plt.figure(figsize=(5,5))
sns.barplot(x=sota['cate_list'],y=sota['type_rate'])
plt.xticks(rotation= 90)
plt.xlabel('type_list')
plt.ylabel('type_rate')
plt.title('download rate by price')
plt.show()
sota.head()#tipine göre indirilme oranı
cate_list=list(df['Category'].unique())
name_count = Counter(df['Category'])         
most_common_names = name_count.most_common(15) 
x,y = zip(*most_common_names)
x,y = list(x),list(y)

plt.figure(figsize=(10,5))
sns.barplot(x=y, y=x,palette = sns.cubehelix_palette(len(x),reverse=True))
plt.xticks(rotation= 90)
plt.xlabel('Category')
plt.ylabel('Most_common_names')
plt.title('Sort by category')
plt.show()
ml_list=list(df['Category'].unique())
ins_rate=[]
rait_rate=[]
rev_rate=[]

for i in ml_list:
    t=df[df['Category']==i]
    ins_rate.append(sum(t.Installs)/len(t))
    rait_rate.append(sum(t.Rating)/len(t))
    rev_rate.append(sum(t.Reviews)/len(t))
dat_k=pd.DataFrame({'ml_list':ml_list,'ins_rate':ins_rate,'rait_rate':rait_rate,'rev_rate':rev_rate})

new_ins_rate=[]
for i in dat_k.ins_rate:
    a=max(ins_rate)
    new_ins_rate.append(i/a)   

new_rait_rate=[]
for i in dat_k.rait_rate:
    b=max(rait_rate)
    new_rait_rate.append(i/b)
    
new_rev_rate=[]
for i in dat_k.rev_rate:
    c=max(rev_rate)
    new_rev_rate.append(i/c)
    
dat_km=pd.DataFrame({'ml_list':ml_list,'new_ins_rate':new_ins_rate,'new_rait_rate':new_rait_rate,'new_rev_rate':new_rev_rate})
    
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=new_ins_rate,y=ml_list,color='red',alpha = 0.7,label='Installs' )
sns.barplot(x=new_rait_rate,y=ml_list,color='yellow',alpha = 0.8,label='Rating' )
sns.barplot(x=new_rev_rate,y=ml_list,color='blue',alpha = 0.5,label='Rating' )
plt.xticks(rotation= 90)
ax.set(xlabel='raiting and ınstalls', ylabel='Category',title = "- ")
plt.show()
dat_k.head()# kategori bazında raiting ve indirilme oranları
dat_km.head()

f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='ml_list',y='new_ins_rate',data=dat_km,color='lime',alpha=0.8)
sns.pointplot(x='ml_list',y='new_rait_rate',data=dat_km,color='red',alpha=0.8)
sns.pointplot(x='ml_list',y='new_rev_rate',data=dat_km,color='blue',alpha=0.8)
plt.text(40,0.6,'Installs',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Raiting',color='lime',fontsize = 18,style = 'italic')
plt.text(40,0.70,'Reviews',color='blue',fontsize = 19,style = 'italic')
plt.xticks(rotation= 90)
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Installs  VS  Raiting and Reviews',fontsize = 20,color='blue')
plt.grid()

g = sns.jointplot(dat_km.new_ins_rate, dat_km.new_rev_rate, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot("new_ins_rate", "new_rait_rate", data=dat_km,size=5, ratio=3, color="r")
df.ContentRating.unique()

df.ContentRating.dropna(inplace = True)
labels = df.ContentRating.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = df.ContentRating.value_counts().values


# visual
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%')
plt.title('Content Raiting',color = 'blue',fontsize = 15)
sns.lmplot(x="new_ins_rate", y="new_rait_rate", data=dat_km)
sns.lmplot(x="new_ins_rate", y="new_rev_rate", data=dat_km)
sns.lmplot(x="new_rev_rate", y="new_rait_rate", data=dat_km)
plt.show()
sns.kdeplot(dat_km.new_ins_rate, dat_km.new_rev_rate, shade=True, cut=3)
plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=dat_km, palette=pal, inner="points")
plt.show()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(dat_km.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
df.head()
sns.pairplot(dat_km)
plt.show()
su_m = x_m.Category.value_counts()
#print(armed)
plt.figure(figsize=(10,7))
sns.barplot(x=su_m[:7].index,y=su_m[:7].values)
plt.ylabel('Number of Weapon')
plt.xlabel('Weapon Types')
plt.title('Category',color = 'blue',fontsize=15)
df.head()
rait_4 =['pozitive' if i >= 4 else 'negative' for i in df.Rating]
a_df = pd.DataFrame({'raiting':rait_4})
sns.countplot(x=a_df.raiting)
plt.ylabel('Number of application')
plt.title('Rating Rate',color = 'blue',fontsize=15)
