# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import cufflinks as cf
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/superhero-set"))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Any results you write to the current directory are saved as output.
hero=pd.read_csv('../input/superhero-set/heroes_information.csv',index_col='Unnamed: 0',na_values='-')
powers=pd.read_csv('../input/superhero-set/super_hero_powers.csv')
hero.head()
powers.head()
hero.info()
hero=hero.drop('Skin color',axis=1)
hero.info()
powers.info()
hero_g=hero.Gender.value_counts()
df=pd.DataFrame({'label':hero_g.index,
                'value':hero_g.values})
df.iplot(kind='pie',labels='label',values='value', title='Gender', hole = 0.5, color = ['#FAEBD7','#7FFFD4'])

hero_p=hero.Publisher.value_counts()
hero_p=pd.DataFrame(
{'label':['Marvel Comics', 'DC Comics','others'],
'value':hero_p[:2].values.tolist()+[hero_p[2:].sum()]})
hero_p
colors = ['#FEBFB3', '#E1396C', '#96D38C']
hero_p.iplot(kind='pie',labels='label',values='value', title='Comics Publishers', hole = 0.5, color =colors)
hero_alig=hero['Alignment'].value_counts()
df=pd.DataFrame({
    'label':hero_alig.index,
    'value':hero_alig.values
})
colors=['#ef932c','#2cefc3','#f790b1']
df.iplot(kind='pie',labels='label',values='value', title='Hero Alignments', hole = 0.4, color =colors)
fig = plt.figure(figsize=(12,7))
fig.add_subplot(1,1,1)
sns.countplot(x='Publisher',data=hero.head(80))
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
hero_name=hero.name.value_counts().head(10)
df=pd.DataFrame({'label':hero_name.index,
                'value':hero_name.values})
fig = plt.figure(figsize=(12,7))
fig.add_subplot(1,1,1)
sns.barplot(x='label',y='value',data=df)
plt.xticks(rotation=70)
plt.show()

from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords
import numpy as np
mask = np.array(Image.open('../input/machinejpg/machine-learning.jpg'))
stopwords = stopwords.words('english')
cloud = WordCloud(width=1440, height=1080,mask=mask, stopwords=stopwords).generate(" ".join(hero['Race'].dropna().astype(str)))
plt.figure(figsize=(20, 12))
plt.imshow(cloud)
plt.axis('off')


hero['Hair color']=hero['Hair color'].str.replace('No Hair','Bald')
df=hero['Hair color'].value_counts().head(10)
df=pd.DataFrame({
    'label':df.index,
    'value':df.values
})
fig=plt.figure(figsize=(12,7))
fig.add_subplot(1,1,1)
sns.barplot(x='label',y='value',data=df)
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
total=hero['Hair color'].notnull().sum()
percent=[]
unique_enteries=hero['Hair color'].value_counts()
for i in unique_enteries.index:
    per=((hero['Hair color']==i).sum()/total)*100
    percent.append(per)
df=pd.DataFrame({
    'label':unique_enteries.index,
    'percent':percent
})
print(df.head(10))
fig=plt.figure(figsize=(12,7))
fig.add_subplot(1,1,1)
sns.barplot(x='label',y='percent',data=df)
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

fig=plt.figure(figsize=(14,8))
fig.add_subplot(1,2,1)
sns.boxplot(x='Gender',y='Weight',data=hero)
fig.add_subplot(1,2,2)
sns.boxplot(x='Gender',y='Height',data=hero)
plt.show()

sns.jointplot(x='Weight',y='Height',data=hero,kind='kde',size=9)
plt.show()

hero['Weight']=hero['Weight'].fillna(method='ffill')
sns.pairplot(hero,kind='reg',size=4,hue='Gender')
powers=powers*1
powe=powers.iloc[:,1:]
total=[]
for i in powe.columns:
    total.append(powe[i].sum())
df=pd.DataFrame(
{
    'label':powe.columns,
    'total':total})
df=df.sort_values('total',ascending=False)
df.head()
#df=df.drop([167])
fig=plt.figure(figsize=(12,7))
fig.add_subplot(1,1,1)
sns.set_palette("husl")
sns.barplot(x='label',y='total',data=df.head(15))
plt.xticks(rotation=60)
plt.show()
powers.columns
powers['total powers']=powers.iloc[:,1:].sum(axis=1)
df=powers[['hero_names','total powers']].sort_values('total powers',ascending=False)
fig=plt.figure(figsize=(14,7))
fig.add_subplot(1,1,1)
sns.barplot(x='hero_names',y='total powers',data=df.head(20),palette="BuGn_d")
plt.xticks(rotation=60)
plt.show()

male=hero[hero['Gender']=='Male']
Female=hero[hero['Gender']=='Female']
male_hero=np.array(male['name'])
Female_hero=np.array(Female['name'])
def is_hero(row):
        if(row in male_hero):
            return True
        else:
            return False
def is_herof(row):
        if(row in Female_hero):
            return True
        else:
            return False
powersm=powers[powers['hero_names'].apply(is_hero)]
df=powersm[['hero_names','total powers']].sort_values('total powers',ascending=False)
fig=plt.figure(figsize=(20,8))
fig.add_subplot(1,2,1)
sns.barplot(x='hero_names',y='total powers',data=df.head(20),palette="BuGn_d")
plt.title('Male Hero')
plt.xticks(rotation=60)
powersf=powers[powers['hero_names'].apply(is_herof)]
df=powersf[['hero_names','total powers']].sort_values('total powers',ascending=False)
fig.add_subplot(1,2,2)
sns.barplot(x='hero_names',y='total powers',data=df.head(20),palette="BuGn_d")
plt.title('Female Hero')
plt.xticks(rotation=60)
plt.show()

df=hero.merge(powers[['hero_names','total powers']],how='inner',right_on='hero_names',left_on='name')
sns.pairplot(df,kind='reg',size=4,hue='Gender')
print(np.corrcoef(df['total powers'],df['Weight']))
print(np.corrcoef(df['total powers'],df['Height']))
