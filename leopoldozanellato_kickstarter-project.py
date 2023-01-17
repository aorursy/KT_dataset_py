# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install us
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import us
import string
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
iowa_file_path = '../input/kickstarter-project-statistics/most_backed.csv'

df = pd.read_csv(iowa_file_path, index_col=0)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.head()
df.describe()
df.info()
maximo = df[df['num.backers'] == df['num.backers'].max()] # get projetc with max num.backers
df[df['num.backers'] == df['num.backers'].max()]
print(f"Blurb: {maximo['blurb'].iloc[0]}") # blurb
minimo = df[df['num.backers'] == df['num.backers'].min()] # get projetcs with min num.backers
minimo['blurb'].count()
minimo
def autolabel(rects):
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom')
"""Attach a text label above each bar in *rects*, displaying its height.I'm going to use in some charts"""
categors = df['category'].value_counts()   # Categories with more Projects, top 10
top10 = categors[:10]

fig , ax = plt.subplots(figsize=(12,7))
rect1 = ax.bar(x=top10.index,height=top10.values)
ax.set_xticklabels(top10.index,rotation=25,)
ax.set_title("Top 10 Categories with more Projects")
ax.set_ylabel('Projects')
ax.set_xlabel('Categories')
autolabel(rect1)
df['currency'].unique()
plt.figure(figsize=(8,8))
df['currency'].hist()
plt.title('Projects by Currency')
numbackers = df[['category','num.backers']].groupby('category').sum().sort_values(by='num.backers',ascending=False)
numbackerstop = numbackers[:10]      #categories by num.backers
plt.figure(figsize=(10,7))
sns.barplot(x=numbackerstop.index,y=numbackerstop['num.backers'],data=numbackers)
plt.xticks(rotation=25)
plt.title('Categories by num.backers')
plt.figure(figsize=(7,7))
sns.jointplot(y=df['amt.pledged'], x=df['num.backers'], data=df, kind='reg')
plt.figure(figsize=(7,7))
sns.jointplot(y=df['goal'], x=df['num.backers'], data=df, kind='reg')
location = df['location'].value_counts() # projects by cities
location_cities = pd.DataFrame({'Cidades':location.index,'Quantidade':location.values})
location_cities.set_index('Cidades',inplace=True)
top_location = location_cities[:40] # top 40 cities
# plt.bar(height=top_location['Quantidade'], x=top_location.index, data=top_location)
plt.figure(figsize=(15,10))
sns.barplot(x=top_location['Quantidade'], y=top_location.index,data=top_location)
plt.title('Cities with more Projects')
plt.ylabel('Cities')
# I want to know from which country are the projetcs, so I did this function.
# The first will get the state and see if its a American state
# The second will get if the project is from EUA or NOT
def get_country(x):
    country = str(us.states.lookup(x))
    if country == 'None':
        return x
    else:
        return 'EUA'
def get_coun(x):
    if x == 'EUA':
        return 'EUA'
    else:
        return 'NOT EUA'
df['País'] = df['location'].apply(lambda x: x.split(sep=','))
df['País'] = df['País'].apply(lambda x: (x[1]).strip())
df['País'] = df['País'].apply(lambda x: get_country(x))
df['World'] = df['País'].apply(lambda x: get_coun(x))
location_country = df['País'].value_counts()
location_country  #countrys with more projects
location_country = location_country[:10]
location_country1 = pd.DataFrame({'Países':location_country.index,'Quantidade':location_country.values})
location_country1.set_index('Países',inplace=True) #transform in a Dataframe
fig , ax = plt.subplots(figsize=(12,7))
rect1 = ax.bar(x=location_country1.index, height='Quantidade', data=location_country1)
plt.title('Projects by Country Top 10')
ax.set_ylabel('Projects')
ax.set_xlabel('Countries')
autolabel(rect1)

plt.figure(figsize=(10,10))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='white',stopwords=stopwords,max_words=200,max_font_size=40,random_state=42).generate(str(df['title']))
plt.imshow(wordcloud)
plt.axis('off')

# Now we can look projects by Amt.Pledged, and here is the best one

df[df['amt.pledged']==df['amt.pledged'].max()]
pledged = df[['amt.pledged','category']].groupby('category').sum().sort_values(by='amt.pledged',ascending = False)
pledged['%'] = pledged['amt.pledged'] / sum(pledged['amt.pledged']) * 100
top_pledged = pledged[:10]
pledged.head()
plt.figure(figsize=(12,8))
sns.barplot(x=top_pledged.index , y=top_pledged['amt.pledged'])
plt.title('Categories with more Amt.Pledged')
plt.xticks(rotation=25)
sns.pairplot(df,hue='World')

# now we can see a Pair Plot if the project is from eua or not
