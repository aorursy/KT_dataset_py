import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("../input/googleplaystore.csv")

data.head()
data.count()
data['Category'].unique()
data[data['Category']=='1.9']
data.loc[10472]
data.loc[10472]=data.loc[10472].shift() # hole shift
#swap fisrt and second column
data['App'].loc[10472] = data['Category'].loc[10472]
data['Category'].loc[10472] = np.nan
data.loc[10472]
data[data.duplicated()].count()
data=data.drop_duplicates()
data[data.duplicated()].count()
data.dtypes
data['Rating'].unique()
data['Rating'] = pd.to_numeric(data['Rating'],errors='coerce')
data['Reviews'].unique()
data['Reviews'] = pd.to_numeric(data['Reviews'],errors='coerce')
data['Size'].unique()
data['Size'].replace('Varies with device', np.nan, inplace = True ) 
data['Size']=data['Size'].str.extract(r'([\d\.]+)', expand=False).astype(float) * \
    data['Size'].str.extract(r'([kM]+)', expand=False).fillna(1).replace(['k','M'],[1,1000]).astype(int)
data['Installs'].unique()
data['Installs']=data['Installs'].str.replace(r'\D','').astype(float)
data['Price'].unique()
data['Price']=data['Price'].str.replace('$','').astype(float)
plt.figure(figsize=(10,10))
g = sns.countplot(y="Category",data=data, palette = "Set2")
plt.title('Total apps of each Category',size = 20)
plt.figure(figsize=(10,10))
g = sns.barplot(x="Installs", y="Category", data=data, capsize=.6)
plt.title('Installations in each Category',size = 20)
data[data[['Installs']].mean(axis=1)>1e5]['Category'].unique()
plt.figure(figsize=(10,10))
plt.scatter( x=data['Rating'], y=data['Installs'] , color = 'blue')
g = sns.lineplot(x="Rating", y="Installs",color="red", data=data) 
plt.yscale('log')
plt.xlabel('Rating')
plt.ylabel('Installs')
plt.title('Rating-Installs (Scatter & line plot)',size = 20)
plt.show()
    

g = sns.lmplot(y="Installs",x="Reviews", data=data,size=(10))
plt.xscale('log')
plt.yscale('log')
plt.title('Reviews-Installs ',size = 20)
plt.figure(figsize=(10,10))
g = sns.boxplot(x="Installs", y="Size", data=data)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
plt.title('Installs-Size(kilobyte) ',size = 20)
plt.figure(figsize=(10,10))
ax = sns.barplot(y="Installs", x="Content Rating", data=data, capsize=.5)
ax.set_xticklabels(ax.get_xticklabels(),  ha="left")
plt.title('Content Rating-Installs',size = 20)

##pie plot
labels=data['Content Rating'].unique()
explode = (0.1, 0, 0, 0, 0)
size=list()
for content in labels:
    size.append(data[data['Content Rating']==content]['Installs'].mean())

##merging Unrated & Adults 
labels[4] = 'Unrated &\n Adults only 18+'
labels = np.delete(labels,5)
size[4]=size[4]+size[5]
size.pop()

plt.figure(figsize=(10,10))
colors = ['#ff6666','#66b3ff','#99ff99','#ffcc99', '#df80ff']
plt.pie(size, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.title('Percentage of Installs for each Content Rating',size = 20)
plt.show()
plt.figure(figsize=(10,10))

labels=['Apps with less than 100,000 Downloads', 'Apps with more than 100,000 Downloads']
size=list()
size.append(data['App'][data['Installs']<1e5].count()) 
size.append(data['App'][data['Installs']>=1e5].count()) 

labels_inner=['Free', 'Paid', 'Free', 'Paid']
size_inner=list()
size_inner.append(data['Type'][data['Type']=='Free'][data['Installs']<1e5].count()) 
size_inner.append(data['Type'][data['Type']=='Paid'][data['Installs']<1e5].count()) 
size_inner.append(data['Type'][data['Type']=='Free'][data['Installs']>=1e5].count())
size_inner.append(data['Type'][data['Type']=='Paid'][data['Installs']>=1e5].count()) 

colors = ['#99ff99', '#66b3ff']
colors_inner = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']

explode = (0,0) 
explode_inner = (0.1,0.1,0.1,0.1)

#outer pie
plt.pie(size,explode=explode,labels=labels, radius=3, colors=colors)
#inner pie
plt.pie(size_inner,explode=explode_inner,labels=labels_inner, radius=2, colors=colors_inner)
       
#Draw circle
centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.show()
g = sns.lineplot(x="Price", y="Installs", data=data)

plt.figure(figsize=(10,10))
g = sns.lineplot(x="Price", y="Installs", data=data)
g.set(xlim=(0, 10))
plt.title('Price (0-10$) - Installs',size = 20)
corpus=list(data['App'])
vectorizer = CountVectorizer(max_features=50, stop_words='english')
X = vectorizer.fit_transform(corpus)
names=vectorizer.get_feature_names()
values=X.toarray().mean(axis=0)

plt.figure(figsize=(15,15))
sns.barplot(x=values, y=names, palette="viridis")
plt.title('Top 50 most frequently occuring words',size = 20)