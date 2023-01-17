import pandas as pd
import numpy as np
import string

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn import datasets

import requests
from bs4 import BeautifulSoup

from wordcloud import WordCloud, STOPWORDS 
import networkx as nx

SEED = 2020
np.random.seed(SEED)
# Let's generate random data using numpy's random module
df = pd.DataFrame(dict(time=pd.date_range('2020-06-24',periods=100),
                       stock_price=abs(np.random.randn(100).cumsum())))

fig, ax = plt.subplots(figsize=(20,8))
sns.lineplot(x='time',y='stock_price',data=df, ax=ax)
plt.title('Stock Market Data',fontsize=15)
# Let's generate Data related to the sales of mobile phones
sales = np.random.choice(['Apple','Samsung','Oneplus','Oppo','Vivo','Nokia'],5000)

fig, ax = plt.subplots(figsize=(20,8))
sns.countplot(x=sales,ax=ax)
plt.title('Total Number of Mobile Phones sold',fontsize=15)
plt.xlabel('Brand')
# Let's generate random data for this
cases = pd.Series(np.random.choice(['Maharashtra','Delhi','Gujarat','Madhya Pradesh','Kerala'],p=[0.5,0.2,0.15,0.1,0.05],size=5000))

fig, ax = plt.subplots(figsize=(20,8))
ax = plt.pie(labels=cases.value_counts().index.values,x=cases.value_counts().values,startangle=90)
plt.title('Distribution of Corona Virus Cases in Several States',fontsize=15)
age = np.random.randint(0,100,size=5000)

fig, ax = plt.subplots(figsize=(20,8))
sns.distplot(age,kde=False,ax=ax)
plt.title('Distribution of Age of the patients',fontsize=15)
plt.xlabel('Age')
weight, height = datasets.make_regression(n_samples=500,n_features=1,noise=5)

# Some preprocessing to give it some realistic touch
weight = abs(weight)
height = abs(height)
weight /= np.max(weight)
height /= np.max(height)
weight = weight*97 + 3
height = height*6 + 1


fig, ax = plt.subplots(figsize=(20,8))
sns.scatterplot(weight.reshape([-1,]),height.reshape([-1,]),ax=ax)
plt.title('Height vs Weight',fontsize=15)
plt.xlabel('Weight in kgs')
plt.ylabel('Height in feets')
x = np.random.rand(5000)

fig, ax = plt.subplots(figsize=(20,8))
sns.kdeplot(x,shade=True,ax=ax)
plt.title('Kernel Density Estimation of x',fontsize=15)
plt.xlabel('x')
# Let's generate random data for this
customers = np.random.choice(['John','Peter','Arjun','Mukesh','Rohit'],5000)
spendings = np.random.randint(1,100000,5000)

# Putting outliers in the data
spendings[np.random.choice(list(range(5000)),replace=False,size=20)] = np.random.choice(list(range(100000,200000)),size=20)

fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(customers,spendings,ax=ax)
plt.title('Distribution of Spendings of customers',fontsize=15)
plt.xlabel('Customers')
plt.ylabel('Spendings ')
url = 'https://www.forbes.com/sites/cognitiveworld/2019/07/23/understanding-explainable-ai/#763dbbc87c9e'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

textdata = []
for p in soup.find_all('span'):
    textdata += p.text.split(" ")
    
stopwords = set(STOPWORDS)

textdata = list(map(lambda x: x.lower(),textdata))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10, max_words=50).generate(" ".join(textdata))

fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(wordcloud, interpolation='bilinear',aspect='auto')
ax.axis("off")
plt.show()
# Declaring Graph
G = nx.Graph()

# Using lower case alphabets as nodes
nodes = list(string.ascii_lowercase)

# Adding nodes 
G.add_nodes_from(nodes)

# Adding edges randomly
for i in range(50):
    edge = np.random.choice(nodes,2,replace=False)
    G.add_edge(*edge)

plt.figure(figsize=(25,25))

options = {
    'edge_color': '#FFDEA2',
    'width': 1,
    'with_labels': True,
    'font_weight': 'regular',
}

nx.draw(G,node_size=[700 for node in G], **options)
ax = plt.gca()
plt.show()
# Let's generate random data
df = pd.DataFrame(data=np.random.normal(size=(500,10)),columns=list(string.ascii_uppercase[:10]))

# Correlation matrix
mat = df.corr()

# Mask for the lower triangle
mask = np.triu(np.ones_like(mat, dtype=np.bool))

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(mat,mask=mask,ax=ax)
plt.title('Correlation Matrix',fontsize=15)