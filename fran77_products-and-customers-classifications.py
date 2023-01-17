# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding = "ISO-8859-1")
data.head()
data['No'] = data['InvoiceNo'].str[0]
data['No'].value_counts()
data[data['No'] == 'C'].head(10)
data[data['No'] == 'C'].Quantity.describe()
# Products with start with C are returned articles
data[data['No'] == 'A']
# Bad Operation ?
data['Code'] = data['StockCode'].str[0]
data['Code'].value_counts()
# Understanding letters code
data[data['Code'] == 'P'].Description.value_counts()
# P : POSTAGE Products
data[data['Code'] == 'D'].Description.value_counts()
data[(data['Code'] == 'D') & (data['StockCode'] != 'D') & (data['StockCode'] != 'DOT')]
# DCGS = Discount
data[data['Code'] == 'C'].Description.value_counts()
data[data['Code'] == 'C'].Country.value_counts()
data[data['Code'] == 'M'].Description.value_counts()
data[data['Code'] == 'B'].Description.value_counts()
data[data['Code'] == 'S'].Description.value_counts()
data[data['Code'] == 'A'].Description.value_counts()
data[data['Code'] == 'g'].Description.value_counts()
# G = Gift
data[data['Code'] == 'm'].Description.value_counts()
# Change this stock code

data.loc[data.Code == 'm', 'Code'] = 'M'
data.UnitPrice.describe()
# Understanding negative values

data[data.UnitPrice < 0]
data[data.UnitPrice >= 0].UnitPrice.describe()
# Some articles are free (Unit Price = 0)
# Distribution of prices < 10

plt.figure(figsize=(10,6))

sns.distplot(data[(data.UnitPrice < 10) & (data.UnitPrice >= 0)].UnitPrice, hist = False)
data.Quantity.describe()
# Distribution of quantity

plt.figure(figsize=(10,6))

sns.distplot(data[data.Quantity > 0].Quantity, hist = False)
data.Country.value_counts()[:10]
# Negative quantities as positive

data['Quantity'] = abs(data['Quantity'])
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
# Distribution of quantity

plt.figure(figsize=(10,6))

sns.distplot(data[data.TotalPrice > 0].TotalPrice, hist = False)
data['Date'] = pd.to_datetime(data['InvoiceDate'])
data['Date'].head(10)
data['year'] = data['Date'].dt.year

data['month'] = data['Date'].dt.month

data['day'] = data['Date'].dt.day

data['hour'] = data['Date'].dt.hour
data.year.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(data.month)
# Majority in November and December (for Christmas)
plt.figure(figsize=(10,6))

sns.countplot(data.day)
plt.figure(figsize=(10,6))

sns.countplot(data.hour)
# check missing values for each column 

data.isnull().sum().sort_values(ascending=False)
prod = data[~data.Description.isnull()]
prod['Description'].head(10)
from wordcloud import WordCloud



wordcloud = WordCloud(max_words=1000,margin=0).generate(' '.join(prod['Description']))

plt.figure(figsize = (15, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# fill description with nan



prod['text'] = prod['Description'].fillna('')
# lower description



prod['text'] = prod['text'].str.lower()
# stopwords



from nltk.corpus import stopwords

stop = stopwords.words('english')



prod['text'] = prod['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
from wordcloud import WordCloud



wordcloud = WordCloud(max_words=1000,margin=0).generate(' '.join(prod['text']))

plt.figure(figsize = (15, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans



vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(prod['text'])
# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    ssd.append(km.inertia_)
# Plot sum of squared distances / elbow method

plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
# Best number of clusters is 5
# Create and fit model

kmeans = KMeans(n_clusters=5)

model = kmeans.fit(X)
pred = model.labels_

prod['Cluster_prod'] = pred
prod = prod[['Description', 'text', 'Cluster_prod']]
prod.head()
fig = plt.figure(figsize = (20, 15))

for c in range(len(prod['Cluster_prod'].unique())):

    ax = fig.add_subplot(3,2,c+1)

    ax.set_title('Cluster %d'%c)

    cluster = prod[prod.Cluster_prod == c]

    wordcloud = WordCloud( max_words=1000,margin=0).generate(' '.join(cluster['text']))

    ax.imshow(wordcloud)

    ax.axis("off")
# Cluster A : Bags

# Cluster B : Signs

# Cluster C : Kitchen

# Cluster D : Decoration

# Cluster E : Retrospots
data.columns
cus = data[['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country', 'TotalPrice']]
cus = cus[~cus.CustomerID.isnull()]
cus_prod = cus.groupby('CustomerID')['Quantity'].sum().reset_index()

cus_prod.columns = ['CustomerID', 'TotalProducts']
cus = cus.merge(cus_prod, on='CustomerID')
cus['InvoiceDate'] = cus['InvoiceDate'].str.split(' ').str[0]
transactions = cus[['CustomerID', 'InvoiceDate']].drop_duplicates()
transactions = transactions.groupby('CustomerID')['InvoiceDate'].count().reset_index()

transactions.columns = ['CustomerID', 'Transactions']
cus = cus.merge(transactions, on='CustomerID')
cus = cus.drop(['Quantity', 'UnitPrice', 'InvoiceDate'], axis=1)
# drop duplicates

cus = cus.drop_duplicates()
len(cus)
cus.head()
# Label encoder

from sklearn import preprocessing



le = preprocessing.LabelEncoder()

cus['Country'] = le.fit_transform(cus.Country.values)
# cus['No'] = le.fit_transform(cus.No.values)

# cus['Code'] = le.fit_transform(cus.Code.values)
# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(cus)

    ssd.append(km.inertia_)
# Plot sum of squared distances / elbow method

plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
# Best number of clusters is 3
# Create and fit model

kmeans = KMeans(n_clusters=3)

model = kmeans.fit(cus)
pred = model.labels_

cus['Cluster_cus'] = pred
cus.head()
# Create PCA for data visualization / Dimensionality reduction to 2D graph

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_model = pca.fit_transform(cus)

cus_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])

cus_transform['Cluster_cus'] = pred
plt.figure(figsize=(10,10))

g = sns.scatterplot(data=cus_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:3], hue='Cluster_cus')

title = plt.title('Personality Clusters with PCA')
customers = cus.groupby('Cluster_cus').mean()

customers = customers.reset_index()
customers.columns
customers[['Cluster_cus', 'CustomerID', 'Country', 'TotalPrice', 'TotalProducts', 'Transactions']]
# Cluster A : From UK : few transactions and products, small total price

# Cluster B : Regular with a good amount of products (weekly shopping)

# Cluster C : Big amount of products, maybe for stock
prod_cust = data.merge(cus[['Cluster_cus', 'CustomerID']], on='CustomerID')
prod_cust = prod_cust.drop_duplicates()
prod_cust['text'] = prod_cust['Description'].fillna('')

prod_cust['text'] = prod_cust['text'].str.lower()

prod_cust['text'] = prod_cust['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
fig = plt.figure(figsize = (20, 15))

for c in range(len(prod_cust['Cluster_cus'].unique())):

    ax = fig.add_subplot(3,1,c+1)

    ax.set_title('Cluster %d'%c)

    cluster = prod_cust[prod_cust.Cluster_cus == c]

    wordcloud = WordCloud(max_words=1000,margin=0).generate(' '.join(cluster['text']))

    ax.imshow(wordcloud)

    ax.axis("off")