import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dff  = pd.read_csv("../input/ecommerce-data/data.csv", encoding="ISO-8859-1", dtype={'CustomerID': str,'InvoiceID': str})

dff
dff.info()
dff.shape
duplicated = dff.duplicated().sum()

print(duplicated)

dff.drop_duplicates(inplace= True)
#transfore the data type

dff['InvoiceDate']= pd.to_datetime(dff['InvoiceDate'])

dff.describe()
#we have negtive value for quantity and price

df = dff[(dff['Quantity']>0) & (dff['UnitPrice']>0)]
df[['Quantity', 'UnitPrice']].describe()
fig, ax = plt.subplots(figsize=(10,8))

ax.scatter(df['Quantity'], df['UnitPrice'])

ax.set_xlabel('Quantity')

ax.set_ylabel('UnitPrice')

plt.show()
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(df[['Quantity','UnitPrice']]))

df = df[(z < 3).all(axis=1)]
df.describe()
df = df[(df['Quantity']>=0) | (df['UnitPrice']>=0)]
sns.boxplot(df['Quantity'])
sns.boxplot(df['UnitPrice'])
dff.shape
df.isna().sum()
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['Hour']=df['InvoiceDate'].dt.hour

df['Month']=df['InvoiceDate'].dt.month

df['Weekdays']= df['InvoiceDate'].dt.weekday
df['Hour'].unique()
rfm = df.copy()
rfm
#because we are looking at the customer and not the product we drop InvoiceNo and Stcokcode

ab = df.groupby('CustomerID').agg({'InvoiceDate': 'min','TotalAmount': 'sum'})

ab.rename(columns={'InvoiceDate': "Recency",('InvoiceDate', 'nunique'): "Frequency","TotalAmount": 'Monetization'},  inplace = True)

frequency = df.groupby('CustomerID').agg({'InvoiceDate':'nunique'})

frequency.rename(columns={'InvoiceDate': "Frequency"}, inplace = True)



rfm = pd.merge(ab,frequency, on='CustomerID')



#rfm.rename(columns={('InvoiceDate',     'min'): "Recency",('InvoiceDate', 'nunique'): "Frequency","TotalAmount": 'Monetization'})
a = rfm.reset_index()

rfm.rename(columns={'InvoiceDate': "Recency",'InvoiceDate': "Frequency","TotalAmount": 'Monetization'})
rfm.describe()
#We save the most recent date to, then , calulate the recency

e = df['InvoiceDate'].min()

print('minimun :'+ str(e))

rfm.info()
###create receny, cad calculate the last time he bought something compare to e

rfm['Recency'] = rfm['Recency'].apply(lambda x : (x - e).days)
#why do we rank it and how?

rfm['Rank_Recency'] = pd.qcut( rfm['Recency'],q=5, labels = range(6, 1, -1))

rfm['Rank_Recency'] = pd.to_numeric(rfm['Rank_Recency'])
def freq(x):

    if x ==1:

        return 1

    elif x == 2:

        return 2

    elif x == 3:

        return 3

    elif x == 4:

        return 4

    else: 

        return 5



rfm['Rank_Frequency'] =rfm['Frequency'].apply(freq)



#rfm['Rank_Frequency'] = pd.qcut( rfm['Frequency'],q=5, labels = range(1, 6, 1))

#rfm['Rank_Frequency'] = pd.to_numeric(rfm['Rank_Frequency'])
rfm['Rank_Monetization'] = pd.qcut( rfm['Monetization'],q=5, labels = range(1, 6, 1))

rfm['Rank_Monetization'] = pd.to_numeric(rfm['Rank_Monetization'])
rfm['RFM_Score'] = rfm['Rank_Recency'].astype(str)+ rfm['Rank_Frequency'].astype(str) + rfm['Rank_Monetization'].astype(str)

rfm['Score'] = rfm['Rank_Recency']+ rfm['Rank_Frequency']+ rfm['Rank_Monetization']
rfm.describe()
rfm.info()
def client_segment(x):

    if x == 15:

        return 'Champions'

    elif  x >= 14:

        return 'Loyal Customers'

    elif  x >= 11:

        return 'Can’t Lose Them'

    elif  x >= 9:

        return 'Potential Loyalist'

    elif  x >= 7:

        return 'Promising'

    elif  x >= 6:

        return 'Needs Attention'

    elif  x >= 5:

        return 'At Risk'

    else:

        return 'Lost'
rfm['Clients'] = rfm['Score'].apply(client_segment)

rfm

clients = rfm[['Clients', 'Frequency', 'Monetization', 'Recency']].groupby('Clients').median()

clients.reset_index(inplace = True)
ax = sns.barplot(x="Frequency", y="Clients", data=clients)

ax.set_ylabel('Clients')

ax.set_title('Median Visits')
ax = sns.barplot(x="Monetization", y="Clients", data=clients)

ax.set_ylabel('Clients')

ax.set_title('Median Expenditure')
ax = sns.barplot(x="Recency", y="Clients", data=clients)

ax.set_ylabel('Clients')

ax.set_title('Median time from last shop ')
rfm['Clients'].unique()
import squarify



squarity =rfm['Clients'] .value_counts()

color=['grey','orange','pink','purple', 'brown', 'blue', 'green', 'red']





fig = plt.gcf()

ax = fig.add_subplot()

fig.set_size_inches(12, 8)

squarify.plot(sizes= squarity , 

              label=['Promising',

                     'Can’t Lose Them',

                     'Potential Loyalist',

                     'Loyal Customer',

                     'Promising', 

                     'Needs Attention',

                     'At Risk',

                     'Champions',

                     'Lost',] ,color = color, alpha=0.5,)

plt.title("RFM Segments",fontsize=18,fontweight="bold")

plt.axis('off')

plt.show()
squarity

clients
rfm['Clients'].value_counts()
cluster = rfm.drop(['Rank_Recency','Rank_Frequency','Rank_Monetization','RFM_Score','Score','Clients'], axis = 1)

from sklearn.preprocessing import StandardScaler,  MinMaxScaler

from sklearn.cluster import KMeans

from sklearn import metrics

from sklearn.cluster import AgglomerativeClustering
X= MinMaxScaler().fit_transform(cluster)

#x = StandardScaler().fit_transform(X)
sum_of_squared_distances = []

K = range(1,15)

for k in K:

    k_means = KMeans(n_clusters=k)

    model = k_means.fit(X)

    sum_of_squared_distances.append(k_means.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('sum_of_squared_distances')

plt.title('Elbow Method')

plt.show()
from sklearn.metrics import silhouette_score



silhouette_scores = [] 



for n_cluster in range(2, 8):

    silhouette_scores.append( 

        silhouette_score(X, KMeans(n_clusters = n_cluster).fit_predict(X))) 

    

# Plotting a bar graph to compare the results 

k = [2, 3, 4, 5, 6,7] 

plt.bar(k, silhouette_scores) 

plt.xlabel('Number of clusters', fontsize = 10) 

plt.ylabel('Silhouette Score', fontsize = 10) 

plt.show() 
k_means_5 = KMeans(n_clusters=6)

model = k_means_5.fit(X)

y_hat_5 = k_means_5.predict(X)

labels_5 = k_means_5.labels_

metrics.silhouette_score(X, labels_5, metric = 'euclidean')

#metrics.calinski_harabasz_score(X, labels_5)
cluster['Cluster'] = labels_5
table = cluster.groupby('Cluster').agg({'Recency': 'mean', 'Frequency':'mean', 'Monetization': 'mean'})

table['Number of user'] = cluster['Cluster'].value_counts()

table