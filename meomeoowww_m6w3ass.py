import pandas as pd #import pandas

import numpy as np #import numpy

import matplotlib.pyplot as plt #import matplotlib.pyplot with plt as alias

%matplotlib inline
df = pd.read_csv("../input/ecommerce-data/data.csv", encoding = 'ISO-8859-1') # import csv file
df.head(1)
df.info()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) #InvoiceDate to datetime

df = df.dropna() #drop NA
df[df.duplicated(keep=False)].sort_values(by=['InvoiceNo', 'StockCode']).head(2) # select duplicates and show header 2
print(str('We have {} duplicates.').format(len(df[df.duplicated(keep=False)].sort_values(by=['InvoiceNo', 'StockCode']))))
df = df.drop_duplicates()

df.shape
df['Total'] = df['Quantity'] * df['UnitPrice']
df.describe()
#first option

dfp = df[df['Quantity']>0]

dfn = df[df['Quantity']<=0]



#second option

dfd = df

dfd['Purchase'] = df['Quantity'].apply(lambda x: 1 if x > 0 else 0)
df.nunique()
print(str('As we can see we have {} unique customers for {} description products over {} countries.').format(df.nunique()['CustomerID'], df.nunique()['Description'], df.nunique()['Country']))
recod_df = pd.get_dummies(df[['Description', 'Country']], drop_first = True)

dfc = pd.concat([df, recod_df], axis=1)

dfc = dfc.drop(['InvoiceNo', 'StockCode', 'Description', 'Country', 'Purchase', 'InvoiceDate','CustomerID'], axis=1)
dfc.head(1)
from sklearn.preprocessing import normalize



a = dfc[['Quantity', 'UnitPrice', 'Total']]

dfc_scaled = normalize(a)

dfc_scaled = pd.DataFrame(dfc_scaled, columns=a.columns)

dfc_scaled.head(1)
y = dfc_scaled['Total']

y1 = dfc_scaled[['Total']]

X = dfc_scaled.drop('Total', axis=1)
#Using Kmeans we will try to find first the optimal number of clusters with elbow method

from sklearn.cluster import KMeans #import kmeans



listkm = [] # define an empty list to add inertia at each number of clusters



for i in range(1,8):

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(y1)

    listkm.append(km.inertia_)

    

# Plot it

plt.plot(range(1,8),listkm, marker ='s')

plt.title('Plot showing inertia versus number of clusters')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()
kms = KMeans(n_clusters=3, random_state=1).fit(X) #kmeans fitting to have our model

predict = kms.predict(X) #predicting

ctds = kms.cluster_centers_

print(ctds)
print("Original array:")

print(predict)

unique_elements, counts_elements = np.unique(predict, return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
plt.scatter(X.iloc[predict == 0, 0], X.iloc[predict == 0, 1], s = 50, c = 'green', label = 'Group 1')

plt.scatter(X.iloc[predict == 1, 0], X.iloc[predict == 1, 1], s = 50, c = 'yellow', label = 'Group 2')

plt.scatter(X.iloc[predict == 2, 0], X.iloc[predict == 2, 1], s = 50, c = 'red', label = 'Group 3')

plt.scatter(kms.cluster_centers_[:, 0], kms.cluster_centers_[:, 1], s = 100, c = 'purple', label = 'Centroids')

plt.title('Customer clusters on price and quantity')

plt.xlabel('Quantity')

plt.ylabel('Unit Price')

plt.legend()

plt.show()
#from sklearn.decomposition import PCA

#pca = PCA().fit(dfc)

#pca_ax2 = pca.transform(dfc)
#import scipy.cluster.hierarchy as sch



#dist = sch.distance.pdist(dfc, lambda u, v: u != v)

#merging = sch.linkage(df_scaled, method='ward')



#plt.figure(figsize=(10,10))

#sch.dendrogram(merging,leaf_font_size=6, leaf_rotation=90)
from wordcloud import WordCloud, STOPWORDS



df_word1 = df[predict == 0]['Description']

df_word2 = df[predict == 1]['Description']

df_word3 = df[predict == 2]['Description']



patchwork1 = " ".join(word for word in df_word1)

patchwork2 = " ".join(word for word in df_word2)

patchwork3 = " ".join(word for word in df_word3)



# Generate a word cloud image

wordcloud1 = WordCloud(background_color="white").generate(patchwork1)

wordcloud2 = WordCloud(background_color="white").generate(patchwork2)

wordcloud3 = WordCloud(background_color="white").generate(patchwork3)
#plot each cluster

plt.figure(figsize=(36, 12))

plt.subplots_adjust(top=1.2)



plt.subplot(131)

plt.imshow(wordcloud1, interpolation='bilinear')

plt.axis("off")

plt.title('Cluster 1 : Home furnitures', fontsize=30)

plt.subplot(132)

plt.imshow(wordcloud2, interpolation='bilinear')

plt.axis("off")

plt.title('Cluster 2 : Travel Bags and others', fontsize=30)

plt.subplot(133)

plt.imshow(wordcloud3, interpolation='bilinear')

plt.axis("off")

plt.title('Cluster 3 : Birthday events ', fontsize=30)



plt.suptitle('Wordcloud by Cluster', fontsize=70)

plt.show()