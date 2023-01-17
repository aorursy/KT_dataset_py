!ls ../input/data-cluster-ta
# Basic Operations

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



# Preprocessing 

from mlxtend.preprocessing import TransactionEncoder



# Mining and Stuff

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

from sklearn.cluster import KMeans
df = pd.read_excel("../input/tanah-airku-data-baru/databaru.xlsx", header=None)

df = df.replace(to_replace ='_.*', value = '', regex = True);

df.sample(10)
## Join the array so we can see the frequent item



temp = (df.to_numpy().ravel())

maxFreq = pd.DataFrame(temp)
plt.rcParams['figure.figsize'] = (10, 10)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(maxFreq[0]))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Items',fontsize = 20)

plt.show()
# Frequent Item with Barchart



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))

maxFreq[0].value_counts().plot.bar()

plt.title('frequency of most popular items', fontsize = 20)

plt.xticks(rotation = 90 )

plt.grid()

plt.show()
# Making an array of items from the transscation data

trans = []

for i in range(0, 30):

    trans.append([str(df.values[i,j]) for j in range(0, 10)])



# conveting it into an numpy array

trans = np.array(trans)



# checking the shape of the array

trans
# Encoding Process



te = TransactionEncoder()

te_ary = te.fit(trans).transform(trans)

processed = pd.DataFrame(te_ary, columns = te.columns_)



# getting the shape of the data

processed
#freq_item = apriori(processed, min_support = 0.5, use_colnames = True)

freq_item = apriori(processed, min_support = 0.3, use_colnames=True)

freq_item['length'] = freq_item['itemsets'].apply(lambda x: len(x))

freq_item
freq_item[ (freq_item['length'] >= 2) &

                   (freq_item['support'] >= 0.7)].sort_values(by=['support'], ascending=False)
## Load Data



dfc = pd.read_excel("../input/data-cluster-ta/datacluster.xlsx",sheet_name=1, header=None);

dfc.columns = ["Nama","Waktu"]

dfc
## Preprocess

## Get Frequency and Average Time



temp = dfc.groupby("Nama").mean().reset_index()

test = dfc.groupby("Nama").size().reset_index()



temp["Count"] = test[0]



temp.head()
## Slice Data 

x = temp.iloc[:,1:3]

x.head()
kmeans = KMeans(4)

kmeans.fit(x)
clusted = kmeans.fit_predict(x)

clusted
data_cluster = temp.copy()

data_cluster["Cluster"] = clusted

data_cluster
fig, ax = plt.subplots()



plt.scatter(temp["Waktu"],temp["Count"], c=data_cluster["Cluster"], s=500, cmap="gist_rainbow")

# plt.figure(figsize=(200,200))

plt.title("Cluster", size=36)

plt.ylabel("View Rate", size=20)

plt.xlabel("Access Time", size=20)

plt.grid()



waktu = np.array(temp["Waktu"])

count = np.array(temp["Count"])

n = np.array(data_cluster["Cluster"]);



for i, txt in enumerate(n):

    plt.annotate(txt,xy=(waktu[i],count[i]))

    

plt.show()