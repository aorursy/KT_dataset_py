# Importation of useful libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import seaborn as sns



import random 

import datetime as dt

import re

import pickle

import nltk, warnings

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from string import digits, punctuation





from scipy.stats import chi2_contingency



from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import preprocessing, model_selection, metrics, feature_selection

from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn import neighbors, linear_model, svm, tree, ensemble

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



from wordcloud import WordCloud, STOPWORDS



warnings.filterwarnings("ignore")

plt.style.use('bmh')

%matplotlib inline

import os

print(os.listdir("../input"))
# Importing the database 



data = pd.read_csv("../input/data.csv", encoding="ISO-8859-1", dtype={'CustomerID': str,'InvoiceID': str})
data.head(5)
data.info()
plt.figure(figsize=(5, 5))

data.isnull().mean(axis=0).plot.barh()

plt.title("Ratio of missing values per columns")
nan_rows = data[data.isnull().T.any().T]

nan_rows.head(5)
data[data['InvoiceNo']== '536414']
data[data['InvoiceNo']== '536544'][:5]
data = data.dropna(subset=["CustomerID"])
plt.figure(figsize=(5, 5))

data.isnull().mean(axis=0).plot.barh()

plt.title("Ratio of missing values per columns")
print('Dupplicate entries: {}'.format(data.duplicated().sum()))

data.drop_duplicates(inplace = True)
data.Country.nunique()
customer_country=data[['Country','CustomerID']].drop_duplicates()

customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
data.describe()
data[(data['Quantity']<0)].head(5)
# Constucting a basket for later use

temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()

nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})
nb_products_per_basket.InvoiceNo = nb_products_per_basket.InvoiceNo.astype(str)

nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))

len(nb_products_per_basket[nb_products_per_basket['order_canceled']==1])/len(nb_products_per_basket)*100
nb_products_per_basket[nb_products_per_basket['order_canceled']==1][:5]
data[data['CustomerID'] == '12346']
test = data[data['Quantity'] < 0][['CustomerID','Quantity',

                                                   'StockCode','Description','UnitPrice']]

for index, col in  test.iterrows():

    if data[(data['CustomerID'] == col[0]) & (data['Quantity'] == -col[1]) 

                & (data['Description'] == col[2])].shape[0] == 0: 

        print(test.loc[index])

        print('Our initial hypothesis is wrong')

        break
data[data['CustomerID'] == '14527'].head(5)
data_check = data[(data['Quantity'] < 0) & (data['Description'] != 'Discount')][

                                 ['CustomerID','Quantity','StockCode',

                                  'Description','UnitPrice']]



for index, col in  data_check.iterrows():

    if data[(data['CustomerID'] == col[0]) & (data['Quantity'] == -col[1]) 

                & (data['Description'] == col[2])].shape[0] == 0: 

        print(index, data_check.loc[index])

        print('The second hypothesis is also wrong')

        break
data[(data['CustomerID'] == '15311') & (data['Description'] == 'SET OF 3 COLOURED  FLYING DUCKS')]
df_cleaned = data.copy(deep = True)

df_cleaned['QuantityCanceled'] = 0



entry_to_remove = [] ; doubtfull_entry = []



for index, col in  data.iterrows():

    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        

    df_test = data[(data['CustomerID'] == col['CustomerID']) &

                         (data['StockCode']  == col['StockCode']) & 

                         (data['InvoiceDate'] < col['InvoiceDate']) & 

                         (data['Quantity']   > 0)].copy()

    #_________________________________

    # Cancelation WITHOUT counterpart

    if (df_test.shape[0] == 0): 

        doubtfull_entry.append(index)

    #________________________________

    # Cancelation WITH a counterpart

    elif (df_test.shape[0] == 1): 

        index_order = df_test.index[0]

        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']

        entry_to_remove.append(index)        

    #______________________________________________________________

    # Various counterparts exist in orders: we delete the last one

    elif (df_test.shape[0] > 1): 

        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        

        for ind, val in df_test.iterrows():

            if val['Quantity'] < -col['Quantity']: continue

            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']

            entry_to_remove.append(index) 

            break    
print("entry_to_remove: {}".format(len(entry_to_remove)))

print("doubtfull_entry: {}".format(len(doubtfull_entry)))
df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)

df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)

remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]

print("nb of entries to delete: {}".format(remaining_entries.shape[0]))

remaining_entries[:5]
df_cleaned.drop(remaining_entries.index, axis = 0, inplace = True)
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()

list_special_codes
df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'POST']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'D']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'C2']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'M']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'BANK CHARGES']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'PADS']

df_cleaned = df_cleaned[df_cleaned['StockCode']!= 'DOT']
df_cleaned.describe()
df_cleaned[(df_cleaned['UnitPrice'] == 0)].head(5)
def unique_counts(data):

   for i in data.columns:

       count = data[i].nunique()

       print(i, ": ", count)

unique_counts(df_cleaned)
# Total price feature



df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
revenue_per_countries = df_cleaned.groupby(["Country"])["TotalPrice"].sum().sort_values()

revenue_per_countries.plot(kind='barh', figsize=(15,12))

plt.title("Revenue per Country")
No_invoice_per_country = df_cleaned.groupby(["Country"])["InvoiceNo"].count().sort_values()

No_invoice_per_country.plot(kind='barh', figsize=(15,12))

plt.title("Number of Invoices per Country")
le = LabelEncoder()

le.fit(df_cleaned['Country'])
l = [i for i in range(37)]

dict(zip(list(le.classes_), l))
df_cleaned['Country'] = le.transform(df_cleaned['Country'])
with open('labelencoder.pickle', 'wb') as g:

    pickle.dump(le, g)
df_cleaned.head(5)
df_cleaned['InvoiceDate'].min()
df_cleaned['InvoiceDate'].max()
# I'll just fix the date to be one day after the last entry in the databse



NOW = dt.datetime(2011,12,10)

df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
custom_aggregation = {}

custom_aggregation["InvoiceDate"] = lambda x:x.iloc[0]

custom_aggregation["CustomerID"] = lambda x:x.iloc[0]

custom_aggregation["TotalPrice"] = "sum"





rfmTable = df_cleaned.groupby("InvoiceNo").agg(custom_aggregation)
rfmTable["Recency"] = NOW - rfmTable["InvoiceDate"]

rfmTable["Recency"] = pd.to_timedelta(rfmTable["Recency"]).astype("timedelta64[D]")
rfmTable.head(5)
custom_aggregation = {}



custom_aggregation["Recency"] = ["min", "max"]

custom_aggregation["InvoiceDate"] = lambda x: len(x)

custom_aggregation["TotalPrice"] = "sum"



rfmTable_final = rfmTable.groupby("CustomerID").agg(custom_aggregation)
rfmTable_final.columns = ["min_recency", "max_recency", "frequency", "monetary_value"]
rfmTable_final.head(5)
first_customer = df_cleaned[df_cleaned['CustomerID']=='12747']

first_customer.head(5)
quantiles = rfmTable_final.quantile(q=[0.25,0.5,0.75])

quantiles = quantiles.to_dict()
segmented_rfm = rfmTable_final
def RScore(x,p,d):

    if x <= d[p][0.25]:

        return 1

    elif x <= d[p][0.50]:

        return 2

    elif x <= d[p][0.75]: 

        return 3

    else:

        return 4

    

def FMScore(x,p,d):

    if x <= d[p][0.25]:

        return 4

    elif x <= d[p][0.50]:

        return 3

    elif x <= d[p][0.75]: 

        return 2

    else:

        return 1
segmented_rfm['r_quartile'] = segmented_rfm['min_recency'].apply(RScore, args=('min_recency',quantiles,))

segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))

segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))

segmented_rfm.head()
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)

segmented_rfm.head()
segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False)
segmented_rfm.head(5)
segmented_rfm = segmented_rfm.reset_index()
segmented_rfm.head(5)
df_cleaned = pd.merge(df_cleaned,segmented_rfm, on='CustomerID')
df_cleaned.columns
df_cleaned = df_cleaned.drop(columns=['r_quartile', 'f_quartile', 'm_quartile'])
df_cleaned['Month'] = df_cleaned["InvoiceDate"].map(lambda x: x.month)
df_cleaned['Month'].value_counts()
df_cleaned['Weekday'] = df_cleaned["InvoiceDate"].map(lambda x: x.weekday())

df_cleaned['Day'] = df_cleaned["InvoiceDate"].map(lambda x: x.day)

df_cleaned['Hour'] = df_cleaned["InvoiceDate"].map(lambda x: x.hour)
df_cleaned.head(5)
X = df_cleaned["Description"].unique()



stemmer = nltk.stem.porter.PorterStemmer()

stopword = nltk.corpus.stopwords.words('english')



def stem_and_filter(doc):

    tokens = [stemmer.stem(w) for w in analyzer(doc)]

    return [token for token in tokens if token.isalpha()]



analyzer = TfidfVectorizer().build_analyzer()

CV = TfidfVectorizer(lowercase=True, stop_words="english", analyzer=stem_and_filter, min_df=0.00, max_df=0.3)  # we remove words if it appears in more than 30 % of the corpus (not found stopwords like Box, Christmas and so on)

TF_IDF_matrix = CV.fit_transform(X)

print("TF_IDF_matrix :", TF_IDF_matrix.shape, "of", TF_IDF_matrix.dtype)
svd = TruncatedSVD(n_components = 100)

normalizer = Normalizer(copy=False)



TF_IDF_embedded = svd.fit_transform(TF_IDF_matrix)

TF_IDF_embedded = normalizer.fit_transform(TF_IDF_embedded)

print("TF_IDF_embedded :", TF_IDF_embedded.shape, "of", TF_IDF_embedded.dtype)
score_tfidf = []



x = list(range(5, 155, 10))



for n_clusters in x:

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)

    kmeans.fit(TF_IDF_embedded)

    clusters = kmeans.predict(TF_IDF_embedded)

    silhouette_avg = silhouette_score(TF_IDF_embedded, clusters)



    rep = np.histogram(clusters, bins = n_clusters-1)[0]

    score_tfidf.append(silhouette_avg)
plt.figure(figsize=(20,16))



plt.subplot(2, 1, 1)

plt.plot(x, score_tfidf, label="TF-IDF matrix")

plt.title("Evolution of the Silhouette Score")

plt.legend()
n_clusters = 135



kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30, random_state=0)

proj = kmeans.fit_transform(TF_IDF_embedded)

clusters = kmeans.predict(TF_IDF_embedded)

plt.figure(figsize=(10,10))

plt.scatter(proj[:,0], proj[:,1], c=clusters)

plt.title("ACP with 135 clusters", fontsize="20")
tsne = TSNE(n_components=2)

proj = tsne.fit_transform(TF_IDF_embedded)



plt.figure(figsize=(10,10))

plt.scatter(proj[:,0], proj[:,1], c=clusters)

plt.title("Visualization of the clustering with TSNE", fontsize="20")
plt.figure(figsize=(20,8))

wc = WordCloud()



for num, cluster in enumerate(random.sample(range(100), 12)) :

    plt.subplot(3, 4, num+1)

    wc.generate(" ".join(X[np.where(clusters==cluster)]))

    plt.imshow(wc, interpolation='bilinear')

    plt.title("Cluster {}".format(cluster))

    plt.axis("off")

plt.figure()
pd.Series(clusters).hist(bins=100)
dict_article_to_cluster = {article : cluster for article, cluster in zip(X, clusters)}
with open('product_clusters.pickle', 'wb') as h:

    pickle.dump(dict_article_to_cluster, h)
cluster = df_cleaned['Description'].apply(lambda x : dict_article_to_cluster[x])

df2 = pd.get_dummies(cluster, prefix="Cluster").mul(df_cleaned["TotalPrice"], 0)

df2 = pd.concat([df_cleaned['InvoiceNo'], df2], axis=1)

df2_grouped = df2.groupby('InvoiceNo').sum()
custom_aggregation = {}

custom_aggregation["TotalPrice"] = lambda x:x.iloc[0]

custom_aggregation["min_recency"] = lambda x:x.iloc[0]

custom_aggregation["max_recency"] = lambda x:x.iloc[0]

custom_aggregation["frequency"] = lambda x:x.iloc[0]

custom_aggregation["monetary_value"] = lambda x:x.iloc[0]

custom_aggregation["CustomerID"] = lambda x:x.iloc[0]

custom_aggregation["Quantity"] = "sum"

custom_aggregation["Country"] = lambda x:x.iloc[0]





df_grouped = df_cleaned.groupby("InvoiceNo").agg(custom_aggregation)
df2_grouped_final = pd.concat([df_grouped['CustomerID'], df2_grouped], axis=1).set_index("CustomerID").groupby("CustomerID").sum()

df2_grouped_final = df2_grouped_final.div(df2_grouped_final.sum(axis=1), axis=0)

df2_grouped_final = df2_grouped_final.fillna(0)
custom_aggregation = {}

custom_aggregation["TotalPrice"] = ['min','max','mean']

custom_aggregation["min_recency"] = lambda x:x.iloc[0]

custom_aggregation["max_recency"] = lambda x:x.iloc[0]

custom_aggregation["frequency"] = lambda x:x.iloc[0]

custom_aggregation["monetary_value"] = lambda x:x.iloc[0]

custom_aggregation["Quantity"] = "sum"

custom_aggregation["Country"] = lambda x:x.iloc[0]



df_grouped_final = df_grouped.groupby("CustomerID").agg(custom_aggregation)
df_grouped_final.head(5)
df_grouped_final.columns = ["min", "max", "mean", "min_recency", "max_recency", "frequency", "monetary_value", "quantity", "country"]
df_grouped_final.head(5)
df2_grouped_final.head(5)
X1 = df_grouped_final.as_matrix()

X2 = df2_grouped_final.as_matrix()



scaler = StandardScaler()

X1 = scaler.fit_transform(X1)

X_final_std_scale = np.concatenate((X1, X2), axis=1)
x = list(range(2, 12))

y_std = []

for n_clusters in x:

    print("n_clusters =", n_clusters)

    

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)

    kmeans.fit(X_final_std_scale)

    clusters = kmeans.predict(X_final_std_scale)

    silhouette_avg = silhouette_score(X_final_std_scale, clusters)

    y_std.append(silhouette_avg)

    print("The average silhouette_score is :", silhouette_avg, "with Std Scaling")
kmeans = KMeans(init='k-means++', n_clusters = 8, n_init=30, random_state=0)  # random state just to be able to provide cluster number durint analysis

kmeans.fit(X_final_std_scale)

clusters = kmeans.predict(X_final_std_scale)
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(clusters, bins=8)

plt.xlabel("Cluster")

plt.title("Number of customers per cluster")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Cluster {}".format(x) for x in range(8)])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)
df_grouped_final["cluster"] = clusters
final_dataset = pd.concat([df_grouped_final, df2_grouped_final], axis = 1)

final_dataset.head()
final_dataset_V2 = final_dataset.reset_index()
final_dataset_V2.to_csv("final_dataset_V2.csv",index=False)
with open('df_cleaned.pickle', 'wb') as f:

    pickle.dump(df_cleaned, f)
tsne = TSNE(n_components=2)

proj = tsne.fit_transform(X_final_std_scale)



plt.figure(figsize=(10,10))

plt.scatter(proj[:,0], proj[:,1], c=clusters)

plt.title("Visualization of the clustering with TSNE", fontsize="25")
final_dataset[final_dataset['cluster']==0]
final_dataset[final_dataset['cluster']==0].mean()
temp_final_df = final_dataset.reset_index()
cust0 = list(temp_final_df[temp_final_df['cluster']==0]['CustomerID'])
cluster0 = df_cleaned[df_cleaned['CustomerID'].isin(cust0)]

cluster0[['Quantity', 'UnitPrice', 'QuantityCanceled', 'TotalPrice', 'frequency', 'min_recency'

         , 'monetary_value']].mean()
cluster0['Description'].value_counts()[:10]
custom_aggregation = {}

custom_aggregation["Country"] = lambda x:x.iloc[0]

custom_aggregation["RFMScore"] = lambda x:x.iloc[0]



cluster0_grouped = cluster0.groupby("CustomerID").agg(custom_aggregation)
cluster0_grouped['RFMScore'].value_counts()
cluster0_grouped['Country'].value_counts()
cluster0['Month'].value_counts()
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(cluster0['Month'], bins=12)

plt.xlabel("Cluster")

plt.title("Number of invoices per month")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Month {}".format(x) for x in range(1, 13)])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)
temp['Year'] = cluster0[cluster0['Month']==12]['InvoiceDate'].map(lambda x: x.year)

temp['Year'].value_counts()
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(cluster0['Weekday'], bins=7)

plt.xlabel("Cluster")

plt.title("Number of invoices per day of the week")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Day {}".format(x) for x in range(0, 7)])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)
cluster0['Day'].nunique()
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(cluster0['Day'], bins=31)

plt.xlabel("Cluster")

plt.title("Number of invoices per day of the month")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Day {}".format(x) for x in range(1,32)])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)
cluster0['Hour'].nunique()
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(cluster0['Hour'], bins=14)

plt.xlabel("Cluster")

plt.title("Number of invoices per hour of the day")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Hour {}".format(x) for x in (sorted(cluster0['Hour'].unique()))])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)