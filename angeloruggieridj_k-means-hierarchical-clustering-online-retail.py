# Import delle l'analisi esplorativa dei dati
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

# import delle librerie richieste per l'applicazione di algoritmi di clustering
import sklearn
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
#encoding: Encoding to use for UTF when reading/writing
#header
store = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv', sep=",", encoding="ISO-8859-1", header=0)
store.head()
store.describe()
store.shape
store.info()
store.describe()
df_null = round(100*(store.isnull().sum())/len(store), 2)
df_null
store = store.dropna()
store.shape
# Per effettuare operazioni di Join, il tipo di dato CustomerID viene convertito in tipo String
store['Customer ID'] = store['Customer ID'].astype(str)
#Introduzione del nuovo attributo Monetary
store['Amount'] = store['Quantity']*store['Price']
rfm_m = store.groupby('Customer ID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()
# Introduzione del nuovo attributo Frequency

rfm_f = store.groupby('Customer ID')['Invoice'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['Customer ID', 'Frequency']
rfm_f.head()
# Unione dei due dataframe: corrisponde ad un Inner-JOIN SQL

rfm = pd.merge(rfm_m, rfm_f, on='Customer ID', how='inner')
rfm.head()
# Conversione della data nel tipo supportato da Python DateTime per effettuare le dovute operazioni
store['InvoiceDate'] = pd.to_datetime(store['InvoiceDate'],format='%Y-%m-%d %H:%M')
# Calcolo della data massima registrata all'interno del dataset
max_date = max(store['InvoiceDate'])
max_date
# Calcolo della differenza tra la data massima registrata nel dataset e il valore espresso per _InvoiceDate_
store['Diff'] = max_date - store['InvoiceDate']
store.head()
# Raggruppando gli esempi per CustomerID, si prende il valore minore della data
rfm_p = store.groupby('Customer ID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()
# Introduzione della feature Recency, estrapolando dalla data solo il numero di giorni
rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()
# Unione dei dataframe, al fine di ottenere l'ultimo DataFrame complessivo
rfm = pd.merge(rfm, rfm_p, on='Customer ID', how='inner')
#Intestazione delle colonne
rfm.columns = ['Customer ID', 'Amount', 'Frequency', 'Recency']
#Stampa dei primi 5 esempi
rfm.head()
attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers nel dataset", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
# Rimozione degli outliers per Amount utilizzando InterQuartileRange
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Rimozione degli outliers per Recency utilizzando InterQuartileRange
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Rimozione degli outliers per Frequency utilizzando InterQuartileRange
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]
#Scaling degli attributi
rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

sc = StandardScaler()
df_scaled = sc.fit_transform(rfm_df)
#Stampa delle nuove dimensioni
df_scaled.shape
#Conversione a dataframe
df_scaled = pd.DataFrame(df_scaled)
#Intestazione delle colonne
df_scaled.columns = ['Amount', 'Frequency', 'Recency']
#Stampa dei primi 5 esempi standardizzati
df_scaled.head()
# Metodo k-Means con un numero di clusters arbitrario.
# - n_clusters: numero di cluster desiderati (3) - limitazione del k-Means;
# - n_init: esegue l'algoritmo n volte in modo indipendente, con diversi centroidi casuali per scegliere il modello finale come quello con il SSE più basso.
# - max_iter: indica il numero massimo di iterazioni per ogni singola esecuzione (qui, 300). 


#method = KMeans(n_clusters=4, random_state = 1, max_iter=300, tol=1e-04, init='random', n_init=10)
method = KMeans(n_clusters=4, random_state = 1, max_iter=300, tol=1e-04, init='k-means++', n_init=10)
method.fit(df_scaled)
#Stampa delle etichette relative ai cluster
method.labels_
elbow_values = []
range_n_clusters = [1, 2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    method = KMeans(n_clusters=num_clusters, random_state = 1, max_iter=300, tol=1e-04, init='k-means++', n_init=10)
    method.fit(df_scaled)
    
    elbow_values.append(method.inertia_)
    
# Plot del valore della somma della radice delle distanze al crescere del numero dei cluster
plt.plot(range(1, 9), elbow_values, marker='o')
plt.ylabel("Sum of Squared Distance")
plt.xlabel("Numero dei cluster")
# Definizione della lista del numero di cluster da testare
range_n_clusters = list(x for x in range (2,10+1))

for num_clusters in range_n_clusters:
    method = KMeans(n_clusters=num_clusters, max_iter=50)
    method.fit(df_scaled)
    cluster_labels = method.labels_
    # Calcolo coefficiente di silhouette
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    print("Per n_clusters={0}, il coefficiente di Silhouette è pari a {1}".format(num_clusters, silhouette_avg))
method = KMeans(n_clusters=3, random_state = 1, max_iter=300, tol=1e-04, init='k-means++', n_init=10)
method.fit(df_scaled)
method.labels_
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
rfm['Cluster_Id'] = method.labels_
# Stampa dei primi 5 esempi
rfm.head()
features_list = ['Amount', 'Frequency', 'Recency']

for feature in features_list:
    sns.boxplot(x='Cluster_Id', y=feature, data=rfm)
    plt.show()
# Applicazione del metodo Single Linkage
single_linkage = linkage(df_scaled, method="single", metric='euclidean')
dendrogram(single_linkage)
plt.show()
# Applicazione del metodo Complete linkage
complete_linkage = linkage(df_scaled, method="complete", metric='euclidean')
dendrogram(complete_linkage)
plt.show()
# Applicazione del metodo Average linkage
avg_linkage = linkage(df_scaled, method="average", metric='euclidean')
dendrogram(avg_linkage)
plt.show()
# Desiderando un numero di cluster pari a 3, si inizializza il parametro n_clusters=3
cluster_labels = cut_tree(avg_linkage, n_clusters=3).reshape(-1, )
#Stampa delle etichette dei cluster
cluster_labels
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
rfm['Cluster_Labels'] = cluster_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
rfm.head()
#Plot delle Features

features_list = ['Amount', 'Frequency', 'Recency']

for feature in features_list:
    sns.boxplot(x='Cluster_Labels', y=feature, data=rfm)
    plt.show()
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
agglomerative_cluster_labels = ac.fit_predict(df_scaled)
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
rfm['Agglomerative_Clustering'] = agglomerative_cluster_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
rfm.head()
#Plot delle Features

features_list = ['Amount', 'Frequency', 'Recency']

for feature in features_list:
    sns.boxplot(x='Agglomerative_Clustering', y=feature, data=rfm)
    plt.show()
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(metric='euclidean')
dbscan.fit(df_scaled)
dbscan_labels = dbscan.labels_
#Stampa delle etichette dei cluster
dbscan_labels
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
rfm['DensityBased_Labels'] = dbscan_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
rfm.head()
#Plot delle Features

features_list = ['Amount', 'Frequency', 'Recency']

for feature in features_list:
    sns.boxplot(x='DensityBased_Labels', y=feature, data=rfm)
    plt.show()
