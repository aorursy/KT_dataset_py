# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import nltk
import random
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

%matplotlib inline
df = pd.read_csv("../input/data.csv", encoding="ISO-8859-1")
df.info()
# Handle cancellation as new feature
df["Cancelled"] = df["InvoiceNo"].str.startswith("C")
df["Cancelled"] = df["Cancelled"].fillna(False)

# Hnadle incorrect Description
df = df[df["Description"].str.startswith("?") == False]
df = df[df["Description"].str.isupper() == True]
df = df[df["Description"].str.contains("LOST") == False]
df = df[df["CustomerID"].notnull()]
df["CustomerID"] = df["CustomerID"].astype(int)

# Convert Invoice Number to integer as we already consider Cancellation as new feature
df['InvoiceNo'].replace(to_replace="\D+", value=r"", regex=True, inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('int')

# remove shiping invoices
df = df[(df["StockCode"] != "DOT") & (df["StockCode"] != "POST")]
df.drop("StockCode", inplace=True, axis=1)

# remove outliers by qty
qte_false = [74215, 3114, 80995]  # fond during exploration but not done here (found with a boxplot on qty or price)
for qte in qte_false:
    df = df[(df["Cancelled"] == False) & (df["Quantity"] !=qte)]

# Now we can only keep the order without cancellation
df = df[df["Cancelled"] == False]
df.drop("Cancelled", axis=1, inplace=True)

# We can create the feature Price
df["Price"] = df["UnitPrice"] * df["Quantity"]

# convert date to proper datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df.info()
revenue_per_countries = df.groupby(["Country"])["Price"].sum().sort_values()
revenue_per_countries.plot(kind='barh', figsize=(15,12))
plt.title("Revenue per Country")
plt.show()
No_invoice_per_country = df.groupby(["Country"])["InvoiceNo"].count().sort_values()
No_invoice_per_country.plot(kind='barh', figsize=(15,12))
plt.title("Number of Invoices per Country")
plt.show()
best_buyer = df.groupby(["Country", "InvoiceNo"])["Price"].sum().reset_index().groupby(["Country"])["Price"].mean().sort_values()
best_buyer.plot(kind='barh', figsize=(15,12))
plt.title("Average Basket Price per Country")
plt.show()
encoder_countries = best_buyer.rank().to_dict()
decoder_countries = {i: j for i, j in encoder_countries.items()}

df["Country"]  = df["Country"].apply(lambda x:encoder_countries[x])
X = df["Description"].unique()
X = df["Description"].unique()

stemmer = nltk.stem.porter.PorterStemmer()
stopword = nltk.corpus.stopwords.words('english')

def stem_and_filter(doc):
    tokens = [stemmer.stem(w) for w in analyzer(doc)]
    return [token for token in tokens if token.isalpha()]

analyzer = CountVectorizer().build_analyzer()
CV = CountVectorizer(lowercase=True, stop_words="english", analyzer=stem_and_filter)
TF_matrix = CV.fit_transform(X)
print("TF_matrix :", TF_matrix.shape, "of", TF_matrix.dtype)

analyzer = TfidfVectorizer().build_analyzer()
CV = TfidfVectorizer(lowercase=True, stop_words="english", analyzer=stem_and_filter, min_df=0.00, max_df=0.3)  # we remove words if it appears in more than 30 % of the corpus (not found stopwords like Box, Christmas and so on)
TF_IDF_matrix = CV.fit_transform(X)
print("TF_IDF_matrix :", TF_IDF_matrix.shape, "of", TF_IDF_matrix.dtype)
svd = TruncatedSVD(n_components = 100)
normalizer = Normalizer(copy=False)

TF_embedded = svd.fit_transform(TF_matrix)
TF_embedded = normalizer.fit_transform(TF_embedded)
print("TF_embedded :", TF_embedded.shape, "of", TF_embedded.dtype)

TF_IDF_embedded = svd.fit_transform(TF_IDF_matrix)
TF_IDF_embedded = normalizer.fit_transform(TF_IDF_embedded)
print("TF_IDF_embedded :", TF_IDF_embedded.shape, "of", TF_IDF_embedded.dtype)
score_tf = []
score_tfidf = []
mean_tf = []
std_tf = []
mean_tfidf = []
std_tfidf = []

x = list(range(5, 105, 5))

for n_clusters in x:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(TF_embedded)
    clusters = kmeans.predict(TF_embedded)
    silhouette_avg = silhouette_score(TF_embedded, clusters)
#     print("N clusters =", n_clusters, "Silhouette Score :", silhouette_avg)
    rep = np.histogram(clusters, bins = n_clusters-1)[0]
    score_tf.append(silhouette_avg)
    mean_tf.append(rep.mean())
    std_tf.append(rep.std())

for n_clusters in x:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(TF_IDF_embedded)
    clusters = kmeans.predict(TF_IDF_embedded)
    silhouette_avg = silhouette_score(TF_IDF_embedded, clusters)
#     print("N clusters =", n_clusters, "Silhouette Score :", silhouette_avg)
    rep = np.histogram(clusters, bins = n_clusters-1)[0]
    score_tfidf.append(silhouette_avg)
    mean_tfidf.append(rep.mean())
    std_tfidf.append(rep.std())
plt.figure(figsize=(20,16))

plt.subplot(2, 1, 1)
plt.plot(x, score_tf, label="TF matrix")
plt.plot(x, score_tfidf, label="TF-IDF matrix")
plt.title("Evolution of the Silhouette Score")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, mean_tfidf, label="TF-IDF mean")
plt.plot(x, mean_tf, label="TF mean")
plt.plot(x, std_tfidf, label="TF-IDF St.Dev")
plt.plot(x, std_tf, label="TF St.Dev")
plt.ylim(0, 200)
plt.title("Evolution of the mean and Std of both clusters")
plt.legend()

plt.show()
n_clusters = 100

kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30, random_state=42)
kmeans.fit(TF_IDF_embedded)
clusters = kmeans.predict(TF_IDF_embedded)
plt.figure(figsize=(20,8))
wc = WordCloud()

for num, cluster in enumerate(random.sample(range(100), 12)) :
    plt.subplot(3, 4, num+1)
    wc.generate(" ".join(X[np.where(clusters==cluster)]))
    plt.imshow(wc, interpolation='bilinear')
    plt.title("Cluster {}".format(cluster))
    plt.axis("off")
plt.figure()
plt.show()
dict_article_to_cluster = {article : cluster for article, cluster in zip(X, clusters)}
tsne = TSNE(n_components=2)
proj = tsne.fit_transform(TF_IDF_embedded)

plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("Visualisation of the clustering with TSNE", fontsize="25")
plt.show()
cluster = df['Description'].apply(lambda x : dict_article_to_cluster[x])
df2 = pd.get_dummies(cluster, prefix="Article_cluster").mul(df["Price"], 0)
df2 = pd.concat([df['InvoiceNo'], df2], axis=1)
df2_grouped = df2.groupby('InvoiceNo').sum()
custom_aggregation = {}
custom_aggregation["Price"] = "sum"
custom_aggregation["InvoiceDate"] = lambda x:x.iloc[0]
custom_aggregation["CustomerID"] = lambda x:x.iloc[0]
custom_aggregation["Country"] = lambda x:x.iloc[0]
custom_aggregation["Quantity"] = "sum"

df_grouped = df.groupby("InvoiceNo").agg(custom_aggregation)
# let create recency for every Invoice
now = df_grouped["InvoiceDate"].max()  # as the dataset is not done in the present
df_grouped["Recency"] = now - df_grouped["InvoiceDate"]
df_grouped["Recency"] = pd.to_timedelta(df_grouped["Recency"]).astype("timedelta64[D]") # conversion to day from now

# add features required for the next groupby
df_grouped["nb_visit"] = 1
df_grouped["total_spent"] = 1
df2_grouped_final = pd.concat([df_grouped['CustomerID'], df2_grouped], axis=1).set_index("CustomerID").groupby("CustomerID").sum()
df2_grouped_final = df2_grouped_final.div(df2_grouped_final.sum(axis=1), axis=0)
df2_grouped_final = df2_grouped_final.fillna(0)
custom_aggregation = {}
custom_aggregation["Price"] = ["mean", "sum"]
custom_aggregation["nb_visit"] = "sum"
custom_aggregation["Country"] = lambda x:x.iloc[0]
custom_aggregation["Quantity"] = "sum"
custom_aggregation["Recency"] = ["min", "max"]

df_grouped_final = df_grouped.groupby("CustomerID").agg(custom_aggregation)
df_grouped_final["Freq"] = (df_grouped_final["Recency"]["max"]  - df_grouped_final["Recency"]["min"] ) / df_grouped_final["nb_visit"]["sum"]
df_grouped_final.columns = ["avg_price", "sum_price", "nb_visit", "country", "quantity", "min_recency", "max_recency", "freq"]
df_grouped_final.head()
X1 = df_grouped_final.as_matrix()
X2 = df2_grouped_final.as_matrix()

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X_final_std_scale = np.concatenate((X1, X2), axis=1)

scaler = MinMaxScaler()
X1 = scaler.fit_transform(X1)
X_final_minmax_scale = np.concatenate((X1, X2), axis=1)
x = list(range(2, 10))
y_std = []
y_minmax = []
for n_clusters in x:
    print("n_clusters =", n_clusters)
    
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(X_final_std_scale)
    clusters = kmeans.predict(X_final_std_scale)
    silhouette_avg = silhouette_score(X_final_std_scale, clusters)
    y_std.append(silhouette_avg)
    print("The average silhouette_score is :", silhouette_avg, "with Std Scaling")
    
    kmeans.fit(X_final_minmax_scale)
    clusters = kmeans.predict(X_final_minmax_scale)
    silhouette_avg = silhouette_score(X_final_minmax_scale, clusters)
    y_minmax.append(silhouette_avg)
    print("The average silhouette_score is :", silhouette_avg, "with MinMax Scaling")
plt.figure(figsize=(12,6))

plt.plot(x, y_std, label="Using Standard Scaling")
plt.plot(x, y_minmax, label="Using MinMax Scaling")

plt.legend()
plt.xlabel("Nombre de Clusters")
plt.ylabel("Score de Silhouette")
plt.title("Impact du Scaling sur le Clustering")
plt.xticks(x)

plt.show()
kmeans = KMeans(init='k-means++', n_clusters = 6, n_init=30, random_state=42)  # random state just to be able to provide cluster number durint analysis
kmeans.fit(X_final_std_scale)
clusters = kmeans.predict(X_final_std_scale)
plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(clusters, bins=6) # arguments are passed to np.histogram
plt.xlabel("Cluster")
plt.title("Number of Customer per cluster")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Cluster {}".format(x) for x in range(6)])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.

plt.show()
tsne = TSNE(n_components=2)
proj = tsne.fit_transform(X_final_std_scale)

plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("Visualisation of the clustering with TSNE", fontsize="25")
plt.show()
df_grouped_final["cluster"] = clusters
df_grouped_final.head()
# custom_aggregation = {}
# custom_aggregation["avg_price"] = "mean"
# custom_aggregation["nb_visit"] = "mean"
# custom_aggregation["Country"] = lambda x:x.iloc[0]
# custom_aggregation["min_recency"] = "mean"
# custom_aggregation["max_recency"] = ["min", "max"]

# df_grouped_final = df_grouped.groupby("CustomerID").agg(custom_aggregation)

df_analysis = df_grouped_final.groupby("cluster").mean()
df_analysis.head()
price = df_analysis["avg_price"].values
freq = df_analysis["freq"].values
visit = df_analysis["nb_visit"].values

plt.figure(figsize = (20,8))
plt.scatter(price, freq, s=visit*20)
    
for label, x, y in zip(["Customer #{}".format(x) for x in range(6)], price, freq):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
plt.title("Panier Moyen et fréquence d'achat par Cluster")
plt.xlabel("Average Price per Invoice")
plt.ylabel("Time between 2 invoices")

plt.show()
x_max = df_analysis["min_recency"].values
x_min = df_analysis["max_recency"].values
freq = df_analysis["freq"].values

plt.figure(figsize = (20,8))
for i in range(6):
    plt.plot([-x_min[i] , -x_max[i]], [i, i], linewidth=freq[i]/5)

plt.xlim(-365, 0)
plt.title("Balance of invoices per Customer Clusters")
plt.xlabel("Recency")
plt.ylabel("Cluster")
plt.show()
purchase_mean = df2_grouped_final.set_index(clusters).groupby(clusters).mean()
purchase_mean.head()
plt.figure(figsize=(15,12))
ax = plt.subplot(111, projection='polar')
theta = 2 * np.pi * np.linspace(0, 1, 100)
matrix = purchase_mean.as_matrix()

for i in range(6):
    r = matrix[i, :]
    ax.plot(theta, r, label="Customer Group {}".format(i))

ax.set_xticklabels([
    "Cluster 0", "Cluster 12", "Cluster 25", "Cluster 38", 
    "Cluster 50", "Cluster 62", "Cluster 75", "Cluster 88"
])

ax.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title("Cluster of article bought per type of customer", va='bottom')
plt.show()
clusters.shape
classification_dataset = pd.concat([df_grouped_final, df2_grouped_final], axis = 1)
classification_dataset.head()
X = classification_dataset.drop("cluster", axis=1).as_matrix()
y = classification_dataset["cluster"].as_matrix()
min_samples_split = 4 
cv = 5
n_estimators = 40
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
for max_depth in range(3, 10):
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split )
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, n_jobs=8)
    scores = np.array(scores)
    print("Depth {} : Acc {:.3f} - Dev {:.3f}".format(max_depth, scores.mean(), scores.std()))
for max_depth in range(3, 10):
    clf = ExtraTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split )
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=8)
    scores = np.array(scores)
    print("Depth {} : Acc {:.3f} - Dev {:.3f}".format(max_depth, scores.mean(), scores.std()))
for max_depth in range(3, 10):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0, max_depth=max_depth, min_samples_split=min_samples_split )
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=8)
    scores = np.array(scores)
    print("Depth {} : Acc {:.3f} - Dev {:.3f}".format(max_depth, scores.mean(), scores.std()))
for max_depth in range(3, 10):
    clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=0, max_depth=max_depth, min_samples_split=min_samples_split )
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=8)
    scores = np.array(scores)
    print("Depth {} : Acc {:.3f} - Dev {:.3f}".format(max_depth, scores.mean(), scores.std()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf1 = DecisionTreeClassifier(random_state=0, max_depth=9, min_samples_split=4 )
clf2 = RandomForestClassifier(random_state=0, max_depth=9, min_samples_split=4, n_estimators=n_estimators)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
np.unique(y_pred1)
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = ["Cluster {}".format(x) for x in range(6)]
np.set_printoptions(precision=2)

plt.figure(figsize=(20,12))
plt.subplot(1, 2, 1)
cnf_matrix = confusion_matrix(y_test, y_pred1)
plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix, with DecisionTreeClassifier')

plt.subplot(1, 2, 2)
cnf_matrix = confusion_matrix(y_test, y_pred2)
plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix, with RandomForestClassifier')

plt.show()
