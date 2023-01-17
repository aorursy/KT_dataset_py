import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm



from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

data = pd.read_json('../input/combined.json', lines=True)

data.head()
data.info()
data.contents
def cal_accuracy(y_test, y_pred): 

      

    print("Confusion Matrix: ", 

        confusion_matrix(y_test, y_pred)) 

      

    print ("Accuracy : ", 

    accuracy_score(y_test,y_pred)*100) 

      

    print("Report : ", 

    classification_report(y_test, y_pred))
tfidf = TfidfVectorizer(

    #min_df = 5,

    #max_df = 0.95,

    #max_features = 8000,

    stop_words = 'english'

)



tfidf.fit(data.contents)

text = tfidf.transform(data.contents)

#print(text)
def find_optimal_clusters(data, max_k):

    iters = range(2, max_k+1, 2)

    

    sse = []

    for k in iters:

        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)

        print('Fit {} clusters'.format(k))

        

    f, ax = plt.subplots(1, 1)

    ax.plot(iters, sse, marker='o')

    ax.set_xlabel('Cluster Centers')

    ax.set_xticks(iters)

    ax.set_xticklabels(iters)

    ax.set_ylabel('SSE')

    ax.set_title('SSE by Cluster Center Plot')

    

find_optimal_clusters(text, 20)
clusters = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)

print(clusters)

#clusters2 = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(text1)

#print(clusters2)

#cl=KMeans(n_clusters=14, random_state=20).fit_predict(text)

#print(cl)
clusters.flatten
def plot_tsne_pca(data, labels):

    max_label = max(labels)

    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())

    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))

    

    

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)

    label_subset = labels[max_items]

    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)

    ax[0].set_title('PCA Cluster Plot')

    

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)

    ax[1].set_title('TSNE Cluster Plot')

    

plot_tsne_pca(text, clusters)

def get_top_keywords(data, clusters, labels, n_terms):

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

            

get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)
cl=MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20)

cl=cl.fit(text)
X=tfidf.transform([" The  labeled as . Some releases are tagged with topics or related agencies."])

print(cl.predict(X))

X=tfidf.transform(["ark,campaigns,murdered,golden,exported,implemented,securitization,core,diligently,aside"])

print(cl.predict(X))

X=tfidf.transform(["rights,people,accessible,800,civic,514,agreement,access,disabilities,ada"])

print(cl.predict(X))
array=cl.predict(text)
X_train, X_test, y_train, y_test = train_test_split(data.contents, data ,test_size=0.33, random_state=42)
tfidf.fit(X_train)

text = tfidf.transform(X_train)

#tfidf.fit(X_test)

text1 = tfidf.transform(X_test)
cl=MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20)

cl=cl.fit(text)

cl1=MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(text1)

cl1
cl.score(text,text1)
array=cl.predict(text1)

array
cal_accuracy(array,cl1)