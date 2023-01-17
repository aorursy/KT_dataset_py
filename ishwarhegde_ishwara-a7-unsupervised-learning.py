import os

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['figure.figsize'] = (12, 9)

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



RANDOM_STATE = 17
# change this if necessary

PATH_TO_SAMSUNG_DATA = "../input/samsung-data/samsung_HAR"
X_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_train.txt"))

y_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_train_labels.txt")).astype(int)



X_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_test.txt"))

y_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_test_labels.txt")).astype(int)
# Checking dimensions

assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))

assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
# Your code here

X=np.concatenate((X_train,X_test),axis=0)

y=np.concatenate((y_train,y_test),axis=0)
y.shape
unique_classes=np.unique(y)
n_classes = np.unique(y).size



# Your code here

X_scaled = StandardScaler().fit_transform(X)
# Your code here

pca = PCA(n_components=2,random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)





#pca_df = pd.DataFrame(data = X_pca, columns = ['principal component 1', 'principal component 2'])
# Your code here --> 65

#X_pca.n_components_

# Your code here -> 51 percent

X_pca.explained_variance_ratio_
df=pd.DataFrame(np.column_stack((X_scaled,y)))

df.rename(columns = {561:'test'}, inplace = True) 

df["pca-one"]=X_pca[:,0]

df["pca-two"]=X_pca[:,1]

df.head()
plt.figure(figsize=(16,10))



sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="test",

    data=df,

    legend="full",

    palette="rocket",

    alpha=0.3

)

# Your code here



kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE,n_init=100).fit(X_scaled)



df["k_means_cluster"]=np.array(kmeans.labels_)

df.head()
# Your code here

# plt.scatter(, , c=cluster_labels, s=20, cmap='viridis');

plt.figure(figsize=(12,8))

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="k_means_cluster",style="test",

    data=df, palette="viridis",

    legend="full"

)
k_means_clusters=df["k_means_cluster"]

tab = pd.crosstab(y, k_means_clusters, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
# # Your code here

# inertia = []

# for k in tqdm_notebook(range(1, n_classes + 1)):

#     pass



inertia = []

for k in range(1, 8):

    kmeans = KMeans(n_clusters=k, random_state=1).fit(X_scaled)

    inertia.append(np.sqrt(kmeans.inertia_))







plt.plot(range(1, 8), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');



ag = AgglomerativeClustering(n_clusters=n_classes,linkage='ward').fit(X_scaled)

df["agg_cluster"]=np.array(ag.labels_)

df.head()
# Your code here

from sklearn.metrics import adjusted_rand_score

score_k=adjusted_rand_score(df["test"], df["k_means_cluster"])

score_agg=adjusted_rand_score(df["test"],df["agg_cluster"])

print("scores are: {k} and {agg}".format(k=score_k,agg=score_agg))
# # Your code here

scaler =StandardScaler().fit(X_train)

X_train_scaled =scaler.transform(X_train)

X_test_scaled =scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

# %%time

best_svc=GridSearchCV(estimator=svc,param_grid=svc_params,n_jobs=-1,cv=3)

best_svc.fit(X_train_scaled,y_train)

best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
# Your code here

pca = PCA(n_components=0.90,random_state=RANDOM_STATE)

pca_2 = pca.fit(X_scaled)

X_pca_train=pca_2.transform(X_train_scaled)

X_pca_test=pca_2.transform(X_test_scaled)

best_svc_pca=GridSearchCV(estimator=svc,param_grid=svc_params,n_jobs=-1,cv=3)

best_svc_pca.fit(X_pca_train,y_train)

best_svc_pca.best_params_, best_svc_pca.best_score_