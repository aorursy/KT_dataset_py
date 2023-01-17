import pandas as pd

df = pd.read_csv("../input/1k-access/dataset_1ktrain_noscore.csv")

df.describe()
import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns



sns.pairplot(data=df,hue="label",palette="Set2")

plt.show()
plt.figure(figsize=(15,15))

plt.xlabel('Event type',fontsize=24)

plt.ylabel('Response size',fontsize=24)

sns.swarmplot(x='label', y='log_bytes', data=df)

plt.show()
X = df.drop(['label'], axis=1)

mapping = {'safe': 0, 'suspicious': 1}

y = df['label'].replace(mapping)
##Calculation 

%matplotlib inline

import time

import hashlib

import scipy

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import silhouette_score



def optimalK(data, nrefs=3, maxClusters=20):

    """

    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie

    Params:

        data: ndarry of shape (n_samples, n_features)

        nrefs: number of sample reference datasets to create

        maxClusters: Maximum number of clusters to test for

    Returns: (gaps, optimalK)

    """

    gaps = np.zeros((len(range(1, maxClusters)),))

    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})

    for gap_index, k in enumerate(range(1, maxClusters)):



        # Holder for reference dispersion results

        refDisps = np.zeros(nrefs)



        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop

        for i in range(nrefs):

            

            # Create new random reference set

            randomReference = np.random.random_sample(size=data.shape)

            

            # Fit to it

            km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

            km.fit(randomReference)

            

            refDisp = km.inertia_

            refDisps[i] = refDisp



        # Fit cluster to original data and create dispersion

        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

        km.fit(data)

        

        origDisp = km.inertia_



        # Calculate gap statistic

        gap = np.log(np.mean(refDisps)) - np.log(origDisp)



        # Assign this loop's gap statistic to gaps

        gaps[gap_index] = gap

        

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)



    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal



sil = []

kmax = 20

wcss = []



for k in range(2, kmax+1):

  kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)

  labels = kmeans.labels_

  sil.append(silhouette_score(X, labels, metric = 'euclidean'))



for k in range(1,kmax):

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

    

k, gapdf = optimalK(X, nrefs=5)

#print('Optimal k (GAP): ', k)
##Visualisation 

plt.figure(figsize=(30,10))

plt.suptitle('Ideal K',fontsize=20)





plt.subplot(1,3,1)

plt.plot(range(1,20),wcss,"-o")

plt.grid(True)

plt.ylabel('WCSS',fontsize=16)



plt.subplot(1,3,2)

plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)

plt.grid(True)

plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')

plt.xlabel('Cluster Count',fontsize=18)

plt.ylabel('Gap Value',fontsize=16)



plt.subplot(1,3,3)

plt.plot(range(1,20),sil)

plt.grid(True)

plt.ylabel("Silhouette Score",fontsize=16)



plt.show()
##Linear Regression

import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance

from sklearn.metrics import precision_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline

from sklearn import metrics

import joblib



features = df.drop(['label'], axis=1)



train = df.sample(n=900, replace=True, random_state=1)

test = df.drop(train.index)



X_train = train.drop(['label'], axis=1)

mapping = {'safe': 0, 'suspicious': 1}

y_train = train['label'].replace(mapping)



X_test = test.drop(['label'], axis=1)

mapping = {'safe': 0, 'suspicious': 1}

y_test = test['label'].replace(mapping)



regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm



#To retrieve the intercept:

print('Intercept: %.8f' % regressor.intercept_)

#For retrieving the slope:

#print(regressor.coef_)



# save the model to disk

filename = 'LR.model'

joblib.dump(regressor, filename)



y_pred = regressor.predict(X_test).round()



y_test = np.array(list(y_test))

y_pred = np.array(y_pred)



count=0



for i in range(len(y_test)):

    if y_test.flatten()[i] !=  round(y_pred.flatten()[i]):

        count+=1

     

precision = precision_score(y_test, y_pred, average='micro')

print('Precision: %.2f' % float(precision*100), "%")



#print(count)

#print(len(y_test))



#dataset2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#dataset2



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

coeffecients = pd.DataFrame(regressor.coef_,X_test.columns)

coeffecients.columns = ['Coeffecient']

#coeffecients
##Multi-layer Perceptron 

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_score

import joblib



clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(5, 2), random_state=1)



clf.fit(X_train, y_train)

#print([coef.shape for coef in clf.coefs_])

#print(clf.coefs_)



y_pred = clf.predict(X_test)



precision = precision_score(y_test, y_pred, average='micro')

print('Precision: %.2f' % float(precision*100), "%")



features["mlp_pred"] = (clf.predict(features[["unusual_hours", 

                                             "has_bad_rep",

                                             "has_bad_string", 

                                             "method", 

                                             "version", 

                                             "status", 

                                             "log_len_uri", 

                                             "log_bytes", 

                                             "scripting_useragent"]]))



# save the model to disk

filename = 'MLP.model'

joblib.dump(clf, filename)
features["lr_pred"] = (0.13688273 * df["unusual_hours"] 

                     + 0.03655747 * df["has_bad_rep"] 

                     + 0.04017277 * df["has_bad_string"] 

                     - 0.05029479 * df["method"] 

                     + 0.34400104 * df["version"] 

                     + 0.20885592 * df["status"] 

                     + 0.41214625 * df["log_len_uri"]

                     - 0.00410462 * df["log_bytes"]

                     + 0.00000000 * df["scripting_useragent"]

                     - 0.45363860)



df["lr_pred"] = (0.13688273 * df["unusual_hours"] 

                     + 0.03655747 * df["has_bad_rep"] 

                     + 0.04017277 * df["has_bad_string"] 

                     - 0.05029479 * df["method"] 

                     + 0.34400104 * df["version"] 

                     + 0.20885592 * df["status"] 

                     + 0.41214625 * df["log_len_uri"]

                     - 0.00410462 * df["log_bytes"]

                     + 0.00000000 * df["scripting_useragent"]

                     - 0.45363860)



df["mlp_pred"] = (clf.predict(df[["unusual_hours", 

                                    "has_bad_rep",

                                    "has_bad_string", 

                                    "method", 

                                    "version", 

                                    "status", 

                                    "log_len_uri", 

                                    "log_bytes", 

                                    "scripting_useragent"]]))
plt.figure(figsize=(24,4))



plt.suptitle("K Means Clustering",fontsize=20)



plt.subplot(1,5,1)

plt.title("K = 1",fontsize=16)

plt.xlabel("log_len_uri")

plt.ylabel("has_bad_strings")

plt.scatter(features.log_len_uri,features.has_bad_string)





plt.subplot(1,5,2)

plt.title("K = 2",fontsize=16)

plt.xlabel("log_len_uri")

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.log_len_uri[features.labels == 0],features.has_bad_string[features.labels == 0])

plt.scatter(features.log_len_uri[features.labels == 1],features.has_bad_string[features.labels == 1])



# I drop labels since we only want to use features.

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(1,5,4)

plt.title("K = 3",fontsize=16)

plt.xlabel("log_len_uri")

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.log_len_uri[features.labels == 0],features.has_bad_string[features.labels == 0])

plt.scatter(features.log_len_uri[features.labels == 1],features.has_bad_string[features.labels == 1])

plt.scatter(features.log_len_uri[features.labels == 2],features.has_bad_string[features.labels == 2])



# I drop labels since we only want to use features.

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(1,5,3)

plt.title("K = 4",fontsize=16)

plt.xlabel("log_len_uri")

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.log_len_uri[features.labels == 0],features.has_bad_string[features.labels == 0])

plt.scatter(features.log_len_uri[features.labels == 1],features.has_bad_string[features.labels == 1])

plt.scatter(features.log_len_uri[features.labels == 2],features.has_bad_string[features.labels == 2])

plt.scatter(features.log_len_uri[features.labels == 3],features.has_bad_string[features.labels == 3])



# I drop labels since we only want to use features.

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(1,5,5)

plt.title("Original Labels",fontsize=16)

plt.xlabel("log_len_uri")

plt.scatter(df.log_len_uri[df.label == "suspicious"],df.has_bad_string[df.label == "suspicious"])

plt.scatter(df.log_len_uri[df.label == "safe"],df.has_bad_string[df.label == "safe"])



plt.subplots_adjust(top=0.8)

plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage



merg = linkage(features,method="ward")



plt.figure(figsize=(18,6))

dendrogram(merg, leaf_rotation=90)

plt.xlabel("data points")

plt.ylabel("euclidian distance")



plt.suptitle("DENDROGRAM",fontsize=18)

plt.show()
from sklearn.cluster import AgglomerativeClustering



kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

features["labels"] = kmeans.fit_predict(features[['lr_pred', 'mlp_pred']])



# cross tabulation table for kmeans

df1 = pd.DataFrame({'prediction':features['labels'] ,"label":df['label'] })

ct1 = pd.crosstab(df1['label'],df1['prediction'])





# hierarchy

hc_cluster = AgglomerativeClustering(n_clusters=2)

hc_predict = hc_cluster.fit_predict(features[['lr_pred', 'mlp_pred']])



# cross tabulation table for Hierarchy

df2 = pd.DataFrame({'prediction':hc_predict,"label":df['label']})

ct2 = pd.crosstab(df2['label'],df2['prediction'])





plt.figure(figsize=(24,8))

plt.suptitle("CROSS TABULATIONS",fontsize=30)

ax = plt.subplot(1,2,1)

label_font = {'size':'24'}

ax.set_xlabel('Prediction', fontdict=label_font);

ax.set_ylabel('Label', fontdict=label_font);

ax.tick_params(axis='both', which='major', labelsize=24)  # Adjust to fit



plt.title("KMeans", fontsize=28)

sns.heatmap(ct1,annot=True,cbar=False, fmt=".1f", cmap="Blues", annot_kws={'size':24})#, square=True)



ay = plt.subplot(1,2,2)



ay.set_xlabel('Prediction', fontdict=label_font);

ay.set_ylabel('Label', fontdict=label_font);

ay.tick_params(axis='both', which='major', labelsize=24)  # Adjust to fit

plt.title("Hierarchy", fontsize=28)

sns.heatmap(ct2,annot=True,cbar=False, fmt=".1f", cmap="Blues", annot_kws={'size':24})#, square=True)



plt.show()
import pandas as pd

import joblib



d10k = pd.read_csv("../input/10k-access/dataset_web_10K.csv", delimiter=',')



# load the models from disk

MLP_model = joblib.load('MLP.model')

LR_model = joblib.load('LR.model')



#print(d10k[['scripting_useragent']])

d10k["mlp_pred"] = (MLP_model.predict(d10k[["unusual_hours", 

                                                 "has_bad_rep",

                                                 "has_bad_string", 

                                                 "method", 

                                                 "version", 

                                                 "status", 

                                                 "log_len_uri", 

                                                 "log_bytes", 

                                                 "scripting_useragent"]]))



d10k["lr_pred"] = (LR_model.predict(d10k[["unusual_hours", 

                                                 "has_bad_rep",

                                                 "has_bad_string", 

                                                 "method", 

                                                 "version", 

                                                 "status", 

                                                 "log_len_uri", 

                                                 "log_bytes", 

                                                 "scripting_useragent"]]))



kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

d10k["labels"] = kmeans.fit_predict(d10k[['lr_pred', 'mlp_pred']])



plt.figure(figsize=(15,10))

plt.title("Unknown items", fontsize=18)

plt.grid(True)

plt.scatter(d10k.log_len_uri[d10k.labels == 0],d10k.lr_pred[d10k.labels == 0])

plt.scatter(d10k.log_len_uri[d10k.labels == 1],d10k.lr_pred[d10k.labels == 1])

plt.xlabel("log_len_uri",fontsize=14)

plt.ylabel("lr_pred",fontsize=14)

plt.tight_layout()

plt.show()



d10k.to_csv (r'export_labeled_df.csv', index = False, header=True)

print("Saved csv!")