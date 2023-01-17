import pandas as pd

import numpy as np

from math import sqrt

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import make_scorer,accuracy_score,roc_curve,auc

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer

import matplotlib.pyplot as plt

from sklearn.utils import resample,class_weight

from sklearn.cluster import KMeans,AgglomerativeClustering,MiniBatchKMeans,DBSCAN

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
dfx = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df = dfx.iloc[:,1:]

print(df.shape)

df.iloc[544,-2] = 0

df.iloc[1348,-2] = 0

df.iloc[1553,-2] = 0

df.iloc[2504,-2] = 0

df.iloc[3083,-2] = 0

df.iloc[4766,-2] = 0

print(df['Satisfied'].value_counts())

df_testx = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

df_test = df_testx.iloc[:,1:]

df_test[df_test['TotalCharges']==" "]

df_test.iloc[71,-1] = 0

df_test.iloc[580,-1] = 0

df_test.iloc[637,-1] = 0

df_test.iloc[790,-1] = 0

df_test.iloc[1505,-1] = 0

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'])

labelencode_cols = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,16]

for i in labelencode_cols:

    le = LabelEncoder()

    df.iloc[:,i] = le.fit_transform(df.iloc[:,i])

    df_test.iloc[:,i] = le.transform(df_test.iloc[:,i])
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
X = df.iloc[:,:-1].values

Y = df.iloc[:,-1].values

X_test = df_test.iloc[:,:].values

"""X = df_w6.iloc[:,:9].values

Y = df_w6.iloc[:,-1].values

print(X.shape,Y.shape)

for x in range(0,2):

    smote = SMOTE('minority')

    X,Y = smote.fit_sample(X,Y)

    print(X.shape,Y.shape)

    print(np.unique(Y,return_counts=True))

X6 = df_6.iloc[:,:9].values

Y6 = df_6.iloc[:,-1].values

X = np.concatenate((X,X6))

Y = np.concatenate((Y,Y6))

print(X.shape,Y.shape)"""

p = np.random.permutation(len(Y))

Y = np.asarray(Y)

X,Y = X[p], Y[p]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print(X_train.shape)
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train[:,[-1,-2,-4]])

scaled_X_test = scaler.transform(X_test[:,[-1,-2,-4]])

scaled_X_val = scaler.transform(X_val[:,[-1,-2,-4]])

X_train = np.delete(X_train,[18,17,15],1)

X_test = np.delete(X_test,[18,17,15],1)

X_val = np.delete(X_val,[18,17,15],1)

X_train = np.concatenate((X_train,scaled_X_train),axis=1)

X_test = np.concatenate((X_test,scaled_X_test),axis=1)

X_val = np.concatenate((X_val,scaled_X_val),axis=1)

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(X_train, Y_train)

val_lda = lda.transform(X_val)

test_lda = lda.transform(X_test)

X_train_lda = np.concatenate((X_train,X_lda),axis=1)

X_test_lda = np.concatenate((X_test,test_lda),axis=1)

X_val_lda = np.concatenate((X_val,val_lda),axis=1)
def auc_score(ground_truth, predictions):

     fpr, tpr,thresh = roc_curve(ground_truth, predictions, pos_label=1)    

     return auc(fpr, tpr)

scorer = make_scorer(auc_score, greater_is_better=True)

param_kmeans = {'n_clusters' : range(1,15)}

best_cluster = 0

max_auc = 0

for k in range(2,20):

    kmeans_lite = KMeans(n_clusters=k,random_state=42).fit(X_train_lda)

    #kmeans_lite = MiniBatchKMeans(n_clusters=k, random_state=42).partial_fit(X_train[0:2000,:])

    #kmeans_lite.partial_fit(X_train[2000:,:])

    Ytrain_clustered = kmeans_lite.labels_

    dict = {}

    for i in range(k):

        dict[i] = 0

    for i in range(0,Ytrain_clustered.shape[0]):

        if(Y_train[i]==1):

            dict[Ytrain_clustered[i]] += 1

        else:

            dict[Ytrain_clustered[i]] -= 1

    yes_list = []

    no_list = []

    for i in range(k):

        if(dict[i]>=0):

            yes_list.append(i)

        else:

            no_list.append(i)

    Y_predicted = kmeans_lite.predict(X_val_lda)

    for i in range(Y_predicted.shape[0]):

        if(Y_predicted[i] in yes_list):

            Y_predicted[i] = 1

        else:

            Y_predicted[i] = 0

    fpr, tpr, thresholds = roc_curve(Y_val,Y_predicted)

    auc_lite = auc(fpr, tpr)

    print(auc_lite)

    if(auc_lite>max_auc):

        max_auc = auc_lite

        best_cluster = k

print(max_auc,best_cluster)
clusters = best_cluster

#clusters = 3

#kmeans = KMeans(n_clusters=clusters, random_state=best_randomstate).fit(X_train)

X = np.concatenate((X_train_lda,X_val_lda))

Y = np.concatenate((Y_train,Y_val))

kmeans = KMeans(n_clusters=clusters, random_state=42).fit(X)

Ytrain_clustered = kmeans.labels_

print(Ytrain_clustered.shape)
dict = {}

for i in range(clusters):

    dict[i] = 0

for i in range(0,Ytrain_clustered.shape[0]):

    if(Y[i]==1):

        dict[Ytrain_clustered[i]] += 1

    else:

        dict[Ytrain_clustered[i]] -= 1

print(dict)

yes_list = []

no_list = []

for i in range(clusters):

    if(dict[i]>=0):

        yes_list.append(i)

    else:

        no_list.append(i)

print(yes_list,no_list)

Y_predicted = kmeans.predict(X_val_lda)

for i in range(Y_predicted.shape[0]):

    if(Y_predicted[i] in yes_list):

        Y_predicted[i] = 1

    else:

        Y_predicted[i] = 0

fpr, tpr, thresholds = roc_curve(Y_val,Y_predicted)

print("Validation Accuracy: " + str(auc(fpr, tpr)))

Y_predicted_test = kmeans.predict(X_test_lda)

for i in range(Y_predicted_test.shape[0]):

    if(Y_predicted_test[i] in yes_list):

        Y_predicted_test[i] = 1

    else:

        Y_predicted_test[i] = 0
out = pd.DataFrame({'custId':df_testx.iloc[:,0],'Satisfied':Y_predicted_test})

"""newname = './submission1.csv'

out.to_csv(newname,index=False)"""
best_cluster = 0

max_auc = 0

for k in range(2,40):

    kmeans_lite = KMeans(n_clusters=k,random_state=42).fit(X_train)

    #kmeans_lite = MiniBatchKMeans(n_clusters=k, random_state=42).partial_fit(X_train[0:2000,:])

    #kmeans_lite.partial_fit(X_train[2000:,:])

    Ytrain_clustered = kmeans_lite.labels_

    dict = {}

    for i in range(k):

        dict[i] = 0

    for i in range(0,Ytrain_clustered.shape[0]):

        if(Y_train[i]==1):

            dict[Ytrain_clustered[i]] += 1

        else:

            dict[Ytrain_clustered[i]] -= 1

    yes_list = []

    no_list = []

    for i in range(k):

        if(dict[i]>=0):

            yes_list.append(i)

        else:

            no_list.append(i)

    Y_predicted = kmeans_lite.predict(X_val)

    for i in range(Y_predicted.shape[0]):

        if(Y_predicted[i] in yes_list):

            Y_predicted[i] = 1

        else:

            Y_predicted[i] = 0

    fpr, tpr, thresholds = roc_curve(Y_val,Y_predicted)

    auc_lite = auc(fpr, tpr)

    print(auc_lite)

    if(auc_lite>max_auc):

        max_auc = auc_lite

        best_cluster = k

print(max_auc,best_cluster)
clusters = best_cluster

#clusters = 3

#kmeans = KMeans(n_clusters=clusters, random_state=best_randomstate).fit(X_train)

X = np.concatenate((X_train,X_val))

Y = np.concatenate((Y_train,Y_val))

kmeans = KMeans(n_clusters=clusters, random_state=42).fit(X)

Ytrain_clustered = kmeans.labels_

print(Ytrain_clustered.shape)
dict = {}

for i in range(clusters):

    dict[i] = 0

for i in range(0,Ytrain_clustered.shape[0]):

    if(Y[i]==1):

        dict[Ytrain_clustered[i]] += 1

    else:

        dict[Ytrain_clustered[i]] -= 1

print(dict)

yes_list = []

no_list = []

for i in range(clusters):

    if(dict[i]>=0):

        yes_list.append(i)

    else:

        no_list.append(i)

print(yes_list,no_list)

Y_predicted = kmeans.predict(X_val)

for i in range(Y_predicted.shape[0]):

    if(Y_predicted[i] in yes_list):

        Y_predicted[i] = 1

    else:

        Y_predicted[i] = 0

fpr, tpr, thresholds = roc_curve(Y_val,Y_predicted)

Y_predicted_test = kmeans.predict(X_test)

for i in range(Y_predicted_test.shape[0]):

    if(Y_predicted_test[i] in yes_list):

        Y_predicted_test[i] = 1

    else:

        Y_predicted_test[i] = 0
out = pd.DataFrame({'custId':df_testx.iloc[:,0],'Satisfied':Y_predicted_test})

"""newname = './submission2.csv'

out.to_csv(newname,index=False)"""