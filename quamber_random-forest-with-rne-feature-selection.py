import numpy as np

import pandas as pd

# Classifier used

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# Best Feature selection

from sklearn.feature_selection import RFE

# Input Matrix

dataframe_trn = pd.read_csv("../input/train.csv")

dataframe_tst = pd.read_csv("../input/test.csv")



#Target Vector

labels_trn = dataframe_trn[["target","id"]]



#Dropping columns

dataframe_trn = dataframe_trn.drop(["target","id"],axis=1)

ids = pd.DataFrame(dataframe_tst["id"].values,columns= ["id"])

dataframe_tst = dataframe_tst.drop("id",axis=1)

headers = dataframe_trn.columns.values

# String to numeric Mapping

dict_mapping = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}

labels_trn_numeric = labels_trn["target"].apply(lambda x: dict_mapping[x])
# Reducing Dimensions

#trn_embedded = TSNE(n_components=2).fit_transform(dataframe_trn)



#Plotting the Dataset

#plt.scatter(trn_embedded[:, 0], trn_embedded[:, 1],c=labels_trn_numeric)

#plt.show()
clf = RandomForestClassifier( n_estimators=500,n_jobs=4)

rfe = RFE(clf, 85)

fit = rfe.fit(dataframe_trn, labels_trn_numeric)

features = []

for i , j in zip(dataframe_trn.columns,fit.support_):

    if j == True:

        features.append(str(i))
array = pd.DataFrame(fit.predict_proba(dataframe_tst),columns=["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])

complete_array = pd.concat([ids,array],axis=1)

complete_array.to_csv("submission.csv",sep=",",index=None)