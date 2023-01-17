import  pandas as pd

import numpy as np

from sklearn.cluster import KMeans
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df_submit = pd.DataFrame()

df_submit["custId"] = test["custId"]
clf = KMeans(n_clusters = 2)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler()
from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans

from sklearn.preprocessing import normalize

import imblearn
from collections import Counter

from imblearn.over_sampling import SMOTE
sm = SMOTE()
#0.73756

df2 = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

X = df2.drop(columns = "Satisfied")

y = df2["Satisfied"]



X.drop(columns= ["custId",'TVConnection',"Channel2", "Channel3", "Channel1","Channel4","Children","gender",'AddedServices', "HighSpeed"], inplace = True)

X = pd.get_dummies(data = X, columns=[ 

       'Channel6', "Channel5"

    , 'PaymentMethod'

    , "Subscription"

    , "Internet", 'Married'

       ])

X["TotalCharges"].replace(" ", 0, inplace = True)

#df2 = df2[df2["TotalCharges"].notnull()]

X["TotalCharges"] = X["TotalCharges"].astype(float)

#X = df2.drop(columns = ["Satisfied", "MonthlyCharges"])

#y = df2["Satisfied"]

#X.drop(columns = "MonthlyCharges", inplace = True)

X, y = sm.fit_resample(X, y)

X = scaler.fit_transform(X)

clf = KMeans(n_clusters=2)

clf.fit(X)

y_pred = clf.labels_

score = roc_auc_score(y, y_pred)

if score < 0.5:

    print(1 - score)

else:

    print(score)

    



test2 = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

test2.drop(columns= ["custId",'TVConnection',"Channel2", "Channel3", "Channel1","Channel4","Children","gender",'AddedServices', "HighSpeed"], inplace = True)

test2 = pd.get_dummies(data = test2, columns=[ 

       'Channel6', "Channel5"

    , 'PaymentMethod'

    , "Subscription"

     

    , 'Married', "Internet"

       ])

test2["TotalCharges"].replace(" ", 0, inplace = True)

#df2 = df2[df2["TotalCharges"].notnull()]

test2["TotalCharges"] = test2["TotalCharges"].astype(float)

#test2 = test2.drop(columns = ["MonthlyCharges"])

test2 = scaler.fit_transform(test2)

y_pred = clf.predict(test2)

#y_pred = clf.labels_

df_submit["Satisfied"] = abs(1 - y_pred)

df_submit.to_csv("1.csv", index = False)