import pandas as pd
import numpy as np
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")
pd.set_option('display.max_columns', 500)
cols_to_drop = ['max_glu_serum', 'A1Cresult', 'weight', 'medical_specialty', 'payer_code']
train = train.drop(cols_to_drop, axis=1)
temps = {}
for col in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'diabetesMed', 'race', 'gender', 'age', 'diabetesMed']:
    temps[col] = pd.DataFrame({
        'data': train[col].unique(), 
        'data_new':range(len(train[col].unique()))
    })

    for index, row in temps[col].iterrows():
        train = train.replace(row['data'], row['data_new'])
train = train.replace('?', 0)
for cols in train:
    print(train[col].dtype)
train.head()
train = train.drop(["diag_3", "diag_1", "diag_2"], axis=1)
X = np.array(train.drop(['readmitted_NO'], 1))
y = np.array(train['readmitted_NO'])
X.shape
y.shape
X
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
test = test.drop(cols_to_drop, axis=1)
for col in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'diabetesMed', 'race', 'gender', 'age', 'diabetesMed']:

    for index, row in temps[col].iterrows():
        test = test.replace(row['data'], row['data_new'])
test = test.drop(["diag_3", "diag_1", "diag_2", "index"], axis=1)
test = test.replace('?', 0)
test = test.replace('Yes', 0)
X_ = np.array(test)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled_ = scaler.fit_transform(X_)
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(X, y)
u = clustering.fit_predict(X_)
u
np.unique(u)
u = clustering.predict(X_)
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage="complete", affinity="l2").fit(X_scaled, y)
u = clustering.predict(X_scaled_)
new_df = pd.DataFrame({"index": [i for i in range(len(u))], "target": u})
new_df.to_csv("solution2.csv", index=False)