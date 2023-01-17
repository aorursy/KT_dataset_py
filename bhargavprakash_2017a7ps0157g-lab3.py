import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df.shape
df.head()
df.dtypes
l = {"Satisfied", "tenure", "SeniorCitizen","TotalCharges", "PaymentMethod", "Subscription", "gender", "Married", "Children", "TVConnection", "Channel1", "Channel2", "Channel3", "Channel4",  "Channel5", "Channel6", "Internet", "HighSpeed", "AddedServices"}

for i in l:

    df[i] = df[i].astype('category')

    df[i] = df[i].cat.codes

    df[i] = df[i].astype('float64')

df.dtypes
df.head()
corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X = df[["TVConnection","Channel5","Channel6","TotalCharges","Subscription","tenure"]].copy()

y = df["Satisfied"].copy()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X) 



X_scaled
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



clf = KMeans(n_clusters = 2, random_state=942)

clf.fit(X_scaled)

y_labels = clf.labels_

#y_pred = clf.predict(X_pred_scaled)

accuracy = accuracy_score(y, y_labels)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',

                           assign_labels='kmeans')

labels = model.fit_predict(X_scaled)

#y_pred = model.fit_predict(X_pred_scaled)

accuracy = accuracy_score(y, labels)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.cluster import MiniBatchKMeans



kmeans = MiniBatchKMeans(2)

labels=kmeans.fit_predict(X_scaled)

#y_pred = kmeans.predict(X_pred_scaled)

accuracy = accuracy_score(y, labels)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
df1 = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

df1.shape
df1.dtypes
l = {"tenure", "SeniorCitizen","TotalCharges", "PaymentMethod", "Subscription", "gender", "Married", "Children", "TVConnection", "Channel1", "Channel2", "Channel3", "Channel4",  "Channel5", "Channel6", "Internet", "HighSpeed", "AddedServices"}

for i in l:

    df1[i] = df1[i].astype('category')

    df1[i] = df1[i].cat.codes

    df1[i] = df1[i].astype('float64')

df1.dtypes
X_pred = df1[["TVConnection","Channel5","Channel6","TotalCharges","Subscription","tenure"]].copy()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = MinMaxScaler()

X_pred_scaled = scaler.fit_transform(X_pred) 



X_pred_scaled
clf.fit(X_pred_scaled)

y_pred=clf.labels_

predictions = [round(value) for value in y_pred]
submission = pd.DataFrame({'custId':df1['custId'],'Satisfied':predictions})

submission.shape
filename = 'predictions13.csv'



submission.to_csv(filename,index=False)