import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
data = pd.read_csv("/kaggle/input/college-data/data.csv")
data.head()
data.info()
pd.set_option('display.float', '{:.2f}'.format)
data.describe()
sns.scatterplot('room_board', 'grad_rate', data=data, hue='private')
sns.scatterplot('outstate', 'f_undergrad', data=data, hue='private')
plt.figure(figsize=(12, 8))

data.loc[data.private == 'Yes', 'outstate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'outstate'].hist(label="Non Private College", bins=30)

plt.xlabel('Outstate')
plt.legend()
plt.figure(figsize=(12, 8))

data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)

plt.xlabel('Graduation Rate')
plt.legend()
data.loc[data.grad_rate > 100]
data.loc[data.grad_rate > 100, 'grad_rate'] = 100
plt.figure(figsize=(12, 8))

data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)

plt.xlabel('Graduation Rate')
plt.legend()
from sklearn.cluster import KMeans
kmeans = KMeans(2)
kmeans.fit(data.drop('private', axis=1))
kmeans.cluster_centers_
data['private'] = data.private.astype("category").cat.codes
data.private
data.head()
kmeans.labels_
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(data.private, kmeans.labels_))
print(classification_report(data.private, kmeans.labels_))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(data.private, kmeans.labels_))
print(classification_report(data.private, kmeans.labels_))