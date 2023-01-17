import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from mpl_toolkits import mplot3d
pd.set_option('display.max_colwidth',500)

pd.set_option('display.max_rows', 300)
csv_path = '../input/dmassign1/data.csv'

df = pd.read_csv(csv_path)

df.head()
indices = df['ID'][1300:13000]
df = df.replace('?',np.NaN)

string_columns = ['ID', 'Col188', 'Col189', 'Col190', 'Col191', 'Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197']

df = df.drop(string_columns, axis=1)

df = df.apply(lambda x: x.astype(np.float64) if x.name not in ['ID'] + ['Class'] else x)

df.head()
X = df.drop(['Class'], axis=1)

y = df['Class']

print(X.head(), y.head())
X.fillna(X.mean(), inplace=True)
X.info()
print(X.shape, y.shape)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

print(scaled_data.shape)

scaled_data
# pca = PCA(n_components=3)

# tsne = TSNE(n_components=3, verbose=1, random_state=42, n_jobs=-1)

# X_tsne = tsne.fit_transform(scaled_data)
tsne = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=-1, perplexity=22)

X_tsne = tsne.fit_transform(scaled_data)
print(X_tsne.shape)

X_tsne
# ax = plt.axes(projection='3d')

# ax.scatter3D(X_tsne[0:1300,0], X_tsne[0:1300,1], X_tsne[0:1300,2], c=y[0:1300], cmap='hsv');

plt.scatter(X_tsne[0:1300, 0], X_tsne[0:1300, 1], c=y[0:1300], cmap='hsv')
classifier = KMeans(n_clusters=30, random_state=42)

classifier.fit(X_tsne)
np.unique(classifier.labels_, return_counts=1)
# print(classifier.labels_, y.astype(np.int32).to_numpy() - 1)
y_pred = classifier.labels_

print(y_pred.shape)

y_pred
y_true = y.dropna().astype(np.int32).to_numpy()

print(y_true)

y_true.shape
mapping = dict()

for i in range(30):

  l = y_true[y_pred[0:1300] == i]

  counts = np.bincount(l)

  mapping[i] = np.argmax(counts)
print(mapping)
labels = np.ndarray(13000)

for i in range(len(y_pred)):

  labels[i] = mapping[y_pred[i]]

print(np.unique(labels, return_counts=1))

labels
print(labels.tolist())
result_df = pd.DataFrame(data = list(zip(indices, labels[1300:13000])), columns = ['ID', 'Class'])
result_df['Class'] = result_df['Class'].astype(np.int64)
result_df['Class'].value_counts()
count=0

for i in range(1300):

  if labels[i] == y_true[i]:

    count+=1

    # print(i)

print(count)

# 594
result_df.shape
result_df.head(15)
result_df.to_csv('submission.csv', index=False)
len(result_df)
from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html) 

create_download_link(result_df)