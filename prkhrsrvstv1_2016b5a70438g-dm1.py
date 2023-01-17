# Imports

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, Birch

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler



%matplotlib inline
# Read in data

X = pd.read_csv("../input/dmassign1/data.csv", index_col="ID")

X, y = X.drop(columns="Class"), X["Class"]

IDs = X.index
features = X

# Separate out categorical features

numerical_features = features.select_dtypes(include="number").columns

categorical_features = features.select_dtypes(exclude="number").columns
# Filter out the string values from categorical features

categorical_features_to_clean = []

for col in categorical_features:

  numeric_vals = set()

  string_vals = set()

  for val in features[col]:

    if pd.api.types.is_number(val):

      numeric_vals.add(val)

    else:

      string_vals.add(val)

  if len(numeric_vals) != 0:

    categorical_features_to_clean.append(col)
# Clean the disguised numerical features that need cleaning

# But need to replace '?' with NaN first

features.replace("?", np.nan, inplace=True)

for col in features.columns:

  try:

    features[col] = np.float64(features[col])

    features[col] = features[col].fillna(np.mean(features[col]))

  except ValueError:

    continue
# Separate out categorical features again

numerical_features = features.select_dtypes(include="number").columns

categorical_features = features.select_dtypes(exclude="number").columns
# Col197 has some dirty repeated labels like ('sm', 'SM'), ('me', 'ME', 'M.E.'), ('la', 'LA'). Handle these.

features["Col197"].replace({"sm":"SM", "me":"ME", "M.E.":"ME", "la":"LA"}, inplace=True)
# Get dummies for object-type columns

features = pd.get_dummies(features)

# Replace nans with mean for categorical columns

features = features.fillna(features.mean())

X = features

# Scale the data (only numerical attributes)

X.loc[:, "Col1":"Col188"] = RobustScaler().fit_transform(X.loc[:, "Col1":"Col188"])

# Reduce dimensions

X = PCA(n_components=20).fit_transform(X)
# Obtain predictions

clusterer = Birch(n_clusters=200, threshold=0.1, branching_factor=100).fit(X)

cluster_labels = pd.Series(clusterer.predict(X))

def map_cluster_to_class(cluster_labels, class_labels):

  temp = {}

  mapping = {}



  for cluster in np.unique(cluster_labels):

    mapping[cluster] = 0

    for class_label in np.unique(class_labels):

      temp[(cluster, class_label)] = 0

  

  for cluster in np.unique(cluster_labels):

    for i in range(len(class_labels)):

      if cluster_labels[i] == cluster:

        temp[(cluster, class_labels[i])] += 1

  

  for cluster in np.unique(cluster_labels):

    mapping[cluster] = max(np.unique(class_labels), key=lambda x: temp[(cluster, x)])

  

  return mapping
# Get a dict to replace change cluster_labels to prediction labels

replace_dict = map_cluster_to_class(cluster_labels[:sum(y.notna())], y[y.notna()])

for i in range(200):

  if i not in replace_dict.keys():

    replace_dict[i] = 2

# Replace cluster_labels by actual prediction labels

cluster_labels.replace(replace_dict, inplace=True)

# Accuracy on labeled data

print(np.mean(np.array((cluster_labels[:sum(y.notna())])) == np.array(y[y.notna()])))

# Final class-distribution on whole dataset

print(cluster_labels.value_counts())

# Visualize and compare class-distribution on labeled data

sns.distplot(np.array(cluster_labels[:sum(y.notna())]), bins=5, kde=False)

sns.distplot(np.array(y[y.notna()]), bins=5, kde=False)

# Create submission dataframe

submission = pd.DataFrame()

submission["ID"] = IDs[sum(y.notna()):]

submission["Class"] = np.int64(cluster_labels[sum(y.notna()):])

submission.to_csv("submission.csv", index=False)
# Visualize predicted class-distribution on whole dataset

sns.distplot(submission["Class"], kde=False, bins=5)
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

create_download_link(submission)