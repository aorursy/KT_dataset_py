import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, FeatureAgglomeration, AgglomerativeClustering
df = pd.read_csv('/kaggle/input/dmassign1/data.csv')

pd.set_option('display.max_columns', 500)

print(len(df))

df.head()
df.replace("?", np.nan, inplace=True)
missing_cols = df.isnull().sum()

# pd.set_option('display.max_rows', 70)

# (missing_cols[missing_cols>0]).sort_values()
df.info()
columns = df.columns

print(columns)

columns_numeric = columns[1:189]

df[columns_numeric] = df[columns_numeric].apply(pd.to_numeric)

df['Class'] = df['Class'].astype('object')
df.info()
class_vals = df['Class']

class_vals.unique()
df.fillna(df.mean(), inplace=True)
missing = df.isnull().sum()

print(missing[missing > 0])
df['Class'].unique()
idx = df[df['Col192'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)
idx = df[df['Col193'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)

idx = df[df['Col194'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)

idx = df[df['Col195'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)

idx = df[df['Col196'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)

idx = df[df['Col197'].isnull() & df['Class'].notnull()].index

df.drop(idx, inplace=True)
len(df)
missing = df.isnull().sum()

print(missing[missing > 0])
df.describe(include=object)
df['Col192'].fillna('p2', inplace=True)

df['Col194'].fillna('ad', inplace=True)

df['Col195'].fillna('Jb3', inplace=True)

missing = df.isnull().sum()

print(missing[missing > 0])
df['Class'] = df['Class'].astype('float64')
df.dtypes[df.dtypes == 'object']
print(df['Col189'].unique(), df['Col189'].nunique())

print(df['Col190'].unique(), df['Col190'].nunique())

print(df['Col191'].unique(), df['Col191'].nunique())

print(df['Col192'].unique(), df['Col192'].nunique())

print(df['Col193'].unique(), df['Col193'].nunique())

print(df['Col194'].unique(), df['Col194'].nunique())

print(df['Col195'].unique(), df['Col195'].nunique())

print(df['Col196'].unique(), df['Col196'].nunique())

print(df['Col197'].unique(), df['Col197'].nunique())
df['Col189'] = (df['Col189'] == 'yes').astype('int64')

df['Col189'].unique()
df['Col197'].replace({'me':'ME', 'sm':'SM', 'M.E.':'ME', 'la':'LA'}, inplace=True)

print(df['Col197'].unique(), df['Col197'].nunique())
df_obj_dummies = pd.get_dummies(data=df, columns=['Col190','Col191', 'Col192', 'Col193','Col194','Col195','Col196','Col197'])

df_obj_dummies.head()
len(df_obj_dummies.columns)
df_obj_dummies.drop(['ID','Class'],axis=1, inplace=True)

len(df_obj_dummies.columns)
len(df_obj_dummies)
corr = df_obj_dummies.corr()

corr.iloc[177, 178]
# scaled_data = StandardScaler().fit_transform(df_obj_dummies)

scaled_data = RobustScaler().fit_transform(df_obj_dummies)

scaled_df = pd.DataFrame(scaled_data, columns=df_obj_dummies.columns)

scaled_df.head()
len(scaled_df.columns), len(scaled_df)
scaled_data.shape
# y_true = y_true.astype('float32')



y_true = np.array(df['Class'][:1291])

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

dtree = RandomForestClassifier(random_state=7)

y_read = dtree.fit(scaled_df.iloc[:1291],y_true).predict(scaled_df.iloc[:1291])

# y_read = dtree.fit(pca_data_54[:1291],y_true).predict(pca_data_54[:1291])

dtree.feature_importances_



print(sum(y_true == y_read)/1292)
from sklearn.feature_selection import RFECV

from sklearn.model_selection import KFold

rfecv = RFECV(estimator=dtree, step=1, cv=StratifiedKFold(2), scoring='accuracy')

rfecv.fit(scaled_data[:1291], y_true)

selected = rfecv.support_
scaled_df.columns[selected]
scaled_data = np.array(scaled_df[scaled_df.columns[selected]])

scaled_data.shape
#PCA with 54 components

pca_54 = PCA(n_components=52)

pca_data_54 = pca_54.fit(scaled_data).transform(scaled_data)

print(pca_data_54.shape)

print(pca_54.explained_variance_ratio_.cumsum())
pca_data_54[:1291].shape
optimum_num_clusters=0

y_true = np.array(df['Class'][:1291])

optimum_acc = 0.0

for i_cluster in range(40, 70):

  print(f"Checking {i_cluster} number of clusters")

  kmeans = KMeans(n_clusters=i_cluster, random_state=42, max_iter=500)

  pred = kmeans.fit_predict(pca_data_54)

  pred_classified = pred[:1291]

  # print(pred.shape)

  cluster_dict = {}



  for cl_num in np.unique(pred):

    cluster_dict[cl_num] = []



  for ele, cl_num in enumerate(pred_classified):

    cluster_dict[cl_num].append(ele)



  pred_new = [0 for i in range(1291)]



  for cl_num in cluster_dict.keys():

    ele_arr = cluster_dict[cl_num]

    cl_grp = [0,0,0,0,0,0]

    for ele in ele_arr:

      label = int(y_true[ele])

      cl_grp[label] += 1

    grp_label = np.where(cl_grp == np.max(cl_grp))[0][0]

    for ele in ele_arr:

      pred_new[ele] = grp_label



  print(np.unique(pred_new))



  num_correct = (pred_new == y_true).sum()

  acc = num_correct/1291

  print(acc)



  if(acc > optimum_acc):

    print(f"Found better config with {i_cluster} number of clusters")

    optimum_acc = acc

    optimum_num_clusters = i_cluster
print(optimum_num_clusters)

print(optimum_acc)
kmeans = KMeans(n_clusters=optimum_num_clusters, random_state=42, max_iter=500)

pred = kmeans.fit(pca_data_54).predict(pca_data_54)

print(len(pred))

pred_classified = pred[:1291]



cluster_dict = {}



for cl_num in np.unique(pred):

  cluster_dict[cl_num] = []



for ele, cl_num in enumerate(pred_classified):

  cluster_dict[cl_num].append(ele)



pred_new = [0 for i in range(len(pred))]



for cl_num in cluster_dict.keys():

  ele_arr = cluster_dict[cl_num]

  cl_grp = [0,0,0,0,0,0]

  for ele in ele_arr:

    label = int(y_true[ele])

    cl_grp[label] += 1

  grp_label = np.where(cl_grp == np.max(cl_grp))[0][0]

  for i, ele in enumerate(pred):

    if(ele == cl_num):

      pred_new[i] = grp_label



print(np.unique(pred_new))



num_correct = (pred_new[:1291] == y_true).sum()

acc = num_correct/1291

print(acc)
df['Class'].value_counts()
np.unique(np.array(pred_new), return_counts=True)
count=0

weird = []

for i,ele in enumerate(pred_new):

  if ele == 0: #clusters with only test data points

    count+=1

    weird.append(pred[i])

    pred_new[i] = 3 #replace with label that is assigned to the least number of points

print(count)

print(np.unique(np.array(weird)))

print(np.unique(np.array(weird)).shape)
np.unique(pred_new)
ids = np.array(df['ID'].iloc[1300:])

ids.shape, len(pred_new[1300:])
final = pd.DataFrame({'ID':df['ID'].iloc[1291:], 'Class':pred_new[1291:]})

final.head()
len(final)
final.dtypes
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data_rfecv_stratified_rscale_pca_52_test.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)



create_download_link(final)