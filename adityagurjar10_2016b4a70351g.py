import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm
df_orig = pd.read_csv('../input/dmassign1/data.csv')

df_orig.head()
y_given = df_orig['Class'].iloc[0:1300].astype(int)

df = df_orig.iloc[:,0:189]

#ids = df_orig['ID']

#df = df_orig.drop_duplicates(subset=df.columns.difference(['ID']))

df.shape
df.head(30)
df.replace('?',np.nan, inplace=True)
null_columns = df.columns[df.isnull().any()]

null_columns
# df.isnull().sum()
df.isnull().sum().sum()
#df.dropna(inplace=True)
# 1:
#df=df.drop_duplicates()
df.info()
y_given.isnull().sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(df.iloc[:, 1:189])

df.iloc[:, 1:189] = imputer.transform(df.iloc[:, 1:189])
df.isnull().sum().sum()
# from sklearn.impute import SimpleImputer



# imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

# imputer = imputer.fit(df.iloc[:, 189:199])

# df.iloc[:, 189:199] = imputer.transform(df.iloc[:, 189:199])
df.isnull().sum().sum()
df.shape
# onehot = pd.get_dummies(df.iloc[:,189:])

# df.drop(df.iloc[:,189:], inplace=True,axis=1)

# df = df.join(onehot)

# df.head()
df.drop(columns='ID',inplace=True)

df.head()
scaler=StandardScaler()

scaled_data=scaler.fit(df).transform(df)

scaled_df1=pd.DataFrame(scaled_data,columns=df.columns)

scaled_df1.tail()
# #Fitting the PCA algorithm with our Data

# pca = PCA().fit(scaled_df1)

# #Plotting the Cumulative Summation of the Explained Variance



# plt.figure()

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlabel('Number of Components')

# plt.ylabel('Variance (%)') #for each component

# plt.title('Pulsar Dataset Explained Variance')

# plt.show()
# pca_model=PCA(n_components=75)

# pca_model_data=pca_model.fit(scaled_df1).transform(scaled_df1)

# pca_model_data.shape
from sklearn.cluster import AgglomerativeClustering as AC



ac_model = AC(n_clusters=25, affinity='cosine', linkage='average',compute_full_tree='auto')
final_label = ac_model.fit_predict(scaled_df1)

final_label
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(scaled_df1, "average",metric="cosine")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
final_label.shape
from sklearn.metrics import confusion_matrix

final_predictions = pd.Series(final_label+1,index=df.index,dtype = np.float64)

classes = (confusion_matrix(y_given[:1300],final_predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

final_predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)
final_predictions.value_counts()
preds = final_predictions.iloc[:1300]

final_predictions = final_predictions.iloc[1300:]
final_predictions.shape
final_predictions.astype('int32').head()
ans = pd.read_csv('../input/dmassign1/sample_submission.csv')

ans['Class'] = list(final_predictions)
ans['Class'] = ans['Class'].astype(int)
ans.head()
ans.to_csv('preds8.csv', index=False)
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(y_given[:1300],preds)
confusion_matrix(y_given[:1300],final_predictions[:1300])
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

create_download_link(ans)