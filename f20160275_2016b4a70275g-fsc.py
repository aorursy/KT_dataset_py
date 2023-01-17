import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/dmassign1/data.csv")
df.head()
df.info()
y = df['Class']
x = df.iloc[:,0:-1]
x.head()
x.replace('?',np.nan,inplace=True)
missing = x.isnull().sum()

# missing[missing>0]
missing.sum()
x_dtype_nunique = pd.concat([x.dtypes, x.nunique()],axis=1)

x_dtype_nunique.columns = ["dtype","unique"]

# x_dtype_nunique[:][150:]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(x.iloc[:, 1:189])

x.iloc[:, 1:189] = imputer.transform(x.iloc[:, 1:189])
x.iloc[:,1:189].isnull().sum().sum()
x.isnull().sum().sum()
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer = imputer.fit(x.iloc[:, 189:])

x.iloc[:,189:] = imputer.transform(x.iloc[:, 189:])
x.isnull().sum().sum()
cols = list(x.columns)[189:]
x= x.drop('ID',axis=1)
x = pd.get_dummies(x,columns=cols)
x_dup = x.drop_duplicates(subset=x.columns.difference(['ID']))
x_dup.info()
x.info()
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

x_sc=scaler.fit_transform(x)

x_dup_sc = scaler.transform(x_dup)
#Fitting the PCA algorithm with our Data

from sklearn.decomposition import PCA

pca = PCA().fit(x_sc)

#Plotting the Cumulative Summation of the Explained Variance



plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
pca=PCA(n_components=75)

principlecompo=pca.fit_transform(x_sc)

principlecompo_dup=pca.transform(x_dup_sc)
col2 = ['pc'+str(i) for i in range(75)]
principlecompo = pd.DataFrame(principlecompo, columns=col2)

principlecompo_dup = pd.DataFrame(principlecompo_dup, columns=col2)
principlecompo_dup.head()
pd.DataFrame()
# from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# from scipy.cluster.hierarchy import fcluster

# linkage_matrix1 = linkage(x_dup_sc, "average",metric="cosine")

# ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
from sklearn.cluster import AgglomerativeClustering



model = AgglomerativeClustering(n_clusters = 40,affinity='cosine',linkage="average")

model.fit_predict(x_dup_sc)

pred = model.fit_predict(x_sc)
from sklearn.metrics import confusion_matrix



predictions = pd.Series(pred+1,index=df.index,dtype = np.float64)

classes = (confusion_matrix(y[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)
predictions.value_counts()
from sklearn.metrics import accuracy_score

# confusion_matrix(y[:1300],predictions[:1300])

accuracy_score(y[:1300],predictions[:1300])
final= pd.read_csv('/kaggle/input/dmassign1/sample_submission.csv')
final.head()
final['Class'] = list(predictions[1300:])
final['Class'] = final['Class'].astype(int)
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

create_download_link(final)