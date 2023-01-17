import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering as AC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score,confusion_matrix

from IPython.display import HTML

import base64
data_orig = pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',')

data = data_orig

data.info()
data.replace(to_replace='?', value=np.nan, inplace=True)

X = data.copy()

y = data['Class'].copy()

imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer1 = imputer1.fit(X.iloc[:, 1:189])

X.iloc[:, 1:189] = imputer1.transform(X.iloc[:, 1:189])



imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer2 = imputer2.fit(X.iloc[:, 189:199])

X.iloc[:, 189:199] = imputer2.transform(X.iloc[:, 189:199])
cat_list = ['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']

for col in cat_list:

    x = pd.get_dummies(X[col], prefix=col, prefix_sep='_').copy()

    X = X.drop(col,axis = 1)

    X = X.join(x)
X_org = X.drop(['ID','Class'],axis=1).copy()

X_wd = X.drop_duplicates(subset=X.columns.difference(['ID'])).copy()

X_wd_wid = X_wd.drop(['ID','Class'],axis=1).copy()



stdsc = preprocessing.StandardScaler()

np_scaled = stdsc.fit_transform(X_wd_wid)

np_scaled1 = stdsc.transform(X_org)



X_wd_wid_sc = pd.DataFrame(np_scaled)

X_org_sc = pd.DataFrame(np_scaled1)

X_wd_wid_sc.head()
pca1 = PCA(n_components=50)

pca1.fit(X_wd_wid_sc)

T1 = pca1.transform(X_org_sc)



pc_cols = ["pc"+str(i) for i in range(50)]

pca_df = pd.DataFrame(data=T1, columns = pc_cols)
#HEIRARCHICAL CLUSTERING OF DATA TO 20 CLUSTERS

aggclus = AC(n_clusters = 20,affinity='cosine',linkage='average',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(pca_df)



#WE MAP THESE 20 CLUSTERS TO 5 CLUSTERS

predictions = pd.Series(y_aggclus+1,index=data.index,dtype = np.float64)

classes = (confusion_matrix(y[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)



#MODEL EVALUATION ON LABELLED SET

print("Accuracy on labelled part of data:-")

print(accuracy_score(y[:1300],predictions[:1300]))
out_df_ac1 = pd.DataFrame(columns=['ID','Class'])

out_df_ac1['ID'] = data['ID'].copy()

out_df_ac1['Class'] = list(predictions)

out_df_ac1['Class'] = out_df_ac1['Class'].astype(int)

odf_ac1 = out_df_ac1.iloc[1300: , :].copy()

odf_ac1.to_csv('AC9.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(odf_ac1)