import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
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
data=pd.read_csv('/kaggle/input/dmassign1/data.csv',low_memory=False)

data.head(10)
data.dtypes.value_counts()
data.describe()
data.describe(include=object)
data.info()
data.isnull().any()
data.replace('?',np.NaN, inplace = True)
data['Col189']=data['Col189'].map({'yes':1, 'no':0})
data['Col189'].unique()
data['Col190']=data['Col190'].map({'sacc1':1, 'sacc2':2, 'sacc3':3, 'sacc4':4, 'sacc5':5})
data['Col190'].unique()
data['Col191']=data['Col191'].map({'time1':1, 'time2':2, 'time3':3})
data['Col191'].unique()
data['Col192'] = data['Col192'].map({'p1':1,'p2':2,'p3':3,'p4':4,'p5':5,'p6':6,'p7':7,'p8':8,'p9':9,'p10':10})
data['Col192'].unique()
data['Col197'].unique()
data['Col197'] = data['Col197'].replace({'me':'ME', 'sm':'SM', 'M.E.':'ME','la':'LA'})

data['Col197'].unique()
data['Col197'] = data['Col197'].map({'SM':1, 'ME':2, 'LA':3, 'XL':4})
data['Col197'].unique()
data['Col195'].unique()
data['Col195'] = data['Col195'].map({'Jb1': 1, 'Jb2': 2, 'Jb3': 3, 'Jb4': 4})
data.isnull().any()
data.columns[data.isnull().any()] #only last 11700 rows of class are null
nonClass = data.columns.tolist()
nonClass = nonClass[:-1]
data[nonClass] = data[nonClass].fillna(data[nonClass].mean()) #for numerical
data[nonClass] = data[nonClass].fillna(data[nonClass].mode().iloc[0]) #for categorical
#removing categorical data
data.columns[data.isnull().any()]
data = pd.get_dummies(data=data,columns = ['Col193','Col194','Col195','Col196'])
columns = data.columns.tolist()
data.head()
numerical = data.columns.tolist()

numerical = numerical[1:-15]

data[numerical] = data[numerical].astype(float)
data.head()
X = data.drop(['ID','Class'],axis=1)

y = data['Class']

print(X.columns[X.isnull().any()])

X_temp = X

X_temp = normalize(X_temp,norm='l2')

X_temp = StandardScaler().fit_transform(X_temp)

X_temp = pd.DataFrame(X_temp,columns = X.columns)
X_temp.head()
X_temp.shape
pca = PCA().fit(X_temp)

plt.figure(figsize = (7,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('n_components')

plt.ylabel('variance')

plt.show()
# About 60 components are able to explain the variance
X_pca = PCA(n_components=60, random_state=42).fit_transform(X_temp)
k = 13
#clf = AgglomerativeClustering(n_clusters = k, affinity = 'cosine',linkage='average')

clf = KMeans(n_clusters=k, random_state = 100)
clf.fit(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clf.labels_, s=50, cmap='viridis')



centers = clf.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
y_pred = clf.labels_ + 1
uq,ele = np.unique(y_pred[:1300],return_counts=True)

print(np.asarray((uq,ele)))
def labelling_checker(actual_labels, pred_labels,k):

    for i in range(1,k+1):

        for j in range(1,6):

            cur_count = 0

            for k in range(len(actual_labels)):

                if(actual_labels[k] == j and pred_labels[k]==i):

                    cur_count = cur_count + 1

            print(i,j,cur_count)

        print(" ")

        
labelling_checker(y[:1300],y_pred[:1300],k)
mapped_y = []

for i in range(len(y_pred)):

    if(y_pred[i] == 1):

        mapped_y.append(1)

    elif(y_pred[i] == 2):

        mapped_y.append(5)

    elif(y_pred[i] == 3):

        mapped_y.append(2)

    elif(y_pred[i] == 4):

        mapped_y.append(2)

    elif(y_pred[i] == 5):

        mapped_y.append(3)

    elif(y_pred[i] == 6):

        mapped_y.append(1)

    elif(y_pred[i] == 7):

        mapped_y.append(5)

    elif(y_pred[i] == 8):

        mapped_y.append(3)

    elif(y_pred[i] == 9):

        mapped_y.append(2) 

    elif(y_pred[i] == 10):

        mapped_y.append(1)

    elif(y_pred[i] == 11):

        mapped_y.append(4)

    elif(y_pred[i] == 12):

        mapped_y.append(3)

    elif(y_pred[i] == 13):

        mapped_y.append(2)
accuracy_score(y[:1300],mapped_y[:1300])
ids = data['ID']
ids = ids[1300:]
ids = ids.tolist()
len(ids)
final_csv = pd.concat([pd.Series(ids),pd.Series(mapped_y[1300:],dtype=int)], axis = 1)

final_csv.columns = ['ID','Class']

final_csv.to_csv('final_csv.csv', index=False)

final_csv.head()
data.shape
data.head()
create_download_link(final_csv)