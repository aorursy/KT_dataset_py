import sys

!{sys.executable} -m pip install sklearn

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
data_origin = pd.read_csv("../input/dmassign1/data.csv")

data = data_origin.iloc[:,:-1]

class_val = data_origin.iloc[:,-1]

data
class_val

class_val_train = class_val[:1300]

class_val_train.shape
data.replace({'?':np.NaN},inplace=True)
for idx, col in enumerate(data):

    print('#%d: %s' % (idx,data[col].isnull().sum()))
data.fillna(data.median(),inplace=True)
data['Col189'].fillna(value=data['Col192'].value_counts().index[0],inplace=True)

data['Col190'].fillna(value=data['Col193'].value_counts().index[0],inplace=True)

data['Col191'].fillna(value=data['Col194'].value_counts().index[0],inplace=True)

data['Col192'].fillna(value=data['Col195'].value_counts().index[0],inplace=True)

data['Col193'].fillna(value=data['Col196'].value_counts().index[0],inplace=True)

data['Col194'].fillna(value=data['Col192'].value_counts().index[0],inplace=True)

data['Col195'].fillna(value=data['Col193'].value_counts().index[0],inplace=True)

data['Col196'].fillna(value=data['Col194'].value_counts().index[0],inplace=True)

data['Col197'].fillna(value=data['Col195'].value_counts().index[0],inplace=True)
for idx, col in enumerate(data):

    print('#%d: %s' % (idx,data[col].isnull().sum()))
data.replace({'yes':0,'no':1}, inplace = True)
data = pd.get_dummies(data, columns=['Col190'], prefix = ['Col190'])

data = pd.get_dummies(data, columns=['Col191'], prefix = ['Col191'])

data = pd.get_dummies(data, columns=['Col192'], prefix = ['Col192'])

data = pd.get_dummies(data, columns=['Col193'], prefix = ['Col193'])

data = pd.get_dummies(data, columns=['Col194'], prefix = ['Col194'])

data = pd.get_dummies(data, columns=['Col195'], prefix = ['Col195'])

data = pd.get_dummies(data, columns=['Col196'], prefix = ['Col196'])

data = pd.get_dummies(data, columns=['Col197'], prefix = ['Col197'])
data = data.drop(['ID'], 1)
scaler=StandardScaler()

scaled_data=scaler.fit(data).transform(data)

scaled_df=pd.DataFrame(scaled_data,columns=data.columns)

scaled_df.tail()
(scaled_df.dtypes == object).any()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(scaled_df)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=0)

kmeans.fit(data)

cluster_prediction = kmeans.predict(data)
np.unique(cluster_prediction, return_counts=True)
rows,cols = 25,5

cluster_label_value = [[0 for i in range(cols)] for j in range(rows)]



    

for idx, value in enumerate(class_val_train):

    cluster_val = cluster_prediction[idx]

    cluster_label_value[cluster_val][int(value)-1]+=1



for r in range(rows):

    for c in range(cols):

        print(cluster_label_value[r][c], end=' ')

    print('\n')

    



cluster_to_label = [0 for i in range(rows)]

for r in range(rows):

    maxval=0

    maxind=0

    for c in range(cols):

        if maxval < cluster_label_value[r][c]:

            maxval = cluster_label_value[r][c]

            maxind = c

        cluster_to_label[r]=maxind

    

        

print(cluster_to_label)

test_val = []

test_string = []



for idx,prediction in enumerate(cluster_prediction[1300:]):

    test_val.append(cluster_to_label[prediction]+1)

    test_string.append("id{}".format(idx+1300))



test_new1 = []



for idx,prediction in enumerate(cluster_prediction[1300:]):

    test_new1.append(("id{}".format(idx+1300),cluster_to_label[prediction]+1))



test_new1
df = pd.DataFrame(test_new1, columns =['ID', 'Class'])

df
export_csv = df.to_csv(r'file.csv',index=False)
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

create_download_link(df)