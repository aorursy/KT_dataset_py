# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dmassign1/data.csv')

df_orig = df
sns.countplot(df['Class'])
df.dtypes.value_counts()
# sm=0

# for i in df.columns.values:

#     sm += len(df[df[i]==ii64.min])

#     sm += len(df[df[i]=='?'])

#     #print(len(df[df[i]=='?']))

# print(sm)
ii64 = np.iinfo(np.int64)



for i in df.columns.values:

    df[i].replace('?', ii64.min, inplace=True)

    
categorical_cols = []

for i in df.columns.values:

    if(df[i].dtype == 'object'):

        try:

            df[i] = df[i].astype(float)

        except:

            categorical_cols.append(i)
print(len(categorical_cols))
for i in df.columns.values:

    if(df[i].dtype == 'object'):

        df[i].replace(ii64.min, df[i].mode()[0], inplace=True)

    else:

        df[i].replace(ii64.min, df[i].median(), inplace=True)
categorical_cols
df = pd.get_dummies(df, columns = ['Col189', 'Col190', 'Col191', 'Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197'])
df_dum = df
df = df_dum
len(df.columns.values)
from sklearn.preprocessing import StandardScaler as SS
df.drop(['ID', 'Class'], axis=1, inplace=True)
df = SS().fit_transform(df)
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(df)

T1 = pca1.transform(df)
from sklearn.cluster import AgglomerativeClustering as AC

agg = AC(n_clusters = 10, affinity='euclidean', linkage='ward', compute_full_tree='auto')

y_agg = agg.fit_predict(df)
plt.scatter(T1[:, 0], T1[:, 1], c=y_agg)
unique_elements, counts_elements = np.unique(y_agg, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
from sklearn.cluster import KMeans

kmean = KMeans(n_clusters=20, random_state=42)

y_pred = kmean.fit_predict(df)
plt.scatter(T1[:, 0], T1[:, 1], c=y_pred)
unique_elements, counts_elements = np.unique(y_pred, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
len(df)
cls_20 = {}

for i in range(0, 20):

    cls_20[i]=[0, 0, 0, 0, 0]
for i in range(0, 1300):

    j = y_pred[i]

    lbl = int(df_orig['Class'][i])

    cls_20[j][lbl-1] += 1
cls_20
mapping = [4,

          1,

          1,

          1,

          5,

          1,

          1,

          1,

          1,

          1,

          1,

          1,

          1,

          1,

          3,

          4,

          1,

          1,

          1,

          5 ]
len(mapping)
#save predictions in list



res_arr = [[],[]]



for i in range(1300, 13000):

    res_arr[0].append(df_orig['ID'][i])

    res_arr[1].append(int(mapping[y_pred[i]]))

unique_elements, counts_elements = np.unique(res_arr[1], return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
#create submission csv file

submission = pd.DataFrame(

                {'Id' : res_arr[0],

                 'Class': res_arr[1]})



submission.to_csv('sub_0033_dm1.csv', index=False)
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