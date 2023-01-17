# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',',index_col=0, na_values = '?')

df = data_orig

df_train = data_orig.iloc[:1300]

pd.set_option('display.max_columns', 230)
for col in df.columns:

    if df[col].dtypes == np.int64 or df[col].dtypes == np.float64:

        df[col] = df[col].fillna(df[col].median())

    else:

        df[col] = df[col].fillna(df[col].value_counts().index[0]) 
obj_col = df.columns[df.dtypes == 'object']

df_onehot = df.drop(obj_col, axis='columns')

df_onehot.info()
from sklearn.preprocessing import StandardScaler

df1 = df_onehot.copy()

scaler = StandardScaler()

scaled_data = scaler.fit(df1).transform(df1)

scaled_df=pd.DataFrame(scaled_data,columns=df1.columns, index=df1.index)

scaled_df = scaled_df.drop('Class', axis=1)

scaled_df
from sklearn.manifold import TSNE



model=TSNE(n_iter=10000,n_components=2,perplexity=100)

#scaled_df.drop(columns='age')

model_data=model.fit_transform(scaled_df)

model_data
plt.scatter(model_data[:,0], model_data[:,1])
model_data = pd.DataFrame(model_data, index = scaled_df.index)
from sklearn.cluster import KMeans



wcss = []

for i in range(20, 70):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(model_data)

    wcss.append(kmean.inertia_)

    

plt.plot(range(20,70),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(16, 8))

pred=[]

ks = list(range(50,75))

for i in ks:

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(model_data)

    temp = kmean.predict(model_data)

    print(". ", end="")

    pred.append(temp)



all_predictions = pd.DataFrame(np.array(pred).T, index=scaled_df.index, columns=ks)
df_train = data_orig.iloc[:1300]

values = []

for K in all_predictions.columns:

    pred_pd = all_predictions[K]

    

    y_true = df_train['Class']

    y_pred = pred_pd.iloc[:1300]

    

    predictions = (pred_pd+1).astype(np.float64)

    classes = (confusion_matrix(y_true, y_pred+1).argmax(axis=0)+1)

    predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

    

    predictions = predictions.astype(int)

#     predictions = pd.DataFrame(predictions)

#     print(predictions)

    values.append((K, (predictions[:1300] == y_true).sum().sum()/1300))

#     print(predictions)

#     break
plt.plot([i for (i, _) in values ], [i for (_, i) in values ])
kmean = KMeans(n_clusters = 74, random_state = 42)

kmean.fit(model_data)

pred = kmean.predict(model_data)

pred_pd = pd.DataFrame(pred)
y_true = df_train['Class']

y_true.astype(int)

predictions = (pred+1).astype(np.int64)

y_pred = predictions[0:1300]



classes = (confusion_matrix(df_train['Class'], predictions[:1300]).argmax(axis=0)+1)

predictions = pd.DataFrame(predictions, columns = ['Class'],index = df.index )

predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

predictions
test_pred = predictions[1300:]

test_pred
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

create_download_link(test_pred)