import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm

import seaborn as sns

from sklearn.metrics import silhouette_score
df = pd.read_csv('../input/dmassign1/data.csv')
#taking care of missing values and categorical variables

df = df.replace({'?': np.nan})

df = df.fillna(df.mean())

y = df['Class']

y = y[0:1300]

df = df.drop(['Class'], axis = 1)

df.fillna(value = df.mode().loc[0], inplace = True)

cols = ['Col179','Col180','Col181','Col182','Col183','Col184','Col185','Col186','Col187']

for i in cols:

    df[i] = df[i].astype('float64')

onehot = ['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']

for i in onehot:

    df = pd.get_dummies(df, columns= [i], prefix = [i])

df = df.drop(['ID'], axis = 1)

for i in df.columns:

    df[i] = df[i].astype('float64')

#standardizing the entire data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler() 

df2 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df2.head(10)

df2.astype('float64')

df3 = pd.read_csv('/kaggle/input/dmassign1/data.csv')

y = df3['Class'].iloc[:1300]
y.value_counts()
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 14))   

met = shc.linkage(df2, metric='cosine', method='average') 

dend = shc.dendrogram(met, p=6, truncate_mode='level')
from sklearn.cluster import AgglomerativeClustering 



ac3 = AgglomerativeClustering(n_clusters = 28, affinity = 'cosine', linkage = 'average') 

labels = ac3.fit_predict(df2)

labels = pd.DataFrame(data = labels, columns = ['Class'])

labels = labels +1

labels2 = labels.iloc[0:1300, 0]



# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=18, init='k-means++', max_iter=50)

# labels = kmeans.fit_predict(df2)

# labels = pd.DataFrame(data = labels, columns = ['Class'])

# labels = labels +1

# labels2 = labels.iloc[0:1300, 0]

# df2.head()

labels2.value_counts()

# y.value_counts()
from sklearn.metrics import confusion_matrix 

conf = confusion_matrix(y, labels2)[:5, :]  

import matplotlib.pyplot as plt      

plt.figure(figsize=(10, 8))

p = plt.subplot()

sns.heatmap(conf, annot=True, ax = p, cmap='Blues')

p.set_xlabel('Clusters')

p.set_ylabel('Class_labels')

p.set_title('CM')

p.yaxis.set_ticklabels([1, 2, 3, 4, 5])

p.xaxis.set_ticklabels(list(range(1,30)))
#mapping the 28 clusters formed to the 5 actual classes that they belong to 



pairs = []

cluster_no=1

 

for i in conf.T:

    

    actual = np.where(i == i.max())[0][0]+1

    pairs.append([cluster_no, actual])

    cluster_no+=1

    
pairs
ind = pd.read_csv('../input/dmassign1/data.csv')

# y = ind['Class']

# ind = ind.iloc[:, 0]

# ind = pd.DataFrame(data = ind)

ind.head()
 

def mapping(pred, pairs):

    

    actual_label_list = []

    for i in pred:

        actual_label = pairs[i-1][1]+50

        actual_label_list.append(actual_label)

        

    actual_label_list = [i-50 for i in actual_label_list]

    

    return actual_label_list

 

final_mapped =  pd.DataFrame(mapping(labels['Class'].values.tolist(), pairs), columns=['Class'], 

                             index=ind['ID']) 



# mapped_labels.columns = ['Class']

final_mapped.head(50)
ans = final_mapped.iloc[1300:13000, :]

ans.head()

ans.to_csv('dm_16.csv')
# from IPython.display import HTML

# import pandas as pd

# import numpy as np

# import base64def 

# create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

#     csv = df.to_csv(index=False)

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html     =     '<a     download="{filename}"     href="data:text/csv;base64,{payload}

#     "target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

# return HTML(html)

# create_download_link(<submission_DataFrame_name>)











from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(ans)


