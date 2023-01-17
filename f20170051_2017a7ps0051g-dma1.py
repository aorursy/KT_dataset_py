import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
def accuracy_score(list1, list2):

    count = 0

    for i in range(len(list1)):

        if(list1[i]==list2[i]):

            count+= 1

    print(count/len(list1))
import os

os.getcwd()
df = pd.read_csv('/kaggle/input/dmassign1/data.csv',low_memory = False)

df.head()
df.info()
df.describe()
df.replace('?',np.NaN,inplace = True)
le = LabelEncoder()
df['Col189'] = le.fit_transform(df['Col189'])

df['Col190'] = df['Col190'].replace({'sacc1':1, 'sacc2':2, 'sacc4':3, 'sacc5':4})

df['Col191'] = df['Col191'].replace({'time1':1, 'time2':2, 'time3':3})

df['Col192'] = df['Col192'].replace({'p1':1,'p2':2,'p3':3,'p4':4,'p5':5,'p6':6,'p7':7,'p8':8,'p9':9,'p10':10})

df['Col192'].fillna(df['Col192'].value_counts().idxmax(),inplace = True)

df['Col193'].fillna(df['Col193'].value_counts().idxmax(),inplace = True)
df['Col194'].fillna(df['Col194'].value_counts().idxmax(),inplace = True)

df['Col194'] = df['Col194'].replace({'ab':1, 'ac':2, 'ad':3})

df['Col195'].fillna(df['Col195'].value_counts().idxmax(),inplace = True)

df['Col195'] = df['Col195'].replace({'Jb1':1, 'Jb2':2, 'Jb3':3, 'Jb4':4})

df['Col196'].fillna(df['Col196'].value_counts().idxmax(),inplace = True)

df['Col196'] = df['Col196'].replace({'H1':1, 'H2':2, 'H3':3})

df['Col197'] = df['Col197'].replace({'me':'ME', 'sm':'SM', 'M.E.':'ME','la':'LA'})

df['Col197'].fillna(df['Col197'].value_counts().idxmax(),inplace = True)

df['Col197'] = df['Col197'].replace({'SM':1, 'ME':2, 'LA':3, 'XL':4})

df = pd.get_dummies(data=df,columns = ['Col193','Col196','Col191'])
cols = [col for col in df if col != 'Class'] + ['Class']
#cols = cols + ['Class']
df = df[cols]
df.head()
obj_cols = df.select_dtypes(include=[object]).columns.to_list()
obj_cols = obj_cols[1:]
df[obj_cols] = df[obj_cols].astype(float)
data = df.copy()
X = data.drop(['ID','Class'],axis=1)

y = data['Class']
X.fillna(X.mean(),inplace = True)
X.head()
X_data = X
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X,columns = X_data.columns)

X.head()
# X.columns.tolist()
#X2 = X.drop(['Col189','Col190','Col191','Col192','Col194','Col195','Col196','Col197','Col193_F0','Col193_F1','Col193_M0','Col193_M1'],axis=1)
X_data.head()
X2 = X_data


X2 = StandardScaler().fit_transform(X2)

X2 = normalize(X2,norm='l1')

X2 = pd.DataFrame(X2,columns = X_data.columns)
from sklearn.decomposition import PCA
pca = PCA(n_components=70)

principalComponents = pca.fit_transform(X2)

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_ratio_, color='blue')

plt.xlabel('PCA features')

plt.ylabel(range(10))

plt.xticks(features)

# Save components to a DataFrame

PCA_components = pd.DataFrame(principalComponents)
from sklearn.decomposition import PCA

pca = PCA().fit(X2)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title('PCA variance cumulative')

plt.xlabel('n_components')

plt.ylabel('variance')

plt.show()
X2.shape
X2.head()
#clf2 = AgglomerativeClustering(n_clusters = k, affinity = 'cosine', linkage='average') 

#clf3 = DBSCAN(eps = 1)

#clf4 = Birch(threshold = 0.25, n_clusters = k, branching_factor=25)
#Algorithm for mapping

def check_acc(old_y,new_y,k,mp):

    for i in range(1,k+1):

        mx = -1;

        mxInd = 1;

        for j in range(1,6):

            count = 0

            num_count = 0

            for k in range(len(new_y)):

                if(old_y[k] == j):

                    num_count += 1

                if(old_y[k]==j and new_y[k]==i):

                    count += 1

            frac = count/num_count

            if(mx<frac):

                mx = frac

                mxInd = j

            #print(i,j,count/num_count)

        mp.append(mxInd)

        #print(mp[i-1])

        #print(" ")
for k in range(5,20):

    kmeans = KMeans(n_clusters = k,random_state=100).fit(principalComponents)

    y2 = kmeans.labels_

    y2 = y2 + 1

    mp = []

    mp = []

    check_acc(y[:1300],y2[:1300],k,mp)

    edited_y = []

    for i in range(len(y2)):

        for j in range(1,k+1):

            if(y2[i]==j):

                edited_y.append(mp[j-1])

    print(k)

    accuracy_score(edited_y[:1300],y[:1300])

    unique_elements, counts_elements = np.unique(edited_y[1300:], return_counts=True)

    print(np.asarray((unique_elements, counts_elements)))
k = 16 #final
kmeans = KMeans(n_clusters = k, random_state=100).fit(principalComponents)
y2 = kmeans.labels_

y2 = y2 + 1
unique_elements, counts_elements = np.unique(y2[:1300], return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(y[:1300], return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
mp = []

check_acc(y[:1300],y2[:1300],k,mp)
edited_y = []

for i in range(len(y2)):

    for j in range(1,k+1):

        if(y2[i]==j):

            edited_y.append(mp[j-1])
accuracy_score(edited_y[:1300],y[:1300])
unique_elements, counts_elements = np.unique(edited_y[1300:], return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
ids = df['ID']

ids = ids[1300:]

ids = ids.tolist()
submission = pd.concat([pd.Series(ids),pd.Series(edited_y[1300:])],axis=1)

submission.columns = ['ID','Class']

submission.to_csv('submission.csv', index=False)

submission.head()
#Ultimate Mapping

maps = {}

for i in range(len(y)):

    key = i

    maps.setdefault(key, [])
# accuracy_score(edited_y[:200],y[:200])

# accuracy_score(edited_y[200:400],y[200:400])

# accuracy_score(edited_y[400:600],y[400:600])

# accuracy_score(edited_y[600:800],y[600:800])

# accuracy_score(edited_y[800:1000],y[800:1000])

# accuracy_score(edited_y[1000:1200],y[1000:1200])
for i in range(len(y)):

    maps[i].append(edited_y[i])
for i in range(10):

    print(maps[i], y[i])
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
create_download_link(submission)