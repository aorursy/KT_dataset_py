import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv('../input/dataset.csv')
data.head()
data['Age'][:5]
data.info()
data.describe()
for col in data.columns:

    print(col,len(data[col].unique()))
numerical = ['Credit1','InstallmentCredit','Yearly Period']
categorical_less = ['id','Housing','Plan','Phone','Expatriate',"#Authorities",'Class']



for col in categorical_less[1:]:

    print(data[col].value_counts())
all_columns = numerical + categorical_less
data_subset = data[all_columns]
data_subset.head()
data_subset['Class'] = data_subset['Class'].replace({np.nan:-1})
data_subset = data_subset.replace({'?':np.nan})
nans = lambda df: df[df.isnull().any(axis=1)]

id_null = nans(data_subset)['id']

len(data_subset.dropna())
data_subset = data_subset.dropna()
data_subset.info()
data_subset.head()
convert_type = ['InstallmentCredit','Yearly Period']

for col in convert_type:

    data_subset[col] = data_subset[col].astype('float')
data_subset.describe()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.Normalizer()

data_subset[numerical] = min_max_scaler.fit_transform(data_subset[numerical])



data_subset.head()
data_dummies = pd.get_dummies(data_subset,columns=['Housing','Plan','Phone','Expatriate'])
data_dummies
clust_data = data_dummies[['Credit1','InstallmentCredit','Yearly Period','#Authorities'

,'Housing_H1','Housing_H2','Housing_H3','Plan_PL1','Plan_PL2','Plan_PL3','Phone_no','Phone_yes','Expatriate_False','Expatriate_True']]
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(clust_data)

T1 = pca1.transform(clust_data)
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(clust_data)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.cluster import KMeans



kmean = KMeans(n_clusters = 15, random_state = 42)

kmean.fit(clust_data)

pred = kmean.predict(clust_data)
data_dummies['Class'][:174].values
pred = pred[:174]

true = data_dummies['Class'][:174].values
def cnt(true,pred,a,b):

    j = 0

    for i in range(len(pred)):

        if(pred[i]==a and true[i]==b):

            j+=1

    return j 



def matching(true,pred,num_clust):

    zero = []

    one = []

    two = []

    for i in range(num_clust):

        mx = -1

        mxVal = -1 

        for j in range(3):

            c = cnt(true,pred,i,j)

            if(c>mx):

                mx = c

                mxVal = j

        if(mxVal==0):

            zero.append(i)

        elif mxVal==1:

            one.append(i)

        else:

            two.append(i)

        print(i,mx,mxVal)

    return zero,one,two
zero,one,two = matching(true,pred,15)
two
pd.Series(pred).value_counts()
from sklearn.cluster import KMeans



kmean = KMeans(n_clusters = 15, random_state = 42)

kmean.fit(clust_data)

pred = kmean.predict(clust_data)
mapped_pred = []

for i in range(len(pred)):

    if(pred[i] in zero):

        mapped_pred.append(0)

    elif(pred[i] in one):

        mapped_pred.append(1)

    elif(pred[i] in two):

        mapped_pred.append(2)

    
centroids = kmean.cluster_centers_

centroids = pca1.transform(centroids)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
mapped_pred[:173]
pd.Series(mapped_pred[:174]).value_counts()
actual = data_dummies['Class'][:174]
actual.value_counts()
actual[:10]
len(mapped_pred[174:])
len(data_dummies['id'][174:])
len(id_null.values)
ids =list( data_dummies['id'][174:].values) + list(id_null.values[1:])
len(ids)
mapped_pred = mapped_pred + 6*[1]
final_pred = mapped_pred[174:]
len(final_pred)
res1 = pd.DataFrame(ids)

res1['Class'] = final_pred

res1 = res1.rename(columns={0: "id"})
res1.head()
res1.to_csv('sub.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(res1)
"66.121 with15 clusters,def 1,drop age and Normalizer"