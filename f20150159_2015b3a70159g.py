import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



data_orig = pd.read_csv('../input/dataset.csv', sep=',')

data = data_orig

print(data.head())
data = data.replace('?',np.nan)



data['Account2'] = data['Account2'].str.upper()

data['Sponsors'] = data['Sponsors'].str.upper()

data['Plotsize'] = data['Plotsize'].str.upper()

data['Plotsize'] = data['Plotsize'].replace('M.E.','ME')



data['History'] = data['History'].replace(np.nan,data['History'].mode()[0])

data['Motive'] = data['Motive'].replace(np.nan,data['Motive'].mode()[0])

data['Employment Period'] = data['Employment Period'].replace(np.nan,data['Employment Period'].mode()[0])



data['Monthly Period'].fillna(-1,inplace=True)

data['Monthly Period'] = data['Monthly Period'].astype(int)

data['Monthly Period'] = data['Monthly Period'].replace(-1,data['Monthly Period'].mean())



data['Credit1'].fillna(-1,inplace=True)

data['Credit1'] = data['Credit1'].astype(int)

data['Credit1'] = data['Credit1'].replace(-1,data['Credit1'].mean())



data['InstallmentRate'] = data['InstallmentRate'].replace(np.nan,data['InstallmentRate'].mode()[0])



data['InstallmentCredit'].fillna(-1,inplace=True)

data['InstallmentCredit'] = data['InstallmentCredit'].astype(float)

data['InstallmentCredit'] = data['InstallmentCredit'].replace(-1,data['InstallmentCredit'].mean())



data['Yearly Period'].fillna(-1,inplace=True)

data['Yearly Period'] = data['Yearly Period'].astype(float)

data['Yearly Period'] = data['Yearly Period'].replace(-1,data['Yearly Period'].mean())





data['Tenancy Period'] = data['Tenancy Period'].replace(np.nan,data['Tenancy Period'].mode()[0])



data['Tenancy Period'] = data['Tenancy Period'].replace(-1,data['Tenancy Period'].mode()[0])



data['Age'].fillna(-1,inplace=True)

data['Age'] = data['Age'].astype(int)

data['Age'] = data['Age'].replace(-1,data['Age'].mean())



data['Account1'] = data['Account1'].replace(np.nan,data['Account1'].mode()[0])
data_high_dim = data.drop(['id','Class','Yearly Period','Monthly Period','Age','#Credits','Motive'],1)
data_high_dim = pd.get_dummies(data_high_dim,columns=[

            'Plotsize', 'Plan', 'Housing','History',

            'Account1','Account2','Phone','Post','Tenancy Period', 

           'Employment Period','InstallmentRate','#Authorities',

            'Expatriate','Sponsors','Gender&Type'

       ])
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.StandardScaler()

np_scaled = min_max_scaler.fit_transform(data_high_dim)

dataC1 = pd.DataFrame(np_scaled)

dataC1.head()
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)

pca2.fit(dataC1)

T2 = pca2.transform(dataC1)

reduced_PCA = PCA(n_components=4,random_state=42).fit_transform(dataC1)

from sklearn.cluster import KMeans

lim = 10

wcss = []

for i in range(2, lim):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(reduced_PCA)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,lim),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']

plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 3, random_state = 42, n_init=200)

kmean.fit(reduced_PCA)

pred = kmean.predict(reduced_PCA)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T2[j,0]

            meany+=T2[j,1]

            plt.scatter(T2[j, 0], T2[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
res = []

for i in range(len(pred)):

    if(pred[i]==0) :

        res.append(2)

    elif (pred[i]==1):

        res.append(1)

    else:

        res.append(0)     

len(res)
y=data['Class'].iloc[0:175].astype(int)



from sklearn.metrics import accuracy_score



accuracy_score(y,res[0:175])
res=res[175:]

len(res)
data_unsolved = data_orig.loc[175:]



data_unsolved.reset_index(drop=True, inplace=True)



res1 = pd.DataFrame(res)

final = pd.concat([data_unsolved["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()
final.to_csv('sub_init_58.csv', index = False)
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