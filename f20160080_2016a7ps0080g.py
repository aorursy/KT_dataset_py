import numpy as np

import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import seaborn as sns
data_orig = pd.read_csv("../input/dataset.csv", sep=',')

data = data_orig
data.info()
data
data.replace({'?':np.NaN},inplace=True)
data.isnull().sum()
data=data.rename(index=str,columns={"Yearly Period":"YearlyPeriod"})

data['InstallmentCredit']=data.InstallmentCredit.astype(float)

data['YearlyPeriod']=data.YearlyPeriod.astype(float)
data['Monthly Period'].replace({

  np.NaN:data['Monthly Period'].median(),

  },inplace=True)

data['Credit1'].replace({

  np.NaN:data['Credit1'].median(),

  },inplace=True)

data['InstallmentRate'].replace({

  np.NaN:data['InstallmentRate'].median(),

  },inplace=True)

data['Tenancy Period'].replace({

  np.NaN:data['Tenancy Period'].median(),

  },inplace=True)

data['Age'].replace({

  np.NaN:data['Age'].median(),

  },inplace=True)



data['InstallmentCredit'].replace({

  np.NaN:data['InstallmentCredit'].mean(),

  },inplace=True)

data['YearlyPeriod'].replace({

  np.NaN:data['YearlyPeriod'].mean(),

  },inplace=True)

data['Account1'].replace({

  np.NaN:"aa",

  },inplace=True)

data['History'].replace({

  np.NaN:"c2",

  },inplace=True)

data['Motive'].replace({

  np.NaN:"p3",

  },inplace=True)
data.duplicated().sum()
datacompare=data

data=data.drop(["Class"],1)



data=data.drop(["id"],1)
data['Account1']=preprocessing.LabelEncoder().fit(data['Account1']).transform(data['Account1'])

data['History']=preprocessing.LabelEncoder().fit(data['History']).transform(data['History'])

data['Motive']=preprocessing.LabelEncoder().fit(data['Motive']).transform(data['Motive'])

data['Account2']=preprocessing.LabelEncoder().fit(data['Account2']).transform(data['Account2'])

data['Employment Period']=preprocessing.LabelEncoder().fit(data['Employment Period']).transform(data['Employment Period'])

data['Gender&Type']=preprocessing.LabelEncoder().fit(data['Gender&Type']).transform(data['Gender&Type'])

data['Sponsors']=preprocessing.LabelEncoder().fit(data['Sponsors']).transform(data['Sponsors'])

data['Plan']=preprocessing.LabelEncoder().fit(data['Plan']).transform(data['Plan'])

data['Plotsize']=preprocessing.LabelEncoder().fit(data['Plotsize']).transform(data['Plotsize'])

data['Housing']=preprocessing.LabelEncoder().fit(data['Housing']).transform(data['Housing'])

data['Post']=preprocessing.LabelEncoder().fit(data['Post']).transform(data['Post'])

data['Phone']=preprocessing.LabelEncoder().fit(data['Phone']).transform(data['Phone'])

data['Expatriate']=preprocessing.LabelEncoder().fit(data['Expatriate']).transform(data['Expatriate'])

f, ax = plt.subplots(figsize=(20, 16))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data=data.drop(["Age"],1)    
min_max_scaler = preprocessing.StandardScaler()

np_scaled = min_max_scaler.fit_transform(data)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
pca2 = PCA(n_components=2)

pca2.fit(dataN1)

T2 = pca2.transform(dataN1)

from sklearn.cluster import KMeans
plt.figure(figsize=(10,5))

preds1 = []



kmean = KMeans(n_clusters = 9, random_state = 42)

kmean.fit(dataN1)

pred = kmean.predict(dataN1)

preds1.append(pred)



pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']

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



res=pred
res1 = pd.DataFrame(res)

final = pd.concat([data_orig["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})
final['Class']=final.Class.astype(int)
final['Class'].replace({

    1:0,6:0,4:0,7:0,

    3:1,0:1,8:1,5:1,

    2:2,

    np.NaN:1,

  },inplace=True)
final.to_csv('submission99.csv', index = False)
predicted0=final['id'].tolist()

predicted1=final['Class'].tolist()

# len(predicted0)



data = datacompare

data=data.dropna()

data['Class']=data.Class.astype(int)

temp0=data['id'].tolist()

temp1=data['Class'].tolist()



count=0

for i in range(0,len(temp0)):

    for j in range(0,len(predicted0)):

        if(temp0[i]==predicted0[j]):

            if(temp1[i]==predicted1[j]):

                count+=1

            break;

print(count/175)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = final.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)