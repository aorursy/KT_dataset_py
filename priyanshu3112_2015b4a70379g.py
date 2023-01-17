import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/dataset.csv", sep=',')

X11=data[['id']].iloc[175:1031]

y=data['Class'].iloc[0:175]

data.head()



data.replace('?',np.NaN,inplace=True)



data['Account1'].fillna(data['Account1'].mode()[0],inplace=True)

data['History'].fillna(data['History'].mode()[0],inplace=True)

data['Motive'].fillna(data['Motive'].mode()[0],inplace=True)

data['Monthly Period'].fillna(data['Monthly Period'].median(),inplace=True)

data['Age'].fillna(data['Age'].mode()[0],inplace=True)

data['Tenancy Period'].fillna(data['Tenancy Period'].mode()[0],inplace=True)

data['InstallmentRate'].fillna(data['InstallmentRate'].mode()[0],inplace=True)



data['Yearly Period'].replace(np.NaN,-1000,inplace=True)

data['Yearly Period']=data['Yearly Period'].astype(np.float64)

data['Yearly Period'].replace(-1000,np.NaN,inplace=True)

data['Yearly Period'].fillna(data['Yearly Period'].mean(),inplace=True)



data['Credit1'].replace(np.NaN,-1000,inplace=True)

data['Credit1']=data['Credit1'].astype(np.float64)

data['Credit1'].replace(-1000,np.NaN,inplace=True)

data['Credit1'].fillna(data['Credit1'].mean(),inplace=True)



data['InstallmentCredit'].replace(np.NaN,-1000,inplace=True)

data['InstallmentCredit']=data['InstallmentCredit'].astype(np.float64)

data['InstallmentCredit'].replace(-1000,np.NaN,inplace=True)

data['InstallmentCredit'].fillna(data['InstallmentCredit'].mean(),inplace=True)



data['Account2']=data['Account2'].str.upper()

data['Sponsors']=data['Sponsors'].str.upper()

data['Plotsize']=data['Plotsize'].str.upper()

data['Plotsize']=data['Plotsize'].replace('M.E.','ME')



data['Monthly Period']=data['Monthly Period'].astype(np.float64)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data.drop(['id','Class','Age','Monthly Period','Yearly Period','#Credits','#Authorities','Motive'],axis=1,inplace=True)

data_features=data.columns

col=['Account1','History','Gender&Type','Account2','Employment Period','InstallmentRate','Sponsors',

     'Tenancy Period','Plotsize','Plan','Housing','Post','Phone','Expatriate']

temp=data[data_features]

X=pd.get_dummies(temp,columns=col)

X.head()
from sklearn import preprocessing

standardscaler = preprocessing.StandardScaler()

np_scaled = standardscaler.fit_transform(X)

X = pd.DataFrame(np_scaled)

X.head()
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

pca = PCA().fit(X)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

preds1=[]

reduced_data = PCA(n_components=3,random_state=42).fit_transform(X)

kmeans = KMeans(n_clusters=3,init='k-means++',random_state=42)

pred=kmeans.fit_predict(reduced_data)

preds1.append(pred)

centroids = kmeans.cluster_centers_
y_pred1=[]

for i in range(len(preds1[0])):

    y_pred1.append(preds1[0][i])

unique, counts=np.unique(y_pred1, return_counts=True)

print(counts)

print(y_pred1.count(0))

print(accuracy_score(y,y_pred1[0:175]))
y_pred2=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred2.append(1)

    elif y_pred1[i]==1:

        y_pred2.append(0)

    else:

        y_pred2.append(2)

unique, counts=np.unique(y_pred2, return_counts=True)

print(counts)

print(y_pred2.count(0))

print(accuracy_score(y,y_pred2[0:175]))
y_pred3=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred3.append(2)

    elif y_pred1[i]==1:

        y_pred3.append(1)

    else:

        y_pred3.append(0)

unique, counts=np.unique(y_pred3, return_counts=True)

print(counts)

print(y_pred3.count(0))

print(accuracy_score(y,y_pred3[0:175]))
y_pred4=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred4.append(0)

    elif y_pred1[i]==1:

        y_pred4.append(2)

    else:

        y_pred4.append(1)

unique, counts=np.unique(y_pred4, return_counts=True)

print(counts)

print(y_pred4.count(0))

print(accuracy_score(y,y_pred4[0:175]))
y_pred5=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred5.append(1)

    elif y_pred1[i]==1:

        y_pred5.append(2)

    else:

        y_pred5.append(0)

unique, counts=np.unique(y_pred5, return_counts=True)        

print(counts)

print(y_pred5.count(0))

print(accuracy_score(y,y_pred5[0:175]))
y_pred6=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred6.append(2)

    elif y_pred1[i]==1:

        y_pred6.append(0)

    else:

        y_pred6.append(1)

unique, counts=np.unique(y_pred6, return_counts=True)

print(counts)

print(y_pred6.count(0))

print(accuracy_score(y,y_pred6[0:175]))
X11.head()

X11['Class']=y_pred3[175:1031]

X11.to_csv("ans.csv",index=False)

X11
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



create_download_link(X11)