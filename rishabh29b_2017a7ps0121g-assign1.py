import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
## Without removing '?' rows
df = pd.read_csv('../input/dmassign1/data.csv')
df = df.replace({'?':math.nan})

## Handle mixed type ones
badcols = df.columns[df.applymap(type).nunique() > 1]
for col in badcols:
    majnum = np.unique(df[col].apply(lambda x: str(type(x))).values,return_counts=True)[1]
    majType = np.unique(df[col].apply(lambda x: str(type(x))).values,return_counts=True)[0][np.argmax(majnum)]
    if(majType[8:-2] == 'int' or majType[8:-2] == 'float'):
        df[col] = df[col].astype(float)
    else:
        df[col] = df[col].astype(majType[8:-2])

## Handle int/float stored as strings
for col in df.columns[df.dtypes=='object']:
    if(df[col][0].isdigit()):
        df[col] = df[col].astype(int)
    elif(df[col][0].replace('.', '', 1).replace('-','',1).isdigit()):
        df[col] = df[col].astype(float)

## Handle NaNs
for col in df.columns:
    if(df[col].dtype == 'object'):
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

## Handle Categorical ones
df['Col189'] = df['Col189'].replace({'yes':1,'no':0})
df['Class'] = df['Class'].astype(int)
for x in df.columns[df.dtypes=='object']:
    if x == 'ID':
        continue
    df = pd.concat([df,pd.get_dummies(df[x], prefix=x[-3:])],axis=1)
    df = df[df.columns.drop(x)]

## Check
print(df.isna().sum().sum())
print(len(df.columns))
print(len(df))
## Drop highly correlated columns (lower the dimentionality)
corr = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train3 = df[df.columns.drop(to_drop)]
train3 = train3[train3.columns.drop(['Class','ID'])]
print(len(train3.columns))
print(len(train3))
## Scaling values
scaler=StandardScaler()
train3[train3.columns[1:183]] = scaler.fit_transform(train3[train3.columns[1:183]])
train3.head()
## Clustering Algorithm
kms = KMeans(n_clusters=33, random_state=42)
pred = kms.fit(train3.values)
pred = kms.predict(train3.values)

## Assign Clusters
gt = df['Class'][:1300].values
mat = np.zeros((5,33))
for i in range(0,1300):
    mat[gt[i]-1][pred[i]] += 1
mapping = np.argmax(mat,axis=0)
for i in range(0,len(pred)):
    pred[i] = mapping[pred[i]]+1
## Final dataframe
thedict = {'ID' : df['ID'].values[1300:], 'Class' : pred[1300:]}
ans = pd.DataFrame.from_dict(thedict)
from IPython.display import HTML
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(ans)
