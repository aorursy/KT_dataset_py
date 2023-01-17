# Step-1 Initialize Python modules

import sys

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







# Step-2 Read data



df = pd.read_csv("../input/dataset.csv", sep=',', dtype={

    'Col189':str,

'Col190':str,

'Col191':str,

'Col192':str,

'Col193':str,

'Col194':str,

'Col195':str,

'Col196':str,

'Col197':str})



#Step-3 Clean Data



nddf = df.drop_duplicates()

#no duplicates present



#numeric Encoding



nddf['Col189'].unique()

def col189(val):

    if val=='yes':

        return 1

    if val=='no':

        return 0

    else:

        return np.nan

    

nddf['Col189']=nddf['Col189'].apply(col189)



nddf['Col190'].unique()

def col190(val):

    if val=='sacc1':

        return 1

    if val=='sacc2':

        return 2

    if val=='sacc3':

        return 3

    if val=='sacc4':

        return 4

    if val=='sacc5':

        return 5

    else:

        return np.nan

    

nddf['Col190']=nddf['Col190'].apply(col190)



nddf['Col191'].unique()

def col191(val):

    if val=='time1':

        return 1

    if val=='time2':

        return 2

    if val=='time3':

        return 3

    else:

        return np.nan

    

nddf['Col191']=nddf['Col191'].apply(col191)



nddf['Col192'].unique()

def col192(val):

    if val=='p1':

        return 1

    if val=='p2':

        return 2

    if val=='p3':

        return 3

    if val=='p4':

        return 4

    if val=='p5':

        return 5

    if val=='p6':

        return 6

    if val=='p7':

        return 7

    if val=='p8':

        return 8

    if val=='p9':

        return 9

    if val=='p10':

        return 10

    else:

        return np.nan

    

nddf['Col192']=nddf['Col192'].apply(col192)









nddf['Col193'].unique()

def col193(val):

    if val=='F0':

        return 1

    if val=='F1':

        return 2

    if val=='M0':

        return 3

    if val=='M1':

        return 4

    else:

        return np.nan

    

nddf['Col193']=nddf['Col193'].apply(col193)



nddf['Col194'].unique()

def col194(val):

    if val=='ab':

        return 1

    if val=='ac':

        return 2

    if val=='ad':

        return 3

    else:

        return np.nan

    

nddf['Col194']=nddf['Col194'].apply(col194)



nddf['Col195'].unique()

def col195(val):

    if val=='Jb1':

        return 1

    if val=='Jb2':

        return 2

    if val=='Jb3':

        return 3

    if val=='Jb4':

        return 4

    else:

        return np.nan

    

nddf['Col195']=nddf['Col195'].apply(col195)



nddf['Col196'].unique()

def col196(val):

    if val=='H1':

        return 1

    if val=='H2':

        return 2

    if val=='H3':

        return 3

    else:

        return np.nan

    

nddf['Col196']=nddf['Col196'].apply(col196)





nddf['Col197'].unique()

def col197(val):

    if val=='sm' or val=='SM':

        return 1

    if val=='me' or val =='ME' or val=='M.E.':

        return 2

    if val=='la' or val =='LA':

        return 3

    if val=='XL':

        return 4

    else:

        return np.nan

    

nddf['Col197']=nddf['Col197'].apply(col197)

nddf.info(verbose=True, null_counts=True)



for i in nddf.columns:

    cell=0

    for j in nddf[i]:

        if j=='?':

            nddf[i][cell]=np.nan

        cell=cell+1

        

for i in nddf.columns:

    if i=='ID' or i=='Class':

        continue

    nddf[i]=pd.to_numeric(nddf[i])

    

#encoding everything into numeric

cols = list(nddf)

for i in cols:

    if i=='ID' or i=='Class':

        continue

    nddf[i] = nddf[i].fillna((nddf[i].mode()[0]))

    

#Incomplete data handled







nddf.info(verbose=True, null_counts=True)
# Step-4 Find Correlation: Now numeric encoding has been done

corr = nddf.corr()



abs(corr['Class']).sort_values(ascending=False).head(30)

#to calculate correlation with class
df1 = nddf[['Class','Col152','Col153','Col151','Col85','Col43','Col84','Col154']]



df1.describe()



# #Step -5 Z score normalize everything

scaler = StandardScaler()

scaled_data=scaler.fit(df1).transform(df1)

scaled_df=pd.DataFrame(scaled_data,columns=df1.columns)

scaled_df.tail()





df2 = scaled_df



df2=df2.drop(columns=['Class'])



df2.columns
# Step-6 PCA Visualizing : Finding max std dev and variance from all columns - to select PCA

maxStd =0

for col in df2.columns:

    stdDev = df2.loc[:,col].std()

    if (maxStd<stdDev):

        maxStd = stdDev

        maxCol = col

    

print(maxCol, maxStd)



#PCA Visualization

from sklearn.decomposition import PCA



model=PCA(n_components=4)

model_data=model.fit(df2).transform(df2)



plt.figure(figsize=(8,6))

plt.scatter(model_data[:,0],model_data[:,1],c=scaled_df['Col84'])
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

pca2 = PCA(n_components=4)

pca2.fit(df2)

T2 = pca2.transform(df2)



colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','orange','gray','maroon', 'silver', 'plum', 'gold','seagreen','darkgreen','tan','azure','aqua']



plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 15, random_state = 42)

kmean.fit(df2)

pred = kmean.predict(df2)

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
# K Means Classification mapping

pred1300 = pred[:1300]

true1300 = df1['Class'][:1300]

mapdf = pd.DataFrame(list(zip(pred1300, true1300)), columns=['Pred', 'True'])

mapdf
d = pd.DataFrame(np.zeros((15, 5)))

for index, row in mapdf.iterrows():

    val=row[1]

    d[val-1][row[0]]+=1

    

d
res=[]

for i in range(len(pred)):

    if pred[i]==0 :

        res.append(4)

    elif pred[i]==11:

        res.append(5)

    elif pred[i]==7:

        res.append(4)

    else:

        res.append(1)
from sklearn.metrics import accuracy_score

predicted = res[:1300]

true = df1['Class'][:1300]

accuracy_score(true, predicted)
res1 = pd.DataFrame(res)

final = pd.concat([nddf["ID"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()



final = final[1300:]

final.to_csv("2017A7PS0130G", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "2017A7PS0130G"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="2017A7PS0130G" href="data:text/csv;base64,{payload}"target="_blank">Dataframe2017A7PS0130G</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final)