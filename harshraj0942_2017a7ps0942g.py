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
data10 = pd.read_csv("/kaggle/input/dmassign1/data.csv",sep=",")
data10
data10 = data10.replace(to_replace='?',value=np.nan)
cols = data10.select_dtypes(exclude=['float']).columns

data10[cols] = data10[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
data11 = pd.read_csv("/kaggle/input/dmassign1/data.csv",sep=",")
data11


#col 189 190 191 192 193 194 195 196 197



data11 = pd.get_dummies(data11, columns=['Col189', 'Col190', 'Col191', 'Col192','Col193','Col194','Col195','Col196','Col197'])



col = data11.select_dtypes(exclude=['float']).columns

data11[col] = data11[col].apply(pd.to_numeric, downcast='float', errors='coerce')
data11 = data11.replace(to_replace='?',value=np.nan)
for column1 in data11.columns:

    data11[column1].fillna(data11[column1].mode(), inplace=True)
data11.info()
col = data11.select_dtypes(exclude=['float','int']).columns

data11[col] = data11[col].apply(pd.to_numeric, downcast='float', errors='coerce')
data11.info()
for column1 in data11.columns:

    data11[column1].fillna(data11[column1].mode(), inplace=True)
data11
data11['Class'] = data11.Class.astype(int)

data11[['Class']].dtypes
data11
data11 = data11.drop(['ID'],axis=1)
data11
data11.isnull().any()
corr = data11.corr()
corr
abs(corr['Class']).sort_values()
c=0

for i in range(236):

    if(abs(corr['Class'].iloc[i])>=0.08):

        c=c+1

        print(data11.columns[i])

        print(abs(corr['Class'].iloc[i]))

print(c)
data12=data11[['Col41','Col42','Col43','Col44','Col45','Col72','Col83','Col84','Col85','Col86','Col150','Col151','Col152','Col153','Col154','Col172','Col179']]
data12
data12.isnull().any()
for column1 in data12.columns:

    data12[column1].fillna(data12[column1].mode(), inplace=True)
corr = data12.corr()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data12.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10),

                    square=True, ax=ax, annot = True);
#drop all columns with corr > 0.9



data12 = data12.drop(['Col41'],axis=1)

data12 = data12.drop(['Col42'],axis=1)
data12 = data12.drop(['Col43'],axis=1)

data12 = data12.drop(['Col44'],axis=1)

data12 = data12.drop(['Col83'],axis=1)

data12 = data12.drop(['Col85'],axis=1)

data12 = data12.drop(['Col150'],axis=1)

data12 = data12.drop(['Col151'],axis=1)

data12 = data12.drop(['Col153'],axis=1)

data12
data12
data12
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data12)

dataN1 = pd.DataFrame(np_scaled)

dataN1
dataN1.isnull().any()
for column1 in dataN1.columns:

    dataN1[column1].fillna(dataN1[column1].mode()[0], inplace=True)
dataN1.isnull().any()
from sklearn.cluster import KMeans

wcss = []

for i in range(5, 30):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    wcss.append(kmean.inertia_)

plt.plot(range(5,30),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn import metrics

preds1 = []

for i in range(5,25):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    pred = kmean.predict(dataN1)

    preds1.append(metrics.calinski_harabasz_score(dataN1, kmean.labels_))

plt.plot(range(5,25),preds1)

plt.title('The Calinski-Harabasz Index')

plt.xlabel('Number of clusters')

plt.ylabel('Index')

plt.show()
from sklearn import metrics

preds1 = []



kmean = KMeans(n_clusters = 20, random_state = 42)

kmean.fit(dataN1)

pred = kmean.predict(dataN1)

preds1.append(metrics.calinski_harabasz_score(dataN1, kmean.labels_))



pred
np.set_printoptions(threshold=np.inf)
pred
pred.size
df = pd.read_csv("/kaggle/input/dmassign1/data.csv",sep=",")
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==0):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==0):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==0):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==0):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==0):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)

c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==1):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==1):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==1):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==1):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==1):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==2):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==2):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==2):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==2):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==2):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==3):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==3):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==3):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==3):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==3):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==4):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==4):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==4):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==4):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==4):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==5):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==5):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==5):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==5):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==5):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==6):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==6):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==6):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==6):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==6):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==7):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==7):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==7):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==7):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==7):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==8):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==8):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==8):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==8):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==8):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==9):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==9):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==9):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==9):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==9):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==10):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==10):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==10):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==10):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==10):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==11):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==11):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==11):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==11):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==11):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==12):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==12):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==12):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==12):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==12):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==13):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==13):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==13):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==13):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==13):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==14):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==14):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==14):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==14):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==14):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==15):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==15):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==15):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==15):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==15):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==16):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==16):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==16):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==16):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==16):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==17):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==17):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==17):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==17):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==17):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==18):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==18):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==18):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==18):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==18):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred[i]==19):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred[i]==19):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred[i]==19):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred[i]==19):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred[i]==19):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
res=[]

for i in range(len(pred)):

    

    if   pred[i] == 0:

        res.append(2)

    

    elif pred[i] == 1 or pred[i] == 2 or pred[i] == 5 or pred[i] == 6 or pred[i] == 7 or  pred[i] == 9 or pred[i] == 10 or pred[i] == 11 or pred[i] == 12 or pred[i] == 13 or pred[i] == 14 or pred[i] == 15 or pred[i] == 16 or pred[i]==18 or pred[i]== 17 or pred[i]==19:   

        res.append(1)

   

   

    elif pred[i] == 3:

        res.append(5)

   

    elif pred[i] ==4:

        res.append(4)

    

    elif pred[i] == 8:

        res.append(3)

   
res

res1 = pd.DataFrame(res)

final = pd.concat([df['ID'], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final=final[1300:]
final.to_csv('submission_1_k_means.csv', index = False)
#Now we try hierarchichal clustering
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN1)

T1 = pca1.transform(dataN1)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 20,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN1)

#plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(dataN1, "ward",metric="euclidean")

#ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
pred1=cut_tree(linkage_matrix1, n_clusters = 20).T

pred1
pred1.size
pred1[0][45]
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==0):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==0):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==0):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==0):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==0):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==1):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==1):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==1):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==1):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==1):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==2):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==2):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==2):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==2):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==2):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==3):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==3):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==3):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==3):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==3):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==4):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==4):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==4):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==4):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==4):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==5):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==5):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==5):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==5):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==5):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==6):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==6):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==6):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==6):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==6):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==7):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==7):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==7):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==7):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==7):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==8):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==8):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==8):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==8):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==8):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==9):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==9):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==9):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==9):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==9):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==10):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==10):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==10):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==10):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==10):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==11):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==11):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==11):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==11):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==11):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==12):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==12):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==12):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==12):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==12):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==13):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==13):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==13):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==13):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==13):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==14):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==14):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==14):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==14):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==14):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==15):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==15):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==15):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==15):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==15):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==16):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==16):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==16):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==16):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==16):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==17):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==17):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==17):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==17):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==17):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==18):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==18):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==18):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==18):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==18):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
c1=0

c2=0

c3=0

c4=0

c5=0

#classes



for i in range(1300):

    

    if(df.iloc[i]['Class']==1 and pred1[0][i]==19):

        c1=c1+1

    elif(df.iloc[i]['Class']==2 and pred1[0][i]==19):

        c2=c2+1

    elif(df.iloc[i]['Class']==3 and pred1[0][i]==19):

        c3=c3+1

    elif(df.iloc[i]['Class']==4 and pred1[0][i]==19):

        c4=c4+1

    elif(df.iloc[i]['Class']==5 and pred1[0][i]==19):

        c5=c5+1

    

   

        

   

   

print(c1)

print(c2)

print(c3)

print(c4)

print(c5)
res_h = []



for i in range(pred1.size):

    

    if  pred1[0][i] == 0:

        

        res_h.append(5)

    

    elif pred1[0][i] == 1:

        

        res_h.append(3)

        

    elif pred1[0][i] == 5 or pred1[0][i] == 4 or pred1[0][i] == 6 or pred1[0][i] == 7 or pred1[0][i] == 8 or pred1[0][i] == 9 or pred1[0][i] == 10 or pred1[0][i] == 11 or pred1[0][i] == 12 or pred1[0][i] == 13 or pred1[0][i] == 14 or pred1[0][i] == 15 or pred1[0][i] == 16 or pred1[0][i] == 17 or pred1[0][i] == 18 or pred1[0][i]==19 :   

       

        res_h.append(1)

   

    elif pred1[0][i] == 2:

       

        res_h.append(4)

   

    elif pred1[0][i] == 3:

       

        res_h.append(2)

   

  

   
pred1.size
res_h
res2 = pd.DataFrame(res_h)

final1 = pd.concat([df["ID"], res2], axis=1).reindex()

final1 = final.rename(columns={0: "Class"})

final1 = final1[1300:]
pred1[0][12999]

       
final.to_csv('submission_hierarchical_1.csv', index = False)
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

create_download_link(final)