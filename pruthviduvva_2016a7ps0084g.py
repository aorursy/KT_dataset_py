import sys

!{sys.executable} -m pip install sklearn
import numpy as np

import pandas as pd

%matplotlib inline  

import matplotlib.pyplot as plt

import seaborn as sns



#import sklearn.preprocessing as sk

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
#importing dataset into dataframe object.

df=pd.read_csv("../input/dataset.csv",sep = ",")

df_original=df.copy()

df_original2=df.copy()

df_original3=df.copy()
#cheking the attributes.

df.info()
df_temp=df.copy()



#dropping class column as it is not needed for preprossing.

df=df.drop('Class',axis=1)

#Changin the ? value to readable value np.nan(NaN)

#df=df.replace('?',np.nan)

df=df.replace('?',np.nan)

df_temp=df_temp.replace('?',np.nan)
df.dropna(inplace=True)

df_temp.dropna(inplace = True)

answers_df=df_temp.iloc[0:175]

#answers_df.tail()

answers_df=answers_df[['id','Class']]

answers_df['id'].nunique()
#no duplicates in the dataset.

df.duplicated(keep='first').any()
#changing dtype of attributes.

df['Monthly Period']=df['Monthly Period'].astype(int)

df['Credit1']=df['Credit1'].astype(int)

df['InstallmentRate']=df['InstallmentRate'].astype(int)

df['Tenancy Period']=df['Tenancy Period'].astype(int)

df['Age']=df['Age'].astype(float)

df['InstallmentCredit']=df['InstallmentCredit'].astype(float)

df['Yearly Period']=df['Yearly Period'].astype(float)
df.info()
df['Expatriate'].unique()
def Phone_encode(val):

    if val ==  'yes':

        return 0

    else:

        return 1



df_before=df.copy()

df['Phone']=df['Phone'].apply(Phone_encode)
def Expatriate_encode(val):

    if val ==  True:

        return 0

    else:

        return 1



df['Expatriate']=df['Expatriate'].apply(Expatriate_encode)
df['Expatriate'].value_counts()
f, ax = plt.subplots(figsize=(10, 8))

corr=df.corr()

sns.heatmap(corr, annot=True, fmt=".1f")
df=df.drop('Credit1',axis = 1)

df=df.drop('Yearly Period',axis = 1)
data=df.drop('id',axis=1)
data.info()
#data1=pd.get_dummies(data,columns=["Account1","History","Motive","Account2","Gender&Type","Sponsors","Plotsize","Plan","Housing","Post"])

#data1.head()

data2=data.copy()

data2=pd.get_dummies(data,columns=["Account1","Account2",'Motive',"Employment Period","Plotsize","Post",'Gender&Type','Sponsors','Housing',"Plan"])

data2=data2.drop(['History'],axis=1)

data2.info()

df1=data2.copy()

df1.head()

minmaxscaler=MinMaxScaler()

minmaxscaled_data=minmaxscaler.fit_transform(df1)

minmaxscaled_df=pd.DataFrame(minmaxscaled_data,columns=df1.columns)

minmaxscaled_df.head()
standardscaler=StandardScaler()

standardscaled_data=standardscaler.fit_transform(df1)

standardscaled_df=pd.DataFrame(standardscaled_data,columns=df1.columns)

standardscaled_df.head()
pca1=PCA(n_components=2)

pca1=pca1.fit(minmaxscaled_df)

T1=pca1.transform(minmaxscaled_df)
pca2=PCA(n_components=2)

pca2=pca2.fit(standardscaled_df)

T2=pca2.transform(standardscaled_df)
from sklearn.cluster import KMeans



#wcss=[]



#for i in range(2,50):

#    kmean=KMeans(n_clusters = i , random_state = 42)

#    kmean.fit(minmaxscaled_df)

#    wcss.append(kmean.inertia_)



#plt.plot(range(2,50), wcss)

#plt.title('The Elbow Method')

#plt.xlabel('Number of clusters')

#plt.ylabel('WCSS')

#plt.show()    
labels   = [np.random.choice(['A','B','C','D']) for i in range(1017)]



# Label to color dict (manual)

label_color_dict = {'A':'red','B':'green','C':'blue','D':'magenta'}



# Color vector creation

cvec = [label_color_dict[label] for label in labels]
minmax_kmean = KMeans(n_clusters = 3 , random_state = 42)

minmax_kmean.fit(minmaxscaled_df)

minmax_pred = minmax_kmean.predict(minmaxscaled_df)





plt.figure(figsize=(8,6))

plt.scatter( T1[:,0], T1[:,1] , c=minmax_pred)

plt.title("MinMaxScaling")



minmax_centroids = pca1.transform(minmax_kmean.cluster_centers_)

plt.plot(minmax_centroids[:, 0], minmax_centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
minmax_df1 = df1.copy()

minmax_df1['Class'] = minmax_pred

minmax_df1['id'] = df_original['id']

minmax_predict_df = minmax_df1.iloc[0:172]

minmax_predict_df = minmax_predict_df[['id','Class']]

minmax_predict_df['id'].nunique()
standard_kmean = KMeans(n_clusters = 3, random_state = 42 )

standard_kmean.fit(standardscaled_df)

standard_pred = standard_kmean.predict(standardscaled_df)





plt.figure(figsize=(8,6))

plt.scatter( T2[:,0], T2[:,1] , c=standard_pred)



standard_centroids = pca2.transform(standard_kmean.cluster_centers_)

plt.plot(standard_centroids[:, 0], standard_centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
import collections

counter=collections.Counter(standard_pred)

print(counter)
standard_df1=df1.copy()

standard_df1['Class']=standard_pred

standard_df1['id']=df_original['id']

standard_predict_df=standard_df1.iloc[0:172]

standard_predict_df=standard_predict_df[['id','Class']]

standard_predict_df.head()
def percent_pred(from_df,to_df):

    list_percent=[]

    count_0=0

    count_1=0

    count_2=0

    for i in range(0,172):

        if(from_df.iloc[i]['Class'] == to_df.iloc[i]['Class']):

            if(from_df.iloc[i]['Class'] == 0):

                count_0+=1

            if(from_df.iloc[i]['Class'] == 1):

                count_1+=1

            if(from_df.iloc[i]['Class'] == 2):

                count_2+=1

    list_percent.append((count_0)*100/172)

    list_percent.append((count_1)*100/172)

    list_percent.append((count_2)*100/172)

    return list_percent
def given_percent(from_df):

    list_percent=[]

    count_0=0

    count_1=0

    count_2=0

    for i in range(0,172):

        if(from_df.iloc[i]['Class'] == 0):

            count_0+=1

        if(from_df.iloc[i]['Class'] == 1):

            count_1+=1

        if(from_df.iloc[i]['Class'] == 2):

            count_2+=1

           

    list_percent.append((count_0)*100/172)

    list_percent.append((count_1)*100/172)

    list_percent.append((count_2)*100/172)

    return list_percent
def match_fun(from_df,to_df):

    count=0;

    for i in range(0,172):

        if(from_df.iloc[i]['Class'] == to_df.iloc[i]['Class']):

            count+=1

    return (count*100/172)        
def transform1(val):

    di = {0:0, 1:1 , 2:2}

    val['Class']=val['Class'].replace(di)



def transform2(val):

    di = {0:0, 1:2 , 2:1}

    val['Class']=val['Class'].replace(di)



def transform3(val):

    di = {0:1, 1:0 , 2:2}

    val['Class']=val['Class'].replace(di)

    

def transform4(val):

    di = {0:1, 1:2 , 2:0}

    val['Class']=val['Class'].replace(di)



def transform5(val):

    di = {0:2, 1:0 , 2:1}

    val['Class']=val['Class'].replace(di)

    

def transform6(val):

    di = {0:2, 1:1 , 2:0}

    val['Class']=val['Class'].replace(di)   

list_ans=given_percent(answers_df)

for i in list_ans:

    print(i)
test = standard_predict_df.copy()

transform2(test)

test.head()
test = minmax_predict_df.copy()

transform1(test)

print(match_fun(test,answers_df))   



test = minmax_predict_df.copy()

transform2(test)

print(match_fun(test,answers_df))  



test = minmax_predict_df.copy()

transform3(test)

print(match_fun(test,answers_df))  



test = minmax_predict_df.copy()

transform4(test)

print(match_fun(test,answers_df))  



test = minmax_predict_df.copy()

transform5(test)

print(match_fun(test,answers_df))  



test = minmax_predict_df.copy()

transform6(test)

print(match_fun(test,answers_df))  
test = standard_predict_df.copy()

transform1(test)

print(match_fun(test,answers_df))   



test = standard_predict_df.copy()

transform2(test)

print(match_fun(test,answers_df))  



test = standard_predict_df.copy()

transform3(test)

print(match_fun(test,answers_df))  



test = standard_predict_df.copy()

transform4(test)

print(match_fun(test,answers_df))  



test = standard_predict_df.copy()

transform5(test)

print(match_fun(test,answers_df))  



test = standard_predict_df.copy()

transform6(test)

print(match_fun(test,answers_df))  
#submission_df=standard_predict_df.copy()

#transform4(submission_df)

#submission_df=submission_df.merge(df_original)



#submission_df=submission_df[['id','Class']]

#submission_df.info()
#submission_df.to_csv('submission1.csv', index = False)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','indigo','orange']
plt.figure(figsize=(16, 14))



label_kmean = KMeans(n_clusters = 8, random_state = 42)

label_kmean.fit(standardscaled_df)

label_pred = label_kmean.predict(standardscaled_df)

pred_pd = pd.DataFrame(label_pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(label_pred)):

        if i == label_pred[j]:

            count+=1

            meanx+=T2[j,0]

            meany+=T2[j,1]

            plt.scatter(T2[j, 0], T2[j, 1])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black')
import collections

counter=collections.Counter(label_pred)

print(counter)
#10,26,14,17,19  0,7,8,5,16,

def anothertransform4(val):

    di = { 0:1 , 1:2 , 2:0 , 3:2 , 4:0 , 5:1 , 6:2 , 7:2 , 8:2 , 9:1 , 10:1 , 11:2 , 12:0 , 13:2 , 14:2 ,

          15:1 ,  16:2 , 17:0 , 18:2 , 19:0 , 20:2 , 21:1 , 22:1 , 23:2 , 24:2 , 25:1 , 26:0 , 27:2}

    val['Class']=val['Class'].replace(di)

    
another_df=df1.copy()

another_df['Class']=label_pred

another_df['id']=df_original['id']

another_predict_df=another_df.iloc[0:172]

another_predict_df=another_predict_df[['id','Class']]

another_predict_df.tail()



anothertransform4(another_predict_df)

another_predict_df.tail()
match_fun(another_predict_df,answers_df)
pred_percent=percent_pred(another_predict_df,answers_df)

for i in pred_percent:

    print(i)
df_original3['Class']=another_df['Class']

df_original3=df_original3.fillna(2)

al_submission_df=df_original3[['id','Class']]

al_submission_df=al_submission_df.iloc[175:]

al_submission_df['Class']=al_submission_df['Class'].astype(int)

test=another_df.copy()

anothertransform4(test)

print(match_fun(test,answers_df))
al_submission_df.info()
anothertransform4(al_submission_df)

al_submission_df.to_csv('final_submission.csv', index = False)
label_pred
standard_df2=df1.copy()

standard_df2['Class']=label_pred

standard_df2['id']=df_original['id']

label_predict_df=standard_df2.iloc[0:172]

label_predict_df=label_predict_df[['id','Class']]

label_predict_df.tail()
test = label_predict_df.copy()

transform1(test)

print(match_fun(test,answers_df))   



test = label_predict_df.copy()

transform2(test)

print(match_fun(test,answers_df))  



test = label_predict_df.copy()

transform3(test)

print(match_fun(test,answers_df))  



test = label_predict_df.copy()

transform4(test)

print(match_fun(test,answers_df))  



test = label_predict_df.copy()

transform5(test)

print(match_fun(test,answers_df))  



test = label_predict_df.copy()

transform6(test)

print(match_fun(test,answers_df)) 
df_original.iloc[26:31]
df_original2['Class']=standard_df2['Class']

df_original2=df_original2.fillna(2)

submission_df=df_original2[['id','Class']]

submission_df=submission_df.iloc[175:]

test=standard_df2.copy()

transform4(test)

submission_df['Class']=submission_df['Class'].astype(int)
print(match_fun(test,answers_df))
submission_df.info()
wcss = []

for i in range(2, 19):

    elbow_kmean = KMeans(n_clusters = i, random_state = 42)

    elbow_kmean.fit(minmaxscaled_df)

    wcss.append(elbow_kmean.inertia_)

    

plt.plot(range(2,19),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.neighbors import NearestNeighbors



ns = 74                                                

nbrs = NearestNeighbors(n_neighbors = ns).fit(minmaxscaled_df)

distances, indices = nbrs.kneighbors(minmaxscaled_df)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2, min_samples=10)

pred = dbscan.fit_predict(minmaxscaled_df)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
ns = 17

nbrs = NearestNeighbors(n_neighbors = ns).fit(standardscaled_df)

distances, indices = nbrs.kneighbors(standardscaled_df)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)
dbscan = DBSCAN(eps=0.6, min_samples=17)

pred = dbscan.fit_predict(standardscaled_df)

plt.scatter(T2[:, 0], T2[:, 1], c=pred)
db_df=df1.copy()

db_df['Class']=pred

db_df['id']=df_original['id']

db_predict_df=db_df.iloc[0:172]

db_predict_df=db_predict_df[['id','Class']]

db_predict_df.head()
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(standardscaled_df)

plt.scatter(T2[:, 0], T2[:, 1], c=y_aggclus)
aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(standardscaled_df)

plt.scatter(T2[:, 0], T2[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(standardscaled_df, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=7)
linkage_matrix2 = linkage(minmaxscaled_df, "ward",metric="euclidean")

ddata2 = dendrogram(linkage_matrix2,color_threshold=10)
y_ac=cut_tree(linkage_matrix2, n_clusters = 3).T

y_ac[0]
plt.scatter(T2[:,0], T2[:,1], c=y_ac[0,:], s=100, label='')

plt.show()
three=y_ac[0]
predict_list=df_original['Class'].tolist()

one=predict_list[:175]
hc_df=df1.copy()

hc_df['Class']=three

hc_df['id']=df_original['id']

hc_predict_df=hc_df.iloc[0:172]

hc_predict_df=hc_predict_df[['id','Class']]

hc_predict_df.tail()
test = hc_predict_df.copy()

transform1(test)

print(match_fun(test,answers_df))   



test = hc_predict_df.copy()

transform2(test)

print(match_fun(test,answers_df))  



test = hc_predict_df.copy()

transform3(test)

print(match_fun(test,answers_df))  



test = hc_predict_df.copy()

transform4(test)

print(match_fun(test,answers_df))  



test = hc_predict_df.copy()

transform5(test)

print(match_fun(test,answers_df))  



test = hc_predict_df.copy()

transform6(test)

print(match_fun(test,answers_df))
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = al_submission_df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(al_submission_df)