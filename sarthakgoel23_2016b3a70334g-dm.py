#importing files
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
from sklearn.cluster import KMeans
#df = pd.read_csv("../input/dataset.csv", sep=',')
import pandas as pd
df = pd.read_csv("../input/dmassign1/data.csv")
df.info()
df2 = pd.read_csv("../input/dmassign1/data.csv", sep=',')
label_list= (df2['Class'][:1300]).tolist()
#label_list
list2=[]
for (columnName, columnData) in df.iteritems():
   if '?' in columnData.values :
       #print("Yes, '?' found in List : " , columnName)
       list2.append(columnName)
   #print('Colunm Name : ', columnName)
   #print('Column Contents : ', columnData.values)
print(list2)
df=df.replace('?',np.nan)
for i in list2:
    if(i!= 'Class' and i != 'Col197' and i != 'Col196' and i != 'Col195' and i != 'Col194' and i != 'Col193' and i != 'Col192' and i != 'Col191' and i != 'Col190' and i != 'Col189'): 
        df[i].fillna(df[i].median(), inplace=True)
df=df.drop(['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'],axis=1)
df= df.drop(['ID'],1)
df.isnull().any()
df=df.drop(['Class'],axis=1)
scaler=StandardScaler()
scaled_data=scaler.fit(df).transform(df)
scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
scaled_df
from sklearn.manifold import TSNE
model2=TSNE(n_components=3,n_iter=5000)
#scaled_df.drop(columns='age')+
T2=model2.fit_transform(scaled_df)
T2.shape
T2
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
wcss = []
pred4=[]
right=[]

for i in range(5, 60):
    kmean = KMeans(n_clusters = i, random_state = 42)  
#     kmean = KMeans(n_clusters = i, random_state = )
    kmean.fit(T2)
#     wcss.append(kmean.inertia_)
    
    pred=kmean.predict(T2)
    fc=pd.Series(pred+1,index=scaled_df.index,dtype= np.float64)
    class1=(confusion_matrix((label_list[:1300]),fc[:1300]).argmax(axis=0)+1).astype(np.float64)
    fc.replace({cluster+1:class1[cluster] for cluster in range(0,len(class1))},inplace=True)
    notTrue=((fc[:1300]!=label_list[:1300])).sum()
    yesTrue=1300-notTrue
    
    right.append(yesTrue)
    pred4.append(fc)
    wcss.append(kmean.inertia_)
    
plt.plot(range(5,60),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print('True:',max(right))
print("accuracy:", max(right) *100/1300)
#df5=pd.read_csv('C:/Users/Sarthak Goel/Downloads/sub1.csv')
df5=df2[['ID','Class']]
#df5.drop(["Class"],1)
df5
list6=pred4[44][1300:13000]
list6 = list6.values.tolist()
# list6 
df5["Class"].loc[1300:13000]= list6
df5["Class"]= df5["Class"].astype(int) #final answer call method here
df5=df5.loc[1300:13000]
df5
#df5.to_csv(r'C:/Users/Sarthak Goel/Downloads/sub3.csv',index=False) # call the output function here
# from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(eps=2, min_samples=10)
# pred = dbscan.fit_predict(scaled_df)
# plt.scatter(T2[:, 0], T2[:, 1], c=pred)
# labels1 = dbscan.labels_
# #labels1 = labels1[labels1 >= 0] #Remove Noise Points
# labels1, counts1 = np.unique(labels1, return_counts=True)
# print(len(labels1))
# print(labels1)
# print(len(counts1))
# print(counts1)
# dbscan = DBSCAN(eps=0.6, min_samples=17)
# pred = dbscan.fit_predict(scaled_df)
# plt.scatter(T2[:, 0], T2[:, 1], c=pred)
# labels2 = dbscan.labels_
# #labels2 = labels2[labels2 >= 0] #Remove Noise Points
# labels2, counts2 = np.unique(labels2, return_counts=True)
# print(len(labels2))
# print(labels2)
# print(len(counts2))
# print(counts2)
# from sklearn.cluster import AgglomerativeClustering as AC
# aggclus = AC(n_clusters = 5,affinity='euclidean',linkage='ward',compute_full_tree='auto')
# y_aggclus= aggclus.fit_predict(scaled_df)
# plt.scatter(T2[:, 0], T2[:, 1], c=y_aggclus)
# from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
# from scipy.cluster.hierarchy import fcluster
# linkage_matrix1 = linkage(scaled_df, "ward",metric="euclidean")
# ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
# y_ac=cut_tree(linkage_matrix1, n_clusters = 5).T
# y_ac=y_ac+1
# y_ac=y_ac.tolist()[0]
# ID=df2[['ID']]
# ID=ID.tail(13000-1300)
# ID['Class']=list6
# df5=ID
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
create_download_link(df5)
