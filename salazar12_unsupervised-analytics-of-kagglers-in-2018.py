import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix



multiplec = pd.read_csv('../input/multipleChoiceResponses.csv')

mkeys=multiplec.keys()
#excludes the columns corresponding to text explaining
multiplec=multiplec.drop([ mkeys[2], mkeys[8], mkeys[10],mkeys[23], mkeys[24], mkeys[25],mkeys[26], mkeys[44], mkeys[56], mkeys[64], mkeys[83], mkeys[85], 
                       mkeys[87], mkeys[107], mkeys[109], mkeys[123],mkeys[125], mkeys[150], 
                          mkeys[194], mkeys[223], mkeys[249],mkeys[262], mkeys[264],mkeys[276],
                      mkeys[304],mkeys[306], mkeys[329], mkeys[341], mkeys[371], mkeys[385], mkeys[394],mkeys[283]],axis=1)
multiplec.keys()
multi2=multiplec.iloc[1:,:]

#since the first column is actually composed of integers, we do not include it in the encoding
obj_df = multi2.select_dtypes(include=['object']).copy()
obj_df = obj_df.drop(mkeys[0],axis=1)
#One-Hot encoding using pandas
obj_df=pd.get_dummies(obj_df)
np.shape(obj_df)
#scaling the time column
x = multi2.iloc[:,0].values.astype(float) 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
df_scaled = pd.DataFrame(x_scaled,columns=['Time from Start to Finish (seconds)'])
#adjusting the indices
df_scaled.index+=1
#Creating our encoded data frame
encoded_data = pd.concat([df_scaled,obj_df,multi2.select_dtypes(exclude=['object']).copy()],axis=1)
print(np.shape(encoded_data))
encoded_data.head()
#changes nan values two -1
encoded_data = encoded_data.fillna(value=-1)
enc_keys=encoded_data.keys()
#pca to reduce dimentionality
pca = PCA(n_components=10)
pca.fit(encoded_data)
var=pca.explained_variance_
n_com=np.linspace(1,10,10)
plt.scatter(n_com,var/np.sum(var),marker='x')
plt.xlabel('Eigenvectors')
plt.ylabel('Variance')
print('First eigen-vector var fraction:', var[0]/np.sum(var), ' second var', var[1]/np.sum(var))
data_proj= pca.transform(encoded_data)
#mean inertia for the clusters formed
inert = []
#elbow method
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=500,n_init=10)
    kmeans.fit(data_proj)
    inert.append(kmeans.inertia_)
plt.plot(range(1,11),inert)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
#looking at the 2 cluster case
n_c=2
kmeans_pca = KMeans(n_clusters=n_c,init='k-means++',max_iter=500,n_init=10,random_state=0) 
clus_index = kmeans_pca.fit_predict(data_proj) #saves the cluster labels in the positions of each observation

#Scatter plots of the projections over the first two principal components
for i in range(0,n_c):
    plt.scatter(np.asarray(data_proj)[clus_index == i, 0], np.asarray(data_proj)[clus_index == i,1], alpha=0.7)

plt.scatter(kmeans_pca.cluster_centers_[:,0], kmeans_pca.cluster_centers_[:,1],marker='x',s=100,c='black')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.title('Clusters')


def get_feat(cl_num,cl_index):
    #Creates a bar plot that show the coefficients associated with the 40 most relevant features that the
    #SVM learns 
    #cl_num: the current cluster label
    #cl_index: an array containing the cluster labels obtained from k-means
    print(1)
    labels=np.zeros(len(cl_index))
#gets positions of the zero cluster members
    ii=np.where(cl_index==cl_num)
    labels[ii]=1
    labels=pd.DataFrame(labels)
    supp = SVC(kernel='linear')
    supp.fit(encoded_data,np.ravel(labels))
    print(2)
    
    coef=(abs(supp.coef_.ravel()))
#The coefficients returned correspond to the features, thus the largest coeficients 
#show the most important features. We can take a look at the first 20 coeficients
    top_coef= np.argsort(coef)[-40:]
    feature_names=encoded_data.keys()[top_coef]
    #creates plots
    plt.figure(figsize=(13,13))
    plt.barh(range(len(feature_names)), coef[top_coef], align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    
    plt.savefig(str(cl_num)+'.pdf')
    
    print('The most relevant feature is: ', encoded_data.keys()[top_coef][-1])
    
    return supp, labels, feature_names


def K_fold(clf,K,X,Y):
    #in case we need to test our model accuracy
    #clf: the trained model
    #K: subdivisions of the training data
    #X: features

    print('ya')
    kf = KFold(n_splits=K)
    lin_model = clf

    #using the confusion matrix
    lin_ERR = []
    l_pre=[]
    l_rec=[]
    print('ya2')

    for train_index, test_index in kf.split(X):
        print(1)
        xk_train, xk_test = X.iloc[train_index], X.iloc[test_index]
        print(2)
        yk_train, yk_test = Y.iloc[train_index], Y.iloc[test_index]
        print(3)
        lin_model.fit(xk_train, yk_train)
        print(4)
        lin_pred=lin_model.predict(xk_test)
        print(5)
        lin_conf=confusion_matrix(yk_test,lin_pred)
        
        #precision recal and f1 values
        l_pre.append(lin_conf[1,1]/(lin_conf[1,1]+lin_conf[0,1]))
        l_rec.append(lin_conf[1,1]/(lin_conf[1,1]+lin_conf[1,0]))


        lin_ERR.append((lin_conf[0,1]+lin_conf[1,0])/np.sum(lin_conf))
        
    return np.mean(lin_ERR), np.mean(l_pre), np.mean(l_rec)
#bar plot for cluster 0
model1, lab1, feat1 =get_feat(0,clus_index)
#error, prec, rec = K_fold(model_used,3, encoded_data, labs)
cluster0=encoded_data.iloc[clus_index==0][feat1]
cluster0.head()
#then, we can make a new data ser with these features
cluster1=encoded_data.iloc[clus_index==1][feat1]
cluster1.head()
cluster0['Q41_Part_3_Very important'].value_counts()
cluster1['Q41_Part_3_Very important'].value_counts()
hist0=np.sum(cluster0, axis=0)
hist1=np.sum(cluster1, axis=0)
fig, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].bar(np.arange(1, len(hist0)+1),hist0)
ax[0].set_title('Cluster 0')
ax[1].bar(np.arange(1, len(hist1)+1),hist1)
ax[1].set_title('Cluster 1')

print('.')
max0=np.where(max(hist0)==hist0)
min0=np.where(min(hist0)==hist0)

max1=np.where(max(hist1)==hist1)
min1=np.where(min(hist1)==hist1)

print('Max values in cluster 0')
print(cluster0.keys()[max0])

print('Max values in cluster 1')
print(cluster1.keys()[max1])

print('Do you use python on a regular basis? -Cluster 0')
print(cluster0['Q16_Part_1_Python'].value_counts())

print('Do you use python on a regular basis? -Cluster 1')
print(cluster1['Q16_Part_1_Python'].value_counts())

print('Would you recomend Python to an  aspiring data scientist? -Cluster0')
print(cluster0['Q18_Python'].value_counts())

print('Would you recomend Python to an  aspiring data scientist? -Cluster1')
print(cluster1['Q18_Python'].value_counts())

#gets tsome relevant questions
q_keys=[mkeys[372],mkeys[355], mkeys[342], mkeys[263],mkeys[108],mkeys[124],mkeys[129], mkeys[128], 
        mkeys[86],mkeys[84],]
questions=multiplec.head()[q_keys].iloc[0,:]
data0=multi2.iloc[clus_index==0][q_keys]
data0 = data0.fillna(value=-1)

data1=multi2.iloc[clus_index==1][q_keys]
data1 = data1.fillna(value=-1)
print(questions)
print('Opininons about data and programing languages- Cluster 0')
print('.')
i=0
for k in q_keys:
    col=data0[k]
    print(questions[i])
    print('_____ ', col.value_counts().keys()[0])
    print('_____ ',col.value_counts().keys()[1])
    print('.')
    i+=1
print('Opininons about data and programing languages- Cluster 1')
print('.')
i=0
for k in q_keys:
    col=data1[k]
    print(questions[i])
    print('_____ ', col.value_counts().keys()[0])
    print('_____ ',col.value_counts().keys()[1])
    print('.')
    i+=1


