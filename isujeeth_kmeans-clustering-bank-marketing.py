from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import FastICA
#from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score, precision_score, recall_score,f1_score
from sklearn.metrics import make_scorer, accuracy_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


%matplotlib inline
data_1=pd.read_csv("/kaggle/input/bankpromotion/bank-additional-full.csv",sep=";")
data_2=pd.read_csv("/kaggle/input/bankpromotion/bank-additional.csv",sep=";")
data=pd.concat([data_1,data_2],axis=0)
data.head()


#Correlation Plot
plt.figure(figsize=(14,14))
sns.set(font_scale=1)
sns.heatmap(data.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);
plt.title('Variable Correlation')




#To avoid mulicorinality drop the higly correltaed column
data = data.drop(["emp.var.rate","nr.employed"],axis=1)
data.head()


#label encoding

jobDummies = pd.get_dummies(data['job'], prefix = 'job')
maritalDummies = pd.get_dummies(data['marital'], prefix = 'marital')
educationDummies = pd.get_dummies(data['education'], prefix = 'education')
defaultDummies = pd.get_dummies(data['default'], prefix = 'default')
housingDummies = pd.get_dummies(data['housing'], prefix = 'housing')
loanDummies = pd.get_dummies(data['loan'], prefix = 'loan')
contactDummies = pd.get_dummies(data['contact'], prefix = 'contact')
poutcomeDummies = pd.get_dummies(data['poutcome'], prefix = 'poutcome')
data['month']=data['month'].astype('category')
data['day_of_week']=data['day_of_week'].astype('category')
data['y']=data['y'].astype('category')

# Assigning numerical values and storing in another column
data['month'] = data['month'].cat.codes
data['day_of_week'] = data['day_of_week'].cat.codes
data['y'] = data['y'].cat.codes

data['y'].dtype


data["age"]=data["age"].astype("int")
data["duration"]=data["duration"].astype("int")
data["pdays"]=data["pdays"].astype("int")
data["previous"]=data["previous"].astype("int")
data["campaign"]=data["campaign"].astype("int")
data_int=data.select_dtypes(include=['int','float64','bool'])
#data_int
bank_df=pd.concat([data_int,jobDummies,maritalDummies,educationDummies,defaultDummies,housingDummies,loanDummies
                  ,contactDummies,poutcomeDummies,data['month'],data['day_of_week'],data['y']],axis=1)
bank_df.head()




#checking variable distribution
print(len(bank_df.columns))
df_test = bank_df.iloc[:,0:25]
for index in range(25):
    df_test.iloc[:,index] = (df_test.iloc[:,index]-df_test.iloc[:,index].mean()) / df_test.iloc[:,index].std();
df_test.hist(figsize= (14,16));




#Predictors count
bank_df.groupby('y').size()


#Total features after one-hot-encoding
features = bank_df.columns
len(features)
#Variables and Output
y=np.array(bank_df["y"])
X=np.array(bank_df.iloc[:,0:48])
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#kmeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#kmeans
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)


bank_df['clusters1'] = pred_y

bank_df.clusters1.unique()
bank_df['clusters1'].value_counts()
matrix = confusion_matrix(y, bank_df['clusters1'])
print(matrix)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, bank_df['clusters1'])
print('Accuracy: %f' % accuracy)
#precision tp / (tp + fp)
precision = precision_score(y, bank_df['clusters1'])
print('Precision: %f' % precision)
s = silhouette_score(X, kmeans.labels_)
print('Silhouette Score:', s)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)


bank_df['clusters2'] = pred_y

bank_df.clusters2.unique()
bank_df['clusters2'].value_counts()
s = silhouette_score(X, kmeans.labels_)
print('Silhouette Score:', s)
# Create a PCA instance: pca
pca = PCA(n_components=14)
principalComponents = pca.fit_transform(X)# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
print(pca.explained_variance_ratio_)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
ks = range(1, 11)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:2])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
PCA_components.iloc[:,:2]
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(PCA_components.iloc[:,:2])

bank_df['clusters_pca'] = pred_y


bank_df.clusters_pca.unique()
bank_df['clusters_pca'].value_counts()
s = silhouette_score(PCA_components.iloc[:,:2], kmeans.labels_)
print('Silhouette Score:', s)
plt.subplots(figsize=(9,6))
plt.scatter(x=PCA_components.iloc[:,0], y=PCA_components.iloc[:,1], 
            c=kmeans.labels_, cmap=plt.cm.Spectral);
plt.xlabel('PCA1')
plt.ylabel('PCA2')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans - PCA')
#plt.savefig('initial_clusters', bpi=150)
plt.scatter(PCA_components.iloc[:,0],PCA_components.iloc[:,1], c=kmeans.labels_, cmap='rainbow')


matrix = confusion_matrix(y, bank_df['clusters_pca'])
print(matrix)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, bank_df['clusters_pca'])
print('Accuracy: %f' % accuracy)
#precision tp / (tp + fp)
precision = precision_score(y, bank_df['clusters_pca'])
print('Precision: %f' % precision)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
boolvec=sel.get_support()
boolvec.astype(bool)
boolvec
input_file=bank_df.iloc[:,0:48]
#X_RF=input_file.loc[:, sel.get_support()]
#input_file=sgemm_df.loc[:, sel.get_support()].head()
selected_feat= input_file.columns[(sel.get_support())]
#selected_feat = np.where(boolvec[:,None], X_train,X_train)
len(selected_feat)
print(selected_feat)
#sgemm_df

X_RF=input_file.loc[:, sel.get_support()]

ks = range(1, 11)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(X_RF)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_RF)

bank_df['clusters_rf'] = pred_y


bank_df.clusters_rf.unique()
bank_df['clusters_rf'].value_counts()
s = silhouette_score(X_RF, kmeans.labels_)
print('Silhouette Score:', s)
matrix = confusion_matrix(y, bank_df['clusters_rf'])
print(matrix)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, bank_df['clusters_rf'])
print('Accuracy: %f' % accuracy)
#precision tp / (tp + fp)
precision = precision_score(y, bank_df['clusters_rf'])
print('Precision: %f' % precision)

ICA = FastICA(n_components=2, random_state=42) 
X_ica=ICA.fit_transform(X)

## K-Means Clustering Algorithm using ICA
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_ica)

bank_df['clusters_ica'] = pred_y        
#print(correct/len(X_ica))
#yp=kmeans.predict(ica_X_train)
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans - ICA')
s = silhouette_score(X_ica, kmeans.labels_)
print('Silhouette Score:', s)
bank_df['clusters_ica'].value_counts()
# confusion matrix

matrix = confusion_matrix(y, bank_df['clusters_ica'])
print(matrix)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, bank_df['clusters_ica'])
print('Accuracy: %f' % accuracy)
#precision tp / (tp + fp)
precision = precision_score(y, bank_df['clusters_ica'])
print('Precision: %f' % precision)
rca = GaussianRandomProjection(n_components=2, eps=0.1, random_state=42)
X_rca=rca.fit_transform(X)

## K-Means Clustering Algorithm using RCA
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_rca)

bank_df['clusters_rca'] = pred_y        
#print(correct/len(X_ica))
#yp=kmeans.predict(ica_X_train)
plt.scatter(X_rca[:, 0], X_rca[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans - RCA')
s = silhouette_score(X_rca, kmeans.labels_)
print('Silhouette Score:', s)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, bank_df['clusters_rca'])
print('Accuracy: %f' % accuracy)
#precision tp / (tp + fp)
precision = precision_score(y, bank_df['clusters_rca'])
print('Precision: %f' % precision)