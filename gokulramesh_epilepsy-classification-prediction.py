#importing numpy and pandas
import numpy as np
import pandas as pd

#importing for visualization
import matplotlib.pyplot as plt

%matplotlib inline

r_s=123456
#importing dataset
df = pd.read_csv('../input/data.csv')
df.head()
df.shape
#Structural analysis
df.info()
#statistical analysis
df.describe()
df.isnull().values.any()
df.corr()
#Dropping the 'Unnamed: 0' column
df.drop('Unnamed: 0',inplace=True,axis=1)
df.head()
#target variable
#df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)
y=df['y']

y = y.apply(lambda x: 1 if x == 1 else 0)
y.unique()

#dropping the y column
X=df.drop('y',axis=1)
X.head()
#SCALING
from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(X)
x_scaled=scale.transform(X)
x_scaled
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  #two component
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.head()
#Concating Principal components and y varaible
PCAdf= pd.concat([principalDf, y], axis = 1)
PCAdf.head()

PCAdf.shape
from sklearn.manifold import TSNE

TNSEdf = TSNE(random_state=r_s).fit_transform(x_scaled)
TNSEdf.shape
tnsedf = pd.DataFrame(TNSEdf)
tnsedf['y']=y
tnsedf.head()
#K-Means
from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=5)

# Centroid values
#centroids = kmeans.cluster_centers_
#implementing kmeans in pcadf
kmeansPCA = kmeans.fit(principalDf)
#Getting the cluster labels
labels = kmeans.predict(principalDf)
PK_df= PCAdf
PK_df['kclust']=labels
PK_df['kclust'].unique()
#K-Means
from sklearn.cluster import KMeans
# Number of clusters
kmeans = KMeans(n_clusters=5)
#implementing kmeans in tnsedf
kmeanstnse = kmeans.fit(TNSEdf)
#Getting the cluster labels
k_labels = kmeans.predict(TNSEdf)
TK_df = tnsedf
TK_df['cluster']=k_labels
TK_df.info()
TK_df['cluster'].unique()
#splitting into x and y varialbles
'''
p_x=PCAdf.loc[:,['principal component 1','principal component 2']]
p_y=PCAdf ['y']

t_x=tnsedf.loc[:,[0,1]]
t_y=tnsedf['y']
'''

pk_x=PK_df.loc[:,['principal component 1','principal component 2','kclust']]
pk_y=PK_df['y']

tk_x=TK_df.loc[:,[0,1,'cluster']]
tk_y=TK_df['y']
from sklearn.model_selection import train_test_split
'''
#only PCA
px_train, px_test, py_train, py_test = train_test_split(p_x,p_y,
                                                    test_size = 0.2, 
                                                    random_state = 101)'''

#PCA +k-means
pkx_train, pkx_test, pky_train, pky_test = train_test_split(pk_x,pk_y,
                                                            test_size = 0.2, 
                                                            random_state = 101)

'''
#T-sne only
tx_train, tx_test, ty_train, ty_test = train_test_split(t_x,t_y,
                                                        test_size = 0.2, 
                                                        random_state = 101)'''

#T-SNE + k-means
tkx_train, tkx_test, tky_train, tky_test = train_test_split(tk_x,tk_y,
                                                            test_size = 0.2, 
                                                            random_state = 101)
#Confusion matrix and accuraccy score
from sklearn.metrics import confusion_matrix, accuracy_score
#Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 

logmodel.fit(pkx_train,pky_train)
pklogpred = logmodel.predict(pkx_test)


logmodel.fit(tkx_train,tky_train)
tklogpred = logmodel.predict(tkx_test)
print(confusion_matrix(pky_test, pklogpred))
print(round(accuracy_score(pky_test, pklogpred),2)*100)
print(confusion_matrix(tky_test, tklogpred))
print(round(accuracy_score(tky_test, tklogpred),2)*100)
#cross validation
LOGCV = (cross_val_score(logmodel, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(pkx_train, pky_train)
pknnpred = knn.predict(pkx_test)

knn.fit(tkx_train,tky_train)
tknnpred = knn.predict(tkx_test)

print(confusion_matrix(pky_test, pknnpred))
print(round(accuracy_score(pky_test, pknnpred),2)*100)
print(confusion_matrix(tky_test, tknnpred))
print(round(accuracy_score(tky_test, tknnpred),2)*100)
KNNCV = (cross_val_score(knn, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.svm import SVC
svc= SVC(kernel = 'sigmoid')
#There are various kernels linear,rbf
svc.fit(pkx_train, pky_train)
pspred = svc.predict(pkx_test)

svc.fit(tkx_train, tky_train)
tspred = svc.predict(tkx_test)


print(confusion_matrix(pky_test, pspred))
print(round(accuracy_score(pky_test, pspred),2)*100)
print(confusion_matrix(tky_test, tspred))
print(round(accuracy_score(tky_test, tspred),2)*100)
SVCCV = (cross_val_score(svc, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini

rfc.fit(pkx_train, pky_train)
prpred = rfc.predict(pkx_test)

rfc.fit(tkx_train, tky_train)
trpred = rfc.predict(tkx_test)

print(confusion_matrix(pky_test, prpred))
print(round(accuracy_score(pky_test, prpred),2)*100)
print(confusion_matrix(tky_test, trpred))
print(round(accuracy_score(tky_test, trpred),2)*100)
#Cross validation
RFCCV = (cross_val_score(rfc, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()

gaussiannb.fit(pkx_train, pky_train)
pgpred = gaussiannb.predict(pkx_test)

gaussiannb.fit(tkx_train, tky_train)
tgpred = gaussiannb.predict(tkx_test)
print(confusion_matrix(pky_test, pgpred))
print(round(accuracy_score(pky_test, pgpred),2)*100)
print(confusion_matrix(tky_test, tgpred))
print(round(accuracy_score(tky_test, tgpred),2)*100)
NBCV = (cross_val_score(gaussiannb, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()

xgb.fit(pkx_train, pky_train)
pxpred = xgb.predict(pkx_test)

xgb.fit(tkx_train, tky_train)
txpred = xgb.predict(tkx_test)
print(confusion_matrix(pky_test, pxpred))
print(round(accuracy_score(pky_test, pxpred),2)*100)
print(confusion_matrix(tky_test, txpred))
print(round(accuracy_score(tky_test, txpred),2)*100)
XGCV = (cross_val_score(xgb, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost'],
                'Score':  [RFCCV, SVCCV, KNNCV, LOGCV, NBCV, XGCV]})

models.sort_values(by='Score', ascending=False)