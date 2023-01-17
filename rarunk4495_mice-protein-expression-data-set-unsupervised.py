import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df=pd.read_excel("../input/Data_Cortex_Nuclear.xls")
df.head()
df.tail(5)
df.describe(include='all')
df.info()
df.Genotype.unique()
df.Treatment.unique()
df.Behavior.unique()
dist_class=list(df['class'].unique())

dist_class

df.Genotype=df.Genotype.map({"Control":0,"Ts65Dn":1})

df.Treatment=df.Treatment.map({'Memantine':0,'Saline':1})

df.Behavior=df.Behavior.map({'C/S':0,'S/C':1})

df['class']=df['class'].map({'c-CS-m':0,'c-CS-s':1,'c-SC-m':2,'c-SC-s':3,'t-CS-m':4,'t-CS-s':5,'t-SC-m':6,'t-SC-s':7})
df.head(10)
df.tail(10)
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)



missing_data.head()
df['class']=df['class'].astype("category")
x=df.drop(columns=['MouseID','class'])

y=df['class']
from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=np.nan,strategy='median')

x_impute=pd.DataFrame(impute.fit_transform(x),columns=x.columns)
x_impute
null=x_impute.isnull().sum()

print(null[null==0])
x_impute.describe()
sns.boxplot(x_impute.NR2A_N,color='r')
sns.boxplot(x=x_impute.DYRK1A_N)
sns.boxplot(x=x_impute.BDNF_N)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_impute_scaled=sc.fit_transform(x_impute.drop(columns=['Genotype','Treatment','Behavior']))
x_impute_scaled[:,72:78]
x_impute_scaled.shape
x_impute.columns
x_impute_scaled_df=pd.DataFrame(x_impute_scaled,columns=x_impute.columns[0:77])

x_impute_scaled_df['Genotype']=x_impute['Genotype']
x_impute_scaled_df['Genotype']=x_impute['Genotype']
x_impute_scaled_df['Treatment']=x_impute['Treatment']
x_impute_scaled_df['Behavior']=x_impute['Behavior']
x_impute_scaled_df
from sklearn.cluster import KMeans
cluster_errors = []



for i in range(1,21):

    clusters = KMeans(i)

    clusters.fit(x_impute_scaled_df)

    cluster_errors.append(clusters.inertia_)
#WSS

clusters_df = pd.DataFrame({"Num_clusters":range(1,21),"cluster_errors":cluster_errors})

clusters_df[0:12]
sns.pointplot(clusters_df.Num_clusters,clusters_df.cluster_errors,marker=".")
kmeans=KMeans(n_clusters=4).fit(x_impute_scaled_df)
k_means_y=kmeans.labels_
x_impute_scaled_df['k_means_y']=k_means_y
x_impute_scaled_df.head()
x_impute_scaled_df=x_impute_scaled_df.drop(columns=['Genotype','Treatment','Behavior'])
# Final X    x_impute_scaled_df  

# Final Y    k_means_y   
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x_impute_scaled_df,k_means_y,test_size=0.3,random_state=0)
from sklearn.decomposition import PCA



pca=PCA(n_components=4)

xtrain_pca=pca.fit_transform(xtrain)

xtest_pca=pca.transform(xtest)
from matplotlib import pyplot as plt

plt.plot(pca.explained_variance_ratio_)

plt.xlabel('No of components')

plt.ylabel('Cumulative explained variance')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
knn=KNeighborsClassifier()

RF=RandomForestClassifier()

DT=DecisionTreeClassifier()

LR=LogisticRegression()

NB=GaussianNB()

SVM=SVC()
from sklearn import metrics

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")
unsup_accuracy_pca=[]



for name, model in zip(['KNN','RandomForest','Logistic_Reg','Naive Bayes','SVM','DecisonTree'],

                      [knn,RF,LR,NB,SVM,DT]):

    model.fit(xtrain_pca,ytrain)

    ypred_unsup_pca=model.predict(xtest_pca)

    accuracy =metrics.accuracy_score(ypred_unsup_pca,ytest)

    unsup_accuracy_pca.append(accuracy)

    print("Accuracy of unsupervised model with PCA %s: %0.02f"%(name,accuracy))
unsup_accuracy=[]



for name, model in zip(['KNN','RandomForest','Logistic_Reg','Naive Bayes','SVM','DecisonTree'],

                      [knn,RF,LR,NB,SVM,DT]):

    model.fit(xtrain,ytrain)

    ypred_unsup_no_pca=model.predict(xtest)

    accuracy =metrics.accuracy_score(ypred_unsup_no_pca,ytest)

    unsup_accuracy.append(accuracy)

    print("Accuracy of unsupervised model without PCA %s: %0.02f"%(name,accuracy))
xtrain_sup,xtest_sup,ytrain_sup,ytest_sup=train_test_split(x_impute_scaled_df,y,test_size=0.3,random_state=42)
sup_accuracy=[]



for name, model in zip(['KNN','RandomForest','Logistic_Reg','Naive Bayes','SVM','DecisonTree'],

                      [knn,RF,LR,NB,SVM,DT]):

    model.fit(xtrain_sup,ytrain_sup)

    ypred_sup_pca=model.predict(xtest_sup)

    accuracy =metrics.accuracy_score(ypred_sup_pca,ytest_sup)

    sup_accuracy.append(accuracy)

    print("Accuracy of supervised model without PCA %s: %0.02f"%(name,accuracy))
pd.DataFrame({'MODEL':['Unsupervised model with PCA KNN',

                       'Unsupervised model with PCA RandomForest',

                        'Unsupervised model with PCA Logistic_Reg',

                      'Unsupervised model with PCA Naive Bayes',

                       'Unsupervised model with PCA SVM',

                       'Unsupervised model with PCA DecisonTree',

                       'Unsupervised model without PCA KNN',

                      'Unsupervised model without PCA RandomForest',

                      'Unsupervised model without PCA Logistic_Reg',

                      'Unsupervised model without PCA Naive Bayes',

                      'Unsupervised model without PCA SVM',

                      'Unsupervised model without PCA DecisonTree',

                       'Supervised model without PCA KNN',

                       'Supervised model without PCA RandomForest',

                       'Supervised model without PCA Logistic_Reg',

                       'Supervised model without PCA Naive Bayes',

                       'Supervised model without PCA SVM',

                       'Supervised model without PCA DecisonTree'

                       

                       

                      

                      

                      ],

              'ACCURACY':[0.96,0.93,0.94,0.95,0.94,0.92,0.86,0.91,0.89,0.94,0.93,0.87,0.95,0.94,0.98,0.77,0.99,0.85]         

})
pd.DataFrame({'Accuracy of unsupervised model with PCA':unsup_accuracy_pca,

             'Accuracy of unsupervised model without PCA':unsup_accuracy,

             'Accuracy of supervised model without PCA':sup_accuracy}, 

             index=['KNN','RandomForest','Logistic_Reg','Naive Bayes','SVM','DecisonTree'])