# Importing the libraries
import pandas as pd        # for data manipulation
import seaborn as sns      # for statistical data visualisation
import numpy as np         # for linear algebra
import matplotlib.pyplot as plt      # for data visualization
from scipy import stats        # for calculating statistics

# Importing various machine learning algorithm from sklearn

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error,roc_curve,auc,accuracy_score
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


dataframe= pd.read_csv("vehicle.csv")  # Reading the data
dataframe.head()   # showing first 5 datas
dataframe.shape
dataframe.info()
dataframe.isnull().sum()
dataframe.apply(lambda x: len(x.unique()))
dataframe.describe()
dataframe.fillna(dataframe.mean(),axis=0,inplace=True)
dataframe.isnull().sum()
dataframe.describe()
dt=dataframe.iloc[:,0:18].copy()
dt
sns.pairplot(dt)
dataframe['class'].value_counts()
f = plt.subplots(1, figsize=(12,6))

colors = ["#FA5858", "#64FE2E","#ac3e9a"]
labels ="Car", "Bus","Van"

plt.suptitle('Information on Term Deposits', fontsize=20)

plt.pie(dataframe['class'].value_counts(),labels=labels,explode=(0,0.1,0.15),shadow=True,colors=colors, startangle=25,autopct='%1.1f%%')

plt.show()
dataframe.skew()
plt.figure(figsize=(10,10))
plt.subplot(6,1,1)
sns.boxplot(dataframe.compactness)
plt.subplot(6,1,2)
sns.boxplot(dataframe.circularity)
plt.subplot(6,1,3)
sns.boxplot(dataframe.distance_circularity)
plt.subplot(6,1,4)
sns.boxplot(dataframe.radius_ratio)
plt.subplot(6,1,5)
sns.boxplot(dataframe["pr.axis_aspect_ratio"])
plt.subplot(6,1,6)
sns.boxplot(dataframe["scaled_radius_of_gyration.1"])
plt.figure(figsize=(10,10))
plt.subplot(6,1,1)
sns.distplot(dataframe.compactness)
plt.subplot(6,1,2)
sns.distplot(dataframe.circularity)
plt.subplot(6,1,3)
sns.distplot(dataframe.distance_circularity)
plt.subplot(6,1,4)
sns.distplot(dataframe.radius_ratio)
plt.subplot(6,1,5)
sns.distplot(dataframe["pr.axis_aspect_ratio"])
plt.subplot(6,1,6)
sns.distplot(dataframe["scaled_radius_of_gyration.1"])

fig,(a1,a2)=plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'hollows_ratio', data = dataframe, orient = 'v', ax = a1)
sns.distplot(dataframe.hollows_ratio, ax = a2)
dataframe["class"].hist(bins=3)
plt.figure(figsize=(8,8))
sns.scatterplot(x='elongatedness',y='scatter_ratio' ,data=dataframe,hue='class')

fig,(a1,a2)=plt.subplots(nrows = 2, ncols = 1, figsize = (13, 15))
sns.boxplot(x='class',y='max.length_rectangularity',data=dataframe,ax=a1)
sns.scatterplot(x='circularity',y='max.length_rectangularity',hue='class',data=dataframe,ax=a2)

plt.figure(figsize=(10,10))
sns.catplot(x="class",y='scaled_radius_of_gyration.1', data=dataframe)
sns.lineplot(x='circularity',y='distance_circularity',data=dataframe,hue='class')

corelation=dt.corr()
corelation
plt.figure(figsize=(20,20))
a=sns.heatmap(corelation,annot=True)
dataframe.columns
features=['compactness', 'circularity', 'distance_circularity', 'radius_ratio',
       'pr.axis_aspect_ratio', 'max.length_aspect_ratio', 'scatter_ratio',
       'elongatedness', 'pr.axis_rectangularity', 'max.length_rectangularity',
       'scaled_variance', 'scaled_variance.1', 'scaled_radius_of_gyration',
       'scaled_radius_of_gyration.1', 'skewness_about', 'skewness_about.1',
       'skewness_about.2', 'hollows_ratio']
X=dataframe[features]
Y=dataframe['class']         
X=X.apply(zscore)
X
Y.replace(['van','car','bus'],[1,2,3],inplace=True)
Y
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 
train_X.head()
test_X.count()
test_X.head()
parameters={
    'C':[0.01,0.25,0.5,1],
    'kernel':['rbf','linear']
}
model=SVC()
best_SVC=GridSearchCV(model,param_grid=parameters,scoring='accuracy',cv=10)
best_SVC
best_SVC.fit(train_X,train_y)
best_SVC.best_params_
svm_model=SVC(C=1,kernel='rbf',random_state=1)
svm_model
svm_model=svm_model.fit(train_X,train_y)
predict=svm_model.predict(test_X)
print(predict[0:1000])
metrics=confusion_matrix(test_y,predict)
metrics
sns.heatmap(metrics,annot=True,fmt='g',cmap='Blues')
print(classification_report(test_y,predict))
svm_accuracy=accuracy_score(test_y,predict)
svm_accuracy

svm_eval = cross_val_score(estimator = svm_model, X = train_X, y = train_y, cv = 10)
svm_eval.mean()
pca=PCA(n_components=18)
pca
pca.fit(X)
pca.explained_variance_
pca.explained_variance_ratio_
pca.components_
sns.barplot(x=list(range(1,19)),y=pca.explained_variance_)
plt.plot(list(range(1,19)),pca.explained_variance_,'ro-', linewidth=2)
plt.title('Elbow Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.step(list(range(1,19)), np.cumsum(pca.explained_variance_ratio_), where = 'mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('Eigen value')

pca=PCA(n_components=8)
pca.fit(X)
pca.explained_variance_
pca.explained_variance_ratio_
pca_X = pca.transform(X)
sns.pairplot(pd.DataFrame(pca_X))
train_X,test_X,train_y,test_y=train_test_split(pca_X,Y,test_size=0.3,random_state=1)
pd.DataFrame(train_X).count()
parameters={
    'C':[0.01,0.25,0.5,1],
    'kernel':['rbf','linear']
}
model=SVC()
best_PCA_SVC_grid=GridSearchCV(model,param_grid=parameters,scoring='accuracy',cv=10)
best_PCA_SVC_grid
best_PCA_SVC_grid.fit(train_X,train_y)
best_PCA_SVC_grid.best_params_
best_PCA_SVC=SVC(C=1,kernel='rbf',random_state=1)
best_PCA_SVC
best_PCA_SVC=best_PCA_SVC.fit(train_X,train_y)
predict=best_PCA_SVC.predict(test_X)
print(predict[0:1000])
metrics_pca=confusion_matrix(test_y,predict)
metrics_pca
sns.heatmap(metrics_pca,annot=True,fmt='g',cmap='Blues')
print(classification_report(test_y,predict))
pca_svm_accuracy=accuracy_score(test_y,predict)
pca_svm_accuracy

pca_svm_eval = cross_val_score(estimator = best_PCA_SVC, X = train_X, y = train_y, cv = 10)
pca_svm_eval.mean()
data=[[svm_accuracy,svm_eval.mean()],[pca_svm_accuracy,pca_svm_eval.mean()]]
compare=pd.DataFrame(data,columns=["Accuracy","Cross validation Mean"],index=["SVC","SVC with PCA"])
compare
plt.subplot(2,2,1)
sns.heatmap(metrics,annot=True,fmt='g',cmap='Blues')
plt.subplot(2,2,2)
sns.heatmap(metrics_pca,annot=True,fmt='g',cmap='Blues')