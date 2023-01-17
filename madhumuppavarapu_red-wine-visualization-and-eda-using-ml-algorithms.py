import numpy as np

import pandas as pd

# importing visualization maodules

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from mpl_toolkits import mplot3d

from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold

# importing MachineLearning Algorithms

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier

from sklearn.svm import SVC,SVR

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# importing ensembling algorithms

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,RadiusNeighborsClassifier

from sklearn.decomposition import PCA,TruncatedSVD

import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head().style.set_properties(**{'background-color': '#161717','color': 'white','border-color': 'red'})
# finding the null values in the data

print(data.isnull().sum())

sns.heatmap(data.isnull())

plt.show()
# info about the data

data.info()

# description of the data

data.describe().style.set_properties(**{'background-color': '#161717','color': 'white','border-color': 'red'})
# plotting based on quality of the product

fig=plt.figure(figsize=(10,10))

sns.countplot(data=data,y='quality')

plt.show()
# depecting the train_data using pairplot()

fig=plt.figure(figsize=(10,10))

sns.pairplot(data)

plt.show()
# plotting kde graph for the given data

fig=plt.figure(figsize=(20,20))

data.plot(kind='kde',subplots=True,ax=plt.gca())

plt.legend()

plt.show()
# plotting box graph for the given data

fig=plt.figure(figsize=(20,20))

data.plot(kind='box',subplots=True,ax=plt.gca())

plt.legend()

plt.show()
# 3D plots

ax=plt.axes(projection='3d')

ax.plot_wireframe(X=data['fixed acidity'],Y=data['volatile acidity'],Z=data[['quality']])

plt.legend()

plt.show()
# above graph is not so clear so we'll use only 10 records of total_sulphur_dioxide vs sulphates and will plot wireframe plot in 3D

fig=plt.figure(figsize=(10,10))

ax=plt.axes(projection='3d')

ax.plot_wireframe(X=data['fixed acidity'][:10],Y=data['volatile acidity'][:10],Z=data.loc[:10,['quality']])

plt.show()
# surface plot for total_sulphur_dioxide vs sulphates

fig=plt.figure(figsize=(10,10))

ax=plt.axes(projection='3d')

ax.plot_surface(X=data['total sulfur dioxide'][:10],Y=data['sulphates'][:10],Z=data.loc[:10,['quality']])

plt.show()
# visualizing density vs quality

fig=plt.figure(figsize=(20,20))

sns.barplot(data=data,x='density',y='quality')

plt.legend()

plt.show()
# pie chart on quality

plt.figure(1, figsize=(10,10))

data['quality'].value_counts().plot.pie(autopct="%1.1f%%",explode=[0,0.1,0.2,0.1,0.2,0.3],shadow=True)

plt.legend()

plt.show()
fig=plt.figure(figsize=(10,10))

sns.regplot(x='fixed acidity',y='pH',data=data,color='green')

plt.legend()

plt.show()
fig=plt.figure(figsize=(30,20))

sns.stripplot(x='fixed acidity',y='pH',data=data,color='b')

sns.swarmplot(x='fixed acidity',y='pH',data=data,color='k')

sns.violinplot(x='fixed acidity',y='pH',data=data,palette='rainbow')

plt.legend()

plt.show()
# heatmap() for correlation of data

fig=plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),color='blue')

plt.legend()

plt.show()
# bar plot of each and every element

fig=plt.figure(figsize=(20,20))

data.plot(kind='bar',color='black',subplots=True,ax=plt.gca())

plt.legend()

plt.show()
X=data.drop(['quality'],axis="columns")

y=data[['quality']]

X.shape,y.shape
# classification using KNeighborsClassifier

knn_clf=KNeighborsClassifier(n_neighbors=5)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)

knn_clf.fit(X_train,y_train)

knn_clf.predict(X_test)
print('accuracy of train set is {}'.format(knn_clf.score(X_train,y_train)))

print('accuracy of predicted set is {}'.format(knn_clf.score(X_test,knn_clf.predict(X_test))))
# plotting accuracy for a range of numbers 

accuracy=[]

n_neighbors_range=range(1,30)

for i in n_neighbors_range:

    knn_clf=KNeighborsClassifier(n_neighbors=i)

    knn_clf.fit(X_train,y_train)

    accuracy.append(knn_clf.score(X_test,knn_clf.predict(X_test)))

fig=plt.figure(figsize=(5,5))

plt.plot(n_neighbors_range,accuracy)
# classification using DecisionTreeClassifier

dec_clf=DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)

dec_clf.fit(X_train,y_train)

dec_clf.predict(X_test)
print('accuracy of train set is {}'.format(dec_clf.score(X_train,y_train)))

print('accuracy of predicted set is {}'.format(dec_clf.score(X_test,dec_clf.predict(X_test))))
# accuracy of some Supervised Classifier models has been compared with Unsupervised KMeans algorithm

log_reg=LogisticRegression(C=10)

log_reg.fit(X_train,y_train)

knn_clf=KNeighborsClassifier(n_neighbors=5)

knn_clf.fit(X_train,y_train)

svc=SVC(C=10)

svc.fit(X_train,y_train)

mnb=MultinomialNB()

mnb.fit(X_train,y_train)

bnb=BernoulliNB()

bnb.fit(X_train,y_train)

gnb=GaussianNB()

gnb.fit(X_train,y_train)

dec_clf=DecisionTreeClassifier(criterion='entropy')

dec_clf.fit(X_train,y_train)

ran_clf=RandomForestClassifier(n_estimators=5)

ran_clf.fit(X_train,y_train)

rad_clf=RadiusNeighborsClassifier(radius=100.0)

rad_clf.fit(X_train,y_train)

passive=PassiveAggressiveClassifier(C=10)

passive.fit(X_train,y_train)

extra_clf=ExtraTreesClassifier()

extra_clf.fit(X_train,y_train)

bag_clf=BaggingClassifier()

bag_clf.fit(X_train,y_train)

adaboost=AdaBoostClassifier()

adaboost.fit(X_train,y_train)

gradient=GradientBoostingClassifier()

gradient.fit(X_train,y_train)

print()
# plotting accuracy for a range of values in KMeans

kmeans=KMeans(n_clusters=3)

kmeans.fit(X_train,y_train)

accuracy=[]

n_neighbors_range=range(1,30)

for i in n_neighbors_range:

    kmeans=KMeans(n_clusters=3)

    kmeans.fit(X_train,y_train)

    accuracy.append(kmeans.score(X_test,kmeans.predict(X_test)))

fig=plt.figure(figsize=(5,5))

plt.plot(n_neighbors_range,accuracy)
# accuracy of some algorithms stored in a dictionary data structure

accuracy_of_diff_test_ml_algo={'LogisticRegression':log_reg.score(X_test,log_reg.predict(X_test)),'KNeighborsClassifier':knn_clf.score(X_test,knn_clf.predict(X_test)),'SVC':svc.score(X_test,svc.predict(X_test)),'MultinomialNB':mnb.score(X_test,mnb.predict(X_test)),'BernoulliNB':bnb.score(X_test,bnb.predict(X_test)),'GaussianNB':gnb.score(X_test,gnb.predict(X_test)),'DecisionTreeClassifier':dec_clf.score(X_test,dec_clf.predict(X_test)),'RandomForestClassifier':ran_clf.score(X_test,ran_clf.predict(X_test)),'RadiusNeighborsClassifier':rad_clf.score(X_test,rad_clf.predict(X_test)),'PassiveAggressiveClassifier':passive.score(X_test,passive.predict(X_test)),'ExtraTreesClassifier':extra_clf.score(X_test,extra_clf.predict(X_test)),'BaggingClassifier':bag_clf.score(X_test,bag_clf.predict(X_test)),'AdaBoostClassifier':adaboost.score(X_test,adaboost.predict(X_test)),'GradientBoostingClassifier':gradient.score(X_test,gradient.predict(X_test)),'KMeans':kmeans.score(X_test,kmeans.predict(X_test))}

print(accuracy_of_diff_test_ml_algo)

values=accuracy_of_diff_test_ml_algo.values()

keys=accuracy_of_diff_test_ml_algo.keys()

values,keys
# plotting accuracy of some supervised algorthms vs unsupervised

fig=plt.figure(figsize=(20,20))

plt.bar(keys,values)
fig=plt.figure(figsize=(20,20))

data.plot(kind='hist',subplots=True,ax=fig.gca())

plt.legend()

plt.show()
fig=plt.figure(figsize=(20,20))

sns.jointplot(data=data,x='density',y='pH',kind='hex')

plt.legend()

plt.show()
fig=plt.figure(figsize=(20,20))

sns.jointplot(data=data,x='density',y='pH',kind='reg')

plt.legend()

plt.show()
fig=plt.figure(figsize=(20,20))

sns.pairplot(data=data,corner=True)

plt.legend()

plt.show()