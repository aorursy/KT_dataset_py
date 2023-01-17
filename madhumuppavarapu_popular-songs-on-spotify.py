import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,SVR

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# importing ensembling algorithms

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,RadiusNeighborsClassifier

from sklearn.decomposition import PCA,TruncatedSVD
data=pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

data.head()
data.reset_index(drop=True)
print(data.info())
sns.heatmap(data.isnull())
data.isnull().sum()
# finding the number of songs of each genre

data.groupby(data['Genre']).size()
# the columns names are not well represented so we'll rename our columns

data.rename(columns={'Unnamed: 0':'sl.No','Track.Name':'Track_name','Artist.Name':'Artist_name','Beats.Per.Minute':'Beats_per_minute','Loudness..dB..':'Loudness(db)','Valence.':'Valence','Length.':'Length','Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)

data.head()
# finding the number of songs sung by each Artist

data.groupby(data['Artist_name']).size()
# Artist with most popular track

data['Artist_name'][data['Popularity']==data['Popularity'].max()]
# plotting between Genre and count

sns.countplot(data=data,y='Genre')
# popularity of each and every track_name

plt.pie(np.array(data['Popularity']),labels=np.array(data.Track_name),radius=10,autopct='%0.1f%%',shadow=True,startangle=90)

plt.show()
# Danceability of each and every track_name

plt.pie(np.array(data['Danceability']),labels=np.array(data.Track_name),radius=10,autopct='%0.1f%%',shadow=True,startangle=90,explode=[0,0.5,0,0,1.0,0,0.5,0,0.5,0,0,0,0.5,0.5,0,0,1.0,0.7,0,0,0.8,0,0,1.5,0.5,0,1.0,0,0.5,0,0,1.0,0,0,0,1.0,1.0,1.0,2.0,0,0,0,0,0,0,1.0,0.5,0,0,0.5])

plt.show()
# representing the data in a pairplot()

sns.pairplot(data)
# countplot representing total no. of songs by a group of artists

sns.countplot(data=data,y=data['Artist_name'].value_counts())
fig=plt.figure(figsize=(10,30))

sns.barplot(data=data,y='Artist_name',x=data['Artist_name'].nunique())
#barplot of each numerical column in the data

fig=plt.figure(figsize=(20,20))

ax=plt.gca()

data.plot(kind='hist',subplots=True,ax=ax)

plt.show()
fig=plt.figure(figsize=(20,20))

ax=plt.gca()

data.plot(kind='kde',subplots=True,ax=ax)

plt.show()
# finding correlation of the data

fig=plt.figure(figsize=(10,10))

sns.heatmap(data.corr())
fig=plt.figure(figsize=(20,20))

ax=plt.gca()

data.plot(kind='box',subplots=True,ax=ax)

plt.show()
# we will check the popularity based on danceability,acousticness and energy 

fig=plt.figure(figsize=(10,10))

plt.subplot(3,1,1)

plt.xlabel('popularity')

plt.ylabel('energy')

plt.title('popularity vs energy')

sns.barplot(data=data,y='Energy',x='Popularity')

plt.subplot(3,1,2)

plt.xlabel('popularity')

plt.ylabel('danceability')

plt.title('popularity vs danceability')

sns.barplot(data=data,y='Danceability',x='Popularity')

plt.subplot(3,1,3)

plt.xlabel('popularity')

plt.ylabel('acoustics')

plt.title('popularity vs acousticness')

sns.barplot(data=data,y='Acousticness',x='Popularity')



# relation between energy and loudness(db)

fig=plt.figure(figsize=(10,10))

sns.regplot(data=data,x='Energy',y='Loudness(db)')
# relation between acousticness and loudness

fig=plt.figure(figsize=(10,10))

sns.jointplot(data=data,x='Acousticness',y='Loudness(db)',kind='kde',color='black')
fig=plt.figure(figsize=(15,15))

sns.countplot(data=data,y='Popularity',hue='Genre')

plt.legend(loc="upper right")
data.columns
# applying some machine learning modelling algorithms

lin_reg=LinearRegression()

X=data.loc[:,['Beats_per_minute',

       'Energy', 'Danceability', 'Loudness(db)', 'Liveness', 'Valence',

       'Length', 'Acousticness', 'Speechiness']]

y=data.loc[:,['Popularity']]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)

y_pred,y_test
print(lin_reg.score(X_train,y_train))

lin_reg.score(X_test,y_pred)
plt.subplot(2,2,1)

sns.kdeplot(X_train,y_train)

plt.subplot(2,2,2)

sns.kdeplot(X_test,y_pred)

plt.subplot(2,2,3)

plt.scatter(X_train.iloc[:,0:1],y_train)

plt.plot(X_test.iloc[:,0:1],y_pred)
# comparing accuracy scores for LinearRegression and Lasso and Ridge

lin_accuracy=lin_reg.score(X_test,y_pred)

ridge=Ridge().fit(X_train,y_train)

ridge_accuracy=ridge.score(X_test,ridge.predict(X_test))

lasso=Lasso().fit(X_train,y_train)

lasso_accuracy=lasso.score(X_test,lasso.predict(X_test))

print(lin_accuracy)

print(ridge_accuracy)

lasso_accuracy
# predicting the genre by taking the numerical values present in the dataset

# - we will use KNeighborsClassifier Algorithm

X=data.loc[:,['Beats_per_minute',

       'Energy', 'Danceability', 'Loudness(db)', 'Liveness', 'Valence',

       'Length', 'Acousticness', 'Speechiness','Popularity']]

y=data.loc[:,['Genre']]

knn_clf=KNeighborsClassifier(n_neighbors=5)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

knn_clf.fit(X_train,y_train)

print(knn_clf.score(X_train,y_train))

knn_clf.score(X_test,knn_clf.predict(X_test))
n_range=range(1,20)

accuracy=[]

for n in n_range:

    knn_clf=KNeighborsClassifier(n_neighbors=n)

    knn_clf.fit(X_train,y_train)

    accuracy.append(knn_clf.score(X_test,knn_clf.predict(X_test)))

plt.scatter(n_range,accuracy)
print(knn_clf.predict(X_test))
gnb=GaussianNB()

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

print(gnb.fit(X_train,y_train))

print(gnb.score(X_train,y_train))

gnb.score(X_test,gnb.predict(X_test))
gnb.predict(X_test)
# we will have to first import PCA from decomposition package

from sklearn.decomposition import PCA

pca=PCA(n_components=5)

pca
# also we will have to scale the data into standard scaler

scaler=StandardScaler()

scaled_X=scaler.fit_transform(X)

pca_scaled=pca.fit_transform(scaled_X)

pca_scaled
X_train,X_test,y_train,y_test=train_test_split(pca_scaled,y,random_state=0,test_size=0.2)

print(gnb.fit(X_train,y_train))

print(gnb.score(X_train,y_train))

gnb.score(X_test,gnb.predict(X_test))
# we will find the correlation of the PCA scaled values

print(pd.DataFrame(pca_scaled).corr())

sns.heatmap(pd.DataFrame(pca_scaled).corr())
sns.pairplot(pd.DataFrame(pca_scaled))
sns.pairplot(pd.DataFrame(pca_scaled),corner=True)
sns.pairplot(pd.DataFrame(pca_scaled),kind='kde')
dec_clf=DecisionTreeClassifier(criterion='entropy')

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

print(dec_clf.fit(X_train,y_train))

print(dec_clf.score(X_train,y_train))

dec_clf.score(X_test,dec_clf.predict(X_test))
from sklearn import tree

fig=plt.figure(figsize=(20,20))

_=tree.plot_tree(dec_clf,feature_names=data.Popularity,class_names=data.Genre,filled=True,max_depth=3)