# import basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
data = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

# Check the data

data.info()
data.head()
sns.countplot(data['species'],palette='spring');
sns.pairplot(data,hue='species');
print("Let's explore distribution of our data.")

fig,axes=plt.subplots(4,1,figsize=(5,20))

sns.boxplot(x=data.species,y=data.flipper_length_mm,ax=axes[0],palette='summer')

axes[0].set_title("Flipper length distribution",fontsize=20,color='Red')

sns.boxplot(x=data.species,y=data.culmen_length_mm,ax=axes[1],palette='rocket')

axes[1].set_title("Culmen length distribution",fontsize=20,color='Red')

sns.boxplot(x=data.species,y=data.culmen_depth_mm,ax=axes[2],palette='twilight')

axes[2].set_title("Culmen depth distribution",fontsize=20,color='Red')

sns.boxplot(x=data.species,y=data.body_mass_g,ax=axes[3],palette='Set2')

axes[3].set_title("Body mass distribution",fontsize=20,color='Red')

plt.tight_layout();
print("Mean body mass index distribution")

data.groupby(['species','sex']).mean()['body_mass_g'].round(2)
100*data.isnull().sum()/len(data)
data['sex'].fillna(data['sex'].mode()[0],inplace=True)

col_to_be_imputed = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']

for item in col_to_be_imputed:

    data[item].fillna(data[item].mean(),inplace=True)
data.species.value_counts()
data.island.value_counts()
data.sex.value_counts()
data[data['sex']=='.']
data.loc[336,'sex'] = 'FEMALE'
# Target variable can also be encoded using sklearn.preprocessing.LabelEncoder

data['species']=data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})



# creating dummy variables for categorical features

dummies = pd.get_dummies(data[['island','sex']],drop_first=True)
# we do not standardize dummy variables 

df_to_be_scaled = data.drop(['island','sex'],axis=1)

target = df_to_be_scaled.species

df_feat= df_to_be_scaled.drop('species',axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_feat)

df_scaled = scaler.transform(df_feat)

df_scaled = pd.DataFrame(df_scaled,columns=df_feat.columns[:4])

df_preprocessed = pd.concat([df_scaled,dummies,target],axis=1)

df_preprocessed.head()
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



kmeans = KMeans(3,init='k-means++')

kmeans.fit(df_preprocessed.drop('species',axis=1))

print(confusion_matrix(df_preprocessed.species,kmeans.labels_))
print(classification_report(df_preprocessed.species,kmeans.labels_))
f"Accuracy is {np.round(100*accuracy_score(df_preprocessed.species,kmeans.labels_),2)}"
wcss=[]

for i in range(1,10):

    kmeans = KMeans(i)

    kmeans.fit(df_preprocessed.drop('species',axis=1))

    pred_i = kmeans.labels_

    wcss.append(kmeans.inertia_)



plt.figure(figsize=(10,6))

plt.plot(range(1,10),wcss)

plt.ylim([0,1800])

plt.title('The Elbow Method',{'fontsize':20})

plt.xlabel('Number of clusters')

plt.ylabel('Within-cluster Sum of Squares');
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



# We need to split data for supervised learning models.

X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop('species',axis=1),target,test_size=0.50)



knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

preds_knn = knn.predict(X_test)

print(confusion_matrix(y_test,preds_knn))
plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(confusion_matrix(y_test,preds_knn),annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24});
print(classification_report(y_test,preds_knn))
print(accuracy_score(y_test,preds_knn))
# print the scores on training and test set



print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))



print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))
error_rate=[]

for i in range(1,10):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate');
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

preds_knn = knn.predict(X_test)

print(confusion_matrix(y_test,preds_knn))

print(classification_report(y_test,preds_knn))