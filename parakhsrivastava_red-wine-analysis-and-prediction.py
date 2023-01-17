import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv('../input/winequality-red.csv')

# Reading Data 
data.head()

# Top 5 rows
data.info()

# Information about data types and null values
data.describe()

# Statistical Analysis
data.isnull().any()

# data[data.isnull()].count()
corr=data.corr()



sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='magma',linecolor="black")

plt.title('Correlation between features');
data['quality'].value_counts()
sns.countplot(x='quality',data=data)

plt.title('Quality Variable Analysis')

plt.xlabel('Quality').set_size(20)

plt.ylabel('Frequency').set_size(20)

plt.show()
sns.pairplot(data,plot_kws={'alpha':0.3})

# data.hist(bins=50,figsize=(15,15))
plt.figure(figsize=(10,8))

sns.boxplot(data['quality'],data['fixed acidity'])
plt.figure(figsize=(10,8))

sns.pointplot(data['quality'],data['pH'],color='grey')

plt.xlabel('Quality').set_size(20)

plt.ylabel('pH').set_size(20)
sns.regplot('alcohol','density',data=data)
sns.regplot('pH','alcohol',data=data)
sns.regplot('fixed acidity','citric acid',data=data)
sns.regplot('pH','fixed acidity',data=data)
bins=[0,4,7,10]

labels=['bad','acceptable','good']

data['group']=pd.cut(data.quality,bins,3,labels=labels)
data.head()
data['group'].value_counts()
sns.set(palette='colorblind')

sns.countplot(x='group',data=data)

plt.title('Group frequencies')

plt.xlabel('Quality group')

plt.ylabel('Frequency')

plt.show()
X = data.iloc[:,:-2].values

y = data.iloc[:,-1].values
y_le = LabelEncoder()

y = y_le.fit_transform(y)
pca = PCA(n_components=8)

x_new = pca.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(x_new,y,test_size=0.2,random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
classifier = SVC(kernel='linear')

classifier.fit(X_train,y_train)

knn_pred=classifier.predict(X_test)

print(confusion_matrix(y_test,knn_pred))

print(accuracy_score(y_test,knn_pred))
classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

knn_pred=classifier.predict(X_test)

print(confusion_matrix(y_test,knn_pred))

print(accuracy_score(y_test,knn_pred))
classifier = LogisticRegression()

classifier.fit(X_train,y_train)

lr_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,lr_pred))

print(accuracy_score(y_test,lr_pred))