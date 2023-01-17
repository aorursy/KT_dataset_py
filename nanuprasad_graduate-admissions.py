import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

import warnings

warnings.filterwarnings('ignore') 
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.head()
data.rename(columns={'Serial No.':'Srno','GRE Score':'GRE','TOEFL Score':'TOEFL','University Rating':'UnivRating'

                     ,'Chance of Admit ':'Chance'},inplace=True)

data.columns
data.drop(columns='Srno', inplace=True)

data.head()
data.describe()
plt.figure(figsize=(8,2))

sns.boxplot(x=data['GRE'])
plt.figure(figsize=(8,2))

sns.boxplot(x=data['TOEFL'])
plt.figure(figsize=(10,8))

sns.heatmap(data=data.corr(),annot=True,cmap='mako',vmin=-1,vmax=1,linewidths=1,linecolor='k')
corr = data.corr()['Chance'].sort_values(ascending=False)

print(corr)
data[['GRE','TOEFL','UnivRating','CGPA']].hist(figsize=(10,8),bins=10,color='g',edgecolor='g')

plt.tight_layout()

plt.show()
print('Mean CGPA Score is :', int(data.CGPA.mean()))

print('Mean TOEFL Score is :', int(data.TOEFL.mean()))

print('Mean GRE Score is :', int(data.GRE.mean()))

print('Mean UnivRating Score is :', int(data.UnivRating.mean()))
sns.countplot(x=data['Research'])

plt.title('Students Research')
plt.figure(figsize=(8,6))

data['Research'].value_counts().plot.pie(shadow=True, explode=[0,0.1],autopct='%1.1f%%')

plt.show()
sns.scatterplot(x='GRE', y='TOEFL', data= data, hue='Research')

plt.show()
def modify(row):

    if row['Chance'] >0.7 :

        return 1

    else :

        return 0

data['Admit'] = data.apply(modify,axis=1)

sns.scatterplot(data=data,x='GRE',y='TOEFL',hue='Admit')

plt.show()

sns.factorplot('Research','Admit',data=data)

plt.show()
data[data['Chance']>0.9].mean().reset_index()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot("Research","GRE",hue="Admit", data=data,split=True)

plt.subplot(2,2,2)

sns.violinplot("Research","TOEFL",hue="Admit", data=data,split=True)

plt.subplot(2,2,3)

sns.violinplot("Research","CGPA",hue="Admit", data=data,split=True)

plt.subplot(2,2,4)

sns.violinplot("Research","UnivRating",hue="Admit", data=data,split=True)

plt.show()
f,ax = plt.subplots(1,2,figsize=(10,6))

data['Admit'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Admitted to University')

sns.countplot(x='Admit',data=data)

ax[1].set_title('Admitted to University')

plt.show()
from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error #for confusion matrix
data.head()
x = data.iloc[:,:-2].values

y = data.iloc[:,-2].values
x[0]
y[0]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)
from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
y_test
y_pred
data.head()
x=data.iloc[:,[0,5]].values    # O represents GRE Score and 5 represnts CGPA 

y=data.iloc[:,8].values        # 8 tells us if the Candidate got Admission or not 
from sklearn.model_selection import train_test_split   

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.preprocessing import StandardScaler 

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(x_train, y_train)

ylog_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,ylog_pred)

cm
sns.heatmap(cm,annot=True)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_train,y_train

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Predicting University Admission')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_test,y_test

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Predicting University Admission')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
from sklearn.neighbors import KNeighborsClassifier

kneigh_classifier = KNeighborsClassifier(p=2)

kneigh_classifier.fit(x_train,y_train)

ykn_pred = kneigh_classifier.predict(x_test)
cm = confusion_matrix(y_test,ykn_pred)

cm
sns.heatmap(cm,annot=True)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_train,y_train

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,kneigh_classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN (Training set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_test,y_test

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,kneigh_classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN (Test set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
x = data.iloc[:,[0,5]].values
from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)

y_kmeans=kmeans.fit_predict(x)
plt.figure(figsize=(8,6))

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Excellent') 

plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='greenyellow',label='Must Improve')  

plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='yellow',label='Good')   

plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='green',label='Outstanding')  #cyan

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='magenta',label='Centroids')

plt.title('Cluster of Students')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show
x = data.iloc[:,[0,5]]

y = data.iloc[:, 8]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
ynb_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, ynb_pred)

cm
sns.heatmap(cm, annot=True)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_train,y_train

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Naive Bayes (Training set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_test,y_test

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Naive Bayes (Training set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
x = data.iloc[:,[0,5]]

y = data.iloc[:, 8]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.svm import SVC

classifier = SVC(kernel='rbf',gamma=0.3, random_state=0)

classifier.fit(x_train,y_train)
ysvm_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, ysvm_pred)

cm
sns.heatmap(cm, annot=True)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_train,y_train

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Kernel SVM (Training set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

x_set,y_set=x_test,y_test

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),

            alpha=0.75,cmap=ListedColormap(('yellow','green')))

plt.xlim(x1.min(),x1.max())

plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Kernel SVM (Training set)')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.legend()

plt.show()
data.head()
x = data.iloc[:,0:7].values

y = data.iloc[:, 8].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
import keras

from keras.models import Sequential

from keras.layers import Dense
#Initializing the ANN

classifier = Sequential()
#adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 7))
#adding the second layer

classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
#adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.7)

y_pred
cm = confusion_matrix(y_test, y_pred)

cm
sns.heatmap(cm,annot=True)