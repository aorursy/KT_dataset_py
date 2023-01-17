from IPython.display import Image

import os

!ls ../input/



Image("../input/charts1/MachL.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#K nearest neighbour algorithm

#First we are going to observe our dataset,

data=pd.read_csv("../input/classification/data.csv")

#drop unnecessary columns from our dataset

data.drop(['id','Unnamed: 32'],axis=1,inplace=True)

data.head()

#As it can be seen from the dataset in diagnosis column we have two types of label,

#melignant =M -->Bad Tumor Type

#benign =B -->Good Tumor Type



M=data[ data['diagnosis']=='M' ]

B=data[ data['diagnosis']=='B' ]



#scatter plot of radius mean-texture_means



plt.scatter( M.radius_mean,M.texture_mean,color="red",label="bad" )

plt.scatter( B.radius_mean,B.texture_mean,color='green',label='Good')



plt.xlabel('radius mean')

plt.ylabel('texture mean')

plt.legend()

plt.show()
#we should convet our data type to integer,so let's make list comprehension



data.diagnosis=[ 1 if each=='M' else 0 for each in data.diagnosis ]



#dependent variable is diagnosis column

y=data.diagnosis.values

#and the rest of the data is called independents,x s ,features that

#affects the dependent variable y,diagnosis

x_data=data.drop(['diagnosis'],axis=1) #features



#normalization

#we should make normalization because of the anormal differences between 

#our values in the columns of the dataset



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train and test splitting

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)



#knn model,import the required library

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier( n_neighbors=8 ) #n_neighbors=k

knn.fit( x_train,y_train ) #train our model

prediction=knn.predict(x_test) #test our model



print("{} nn score: {}".format(8,knn.score(x_test,y_test))) #accuracy
#find k value



score_list=[]



#let's try different numbers of n_neighbors and see the 

#changing results according to it,here we'll try

#the numbers between 1-15,and append the results 

#to the array,finally we'll plot it to see better



for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    current_score=knn2.score( x_test,y_test )

    score_list.append( current_score )

plt.plot( range(1,15),score_list )

plt.xlabel("k values")

plt.ylabel("accuracy")



#when  knn has the highest value k takes the value of 8
#Here I can say that if we choose 4 it'll give us best 

#result,since it hast the highest accuracy value in graph



#our previous accuracy is nearly 0.96 means 96%

#now I'm gonna change it to the 4 and we'll see better 

#accuracy results



knn3=KNeighborsClassifier(n_neighbors=4)

knn3.fit(x_train,y_train)

current_score=knn3.score( x_test,y_test )



current_score

#see it's better,so try and find the best!

#again train and test splitting

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#by chaning the size of test and size data we can change accuracy val.

from sklearn.svm import SVC

from warnings import simplefilter



# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



svm=SVC( random_state=1 )

svm.fit( x_train,y_train )



print('accuracy of svm algorithm: ',svm.score( x_test,y_test ))

from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)



print('accuracy of naive bayes algorithm: ',svm.score( x_test,y_test ))
from IPython.display import Image

import os

!ls ../input/



Image("../input/charts1/chart.png")
# Load libraries

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# load dataset

pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

pima['Pregnancies']=pima['Pregnancies'].astype('float')

pima.head()

#split dataset in features and target variable

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']



y = pima.Outcome # Target variable

X = pima.drop(['Outcome'],axis=1) # Features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from IPython.display import Image

import os

!ls ../input/



Image("../input/charts1/chart2.png")
from sklearn.tree     import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix





bc = load_breast_cancer()

X = bc.data

y = bc.target



# Create our test/train split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)





## build our models

decision_tree = DecisionTreeClassifier()

random_forest = RandomForestClassifier(n_estimators=100)



## Train the classifiers

decision_tree.fit(X_train, y_train)

random_forest.fit(X_train, y_train)



# Create Predictions

dt_pred = decision_tree.predict(X_test)

rf_pred = random_forest.predict(X_test)



# Check the performance of each model

print('Decision Tree Model')

print(classification_report(y_test, dt_pred, target_names=bc.target_names))



print('Random Forest Model')

print(classification_report(y_test, rf_pred, target_names=bc.target_names))



#Graph our confusion matrix



dt_cm = confusion_matrix(y_test, dt_pred)

rf_cm = confusion_matrix(y_test, rf_pred)
data=pd.read_csv('../input/classification/data.csv')



data.drop(['id','Unnamed: 32'],axis=1,inplace=True)



#diagnosis type cannot be object it must be categorical or integer 

#convert them into integer with list comprehension



data.diagnosis=[ 1 if each=='M' else 0 for each in data.diagnosis ]



y=data.diagnosis.values

x_data=data.drop(['diagnosis'],axis=1) #features



#normalization



import numpy as np

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train and test data splitting



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier( n_estimators=100,random_state=1 )

rf.fit(x_train,y_train)



print("Random Forest Classification score: ",rf.score(x_test,y_test))

#estimator how many trees inside of it

#which subsample will you use every time random_state indicates



y_pred=rf.predict(x_test)

y_true=y_test
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)

plt.xlabel('y_pred')

plt.ylabel('y_true')

plt.show()
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



#create dataset using gaussian variable

#class 1

x1=np.random.normal(25,5,1000) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y1=np.random.normal(25,5,1000)



#create dataset using gaussian variable

#class 2

x2=np.random.normal(55,5,1000) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y2=np.random.normal(60,5,1000)



#create dataset using gaussian variable

#class 3

x3=np.random.normal(55,5,1000) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y3=np.random.normal(15,5,1000)



x=np.concatenate((x1,x2,x3),axis=0) #yukardan aşağı birleştirdik 3000 tane değer elde ettik

y=np.concatenate((y1,y2,y3),axis=0) #yukardan aşağı birleştirdik 3000 tane değer elde ettik

dictionary={"x":x,"y":y}



data=pd.DataFrame(dictionary)



plt.scatter(x1,y1,color='black') #we give color black to all cause it will be unsupervised learning

plt.scatter(x2,y2,color='black') #implementation,remove color='black to see classification'

plt.scatter(x3,y3,color='black')

plt.show()
data.head()#concatenated data
from sklearn.cluster import KMeans

wcss=[]

#to find the optimum value of k we try all k values in for loop

#according to the elbow rule we'll decide the k value

for k  in range(1,15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

    

plt.plot( range(1,15),wcss )

plt.xlabel('number of k(cluster value)')

plt.ylabel('wcss')

plt.show()

#most optimum k value is 3
#so lest's choose k=3 and see the model



kmeans2=KMeans(n_clusters=3)

clusters=kmeans2.fit_predict(data)



print(clusters[:20])

#print(clusters[:50])

# we have in the labels 0,1 and 2 's iside of them

#it assigned some labels to each group of data inside of it



data["label"]=clusters #clusterları dataya ekliyoruz

#ekledim clusterları görsellestire

plt.scatter( data.x[data.label==0],data.y[data.label==0],color='red')

plt.scatter( data.x[data.label==1],data.y[data.label==1],color='blue')

plt.scatter( data.x[data.label==2],data.y[data.label==2],color='green')

#see successfully classified data,they are differentited from each other

#and let's see that centroids



#kmeans2.cluster_centers_ is a two dimensional array

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color='yellow')
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



#create dataset using gaussian variable

#class 1

x1=np.random.normal(25,5,100) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y1=np.random.normal(25,5,100)



#create dataset using gaussian variable

#class 2

x2=np.random.normal(55,5,100) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y2=np.random.normal(60,5,100)



#create dataset using gaussian variable

#class 3

x3=np.random.normal(55,5,100) #avg=25,sigma=5,total points=1000 (25-30 arasında 1000 tane değer)

y3=np.random.normal(15,5,100)



x=np.concatenate((x1,x2,x3),axis=0) #yukardan aşağı birleştirdik 3000 tane değer elde ettik

y=np.concatenate((y1,y2,y3),axis=0) #yukardan aşağı birleştirdik 3000 tane değer elde ettik

dictionary={"x":x,"y":y}



data=pd.DataFrame(dictionary)



plt.scatter(x1,y1) #we give color black to all cause it will be unsupervised learning

plt.scatter(x2,y2) #implementation,remove color='black to see classification'

plt.scatter(x3,y3)

plt.show()
#we'll draw dendogram



from scipy.cluster.hierarchy import dendrogram, linkage



merg=linkage( data,method='ward') #clusterların içindeki varianceları küçültür,yayılımları minimize eder

dendrogram(merg,leaf_rotation=90)

plt.xlabel('data points')

plt.ylabel('euclidian distance')

plt.show()
from sklearn.datasets import load_iris



iris=load_iris()

#convert it to dat frame

data=iris.data # numpy array

feature_names=iris.feature_names

y=iris.target



df=pd.DataFrame(data,columns=feature_names)

df['class']=y #0-1-2 sınıflarımız var



x=data

df.head()

from sklearn.decomposition import PCA

#datamızın featurelarını azaltmaya çalışıyoruz

#reduce features into 2,normalize=whitten

pca=PCA( n_components=2,whiten=True )

pca.fit(x)



#boyutu düsürcek modeli ettik,matemaksiksel hesaplamaları yaptık

x_pca=pca.transform(x)

#uygulamak için trnsform etmeliyiz

print('variance ratio: ',pca.explained_variance_ratio_)

print('sum: ',sum(pca.explained_variance_ratio_))

# %97(sum) sini datanın hala kaybetmedik
#pca ile 2d görselleştirme yapacağız

df['p1']=x_pca[:,0]

df['p2']=x_pca[:,1]

#p1 ve p2 bizim reduction sonucunda elde ettiğimiz featurelar 

#bunları dataframe e ekliyoruz



color=["red","green","blue"]



import matplotlib.pyplot as plt



for each in range(3):

    plt.scatter(df[ df['class']==each ].p1,df[ df['class']==each ].p2,color=color[each],label=iris.target_names[each] )

plt.legend()

plt.show()

#versicolor ve virginica arasında biraz karışma var ama yinede iyi şekilde

#birbirlerinden ayrılmışalr,featurelar azaltınca veri kaybı yaşamışız anlamına gelir
from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score

import pandas as pd

import numpy as np



iris=load_iris()

#convert it to dat frame

x=iris.data

y=iris.target

#normalization

x=( x-np.min(x))/(np.max(x)-np.min(x))

#train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)



#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

#en yakın 3 komşuya bakıyoruz

#3 tane accuracy değeri buluyoruz

accuracies=cross_val_score( estimator=knn,X=x_train, y=y_train,cv=10 )

#train datamızı 10 a böldük her seferinde ir kaçını train diğerlerini validation olarak kullandık

print('Accuracy values are: ',accuracies)



print('average accuracy: ',np.mean(accuracies))

print('average std: ',np.std(accuracies))



knn.fit(x_train,y_train)

print('test accuracy: ',knn.score(x_test,y_test))
#grid search cross validation

from sklearn.model_selection import GridSearchCV

#grid in içine tune etmek istediğimiz parametreyi yazıyoruz

grid={'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

#öncesinde n_neighbors u elimizle seçiyorduk

#ama şimdi GridSearchCV ile optimum değeri bulduruyoruz

#daha sonra knn ye atayıp knn_cv değerini belirlemek

#için kullanıyoruz 

knn_cv=GridSearchCV( knn,grid,cv=10 )

knn_cv.fit(x,y)



print("tuned hyperprarameter K:",knn_cv.best_params_)

print("the best accuracy score according to \nthe tuned parameter: ",knn_cv.best_score_)
from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



from sklearn.linear_model import LogisticRegression





#grid search cv with logistic regression

x=x[:100,:]

y=y[:100]





#C parametresi regularization parametresi dir.Fazla yüksek

#olursa overfit olur model datayı ezberler,çok düşük olursa underfit

#olur ondada model datayı iyi öğrenemez

#l1 ve l2 loss functionlardır lasso ve ridge

grid={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x,y)

print('accuracy',logreg_cv.best_score_)
#let us separate them x_train and y_train in fit()

x=x[:100,:]

y=y[:100]



#normalization

x=( x-np.min(x))/(np.max(x)-np.min(x))

#train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)



grid={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)

print('accuracy',logreg_cv.best_score_)



#bu değerlerden yeni bir log_reg modeli oluştur



logreg2=LogisticRegression()

logreg2.fit(x_train,y_train)

print('score2: ',logreg2.score(x_test,y_test))


