from sklearn import datasets  ## import iris dataset from the sklearn package
import pandas as pd     ## import pandas dataframe

import matplotlib.pyplot as plt
iris = datasets.load_iris()
type(iris) ## load as a numpy array
print ("Type of the data :",type(iris['data']))
print (iris.keys())  ## the key value of the dataset dict
print (iris.DESCR)  ## dataset Description
%matplotlib inline
import seaborn as sns   ## import seaborn for advance plotting
iris = sns.load_dataset('iris')  ## import data as iris dataframe
correlation = iris.corr()  # find the pearson coefficient
correlation
sns.heatmap(iris.corr(),cmap='coolwarm',annot=True)
iris.head()
## setting the style first

sns.set(style="whitegrid",color_codes=True) ## change style
sns.lmplot(x='sepal_length',y='petal_length',hue='species',data=iris,markers=['o','s','D'])
## lets make a scatter plot

sns.lmplot(x='sepal_length',y='petal_length',hue='species',data=iris,fit_reg=False,markers=['o','s','D'])
sns.pairplot(iris,size=5,vars=['sepal_length','petal_length'],markers=['o','s','D'],hue="species")
sns.pairplot(iris,hue='species',markers=['o','s','D'])
sns.pairplot(iris,hue='species',kind='reg',markers=['o','s','D'])
## for boxplot

sns.boxplot(data=iris,orient='h')
## loading the iris with the sklearn.datasets

iris = datasets.load_iris()
n_samples,n_features = iris.data.shape
n_samples
n_features
print ("Shape of the data "+str(iris.data.shape))
len(iris.target)==n_samples
## loading data from sns

iris_pandas = sns.load_dataset('iris')  ## import data as iris dataframe
iris_pandas.isnull().sum()
iris_pandas.isna().sum()
iris_numpy = datasets.load_iris()



num_setosa=sum(iris_numpy.target==0)

num_versicolor=sum(iris_numpy.target==1)

num_virginica=sum(iris_numpy.target==2)

sizes=[num_setosa,num_versicolor,num_virginica]



labels = ['num_setosa','num_versicolor','num_virginica']

explode = (.1, 0.1, .1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

##load from numpy 

iris_numpy = datasets.load_iris()
## find feature names

iris_numpy.feature_names
print (iris_numpy['feature_names'])
iris_numpy.target_names
list(iris.target_names)
print ("Target names : ",iris_numpy['target_names'])
iris_numpy.data[0]  ## watch the first row of the data
## lets see the full data

iris_numpy.data
iris.target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn
## extracting target and feture column
X = iris_numpy.data

y = iris_numpy.target
knn.fit(X,y) ## train the model 
## predict data

knn.predict([[3,5,4,2]])
## print the target names

print ("Predicted target name :",iris_numpy['target_names'][knn.predict([[3,5,4,2]])])
X_new = [[3,5,4,2],[5,4,3,2]]
knn.predict(X_new)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
knn.predict(X_new)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X,y)
lr.predict(X_new)
X = iris_numpy.data

y = iris_numpy.target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.predict(X)
## first five prediction

knn.predict(X)[:5]
y_pred = knn.predict(X)
len(y_pred)
knn.score(X,y)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X,y)

y_pred = lr.predict(X)
from sklearn import metrics
print (metrics.accuracy_score(y,y_pred))
import numpy as np



np.mean(y_pred==y)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X,y)

y_pred = knn.predict(X)

print (metrics.accuracy_score(y,y_pred))
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y,y_pred)
plt.imshow(cn,cmap='gray')
sns.heatmap(cn,cmap='coolwarm',annot=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
## splitting 60-40

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4,test_size=.4)
print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
neighbors = list(range(1,31))

scores=[]



for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test,y_pred))
## plotting the accuracy

plt.plot(neighbors,scores)

plt.xlabel("K value")

plt.ylabel("Testing Accuracy")
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X,y)

knn.predict([[3,5,4,2]])
## print the target names

print ("Predicted target name :",iris_numpy['target_names'][knn.predict([[3,5,4,2]])])
%matplotlib inline
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=4,test_size=.4)
training_accuracy=[]

testing_accuracy=[]
neighbors = list(range(1,26))





for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    training_accuracy.append(knn.score(X_train,y_train))

    testing_accuracy.append(knn.score(X_test,y_test))    
plt.plot(neighbors,training_accuracy,label='training accuracy')

plt.plot(neighbors,testing_accuracy,label='testing accuracy')

plt.ylabel("Accuracy")

plt.xlabel("K value")

plt.legend()
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print (scores)

print (scores.mean())



print ("Mean Accuracy "+str(scores.mean()))
k_range = range(1,31)

k_scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')

    k_scores.append(scores.mean())

    

print (k_scores)
plt.plot(k_range,k_scores)

plt.xlabel("k range")

plt.ylabel("scores")
knn = KNeighborsClassifier(n_neighbors=20)

scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean()

print (scores)
lr = LogisticRegression()

scores = cross_val_score(lr,X,y,cv=10,scoring='accuracy').mean()

print (scores)
import sklearn

from sklearn import datasets

iris = datasets.load_iris()
X_temp = iris.data

y_temp = iris.target
X_numpy=X_temp[:,[0,1]]

y_numpy = y_temp
#iris_pandas.head()

X_temp_pandas = iris_pandas.drop('species',axis=1)

y_temp_pandas = iris_pandas['species']
X_pandas=X_temp_pandas[['sepal_length','sepal_width']]

y_pandas = y_temp_pandas
## train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_numpy,y_numpy,test_size=.25,random_state=33)
## feature scaling / stander scalar transform

from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
## find the standerd daviation

import numpy as np

print("Training set mean {:.2f} and Standard Deviation {:.2f}".format(np.average(X_train),np.std(X_train)))

print("Testing set mean {:.2f} and Standard Deviation {:.2f}".format(np.average(X_test),np.std(X_test)))



import matplotlib.pyplot as plt

color_mk = [['red','s'],['green','o'],['blue','x']]

plt.figure("Training data")
##

#for the first loop xs is the value having the first element of X_train train for y_tain=0

#for the first loop ys is the value having the second element of X_train train for y_tain=0



#for the second loop xs is the value having the first element of X_train train for y_tain=1

#for the second loop ys is the value having the second element of X_train train for y_tain=1





#for the third loop xs is the value having the first element of X_train train for y_tain=2

#for the third loop ys is the value having the second element of X_train train for y_tain=2



# it is possible to do that without loop and do it separately 





for i in range(len(color_mk)):

    

    xs = X_train[:,0][y_train==i]

    ys = X_train[:,1][y_train==i]

    plt.scatter(xs,ys,c=color_mk[i][0],marker=color_mk[i][1])

import copy

y_train_setosa = copy.copy(y_train)

y_test_setosa = copy.copy(y_test)
len(y_train_setosa)
def change(df):

    for item in range(len(df)):

        if df[item]>0:

            df[item]=1

    return df
y_train_setosa_c = change(y_train_setosa) 
y_train_setosa_c
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(loss='log',random_state=42)
clf.fit(X_train,y_train_setosa_c)
print (clf.coef_[0])
print (clf.intercept_)
print ('If the flower has 4.6 petal width and 3.2 petal length is a {}'.format(

        iris.target_names[clf.predict(scaler.transform([[4.6, 3.2]]))]))

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data
X
## picking the third feature

X = iris.data[:,3:]

X
## picking the last 2 feature

X = iris.data[:,2:]

X
y=iris.target
## train with with two feature

tree_clf = DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,y)
from sklearn.tree import export_graphviz

export_graphviz(tree_clf,out_file='tree.png',feature_names=iris.feature_names[2:],class_names=iris.target_names,rounded=True,filled=True)
from subprocess import call
call(['dot','-Tpng','tree.dot','-o','tree.png'])
from IPython.display import Image
Image(filename='tree.png')
from sklearn.datasets import load_iris

iris=load_iris()

X,y = iris.data,iris.target
X.shape
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,y)
tree_clf.feature_importances_
for name,score in zip(iris['feature_names'],tree_clf.feature_importances_):

    print (name,score)
from sklearn.multioutput import MultiOutputClassifier

def multiple_target_output(X,y,model):

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.25)

    models = MultiOutputClassifier(model)

    models.fit(x_train,y_train)

    accuracy=models.score(x_test,y_test)

    predicted = models.predict(x_test)

    return accuracy,predicted

    

    