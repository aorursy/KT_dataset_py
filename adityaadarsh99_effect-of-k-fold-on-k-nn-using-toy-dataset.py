#importing lib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # K-NN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
def simple_knn(X,y):
    #splitting text and train dataset
    x_train,x_text,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=1)

    #simple K-NN
    acc=[]
    for i in range(1,30,2):
        clf=KNeighborsClassifier()
        k=[1,5,15,30]
        clf=KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_text)
        acc.append(accuracy_score(y_test,y_pred))

    # comparison of accuracy on different k-values in knn
    print('********** simple knn and k-value comparison **************\n')
    xy=zip(range(1,30,2), np.round(acc,3))
    comp=pd.DataFrame(list(xy),columns=['k-value','acc'])
    print(comp)
    k=comp[comp['acc']==comp['acc'].max()]['k-value'].values  #best value of k
    acc=comp[comp['acc']==comp['acc'].max()]['acc'].values    #acc at best value of k
    print(f'best k-value is {k} with accuracy {acc}')

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def validation_knn(X,y):
    #spliting dataset
    x1,x_text,y1,y_test=train_test_split(X,y,test_size=0.33, random_state=1) #training dataset
    x_train,x_cv,y_train,y_cv=train_test_split(x1,y1,test_size=0.33, random_state=1) #validation dataset

    # using validation dataset to find the optimum value of k in k-nn
    acc=[]
    for i in range(1,15,2):
        k=i
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train,y_train)
        y_cv_pred=clf.predict(x_cv)
        accuacy=accuracy_score(y_cv,y_cv_pred)* float(100)
        acc.append(round(accuacy,3))
    k=acc.index(max(acc))+1
    print("\n************** using validation to find best k ******************")
    print('best accuracy at k-value =',k)

    # using optimum value of k to find the generalization accuracy
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(x1,y1)
    y_pred=clf.predict(x_text)
    accuacy=accuracy_score(y_test,y_pred)
    print(f'generalization accuracy using best k-vlaue-->{k} =',accuacy)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

def kfold_knn(X,y):

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    # using k-fold to find best k-value in KNN
    acc=[]
    for k in range(1,15,2):
        clf=KNeighborsClassifier(n_neighbors=k)
        cv_scores=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
        acc.append(cv_scores.mean()*float(100))

    max_acc=acc[acc.index(max(acc))] # maximum accuarcy
    k=acc.index(max(acc))+1 # best k-value
    print("\n**************************k-fold knn **********************************")
    print('**************** using k-fold to find best K-value **********************')
    print(f'best accuracy is {max_acc} on cv datatset using 10 fold at k-value {k}')

    #using best k-value to find generalistion value
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f'genearalisation accuracy on best k-value at k = {k} is accuacy = {accuracy}')

data=pd.read_csv("../input/1.ushape.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/2.concerticcir1.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/3.concertriccir2.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/4.linearsep.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/5.outlier.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/4.linearsep.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/6.overlap.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/7.xor.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/8.twospirals.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)
data=pd.read_csv("../input/9.random.csv",header=None,index_col=None)
X=data.iloc[:,:2].values
y=data.iloc[:,2].values
simple_knn(X,y)
validation_knn(X,y)
kfold_knn(X,y)