import pickle

ofname = open ('../input/qwerty/dataset_small.pkl','rb')

# x stores input data and y target values

(x,y) = pickle.load(ofname,encoding = 'iso8859')

dims = x.shape[1]

N = x.shape[0]

print ('dims: ' + str(dims) + ', samples: ' + str(N))
from sklearn import neighbors

from sklearn import datasets

# Create an instance of K-nearest neighbor classifier

knn = neighbors .KNeighborsClassifier( n_neighbors = 11)

# Train the classifier

knn.fit(x, y)

# Compute the prediction according to the model

yhat = knn.predict(x)

# Check the result on the last example

print ('Predicted value: ' + str(yhat[ -1]),

', real target: ' + str(y[-1]))
knn.score(x,y)
import numpy as np

import matplotlib.pyplot as plt

plt.pie(np.c_[np.sum(np.where(y == 1, 1, 0)),

    np.sum(np.where(y == -1, 1, 0))][0],

    labels = ['Not fully funded','Full amount'],

    colors = ['r','g'],shadow = False,

    autopct = '%.2f')

plt.gcf().set_size_inches((7, 7))
import numpy as np

from sklearn.neighbors import KNeighborsClassifier



yhat = knn.predict(x)

TP = np.sum(np.logical_and( yhat == -1, y == -1))

TN = np.sum(np.logical_and( yhat == 1, y == 1))

FP = np.sum(np.logical_and( yhat == -1, y == 1))

FN = np.sum(np.logical_and( yhat == 1, y == -1))

print ('TP: '+ str(TP), ', FP: '+ str(FP))

print ('FN: '+ str(FN), ', TN: '+ str(TN))
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

yhat = knn.predict(x)

metrics.confusion_matrix(yhat, y)

# sklearn uses a transposed convention for the confusion

# matrix thus I change targets and predictions
# Train a classifier using .fit()

knn = neighbors .KNeighborsClassifier( n_neighbors = 1)

knn.fit(x, y)

yhat = knn.predict(x)

print ('classification accuracy:' + str(metrics. accuracy_score(yhat , y)))

print ('confusion matrix: \n' + str(metrics. confusion_matrix(yhat , y)))
# Simulate a real case: Randomize and split data into

# two subsets PRC*100\% for training and the rest

# (1-PRC)*100\% for testing

perm = np.random. permutation(y. size)

PRC = 0.7

split_point = int(np.ceil(y.shape[0]*PRC))



X_train = x[perm[: split_point]. ravel() ,:]

y_train = y[perm[: split_point]. ravel()]



X_test = x[perm[ split_point:]. ravel() ,:]

y_test = y[perm[ split_point:]. ravel()]



#Train a classifier on training data

knn = neighbors .KNeighborsClassifier( n_neighbors = 1)

knn.fit(X_train , y_train)

yhat = knn.predict(X_train)

print ('\n TRAINING STATS:')

print ('classification accuracy:' +

str(metrics. accuracy_score(yhat , y_train)))

print ('confusion matrix: \n' +

str(metrics. confusion_matrix( y_train , yhat)))
#Check on the test set

yhat = knn. predict(X_test)

print ('TESTING STATS:')

print ('classification accuracy:', metrics. accuracy_score(yhat , y_test))

print ('confusion matrix: \n'+ str(metrics. confusion_matrix(yhat , y_test)))
# Spitting done by using the tools provided by sklearn:

from sklearn.model_selection import train_test_split



PRC = 0.3

acc = np.zeros((10 ,))

for i in range (10):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = PRC)

    knn = neighbors .KNeighborsClassifier( n_neighbors = 1)

    knn.fit(X_train , y_train)

    yhat = knn.predict(X_test)

    acc[i] = metrics.accuracy_score(yhat , y_test)

acc.shape = (1, 10)

print ('Mean expected error:' + str(np.mean(acc[0])))
from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn import tree

from sklearn import svm

from sklearn import metrics



PRC = 0.1

acc_r=np.zeros((10,4))

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=PRC)

    nn1 = neighbors.KNeighborsClassifier(n_neighbors=1)

    nn3 = neighbors.KNeighborsClassifier(n_neighbors=3)

    svc = svm.SVC(gamma='scale')

    dt = tree.DecisionTreeClassifier()

    

    nn1.fit(X_train,y_train)

    nn3.fit(X_train,y_train)

    svc.fit(X_train,y_train)

    dt.fit(X_train,y_train)

    

    yhat_nn1=nn1.predict(X_test)

    yhat_nn3=nn3.predict(X_test)

    yhat_svc=svc.predict(X_test)

    yhat_dt=dt.predict(X_test)

    

    acc_r[i][0] = metrics.accuracy_score(yhat_nn1, y_test)

    acc_r[i][1] = metrics.accuracy_score(yhat_nn3, y_test)

    acc_r[i][2] = metrics.accuracy_score(yhat_svc, y_test)

    acc_r[i][3] = metrics.accuracy_score(yhat_dt, y_test)





plt.boxplot(acc_r);

for i in range(4):

    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1

    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)

    

ax = plt.gca()

ax.set_xticklabels(['1-NN','3-NN','SVM','Decission Tree'])

plt.ylabel('Accuracy')

plt.savefig("error_ms_1.png",dpi=300, bbox_inches='tight')
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

# Create a 10-fold cross - validation set

kf =cross_validation.KFold(n = y.shape[0],n_folds = 10,shuffle = True ,random_state = 0)



print(rf_cv_score)

# Search for the parameter among the following:

C = np.arange(2, 20,)

acc = np.zeros ((10, 18))

i=0

for train_index , val_index in kf:

    X_train , X_val = X[ train_index], X[val_index]

    y_train , y_val = y[ train_index], y[val_index]

    j=0

    for c in C :

        dt = tree. DecisionTreeClassifier(min_samples_leaf = 1,max_depth = c)

        dt.fit(X_train , y_train)

        yhat = dt. predict(X_val)

        acc [i ][ j ] = metrics. accuracy_score(yhat , y_val)

        j=j+1

    i=i+1
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



# Train_test split

#X_train , X_test , y_train , y_test = cross_validation.train_test_split(X, y, test_size = 0.20)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

# Create a 10-fold cross - validation set

kf = cross_validation. KFold(n = y_train.shape[0],n_folds = 10,shuffle = True ,random_state = 0)

# Search the parameter among the following

C = np.arange(2, 20,)

acc = np.zeros ((10, 18))

i=0

for train_index , val_index in kf:

    X_t , X_val = X_train[ train_index], X_train[val_index]

    y_t , y_val = y_train[ train_index], y_train[val_index]

    j=0

    for c in C :

        dt = tree. DecisionTreeClassifier(min_samples_leaf = 1,max_depth = c)

        dt.fit(X_t , y_t)

        yhat = dt. predict(X_val)

        acc [i ][ j ] = metrics. accuracy_score(yhat , y_val)

        j=j+1

    i=i+1

print ('Mean accuracy: ' + str(np.mean(acc , axis = 0)))

print ('Selected model index: ' + str(np.argmax(np.mean(acc , axis = 0))))
%reset -f

%matplotlib inline

import pickle

ofname = open('../input/qwerty/dataset_small.pkl','rb') 

(X,y) = pickle.load(ofname,encoding = 'iso8859')

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import tree



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

#Train_test split

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20, random_state=42)



#Create a 10-fold cross validation set

kf=cross_validation.KFold(n=y_train.shape[0], n_folds=10, shuffle=True, random_state=0)     

#Search the parameter among the following

C=np.arange(2,20)

acc = np.zeros((10,18))

i=0

for train_index, val_index in kf:

    X_t, X_val = X_train[train_index], X_train[val_index]

    y_t, y_val = y_train[train_index], y_train[val_index]

    j=0

    for c in C:

        dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=c)

        dt.fit(X_t,y_t)

        yhat = dt.predict(X_val)

        acc[i][j] = metrics.accuracy_score(yhat, y_val)

        j=j+1

    i=i+1



print ('Mean accuracy: ' + str(np.mean(acc,axis = 0)))

print ('Selected model index: ' + str(np.argmax(np.mean(acc,axis = 0))))

print ('Complexity: ' + str(C[np.argmax(np.mean(acc,axis = 0))]))

# Train the model with the complete training set with the selected complexity

C=np.arange(2,20)

acc = np.zeros((10,18))

dt = tree. DecisionTreeClassifier(

    min_samples_leaf = 1,

    max_depth = C[np.argmax(np. mean(acc , axis = 0))])

dt.fit(X_train ,y_train)

# Test the model with the test set

yhat = dt. predict(X_test)

print ('Test accuracy: ' + str(metrics. accuracy_score(yhat , y_test)))
# Train the final model

dt = tree. DecisionTreeClassifier(min_samples_leaf = 1,max_depth = C[np.argmax(np. mean(acc , axis = 0))])

dt.fit(x, y)
%reset -f



import pickle

ofname = open('../input/qwerty/dataset_small.pkl','rb') 

(X,y) = pickle.load(ofname,encoding = 'iso8859')

from sklearn.model_selection import learning_curve, GridSearchCV

import numpy as np



from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn import linear_model



from sklearn import metrics



parameters = {'C':[1e4,1e5,1e6],'gamma':[1e-5,1e-4,1e-3]}



N_folds = 5



kf=cross_validation.KFold(n=y.shape[0], n_folds=N_folds,  shuffle=True, random_state=0)



acc = np.zeros((N_folds,))

i=0

#We will build the predicted y from the partial predictions on the test of each of the folds

yhat = y.copy()

for train_index, test_index in kf:

    X_train, X_test = X[train_index,:], X[test_index,:]

    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    clf = svm.SVC(kernel='rbf')

    clf = grid_search.GridSearchCV(clf, parameters, cv = 3) #This line does a cross-validation on the 

    clf.fit(X_train,y_train.ravel())

    X_test = scaler.transform(X_test)

    yhat[test_index] = clf.predict(X_test)

    

print (metrics.accuracy_score(yhat, y))

print (metrics.confusion_matrix(yhat, y))