import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('../input/skin.csv')

data = np.random.permutation(data)



X = data[:10000,:-1]

Y = data[:10000,-1]
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression



skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=33)

logreg = LogisticRegression(solver='newton-cg')
from mpl_toolkits import mplot3d

fold_num = 0



accuracy_all = []

for train_index, test_index in skf.split(X, Y):

    fold_num +=1

    print("This is fold number:",fold_num)

    #split data

    X_train,X_test = X[train_index,:],X[test_index,:]

    Y_train,Y_test = Y[train_index],Y[test_index]

    

    #train data with training set

    logreg.fit(X_train,Y_train)

    print('The weight vectors are:',logreg.intercept_,logreg.coef_)

    fig = plt.figure()

    ax = plt.axes(projection = '3d' )

    ax.scatter3D(X_train[:,0], X_train[:,1], X_train[:,2], c=Y_train, cmap='winter',label='training data');

    x1p,x2p = np.meshgrid(np.linspace(0,300,30),np.linspace(0,300,30))

    x3p = -logreg.intercept_/logreg.coef_[0,2]- (logreg.coef_[0,0]/logreg.coef_[0,2])*x1p - (logreg.coef_[0,1]/logreg.coef_[0,2])*x2p

    ax.plot_surface(x1p,x2p,x3p,label='decision boundary')

    plt.xlabel('Intensity of Blue'); plt.ylabel('Intesity of Green'); ax.set_zlabel('Intesity of Red');

    #plt.legend()

    plt.title('Training model with train data')

    plt.show()

    

    

    #predict on test data

    Y_predict= logreg.predict(X_test)

    correct_predict = sum(Y_predict==Y_test)

    accuracy = 100*correct_predict/len(Y_test)

    fig = plt.figure()

    ax = plt.axes(projection = '3d' )

    ax.scatter3D(X_test[:,0], X_test[:,1], X_test[:,2], c=Y_test, cmap='winter',label='test data');

    ax.plot_surface(x1p,x2p,x3p,label='decision boundary')

    plt.xlabel('Intensity of Blue'); plt.ylabel('Intesity of Green'); ax.set_zlabel('Intesity of Red');

    #plt.legend()

    plt.title('Testing model with test data with accuracy:'+ str(accuracy))

    plt.show()

    print("Fold ",fold_num,"accuracy is:",accuracy)

    accuracy_all.append(accuracy)

print("Average model accuracy for",skf.get_n_splits(),"fold CV is:",np.mean(accuracy_all))

from sklearn.model_selection import cross_val_score

acc = cross_val_score(logreg,X,Y,cv=5,scoring='accuracy')

print("Individual fold accuracy:",acc)

print("Average accuracy:",np.mean(acc))