# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading data from sklearn library

from sklearn import datasets



df = datasets.load_iris()

X = df.data[:, [0, 1]]

y = df.target
#function to plot decision regions

from matplotlib.colors import ListedColormap



def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):

    

    # Initialise the marker types and colors

    markers = ('s','x','o','^','v')

    colors = ('red','blue','lightgreen','gray','cyan')

    color_Map = ListedColormap(colors[:len(np.unique(y))]) #we take the color mapping correspoding to the 

                                                            #amount of classes in the target data

    

    # Parameters for the graph and decision surface

    x1_min = X[:,0].min() - 1

    x1_max = X[:,0].max() + 1

    x2_min = X[:,1].min() - 1

    x2_max = X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),

                           np.arange(x2_min,x2_max,resolution))

    

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    plt.contour(xx1,xx2,Z,alpha=0.4,cmap = color_Map)

    plt.xlim(xx1.min(),xx1.max())

    plt.ylim(xx2.min(),xx2.max())

    

    # Plot samples

    X_test, Y_test = X[test_idx,:], y[test_idx]

    

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],

                    alpha = 0.8, c = color_Map(idx),

                    marker = markers[idx], label = cl

                   )
#Split data into training and test datasets (training will be based on 70% of data)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

#test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion

print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))



#Scaling data

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



#X_train_std and X_test_std are the scaled datasets to be used in algorithms
#To be used later for plotting

X_combined_standard = np.vstack((X_train_std, X_test_std))      #vstack stacks arrays vertically

Y_combined = np.hstack((y_train, y_test))                       #hstack stacks arrays horizontally
#SVM with linear Kernel

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score





C_range = [0.01, 0.1, 1, 10, 100]                                   #Set range for C

svm_linear_acc_table = pd.DataFrame(columns=['C', 'Accuracy'])      #Create table to store C and respective accuracy

svm_linear_acc_table['C'] = C_range                                 #First column of table storing values for C



plt.figure(figsize=(20, 20))

plt.show()

j = 0



for i in C_range:

    svm_linear = SVC(kernel = 'linear', C = i, random_state = 0)     #Initialize SVM model with linear kernel

    svm_linear.fit(X_train_std, y_train)                                #Train SVM model

    y_pred = svm_linear.predict(X_test_std)                             #Predict test values

    svm_linear_acc_table.iloc[j, 1] = accuracy_score(y_test, y_pred)    #store accuracy into accuracy table

    j += 1

    

    #plotting

    plt.subplot(3, 2, j)

    plot_decision_regions(X = X_combined_standard,

                          y = Y_combined,

                          classifier = svm_linear, 

                          test_idx = range(105, 150))

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.title('Linear Kernel using C = %s' %i)

    

print(svm_linear_acc_table)
#SVM with polynomial Kernel

degree_range = [1, 2, 3, 4, 5, 6]       #set range for degree



svm_poly_acc_table = pd.DataFrame(columns=['Degree', 'Accuracy'])      #Create table to store C and respective accuracy

svm_poly_acc_table['Degree'] = degree_range                                 #First column of table storing values for C



plt.figure(figsize=(20, 20))

plt.show()

j = 0



for i in degree_range:

    svm_poly = SVC(kernel = 'poly', degree = i, C = 1, random_state = 0)      #Initialize SVM model with linear kernel

    svm_poly.fit(X_train_std, y_train)                                        #Train SVM model

    y_pred = svm_poly.predict(X_test_std)                                     #Predict test values

    svm_poly_acc_table.iloc[j, 1] = accuracy_score(y_test, y_pred)            #store accuracy into accuracy table

    j += 1

    

    #plotting

    plt.subplot(3, 2, j)

    plot_decision_regions(X = X_combined_standard,

                          y = Y_combined,

                          classifier = svm_poly, 

                          test_idx = range(105, 150))

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.title('Polynomail Kernel using degree = %s' %i)

    



print(svm_poly_acc_table)