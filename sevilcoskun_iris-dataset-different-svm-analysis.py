# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import matplotlib.pylab as plt



#Reading whole dataset

data = pd.read_csv("../input/Iris.csv")



data.columns



#drop ID column because it is not a feature

data.drop(["Id"], axis = 1, inplace = True)



data.info()
#Give Species column integer values;

#Iris-setosa = 0; Iris-versicolor = 1; Iris-virginica = 2

data.Species = [0 if each == "Iris-setosa" else 1 if each == "Iris-versicolor" else 2 for each in data.Species]   

#For analyzng data it is necessary to divide data dimensions and labels(Species)

x = data.drop(["Species"],axis=1)

#Simply select first 2 features

x = x.iloc[:,:2]

y = data.Species.values

x.head()
# Normalize X

mean = np.mean(x)

std = np.std(x)

print('Mean: ', mean, ' - Standard variance: ', std)



x_norm = (x-mean) / std

mean = np.mean(x)

std = np.std(x)

print('Normalized mean: ', mean, ' - Standard variance: ', std)



x_norm.head()

from sklearn.model_selection import train_test_split

#Randomly split your data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=1)

print("x_train: ", x_train.shape, "x_val: ", x_val.shape, "x_test: ", x_test.shape)
#For C from 10-3 to 103 (multiplying at each step by 10)

#a. Train a linear SVM on the training set.

#b. Plot the data and the decision boundaries

#c. Evaluate the method on the validation set



def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out



# I create an instance of SVM and fit out data-->model.

c = 0.001  # SVM regularization parameter

titles = []

models = []

for i in range(6):

    clf = svm.SVC(kernel='linear', C=c)

    model = clf.fit(x, y)

    models.append(model)

    title1 = "Decision Boundaries with C=" + str(c)

    titles.append(title1)

    c = c * 10    



# Set-up 2x3 grid for plotting.

fig, sub = plt.subplots(2, 3, figsize=(15,10))

plt.subplots_adjust(wspace=0.7, hspace=0.7)



#create a meshgrid to represent each data on validation set with consider max and min values on x y axises

#why validation set because we will predict our results with validaiton set

x0, x1 = x_val.iloc[:, 0], x_val.iloc[:, 1]

x_min, x_max = x0.min() - 1, x0.max() + 1

y_min, y_max = x1.min() - 1, x1.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))



for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(x0, x1, c=y_val, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlabel('Sepal length')

    ax.set_ylabel('Sepal width')

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(title)

plt.show()

accuracy = []

#c. Evaluate the method on the validation set

for clf in models:

    pred = clf.predict(x_val)

    

    #Accuracy

    accuracy1 = sum(pred==y_val) / y_val.shape[0] 

    accuracy.append(accuracy1)

    print('Accuracy val set: ', accuracy1)



fig1, ax1 = plt.subplots()

ax1.plot(accuracy, label="score")

ax1.set_xlabel("C Values")

ax1.set_ylabel("Accuracy")

ax1.set_title("Accuracy on Validation on LinearSVM")

plt.show()
#Use the best value of C and evaluate the model on the test set. How well does

#the best c = 0.1

#redo svm for test_split

svm_test = svm.SVC(kernel = 'linear', C=0.1)

svm_test.fit(x_train, y_train)

predict_test = svm_test.predict(x_test)

accuracy_test = sum(predict_test==y_test) / y_test.shape[0] 

print('Accuracy test set: ', accuracy_test)
#Repeat point 4. (train, plot, etc..), but this time use an RBF kernel

#Evaluate the best C on the test set.

#Are there any differences compared to the linear kernel? How are the boundaries different?



def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out



c = 0.001  # SVM regularization parameter

titles = []

models = []

predict = []

accuracy = []



for i in range(7):

    clf = svm.SVC(kernel='rbf', C=c)

    model = clf.fit(x_train, y_train)

    models.append(model)    

    

    title1 = "Decision Boundaries RBF C=" + str(c)

    titles.append(title1)

    c *= 10    



# Set-up 2x2 grid for plotting.

fig2, sub2 = plt.subplots(2,4, figsize=(15,10))

plt.subplots_adjust(wspace=0.7, hspace=0.7)



#create a meshgrid to represent each data on validation set with consider max and min values on x y axises

#why validation set because we will predict our results with validaiton set

x0, x1 = x_val.iloc[:, 0], x_val.iloc[:, 1]

x_min, x_max = x0.min() - 1, x0.max() + 1

y_min, y_max = x1.min() - 1, x1.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))



for clf, title, ax2 in zip(models, titles, sub2.flatten()):

    plot_contours(ax2, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)

    ax2.scatter(x0, x1, c=y_val, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax2.set_xlabel('Sepal length')

    ax2.set_ylabel('Sepal width')

    ax2.set_xticks(())

    ax2.set_yticks(())

    ax2.set_title(title,size='small')

plt.show()
accuracy = []

#c. Evaluate the method on the validation set

for clf in models:

    pred = clf.predict(x_val)

    #Accuracy

    accuracy1 = sum(pred==y_val) / y_val.shape[0] 

    accuracy.append(accuracy1)

    print('Accuracy val set: ', accuracy1)



fig3, ax3 = plt.subplots()

ax3.plot(accuracy, label="score")

ax3.set_xlabel("C Values")

ax3.set_ylabel("Accuracy")

ax3.set_title("Accuracy on Validation on RBF SVM", size='small')

plt.show()
#Use the best value of C and evaluate the model on the test set. How well does

#the best c = 1.0

#redo svm for test_split

svm_test = svm.SVC(kernel = 'rbf', C=1.0)

svm_test.fit(x_train, y_train)

predict_test = svm_test.predict(x_test)

accuracy_test = sum(predict_test==y_test) / y_test.shape[0] 

print('Accuracy test set: ', accuracy_test)
#Perform a grid search of the best parameters for an RBF kernel: we will now tune both gamma and C at the same time. Select an appropriate range for both parameters. Train the model and score it on the validation set.

#Show the table showing how these parameters score on the validation set.



costs = [1e-3,1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

gammas = [1e-3,1e-2, 1e-1, 1e0, 1e1]

kernels = ['rbf']



classifiers = []

for c in costs:

    for g in gammas:

        grid_search = svm.SVC(kernel = 'rbf', C=c, gamma=g)

        grid_search.fit(x_train, y_train)

        classifiers.append((c, g, grid_search))



x0, x1 = x_val.iloc[:, 0], x_val.iloc[:, 1]

x_min, x_max = x0.min() - 1, x0.max() + 1

y_min, y_max = x1.min() - 1, x1.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))



for (k, (c, g, grid_search)) in enumerate(classifiers):

    # evaluate decision function in a grid

    Z = grid_search.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    # visualize decision function for these parameters

    plt.subplot(len(costs), len(gammas), k + 1)

    

    plt.title("gamma=10^%d, C=10^%d" % (np.log10(g), np.log10(c)),size='small')



    # visualize parameter's effect on decision function

    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)

    plt.scatter(x0, x1, c=y_val, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    plt.xticks(())

    plt.yticks(())

    plt.axis('tight')

    plt.show()

#Evaluate the best parameters on the test set. Plot the decision boundaries

#the best c = 1.0 and  gamma=0.1

#redo svm for test_split

svm_test = svm.SVC(kernel = 'rbf', C=1.0, gamma=0.1)

svm_test.fit(x_train, y_train)

predict_test = svm_test.predict(x_test)

accuracy_test = sum(predict_test==y_test) / y_test.shape[0] 

print('Accuracy test set: ', accuracy_test)
from sklearn.model_selection import KFold

x_train.append(x_val)

y_train = np.append(y_train,y_val)



##MAKE WITH GRIDSEARCH CV = 5 passed kf

k_folds = KFold(n_splits = 5, shuffle=True, random_state=1)



costs = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

gammas = [1e-3,1e-2, 1e-1, 1e0, 1e1]

kernels = ['rbf']

#scores = np.empty((len(costs), len(gammas), len(k_folds), 1))

scores = []

acc_score = np.empty((len(costs), len(gammas)))



for i, c in enumerate(costs):

    for j, g in enumerate(gammas):

        k = 1

        k_fold_avg_acc = 0

        for train, test in k_folds.split(x_train):

            #train and test are the index of the dataset

            grid_search = svm.SVC(kernel = 'rbf', C=c, gamma=g)

            grid_search.fit(x_train.iloc[train], y_train[train]) 

            score = grid_search.score(x_train.iloc[test], y_train[test])

            scores.append((g,c,k,score))

            k_fold_avg_acc += score

            print("For gamma=", g, " C=", c," k-fold=",k," the accuracy=",score)

            if k == 5:

                k_fold_avg_acc = k_fold_avg_acc / 5.0

                acc_score[i,j] = np.round(k_fold_avg_acc,3)

            k += 1

            

#Heatmap accuracies

fig5, ax5 = plt.subplots()

im = ax5.imshow(acc_score)



ax5.set_xticks(np.arange(len(gammas)))

ax5.set_yticks(np.arange(len(costs)))



ax5.set_xticklabels(gammas)

ax5.set_yticklabels(costs)



ax5.set_xlabel("Gamma Values")

ax5.set_ylabel("C Values")



plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(costs)):

    for j in range(len(gammas)):

        text = ax5.text(j, i, acc_score[i, j],ha="center", va="center", color="w", size='small')



ax5.set_title("Average Accuracies K-Fold(in C/gamma)")

fig.tight_layout()

plt.show()



#Evaluate the best parameters on the test set

#the best c = 1.0 and  gamma=0.1

grid_search.fit(x_test, y_test)

test_score = grid_search.score(x_test, y_test)

print("The test accuracy = ",test_score)

#Evaluate the best parameters on the test set

#the best c = 1.0 and  gamma=0.1

k_folds = KFold(n_splits = 5, shuffle=True, random_state=1)

f_scores = []



for train, test in k_folds.split(x_test):

    #train and test are the index of the dataset

    grid_search = svm.SVC(kernel = 'rbf', C=1.0, gamma=0.1)

    grid_search.fit(x_test.iloc[train], y_test[train]) 

    #There is an error abput x_train has 2 dimension but train value is one dimension

    f_score = grid_search.score(x_test.iloc[test], y_test[test])

    print("the accuracy = ",f_score)

    f_scores.append(f_score)



fig6, ax6 = plt.subplots()

ax6.plot(f_scores, label="Score")

ax6.set_xlabel("K Folds")

ax6.set_ylabel("Accuracy")

ax6.set_title("Accuracy on KFolds")

plt.show()