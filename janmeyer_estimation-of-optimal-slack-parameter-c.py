import numpy as np

from sklearn import preprocessing, svm

import pandas as pd

from math import sqrt

import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score

from sklearn import metrics

from sklearn.cross_validation import  train_test_split



df =  pd.read_csv('../input/data.csv', header=0)



df.replace('?', -99999, inplace=True)

df.drop(['id'], 1, inplace=True)

df.drop(['Unnamed: 32'], 1, inplace=True)



X = np.array(df.drop(['diagnosis'], 1))

y = np.array(df['diagnosis'])
#Now I split the set the first time and use 

X_train_CV, X_test_CV, y_train_CV, y_test_CV = train_test_split(X, y, test_size=0.2)



reduced_X=X_train_CV

reduced_y=y_train_CV



#List of C-values and other stuff

C_values=[0.001,0.01,0.1,0.5,1,1.5,2,3,4,5,10,15,20,25,50,100]

opt_dict_lin={}

opt_dict_gauss={}

C_dependent_acc=[]



number_of_simulations=200    #Adjust at will!
C_dependent_acc_lin=[]

yerr=[]

for i in range(len(C_values)):



    accuracies = []

    for i1 in range(number_of_simulations):



        X = preprocessing.scale(reduced_X)

        X_train, X_test, y_train, y_test = train_test_split(X, reduced_y, test_size=0.2)



        clf=svm.SVC(C=C_values[i],kernel='linear')

        clf.fit(X_train,y_train)



        predictions = clf.predict(X_test)

        accuracy=metrics.accuracy_score(y_test, predictions)



        #accuracy=clf.score(X_test,y_test)

        accuracies.append(accuracy)



    mean_acc=(sum(accuracies)/len(accuracies))

    standard_error=np.std(accuracies)/sqrt(len(accuracies))

    C_dependent_acc_lin.append(mean_acc)

    yerr.append(standard_error)

    opt_dict_lin[mean_acc]=[C_values[i]]



norms=sorted([n for n in opt_dict_lin])

opt_choice_lin=opt_dict_lin[norms[-1]]
#Out of sample estimation of accuracy

#Note: If I try preprocessing my data using

#X_train_CV = preprocessing.scale(X_train_CV)

#,the performance is screwed up. Why is that?



clf = svm.SVC(C=opt_choice_lin[0], kernel='linear')

clf.fit(X_train_CV, y_train_CV)





predictions = clf.predict(X_test)

accuracy=metrics.accuracy_score(y_test, predictions)









#Now, everything is repeated, only using a Gaussian Kernel instead of a Linear one.



C_dependent_acc=[]

yerr1=[]

for i in range(len(C_values)):



    accuracies = []

    for i1 in range(number_of_simulations):





        X = preprocessing.scale(reduced_X)

        X_train, X_test, y_train, y_test = train_test_split(X, reduced_y, test_size=0.2)



        clf=svm.SVC(C=C_values[i],kernel='rbf')

        clf.fit(X_train,y_train)



        predictions = clf.predict(X_test)

        accuracy=metrics.accuracy_score(y_test, predictions)

        accuracies.append(accuracy)



    mean_acc=(sum(accuracies)/len(accuracies))

    standard_error=np.std(accuracies)/sqrt(len(accuracies))

    C_dependent_acc.append(mean_acc)

    yerr1.append(standard_error)

    opt_dict_gauss[mean_acc] = [C_values[i]]



norms = sorted([n for n in opt_dict_gauss])

opt_choice_gauss = opt_dict_gauss[norms[-1]]



#Testing of optimal value at new data

#X = preprocessing.scale(X_train_CV)





clf = svm.SVC(C=opt_choice_gauss[0], kernel='rbf')

clf.fit(X_train_CV, y_train_CV)



predictions = clf.predict(X_test)

accuracy1=metrics.accuracy_score(y_test, predictions)



#Things are plotted using a log scale

ax = plt.subplot(111)

ax.set_xscale("log")

plt.errorbar(C_values,C_dependent_acc_lin,yerr=yerr,color='b',label='Linear Kernel')

plt.errorbar(C_values,C_dependent_acc,yerr=yerr1,color='r',label='Gaussian Kernel')



plt.scatter(opt_choice_gauss[0],accuracy1,marker='*',s=100,color='r',label='Accuracy of best C of Gaussian kernel on new data')

plt.scatter(opt_choice_lin[0],accuracy,marker='*',s=100,color='b',label='Accuracy of best C of Linear kernel on new data')

plt.xlabel("Log of Slack Parameter 'C'")

plt.ylabel("Accuracy")

#plt.legend(loc='lower right')

print('Accuracy of best C (',opt_choice_lin[0],') at new data with linear kernel:', accuracy)

print('Accuracy of best C (',opt_choice_gauss[0],') at new data with Gaussian kernel:', accuracy1)
print('Accuracy of best C (',opt_choice_lin[0],') at new data with linear kernel:', accuracy)

print('Accuracy of best C (',opt_choice_gauss[0],') at new data with Gaussian kernel:', accuracy1)