import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn import svm
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/admission_basedon_exam_scores.csv')

print('Shape of data= ', df.shape)

df.head()
df_admitted = df[df['Admission status'] == 1]

print('Training examples with admission status 1 are = ', df_admitted.shape[0])

df_admitted.head(3)
df_notadmitted = df[df['Admission status'] == 0]

print('Training examples with admission status 0 are = ', df_notadmitted.shape[0])

df_notadmitted.head(3)
def plot_data(title):    

    plt.figure(figsize=(10,6))

    plt.scatter(df_admitted['Exam 1 marks'], df_admitted['Exam 2 marks'], color= 'green', label= 'Admitted Applicants')

    plt.scatter(df_notadmitted['Exam 1 marks'], df_notadmitted['Exam 2 marks'], color= 'red', label= 'Not Admitted Applicants')

    plt.xlabel('Exam 1 Marks')

    plt.ylabel('Exam 2 Marks')

    plt.title(title)

    plt.legend()

 

plot_data(title = 'Admitted Vs Not Admitted Applicants')
#Lets create feature matrix X and label vector y

X = df[['Exam 1 marks', 'Exam 2 marks']]

y = df['Admission status']



print('Shape of X= ', X.shape)

print('Shape of y= ', y.shape)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state= 1)



print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
# Note here we are using default SVC parameters

clf = svm.SVC()

clf.fit(X_train, y_train)

print('Model score using default parameters is = ', clf.score(X_test, y_test))
def plot_support_vector(classifier):

    """

    To plot decsion boundary and margin. Code taken from Sklearn documentation.



    I/P

    ----------

    classifier : SVC object for each type of kernel



    O/P

    -------

    Plot

    

    """

    clf =classifier

    # plot the decision function

    ax = plt.gca()

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    # create grid to evaluate model

    xx = np.linspace(xlim[0], xlim[1], 30)

    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = clf.decision_function(xy).reshape(XX.shape)



    # plot decision boundary and margins

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,

               linestyles=['--', '-', '--'])

    # plot support vectors

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,

               linewidth=1, facecolors='none', edgecolors='k')  
plot_data(title = 'SVM Classifier With Default Parameters')

plot_support_vector(clf)  
def svm_params(X_train, y_train, X_test, y_test):

    """

    Finds the best choice of Regularization parameter (C) and gamma for given choice of kernel and returns the SVC object for each type of kernel



    I/P

    ----------

    X_train : ndarray

        Training samples

    y_train : ndarray

        Labels for training set

    X_test : ndarray

        Test data samples

    y_test : ndarray

        Labels for test set.



    O/P

    -------

    classifiers : SVC object for each type of kernel

    

    """

    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40]

    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40]

    kernel_types = ['linear', 'poly', 'rbf']

    classifiers = {}

    max_score = -1

    C_final = -1

    gamma_final = -1

    for kernel in kernel_types:                    

        for C in C_values:

            for gamma in gamma_values:

                clf = svm.SVC(C=C, kernel= kernel, gamma=gamma)

                clf.fit(X_train, y_train)

                score = clf.score(X_test, y_test)

                #print('C = %s, gamma= %s, score= %s' %(C, gamma, score))

                if score > max_score:

                    max_score = score

                    C_final = C

                    gamma_final = gamma

                    classifiers[kernel] = clf        

        print('kernel = %s, C = %s, gamma = %s, score = %s' %(kernel, C_final, gamma_final, max_score))

    return classifiers
classifiers = svm_params(X_train, y_train, X_test, y_test)
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['linear']))

plot_support_vector(classifiers['linear'])
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['rbf']))

plot_support_vector(classifiers['rbf'])
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['poly']))

plot_support_vector(classifiers['poly'])