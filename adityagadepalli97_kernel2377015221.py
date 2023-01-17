import random 

random.seed(42)



import numpy as np

import pandas as pd



df = pd.read_csv('../input/ThoracicSurgery.csv')

df.head()
df = pd.get_dummies(df,drop_first=True)
df.head()
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



def plot_target(data):

    classes = data.iloc[:, -1].values

    unique, counts = np.unique(classes, return_counts=True)



    plt.bar(unique,counts)

    plt.title('Class Frequency')

    plt.xlabel('Class')

    plt.ylabel('Frequency')

    plt.show()
plot_target(df)
from imblearn.over_sampling import RandomOverSampler



def balance_data(data):

    

    # Splitting into X (input) and y (target)

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    

    # Adding sparsity

    X.insert(2, "Spars_Var", np.arange(1,len(y)+1,1), True) 

        

    # Define the resampling method

    method = RandomOverSampler(random_state=42)

    

    # Create the resampled feature set

    return method.fit_sample(X, y)
X_resampled, y_resampled = balance_data(df)
# Print the value_counts on the original labels y

print(pd.value_counts(pd.Series(df.iloc[:, -1])))



# Print the value_counts

print(pd.value_counts(pd.Series(y_resampled)))
from sklearn.model_selection import train_test_split



def split(X,y):

    

    # Create training and test sets

    return train_test_split(X, y, test_size = 0.3, random_state=42)
X_train, X_test, y_train, y_test = split(X_resampled,y_resampled)
#Import svm model

from sklearn import svm

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV



def SVM(X_train, X_test, y_train, y_test, kernel, degree, C):

    

    #Create a svm Classifier

    clf = svm.SVC(C = C, kernel= kernel , degree = degree, gamma='auto') 

        

     #Train the model using the training sets

    clf.fit(X_train, y_train)

    

    #Predict the response for test dataset

    return clf.predict(X_train),clf.predict(X_test)
from sklearn import metrics



def performance(y_train,y_test,y_train_pred,y_test_pred):

    #Import scikit-learn metrics module for accuracy calculation

    

    

    # Model Accuracy: how often is the classifier correct?

    

    Train_accuracy = metrics.accuracy_score(y_train, y_train_pred)

    Test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

        

    # Model Precision: what percentage of positive tuples are labeled as such?

    

    Train_precision = metrics.precision_score(y_train, y_train_pred)

    Test_precision = metrics.precision_score(y_test, y_test_pred)



    

    # Model Recall: what percentage of positive tuples are labelled as such?

    

    Train_recall = metrics.recall_score(y_train, y_train_pred)

    Test_recall = metrics.recall_score(y_test, y_test_pred)

    

    

    performance = {'Train_accuracy':Train_accuracy,'Test_accuracy':Test_accuracy,

                   'Train_precision':Train_precision,'Test_precision':Test_precision,

                   'Train_recall':Train_recall,'Test_recall':Test_recall}

    

    return performance
def linear_SVM(X_train, X_test, y_train, y_test, C):

    

    return SVM(X_train, X_test, y_train, y_test, 'linear',1, C)
lin_y_train_pred,lin_y_test_pred = linear_SVM(X_train, X_test, y_train, y_test,1)
lin_results = performance(y_train,y_test,lin_y_train_pred,lin_y_test_pred)

print(lin_results)
def quadratic_SVM(X_train, X_test, y_train, y_test, C):

    

    return SVM(X_train, X_test, y_train, y_test, 'poly',2, C)
quad_y_train_pred,quad_y_test_pred = quadratic_SVM(X_train, X_test, y_train, y_test, 1)
quad_results = performance(y_train,y_test,quad_y_train_pred,quad_y_test_pred)

print(quad_results)
def rbf_SVM(X_train, X_test, y_train, y_test, C):

    

    return SVM(X_train, X_test, y_train, y_test, 'rbf',3, C)
rbf_y_train_pred,rbf_y_test_pred = rbf_SVM(X_train, X_test, y_train, y_test, 1)
rbf_results = performance(y_train,y_test,rbf_y_train_pred,rbf_y_test_pred)

print(rbf_results)
def find_best_C(C,kernel):

    

    all_preds = []

    all_results = []

    

    test_acc = np.zeros(len(C))

    train_acc = np.zeros(len(C))

    

    i = 0

    

    for c in C:

        

        if(kernel == 'linear'):

            y_train_pred,y_test_pred = linear_SVM(X_train, X_test, y_train, y_test,c)

        elif(kernel == 'quad'):

            y_train_pred,y_test_pred = quadratic_SVM(X_train, X_test, y_train, y_test, c)

        elif(kernel == 'rbf'):

            y_train_pred,y_test_pred = rbf_SVM(X_train, X_test, y_train, y_test, c)

        

        all_preds.append([list(y_train_pred),list(y_test_pred)])

        results = performance(y_train,y_test,y_train_pred,y_test_pred)

        all_results.append(results)

        

        train_acc[i] = results['Train_accuracy']

        test_acc[i] = results['Test_accuracy']

        i+=1



    best_c = C[np.argmax(test_acc)]

    print('Kernel: ', kernel)

    print('Best C value: ', best_c)

    print('Train accuracy: ', train_acc[trial_Cs.index(best_c)])

    print('Test accuracy: ',test_acc[trial_Cs.index(best_c)])

    

    return all_preds, all_results, best_c
#trial_Cs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

trial_Cs = [1,0.1,0.01]

lin_all_preds, lin_all_results,linear_best_c = find_best_C(trial_Cs,'linear')

quad_all_preds, quad_all_results,quad_best_c = find_best_C(trial_Cs,'quad')

rbf_all_preds, rbf_all_results,rbf_best_c = find_best_C(trial_Cs,'rbf')
def ensemble_svm(lin_pred,quad_pred,rbf_pred):

    n = len(lin_pred)

    ensemble_pred = np.zeros(n)

    assert len(lin_pred) == len(quad_pred)

    assert len(quad_pred) == len(rbf_pred)

    for i in range(n):

        ensemble_pred[i] = int(round(np.mean((lin_pred[i],quad_pred[i],rbf_pred[i]))))

    return ensemble_pred
ensm_y_train_pred = ensemble_svm(lin_y_train_pred,quad_y_train_pred,rbf_y_train_pred)

ensm_y_test_pred = ensemble_svm(lin_y_test_pred,quad_y_test_pred,rbf_y_test_pred)
ensemble_results = performance(y_train,y_test,ensm_y_train_pred,ensm_y_test_pred)

print(ensemble_results)