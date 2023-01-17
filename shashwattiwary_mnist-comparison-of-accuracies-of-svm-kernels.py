import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



import warnings

warnings.filterwarnings('ignore')
mnist_train = pd.read_csv('../input/train.csv', index_col = False)

mnist_test = pd.read_csv('../input/test.csv', index_col = False)
print(mnist_train.shape)

print(mnist_test.shape)
mnist_train.head()
mnist_test.head()
sns.countplot(mnist_train['label'])

plt.show()
X_ = mnist_train.iloc[:, 1:]

y_ = mnist_train.iloc[:, 0]

print(X_.shape)

print(y_.shape)
def scale_df(X):

    '''Scaling the data set using StandardScaler() and returning the scaled data set'''

    scale = StandardScaler()

    X_scaled = scale.fit_transform(X)

    return(X_scaled)
#stratify = y_ to preserve the distribution of digits 

X_train, X_validate, y_train, y_validate = train_test_split(scale_df(X_), y_, test_size = 0.80, random_state = 30, stratify = y_)

print(X_train.shape)

print(X_validate.shape)

print(y_train.shape)

print(y_validate.shape)
def get_accuracy(X_train, X_validate, y_train, y_validate, k):

    '''fitting the SVC model for various kernels like linear, polynomial and rbf

    and finding the accuracy for each kernel for the train and validate sets. Based on the accuracy

    an appropriate kernel will be chosen for hyperparameter tuning and final model building with optimum hyperparameters'''

    

    #Caling teh scale_df() and storing the scaled df

    #X_train_scaled = scale_df(X_train)

    #X_validate_scaled = scale_df(X_validate)

    

    #Building a likear model for kernel type k passed in the parameter

    SVC_model = SVC(kernel = k)

    

    #Fitting the model for the training set

    SVC_model.fit(X_train, y_train)

    #Predicting the labels for the training set

    y_train_predict = SVC_model.predict(X_train)

    #Accuracy for the training set

    train_accuracy = metrics.accuracy_score(y_train, y_train_predict)

    #Classification Report

    #c_report_train = metrics.classification_report(y_train, y_train_predict)

    

    #Fitting the model for validation set

    SVC_model.fit(X_validate, y_validate)

    #Predicting the labels for the validation set

    y_validate_predict = SVC_model.predict(X_validate)

    #Accuracy for the validation set

    validate_accuracy = metrics.accuracy_score(y_validate, y_validate_predict)

    #Classification Report

    #c_report_validate = metrics.classification_report(y_validate, y_validate_predict)

    

    #returning the accuracy for the train and validate set

    return(train_accuracy, validate_accuracy)
train_accuracy_linear, validate_accuracy_linear = get_accuracy(X_train, X_validate, y_train, y_validate, 'linear')

train_accuracy_poly, validate_accuracy_poly = get_accuracy(X_train, X_validate, y_train, y_validate, 'poly')

train_accuracy_rbf, validate_accuracy_rbf = get_accuracy(X_train, X_validate, y_train, y_validate, 'rbf')

#train_accuracy_sigmoid, c_report_train_sigmoid, test_accuracy_sigmoid, c_report_validate_sigmoid = get_accuracy(X_train, X_validate, y_train, y_validate, 'sigmoid')
print('Kernel = Linear')

print('Train Accuracy = ', train_accuracy_linear)

#print('Train Classification Report: \n', c_report_train_linear)

print('Validate Accuracy = ', validate_accuracy_linear)

#print('Validate Classification Report: \n', c_report_validate_linear)



print('\n Kernel = Polynomial')

print('Train Accuracy = ', train_accuracy_poly)

#print('Train Classification Report: \n', c_report_train_poly)

print('Validate Accuracy = ', validate_accuracy_poly)\

#print('Validate Classification Report: \n', c_report_validate_poly)



print('\n Kernel = RBF')

print('Train Accuracy = ', train_accuracy_rbf)

#print('Train Classification Report: \n', c_report_train_rbf)

print('Validate Accuracy = ', validate_accuracy_rbf)

#print('Validate Classification Report: \n', c_report_validate_rbf)



#print('\n Kernel = Sigmoid')

#print('Train Accuracy = ', train_accuracy_sigmoid)

#print('Train Classification Report: \n', c_report_train_sigmoid)

#print('Validate Accuracy = ', validate_accuracy_sigmoid)

#print('Validate Classification Report: \n', c_report_validate_sigmoid)
#steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel = 'rbf'))]

#pipeline = Pipeline(steps)
parameters = {'gamma': [0.01, 0.1, 10],

              'C': [0.001, 0.01, 0.1]}
model = SVC(kernel = 'rbf')
grid = GridSearchCV(estimator = model, 

                    param_grid = parameters, 

                    cv = 5, 

                    n_jobs = -1, 

                    scoring = 'accuracy', 

                    verbose = 1, 

                    return_train_score = True)
grid.fit(X_train, y_train)
print('Score = %3.2f'%grid.score(X_train, y_train))
cv_results = pd.DataFrame(grid.cv_results_)

cv_results.head()
cv_results['param_C'] = cv_results['param_C'].astype('int')



plt.figure(figsize = (16,6))



#plt.subplot(221)

#gamma_001 = cv_results[cv_results['param_SVM__gamma'] == 0.001]



#plt.plot(gamma_001['param_SVM__C'], gamma_001['mean_test_score'])

#plt.plot(gamma_001['param_SVM__C'], gamma_001['mean_train_score'])

#plt.xlabel('C')

#plt.ylabel('Accuracy')

#plt.title('Gamma = 0.001')

#plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

#plt.xscale('log')



plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma'] == 0.01]



plt.plot(gamma_01['param_C'], gamma_01['mean_test_score'])

plt.plot(gamma_01['param_C'], gamma_01['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.01')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.subplot(132)

gamma_1 = cv_results[cv_results['param_gamma'] == 0.1]



plt.plot(gamma_1['param_C'], gamma_1['mean_test_score'])

plt.plot(gamma_1['param_C'], gamma_1['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.1')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.subplot(133)

gamma_10 = cv_results[cv_results['param_gamma'] == 10]



plt.plot(gamma_10['param_C'], gamma_10['mean_test_score'])

plt.plot(gamma_10['param_C'], gamma_10['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 10')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.show()
print('Best Score', grid.best_score_)

print('Best hyperparameters', grid.best_params_)
model_opt = SVC(C = 0.1, gamma = 0.01, kernel = 'rbf')

model_opt.fit(X_train, y_train)
y_pred_train = model_opt.predict(X_train)

print('Accuracy', metrics.accuracy_score(y_train, y_pred_train))

print('Classification Report: \n', metrics.classification_report(y_train, y_pred_train))
y_pred_validate = model_opt.predict(X_validate)

print('Accuracy', metrics.accuracy_score(y_validate, y_pred_validate))

print('Classification Report: \n', metrics.classification_report(y_validate, y_pred_validate))
y_pred_test = model_opt.predict(scale_df(mnist_test))
y_pred_test
ImageId = np.arange(1, len(y_pred_test)+1)
y_pred_test_df = pd.DataFrame({'ImageId': ImageId, 'Label': y_pred_test})
y_pred_test_df.head()
y_pred_test_df.to_csv('submission.csv', index = False)