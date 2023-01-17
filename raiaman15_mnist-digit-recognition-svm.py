import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split, validation_curve, KFold, cross_val_score, GridSearchCV

from sklearn.preprocessing import scale

from sklearn.svm import SVC
# Checking version of imported libraries

np.__version__, pd.__version__, sns.__version__
# Making miscellaneous setting for better experience

import warnings

warnings.filterwarnings('ignore')
# Importing training dataset (train.csv)

training_dataframe = pd.read_csv('../input/train.csv')



# Importing testing dataset (test.csv)

testing_dataframe = pd.read_csv('../input/test.csv')
# Understanding the training dataset | Shape

training_dataframe.shape
# Understanding the training dataset | Meta Data

training_dataframe.info()
# Understanding the training dataset | Data Content

training_dataframe.describe()
# Understanding the training dataset | Sample Data

training_dataframe.head()
# Understanding the training dataset | Missing Values

sum(training_dataframe.isnull().sum())
# Dropping Duplicate Values

training_dataframe.drop_duplicates(inplace=True)
# Taking a random subset of training dataset (containing 25% of rows from the original dataset)

rcount = int(.25*training_dataframe.shape[0])

subset_training_dataframe = training_dataframe.sample(n=rcount)
# Understanding the processed training dataset | Shape

subset_training_dataframe.shape
# Clecking if all labels are present almost equally in subset training dataset

plt.figure(figsize=(8,4))

sns.countplot(subset_training_dataframe['label'], palette = 'icefire')
# Checking for collinearity in dataset

plt.figure(figsize=(16,8))

sns.heatmap(data=subset_training_dataframe.corr(),annot=False)
# splitting into X and y

X = subset_training_dataframe.drop("label", axis = 1)

y = subset_training_dataframe.label.values.astype(int)
# scaling the features

X = scale(X)
# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
# confirm that splitting also has similar distribution

print(y_train.mean())

print(y_test.mean())
# Model building



# instantiate an object of class SVC() using cost C=1, gamma='auto'

model = SVC(C = 1, gamma='auto')



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# Evaluate the model using confusion matrix 

confusion_matrix(y_true=y_test, y_pred=y_pred)
# Model Accuracy

print("Accuracy :", accuracy_score(y_test, y_pred))
# K-Fold Cross Validation



# Creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# Instantiating a model with cost=1, gamma='auto'

model = SVC(C = 1, gamma='auto')



# computing the cross-validation scores 

# Argument cv takes the 'folds' object, and we have specified 'accuracy' as the metric

cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = 'accuracy', n_jobs=-1)



# print 5 accuracies obtained from the 5 folds

print(cv_results)

print(f'mean accuracy = {cv_results.mean()}')
# Grid Search to Find Optimal Hyperparameter C



# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100, 1000]}



model = SVC(gamma='auto')



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV(estimator = model, param_grid = params, 

                        scoring='accuracy', cv=folds, n_jobs=-1,

                        verbose=1, return_train_score=True)



# fit the model - it will fit 5 folds across all values of C

model_cv.fit(X_train, y_train)  



# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plot of C versus train and test scores



plt.figure(figsize=(4, 4))

plt.plot(cv_results['param_C'], cv_results['mean_test_score'])

plt.plot(cv_results['param_C'], cv_results['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
best_score = model_cv.best_score_

best_C = model_cv.best_params_['C']



print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))
# model with the best value of C

model = SVC(C=best_C, gamma='auto')



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# Optimal Final Linear SVM Model Accuracy

print("Accuracy :", accuracy_score(y_test, y_pred))
# Model building



# instantiate an object of class SVC() using cost C=1, Gamma='auto', Kernel='rbf'

model = SVC(C = 1, gamma='auto', kernel='rbf')



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# Evaluate the model using confusion matrix 

confusion_matrix(y_true=y_test, y_pred=y_pred)
# Model Accuracy

print("Accuracy :", accuracy_score(y_test, y_pred))
# K-Fold Cross Validation



# Creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# Instantiating a model with cost=1, Gamma='auto', Kernel='rbf'

modelkernel=SVC(C = 1, gamma='auto', kernel='rbf')



# computing the cross-validation scores 

# Argument cv takes the 'folds' object, and we have specified 'accuracy' as the metric

cv_results = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy', n_jobs=-1)



# print 5 accuracies obtained from the 5 folds

print(cv_results)

print(f'mean accuracy = {cv_results.mean()}')
# Grid Search to Find Optimal Hyperparameter C, Gamma



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-1, 1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]



# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator=model, param_grid=hyper_params, 

                        scoring='accuracy', cv=folds, n_jobs=-1,

                        verbose=1, return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train) 



# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plot of C and Gamma versus train and test scores



# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# plotting

plt.figure(figsize=(16,4))



# subplot 1/4

plt.subplot(141)

gamma_1 = cv_results[cv_results['param_gamma']==0.1]



plt.plot(gamma_1["param_C"], gamma_1["mean_test_score"])

plt.plot(gamma_1["param_C"], gamma_1["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.1")

plt.ylim([0.0, 1.1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/4

plt.subplot(142)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.6, 1.1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 3/4

plt.subplot(143)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.8, 1.1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 4/4

plt.subplot(144)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.8, 1.1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print(f'The best test score is {best_score} corresponding to hyperparameters {best_hyperparams}')
# model with the best value of C and Gamma

model = SVC(C=best_hyperparams['C'], gamma=best_hyperparams['gamma'], kernel="rbf")



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# Optimal Final Linear SVM Model Accuracy

print("Accuracy :", accuracy_score(y_test, y_pred))
# Predicting values for our Test Split of Training Dataset

test_predict = model.predict(X_test)
# Plotting the distribution of our prediction

d = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}

dataframe_to_export = pd.DataFrame(data=d)

sns.countplot(dataframe_to_export['Label'], palette = 'icefire')
# Les't visualize our Final Model in Action for few unseen images from Training Dataset



a = np.random.randint(1,test_predict.shape[0]+1,5)



plt.figure(figsize=(16,4))

for k,v in enumerate(a):

    plt.subplot(150+k+1)

    _2d = X_test[v].reshape(28,28)

    plt.title(f'Predicted Label: {test_predict[v]}')

    plt.imshow(_2d)

plt.show()
# Predicting values for unseen Test Dataset



# scaling the features

testing_dataframe = scale(testing_dataframe)



test_predict = model.predict(testing_dataframe)
# Plotting the distribution of our prediction

d = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}

dataframe_to_export = pd.DataFrame(data=d)

sns.countplot(dataframe_to_export['Label'], palette = 'icefire')
# Les't visualize our Final Model in Action for few images from Test Dataset



a = np.random.randint(1,test_predict.shape[0]+1,5)



plt.figure(figsize=(16,4))

for k,v in enumerate(a):

    plt.subplot(150+k+1)

    _2d = testing_dataframe[v].reshape(28,28)

    plt.title(f'Predicted Label: {test_predict[v]}')

    plt.imshow(_2d)

plt.show()
# Exporting the Predicted values for evaluation at Kaggle

dataframe_to_export.to_csv(path_or_buf='submission.csv', index=False)