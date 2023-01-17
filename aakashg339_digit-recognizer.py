# Importing libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
# loading dataset

train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")

container = [train_dataset, test_dataset]
container[0].head()
container[1].head()
# Checking for row having null value



print(container[0][container[0].isnull().sum(axis = 1) > 0])

print("\n\n\n")

print(container[1][container[1].isnull().sum(axis = 1) > 0])
# Info of datasets



print(container[0].info())

print("\n\n\n")

print(container[1].info())
# Info of datasets



print(container[0].describe())

print("\n\n\n")

print(container[1].describe())
# Checking the shape of train and test data



print(container[0].shape)

print(container[1].shape)
# Seeing the unique values in column data to be predicted



num = container[0]['label'].unique()

num.sort()

num
# Seeing the distribution of each label



plt.figure(figsize = (10,8))

sns.countplot(x = 'label', data = container[0], order = num)

plt.show()
# Checking the count of each pixel for a given number



num_mean = container[0].groupby('label').sum()

num_mean
# Dividing data in X and y



X = container[0].drop(['label'], axis = 1)

y = container[0]['label']
# Scaling X as pixel value ranges from 0 to 254



X = scale(X)
# Dividing in train and test data



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, test_size = 0.1, random_state = 101)
# Creating an instance of PCA



pca = PCA(svd_solver = 'randomized', random_state = 42)
pca.fit(X_train)
pca.explained_variance_ratio_
# plotting cummulative variance to get better understanding



plt.figure(figsize = (10,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Number of Components")

plt.ylabel("Cummulative explained variance")

plt.show()
# plotting cummulative variance to get better understanding (Zooming in)



plt.figure(figsize = (10,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Number of Components")

plt.ylabel("Cummulative explained variance")

plt.ylim(0.8, 1.025)

plt.grid()

plt.show()
# plotting variance of each component to get better understanding



plt.figure(figsize = (10,8))

plt.plot(pca.explained_variance_ratio_)

plt.xlabel("Number of Components")

plt.ylabel("Cummulative explained variance")

plt.show()
# plotting variance of each component to get better understanding (Zooming in)



plt.figure(figsize = (10,8))

plt.plot(pca.explained_variance_ratio_)

plt.xlabel("Number of Components")

plt.ylabel("Cummulative explained variance")

plt.xlim(-5,100)

plt.grid()

plt.show()
## Tuning hyperparameters



## Creating instance of model and pca

#model = SVC(kernel = 'rbf')

#pca = PCA()



## Creating pipeline instance

#pipe = Pipeline(steps = [('pca', pca), ('model', model)])





#folds = KFold(n_splits = 3, shuffle = True, random_state = 4)



#param_grid = [{'pca__n_components' : [35, 50],

#              'model__C' : [1, 10, 100, 1000], 

#              'model__gamma' : [0.0001, 0.001, 0.01]

#              }]



#model_cv = GridSearchCV(estimator = pipe,

#                        param_grid = param_grid,

#                        scoring = 'accuracy',

#                        cv = folds,

#                        verbose = 1,

#                        return_train_score = True)



#model_cv.fit(X_train, y_train)



#model_cv.best_params_



### Used the above code and got the best parameters as n_components = 35, C = 100, gamma = 0.001

### Therefore we will select 35 features
# Building final model and checking



pca = PCA(n_components = 35)



model = SVC(C = 20, gamma = 0.001, kernel = 'rbf')
pca.fit(X_train)
X_train_tr = pca.transform(X_train)

X_test_tr = pca.transform(X_test)
model.fit(X_train_tr, y_train)
y_pred = model.predict(X_test_tr)
model.score(X_test_tr, y_test)
model.score(X_train_tr, y_train)
# Now we will build our final model with the entire dataset

# For final model dividing dataset in train and test



X_train_f = container[0].drop(['label'], axis = 1)

X_test_f = container[1]

y_train_f = container[0]['label']
# Scaling the data

X_train_f = scale(X_train_f)

X_test_f = scale(X_test_f)
# Using PCA to reduce dimunsions



pca_f = PCA(n_components = 35)
X_train_ft = pca_f.fit_transform(X_train_f)
X_train_ft.shape
X_test_ft = pca_f.transform(X_test_f)
X_test_ft.shape
# Hyperparameter tuning

# Here again running the hypermater to plot graph with results



# Instantiating SVC model

model = SVC(kernel = 'rbf')



# Instantiating flods

folds = KFold(n_splits = 3, shuffle = True, random_state = 4)



param_grid = [{'C' : [1, 10, 100, 1000],

               'gamma' : [0.0001, 0.001, 0.01]}]





model_cv = GridSearchCV(estimator = model,

                        param_grid = param_grid,

                        scoring = 'accuracy',

                        cv = folds,

                        verbose = 1,

                        return_train_score = True)



model_cv.fit(X_train_ft, y_train_f)
cv_results = pd.DataFrame(model_cv.cv_results_)



cv_results
# Plotting the results



plt.figure(figsize = (16,6))



plt.subplot(1,3,1)

gamma_01 = cv_results[cv_results['param_gamma'] == 0.0001]



plt.plot(gamma_01['param_C'], gamma_01['mean_train_score'])

plt.plot(gamma_01['param_C'], gamma_01['mean_test_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.0001')

plt.ylim([0.00, 1.2])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



plt.subplot(1,3,2)

gamma_02 = cv_results[cv_results['param_gamma'] == 0.001]



plt.plot(gamma_01['param_C'], gamma_02['mean_train_score'])

plt.plot(gamma_01['param_C'], gamma_02['mean_test_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.001')

plt.ylim([0.00, 1.2])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



plt.subplot(1,3,3)

gamma_03 = cv_results[cv_results['param_gamma'] == 0.01]



plt.plot(gamma_01['param_C'], gamma_03['mean_train_score'])

plt.plot(gamma_01['param_C'], gamma_03['mean_test_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.01')

plt.ylim([0.00, 1.2])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



plt.show()
model_cv.best_params_
# Again tuened with the below hyperparameter



#param_grid = [{'C' : [10, 15, 20, 30, 40, 50, 100, 500],

#               'gamma' : [0.0005, 0.001, 0.005]}]



# Got best paramenter as C = 20, gamma = 0.001
# Building final model



model = SVC(C = 20, gamma = 0.001, kernel = 'rbf')



model.fit(X_train_ft, y_train_f)
# Final predictions



y_pred = model.predict(X_test_ft)
d = pd.DataFrame({'ImageId': np.arange(1,len(y_pred) + 1), 'Label': y_pred})

d.set_index('ImageId', inplace = True)

d.to_csv('predictions.csv', sep=",")