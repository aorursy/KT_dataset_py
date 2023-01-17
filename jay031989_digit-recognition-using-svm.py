# libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale

# Reading train and test data sets from csv files using pandas

train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
# about the dataset

# dimensions

print("Dimensions: ", train_df.shape, "\n")

# data types

train_df.info()
# Printing first 5 rows of train data set just to get the hang of dataframe

train_df.head()
# Printing first 5 rows of test data set just to get the hang of dataframe

test_df.head()
# Splitting the data into two parts 20% and 80% for easy computaion purpose

train_80per_df,train_20per_df=train_test_split(train_df, test_size = 0.2, random_state = 4)
train_20per_df.head()
# about the dataset

# dimensions

print("Dimensions: ", train_20per_df.shape, "\n")

# data types

train_20per_df.info()
# Checking for outliers

train_20per_df.describe()
# Listing columns of dataset

print(train_20per_df.columns)
# Listing unique digits available in dataset in label column

print("List of digits : ", list(np.sort(train_20per_df.label.unique())))
# Lets check the distribution of the labels

sns.countplot(train_20per_df.label)
# Printing the occurence of each digit in train dataframe

train_20per_df.label.astype('category').value_counts()
# Let us try to check the heatmap if we can get some data insights.

plt.figure(figsize=(18, 10))

sns.heatmap(train_20per_df)
#Lets see digit "1" images in the data.



plt.figure(figsize=(28,28))



digit_1 = train_20per_df.loc[train_20per_df.label==1,:]

digit_image = digit_1.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    four = digit_image.iloc[i].values.reshape(28, 28)

    plt.imshow(four, cmap='gray')

    subplots_loc = subplots_loc +1
#Lets see digit "3" images in the data.





plt.figure(figsize=(28,28))



digit_3 = train_20per_df.loc[train_20per_df.label==3,:]

digit_image = digit_3.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    four = digit_image.iloc[i].values.reshape(28, 28)

    plt.imshow(four, cmap='gray')

    subplots_loc = subplots_loc +1
# splitting into X and y

X=train_20per_df.drop("label",axis=1)

y=train_20per_df.label
# scaling the features

X_scaled = scale(X)



# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)
# Data splitting in train and test data

#X_train, X_test,y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
linear_model=SVC(kernel='linear')

linear_model.fit(X_train,y_train)

# predict

y_pred = linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  

# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = SVC(C=10, gamma=0.001, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")



test_df.head()
# Scaling test data frame

X_Scaled_Test_df=scale(test_df)
# Applying final model to test data 

predicted_digits=model.predict(X_Scaled_Test_df)
predicted_digits.shape
# Creating data frame

data = pd.DataFrame({'Label': predicted_digits})

data.head()
data.to_csv('predicted_digits.csv', sep=",")