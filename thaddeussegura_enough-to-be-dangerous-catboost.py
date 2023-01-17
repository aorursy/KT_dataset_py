#Numpy is used so that we can deal with array's, which are necessary for any linear algebra

# that takes place "under-the-hood" for any of these algorithms.

import numpy as np



#Pandas is used so that we can create dataframes, which is particularly useful when

# reading or writing from a CSV.

import pandas as pd



#Matplotlib is used to generate graphs in just a few lines of code.

import matplotlib.pyplot as plt



#Import the classes we need to test linear, ridge, and lasso to compare

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV



#Need these for selecting the best model

from sklearn.model_selection import cross_val_score



#These will be our main evaluation metrics 

from sklearn.metrics import confusion_matrix, roc_auc_score



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



#import CatBoost

from catboost import CatBoostClassifier, FeaturesData
#read the data from csv

dataset = pd.read_csv('../input/churn-modeling/Churn_Modelling.csv')



#take a look at our dataset.  head() gives the first 5 lines. 

dataset.head()
#Grab X, ignoring row number, customer ID, and surname

X = dataset.iloc[:, 3:13]



#grab the output variable

y = dataset.iloc[:, 13]



#take a look

X[0:10]
#create the indexes 

categorical_features_indices = np.array([1,2])
#split the datasets, leaving 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Create the Model

model = CatBoostClassifier(iterations = 50000, 

                                learning_rate=0.20,

                                depth=8,

                                loss_function='Logloss',

                                subsample = 0.8,

                                custom_loss = ['AUC'] )





#Fit the Model passing in our categorical Feature indexes.

model.fit(X_train, y_train,cat_features=categorical_features_indices)
# Predicting the Test set results

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm
Cat_preds1 = model.predict_proba(X_test)

Cat_class1 = model.predict(X_test)

Cat_score1 = roc_auc_score(y_test, Cat_preds1[:,1])

print("ACCURACY: {:.4f}".format(Cat_score1))


