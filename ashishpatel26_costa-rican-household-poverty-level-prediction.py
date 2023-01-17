# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Displays all the columns of a dataframe
pd.set_option('display.max_columns',None)
# Importing training dataset
df = pd.read_csv('../input/train.csv')
df.head(10)
# Total number of columns in the dataframe
print(len(df.columns))
print(df.count())
# Droping columns with more than 6000 missing values
mask = df.count() > 3000
df.drop(df.columns[~mask], axis = 1, inplace = True)
# Columns having object datatype
df.columns[df.dtypes == 'object']
# printing unique values of above columns
print(df.dependency.unique(),'\n')
print(df.edjefe.unique(),'\n')
print(df.edjefa.unique())
# Droping useless columns 
df.drop(['Id', 'idhogar'], axis = 1, inplace = True)

# Columns having both continuous and categorical data should be encoded using labelencoder or get_dummies
df.drop(['dependency', 'edjefe', 'edjefa'], axis = 1, inplace = True)
# Total number of rows in the dataframe
print(df.count().max())

# Total number of null entries
print(df.isnull().sum().sum())

# printing columns with missing rows
col_mask = df.columns[df.isnull().sum() > 0]
df[col_mask].head(10)
# Droping rows with missing values
df.dropna(inplace = True)
# Total Number of rows left
print(df.count().max())

# Total Number of columns left
print(len(df.columns))

# Total Number of null entries
print(df.isnull().sum().sum())
# Splitting data into dependent and independent variable
# X is the independent variables matrix
X = df.drop('Target', axis = 1)

# y is the dependent variable vector
y = df.Target
# Scaling Features
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X = ss.fit_transform(X)
# Checking p-values of dataframe features

import statsmodels.formula.api as sm
X1 = np.append(arr = np.ones((9552,1)).astype(int), values = X, axis = 1)
X_opt = X1[:, range(0,135)]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Feature Selection using Backward Elimination technique

def backwardelimination(x, sl):
    numvars = len(X[0])
    for i in range(0, numvars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxvar = max(regressor_OLS.pvalues)
        if maxvar> sl:
            for j in range(0, numvars-i):
                if(regressor_OLS.pvalues[j].astype(float) == maxvar):
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x

sl = 0.05
X_opt = X1[:, range(0,135)]
X_modeled = backwardelimination(X_opt, sl)
X_modeled.shape
# Removing additional columns added for backward elimination
X = X[:,1:]
X.shape
# Applying XGBoost Classifier

from xgboost import XGBClassifier
clc = XGBClassifier(n_estimators = 10)
clc.fit(X, y)
scores1 = []
scores2 = []
scores3 = []
scores4 = []
# Applying 5-fold cross-validation to X_modeled matrix

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clc, X = X_modeled, y = y, cv = 5)
scores1.append(accuracies.mean())
accuracies.mean()
# Applying 5-fold cross-validation to X matrix

accuracies = cross_val_score(estimator = clc, X = X, y = y, cv = 5)
scores1.append(accuracies.mean())
accuracies.mean()
# Feature Selection using PCA(principal component analysis)

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X1 = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
ev
# Using 4-Component PCA

pca = PCA(n_components = 4)
X1 = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
ev
# Applying 5-fold cross-validation to X1 matrix
clc1 = XGBClassifier()
clc1.fit(X1, y)

accuracies = cross_val_score(estimator = clc1, X = X1, y = y, cv = 5)
scores1.append(accuracies.mean())
accuracies.mean()
##Visualizing accuracies of different models using barplot
log_cols = ["Model", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

import seaborn as sns

acc_dict = {'All-columns': scores1[0],'Backward Elimination': scores1[1], 'PCA': scores1[2]}

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Model', data=log, color="g")
# Applying Random-Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clc = RandomForestClassifier(n_estimators = 10)
clc.fit(X, y)
# Applying 5-fold cross-validation to X matrix

accuracies = cross_val_score(estimator = clc, X = X, y = y, cv = 5)
scores2.append(accuracies.mean())
accuracies.mean()
# Applying 5-fold cross-validation to X_modeled matrix

accuracies = cross_val_score(estimator = clc, X = X_modeled, y = y, cv = 5)
scores2.append(accuracies.mean())
accuracies.mean()
# Using PCA components
clc1 = RandomForestClassifier(n_estimators = 10)
clc1.fit(X1, y)

accuracies = cross_val_score(estimator = clc1, X = X1, y = y, cv = 5)
scores2.append(accuracies.mean())
accuracies.mean()
##Visualizing accuracies of different models using barplot
log_cols = ["Model", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

acc_dict = {'All-columns': scores2[0],'Backward Elimination': scores2[1], 'PCA': scores2[2]}

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Model', data=log, color="r")
# Applying K-Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
clc = KNeighborsClassifier(n_neighbors = 5)
clc.fit(X, y)
# Applying 5-fold cross-validation to X matrix

accuracies = cross_val_score(estimator = clc, X = X, y = y, cv = 5)
scores3.append(accuracies.mean())
accuracies.mean()
# Applying 5-fold cross-validation to X_modeled matrix

accuracies = cross_val_score(estimator = clc, X = X_modeled, y = y, cv = 5)
scores3.append(accuracies.mean())
accuracies.mean()
# Using PCA components
clc1 = KNeighborsClassifier(n_neighbors = 10)
clc1.fit(X1, y)

accuracies = cross_val_score(estimator = clc1, X = X1, y = y, cv = 5)
scores3.append(accuracies.mean())
accuracies.mean()
##Visualizing accuracies of different models using barplot
log_cols = ["Model", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

acc_dict = {'All-columns': scores3[0],'Backward Elimination': scores3[1], 'PCA': scores3[2]}

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Model', data=log, color="b")
#Importing libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Splitting X1 into training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.25, random_state = 42)
# Splitting independent variable into different classes for neural networks

y_train = pd.get_dummies(y_train)
# Applying Artificial Neural Networks

def func():    
    
    clc = None

    #initializing ANN
    clc = Sequential()

    #Adding input layer and 1st hidden layer
    clc.add(Dense(activation="relu", units=300, kernel_initializer="uniform", input_dim=4))

    #Adding 2nd hidden layer
    clc.add(Dense(activation="relu", units=300, kernel_initializer="uniform"))

    #Adding output layer
    clc.add(Dense(activation="softmax", units=4, kernel_initializer="uniform"))

    #Compiling ANN
    clc.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return clc

estimator = KerasClassifier(build_fn = func, epochs = 10, batch_size = 5)


#fitting ANN
estimator.fit(X_train, y_train)


# Predicting test set results

pred = estimator.predict(X_test)
print(np.unique(pred))

pred1 = pd.DataFrame(pred)
pred1.columns = ['Target']
pred1.head()
y_test = y_test.reset_index()
y_test.drop('index', axis = 1, inplace = True)
y_test.head()
# Calculating Accuracy
acc = (pred1 == y_test).sum()/len(y_test)
scores4.append(acc[0])
# Splitting X_modeled dataset

X1_train, X1_test, y1_train, y1_test = train_test_split(X_modeled, y, test_size = 0.25, random_state = 42)
# Splitting independent variable into different classes for neural networks

y1_train = pd.get_dummies(y1_train)
# Applying Artificial Neural Networks

def func():    
    
    clc = None

    #initializing ANN
    clc = Sequential()

    #Adding input layer and 1st hidden layer
    clc.add(Dense(activation="relu", units=100, kernel_initializer="uniform", input_dim=X1_train.shape[1]))

    #Adding 2nd hidden layer
    clc.add(Dense(activation="relu", units=100, kernel_initializer="uniform"))

    #Adding output layer
    clc.add(Dense(activation="softmax", units=4, kernel_initializer="uniform"))

    #Compiling ANN
    clc.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return clc

estimator = KerasClassifier(build_fn = func, epochs = 10, batch_size = 5)


#fitting ANN
estimator.fit(X1_train, y1_train)


# Predicting test set results

pred = estimator.predict(X1_test)
print(np.unique(pred))

pred1 = pd.DataFrame(pred)
pred1.columns = ['Target']
pred1.head()
# Re-indexing y_test
y1_test = y1_test.reset_index()
y1_test.drop('index', axis = 1, inplace = True)
y1_test.head()
# Calculating Accuracy
acc = (pred1 == y1_test).sum()/len(y1_test)
scores4.append(acc[0])
max(scores3)
## To save score of each model used
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
import seaborn as sns

acc_dict = {'KNeighborsClassifier':max(scores3),'RandomForestClassifier': max(scores2), 'XGBClassifier':max(scores1), 'Neural networks':max(scores4)}

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="black")
# Reading test file

df1 = pd.read_csv('../input/test.csv')
df1.head()
print(df1.columns[df1.dtypes == 'object'])
df1.drop(df1.columns[df1.dtypes == 'object'], axis = 1, inplace = True)
print(df1.count())
print(df1.columns[df1.count() < 7000])
df1.drop(df1.columns[df1.count() < 7000], axis = 1, inplace = True)
print(df1.isnull().sum().sum())
# Replacing Missing values with most frequent values of that columns

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'most_frequent')
df1 = imp.fit_transform(df1)
np.isnan(df1).sum()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

df1 = ss.fit_transform(df1)
# Using 4-Component PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
df1 = pca.fit_transform(df1)
ev = pca.explained_variance_ratio_
ev
y_pred = clc1.predict(df1)
# Importing sample_submission file

df2 = pd.read_csv('../input/sample_submission.csv')
df2.head()
df2.drop('Target', axis = 1, inplace = True)
df2.set_index('Id', inplace = True)
df2['Target'] = y_pred
df2.head()
# Writing to My_submission.csv
df2.to_csv('My_submission.csv')