import os, sys
import itertools, time
import numpy as np 
import pandas as pd

# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport

# postprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, explained_variance_score, mean_squared_log_error


%matplotlib inline
import matplotlib.pyplot as plt
test = pd.read_csv('../input/test.csv')
test.set_index('Id', inplace=True, drop=True)
data = pd.read_csv('../input/train.csv')
data.set_index('Id', inplace=True, drop=True)

y = data[['SalePrice']]
X = data.drop('SalePrice', axis=1)
ProfileReport(data)
numerical = X._get_numeric_data().columns
categorical = X.columns.drop(numerical)
# Transformations must be applied to both training and testing set.
Xtot = pd.concat((X, test))
len(X), len(test), len(Xtot)
# Empty dataframe for building features
X_eng = pd.DataFrame()
# Find skewed features
from scipy.stats import skew

skewed_feats = Xtot[numerical].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
for header in numerical:
    # Set nan values to mean
    column = Xtot[header].copy()
    colmean = np.mean(column)
    colnan = np.isnan(column)
    column[colnan] = colmean
    
    # Take log1p of skewed features
    if header in skewed_feats:
        column = np.log1p(column)
        header = header+'_log1p'
    
    # Feature scaling
    colstd = np.std(column)
    column = (column - colmean)/colstd
    
    # Set column in engineered dataframe
    X_eng[header] = column
def ohe_cols(column, label='', index='index'):
    
    # Combine rare values (values with counts less than threshold)
    threshold = 0.025
    unique_values = column.unique()
    for value in unique_values:
        if (np.sum(column==value)/len(column))<threshold:
            column[column==value] = 'rare'

    # Encode values into integers
    label_encoder = LabelEncoder()
    try:integer_encoded = label_encoder.fit_transform(column)
    except TypeError:
        integer_encoded = label_encoder.fit_transform(column.astype(str))
    headers = label_encoder.classes_.astype(str)
    
    if len(headers)>2:
        # Encode integers into onehot
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded

        headers = [str(label)+"_"+x for x in headers]
    
        df = pd.DataFrame(onehot_encoded, columns=headers, index=column.index)
        
    else:
        df = pd.DataFrame(integer_encoded, columns=[str(label)+'_binary'], index=column.index)
    
    return df
for header in categorical:
    # One hot encode data
    column  = Xtot[header].copy()
    ohe_column = ohe_cols(column, label=header)
    
    # Merge data into dataframe
    X_eng = pd.merge(X_eng, ohe_column, left_index=True, right_index=True)
log1p_y = np.log1p(y)
# Recover input data and test data
Xinput = X_eng[:len(X)].copy()
test = X_eng[len(X):].copy()

# Train test split - 20% testing
Xtrain, Xtest, ytrain, ytest = train_test_split(Xinput, y, test_size=0.2, random_state=15)

# Train cross-validation split - overall 20% cross-validation
Xtrain, Xcv, ytrain, ycv = train_test_split(Xtrain, ytrain, test_size=(0.2/0.8), random_state=15)

# So we have a 60-20-20 train-cv-test split
len(Xtrain), len(Xcv), len(Xtest), len(test)

# machine learning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
models = {'svm-SVR': SVR,
          'svm-LinearSVR': LinearSVR, 
          'svm-NuSVR': NuSVR,
          'RandomForest': RandomForestRegressor,
          'KNeighbors': KNeighborsRegressor,
          'DecisionTree': DecisionTreeRegressor, 
          'ExtraTree': ExtraTreeRegressor,
          'GaussianProcess': GaussianProcessRegressor}
def RUN(X_train, y_train, X_test, y_test, model, cm=False, label='', target=''):

    """
    Fits model to X_train and y_train
    Predicts targets for X_test
    Provides metrics for prediction success of y_test
    """
    
    start = time.time()
    
    print(label)
    
    model.fit(X_train, np.array(y_train[target]))
    
    pred = model.predict(X_test)
    y_test = np.array(y_test[target])
    
    now = time.time()
    timetaken = now-start
    
    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(y_test, pred))
    print("Accuracy: %f, Error: %f, time: %.3f" % (acc, error, timetaken))
    
    if cm:
        cm = confusion_matrix(y_test, pred)
        plot_confusion_matrix(cm, np.arange(1), normalize=True)
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
for value in models:
    _=RUN(Xtrain, np.log1p(ytrain), Xcv, np.log1p(ycv), models[value](), label=value, target='SalePrice')
# Use random forest to find the most important features

forest = RandomForestRegressor()

forest.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))
    
pred = forest.predict(Xcv)
y_test = np.array(np.log1p(ycv.SalePrice))
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))
print("Accuracy: %f, Error: %f" % (acc, error))
    
importances = forest.feature_importances_

featureimportance = np.vstack((Xtrain.columns.values, importances))
featureimportance = pd.DataFrame(featureimportance.T, columns=['Feature', 'Importance'])

plt.figure(figsize=(20,10))
_=plt.bar(Xtrain.columns.values, importances)
_=plt.xticks(rotation=45, fontsize=10)
featureimportance.sort_values('Importance', ascending=False)
clf = NuSVR()

clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))
    
pred = clf.predict(Xcv)
y_test = np.log1p(np.array(ycv.SalePrice))
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))

error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
print("Accuracy: %f, Error: %f" % (acc, error))
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in np.linspace(0.001, 0.1, 10):
    clf = NuSVR(nu=0.5, C=i, kernel='linear')

    clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))

    pred = clf.predict(Xcv)
    y_test = np.log1p(np.array(ycv.SalePrice))

    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
    print("Accuracy: %f, Error: %f, Parameter: %s" % (acc, error, i))
Xnew = Xtrain.copy()
Xcv_new = Xcv.copy()
Xtest_new = Xtest.copy()
test_new = test.copy()

threshold = 0.005
min_importance = 0.

while min_importance<threshold:

    forest = RandomForestRegressor()
    forest.fit(Xnew, np.array(ytrain)[:,0])

    pred = forest.predict(Xcv_new)
    y_test = np.array(ycv.SalePrice)

    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(y_test, pred))
    print("Accuracy: %f, Error: %f" % (acc, error))

    importances = forest.feature_importances_

    min_importance = np.min(importances)
    
    if min_importance<threshold:
        col = Xnew.columns.values[importances == min_importance][0]
        print(col)
        
        Xnew.drop(col, inplace=True, axis=1)
        Xcv_new.drop(col, inplace=True, axis=1)
        Xtest_new.drop(col, inplace=True, axis=1)
        test_new.drop(col, inplace=True, axis=1)
forest = RandomForestRegressor()

forest.fit(Xnew, np.array(ytrain)[:,0])
    
pred = forest.predict(Xcv_new)
y_test = np.array(ycv.SalePrice)
    
acc = explained_variance_score(ycv, pred)
error = np.sqrt(mean_squared_log_error(ycv, pred))
print("Accuracy: %f, Error: %f" % (acc, error))
    
importances = forest.feature_importances_

plt.figure(figsize=(20,10))
_=plt.bar(Xnew.columns.values, importances)
_=plt.xticks(rotation=45, fontsize=15)
for value in models:
    _=RUN(Xnew, ytrain, Xcv_new, ycv, models[value](), label=value, target='SalePrice')

# Running the Regressor on the test sample tells us what we expect the error/accuracy to be.

clf = RandomForestRegressor()

clf.fit(Xtrain, np.array(ytrain)[:,0])
    
pred = clf.predict(Xtest)
y_test = np.array(ytest.SalePrice)
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))
print("Accuracy: %f, Error: %f" % (acc, error))
pred = clf.predict(test)
pi = test.index.values.astype(int)

prediction = pd.DataFrame(np.vstack((pi, pred)).T, columns=['Id', 'SalePrice'])
prediction.Id = prediction.Id.astype(int)
prediction
clf = NuSVR(nu=0.5, C=0.01, kernel='linear')

clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))

pred = clf.predict(Xtest)
y_test = np.log1p(np.array(ytest.SalePrice))

acc = explained_variance_score(np.exp(y_test)-1, np.exp(pred)-1)
error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
print("Accuracy: %f, Error: %f, Parameter: %s" % (acc, error, i))
pred = clf.predict(test)
pred = np.exp(pred)-1
pi = test.index.values
prediction = pd.DataFrame(np.vstack((pi, pred)).T, columns=['Id', 'SalePrice'])
prediction.Id = prediction.Id.astype(int)
prediction