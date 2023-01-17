# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt #Plotting
import seaborn as sns
# Scaling preprocessing library
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing
from sklearn.preprocessing import Imputer
# Math Library
from math import ceil
from functools import reduce
# Boosting Libraries
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the dataset
train = pd.read_csv('../input/train.csv',low_memory=False)
train.head()
# Importing the dataset
test = pd.read_csv('../input/test.csv',low_memory=False)
test.head()
#Filling for NaN values
train = train.fillna(0)
test = test.fillna(0)
train.head()
#Removal of first row
train= train[1:]
#Removal of first row
test= test[1:]
#Feature Selection
x_train = train.loc[:, train.columns != 'label'].values.astype(int)
y_train = train.iloc[:, -1].values.astype(int)
x_train
y_train.astype(float)
x_test =test.iloc[:, test.columns != 'label'] .values.astype(int)
x_test
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
mean_vec = np.mean(X_train, axis=0)
cov_mat = (X_train - mean_vec).T.dot((X_train - mean_vec)) / (X_train.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
mean_vec = np.mean(X_test, axis=0)
cov_mat1 = (X_test - mean_vec).T.dot((X_test - mean_vec)) / (X_test.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat1)
print('NumPy covariance matrix: \n%s' %np.cov(X_train.T))
print('NumPy covariance matrix: \n%s' %np.cov(X_test.T))
#Plotting of covariance matrix
plt.figure(figsize=(20,20))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')
#Generation of Eigenvectors and Eigenvalues from train covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
#Generation of Eigenvalues and Eigenvectors from test covariance matrix
eig_vals1, eig_vecs1 = np.linalg.eig(cov_mat1)

print('Eigenvectors \n%s' %eig_vecs1)
print('\nEigenvalues \n%s' %eig_vals1)
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs1 = [(np.abs(eig_vals1[i]), eig_vecs1[:,i]) for i in range(len(eig_vals1))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs1.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs1:
    print(i[0])
#Reshaping of Eigenpairs Matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(55,1), 
                      eig_pairs[1][1].reshape(55,1)
                    ))
print('Matrix W:\n', matrix_w)
#Reshaping Test eigenpairs Matrix
matrix_w1 = np.hstack((eig_pairs1[0][1].reshape(55,1), 
                      eig_pairs1[1][1].reshape(55,1)
                    ))
print('Matrix W:\n', matrix_w1)
Y = X_train.dot(matrix_w)
Y
Y1 = X_test.dot(matrix_w1)
Y1
#Principal component analysis for feature column dropping
from sklearn.decomposition import PCA
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,55,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA
pca = PCA().fit(X_test)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,55,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
#Dropping of columns from where covariance is almost 1.0
from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=50)
X_pca_train = sklearn_pca.fit_transform(X_train)
from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=50)
X_pca_test = sklearn_pca.fit_transform(X_test)
print(X_pca_train)
print(X_pca_test)
X_pca_train.shape
#Splitting the train set into training data and validation data
trainX, valX, trainY, valY = train_test_split(X_pca_train, y_train, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
accuracy_score(y_train,y_pred)

score =clf.predict_proba(X_test)
def getModel(arr):
    model=Sequential()
    for i in range(len(arr)):
        if i!=0 and i!=len(arr)-1:
            if i==1:
                model.add(Dense(arr[i],input_dim=arr[0],kernel_initializer='normal', activation='relu'))
            else:
                model.add(Dense(arr[i],activation='relu'))
    model.add(Dense(arr[-1],kernel_initializer='normal',activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])
    return model
#Define a model of 5 dense layers
Model=getModel([50,50,70,40,1])
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
#Fitting the Model
Model.fit(np.array(trainX),np.array(trainY),epochs=6,callbacks=[plot_losses])

#Accuracy Score
scores=Model.evaluate(np.array(valX),np.array(valY))
print(scores)
#Probability Prediction
predY=Model.predict_proba(np.array(X_pca_test))

# Uncomment the whole section and run it 
#param_grid = { 
 #   'n_estimators': [200, 500],
  #  'max_features': ['auto', 'sqrt', 'log2'],
   # 'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy']
#} 
#CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
#CV_rfc.fit(x_pca_train, y_train)
#CV_rfc.best_params_
#rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
#rfc1.fit(x_pca_train, y_train)
#pred=rfc1.predict_proba(x_pca_test)