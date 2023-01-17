import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, tests

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
tr_data = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

te_data = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
### USING LABEL ENCODING

lab_enc = LabelEncoder()

y = lab_enc.fit_transform(tr_data[:,0])

X_train = [tr_data[i][1].flatten() for i in range(tr_data.shape[0])]

X_test = [te_data[i][1].flatten() for i in range(te_data.shape[0])]
### USING PCA

n_components = 100

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(X_train) 

X_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
### USING SVM CLASSIFIER AND GRID SEARCH

svc = SVC(kernel='rbf', class_weight='balanced')

param_grid = {'C': [1,5,10,50],'gamma': [0.005,0.01,0.05,0.1]}

grid = GridSearchCV(svc, param_grid, cv=5,verbose=3,n_jobs=-1)
grid.fit(X_pca, y)

print(grid.best_params_)

model_selected = grid.best_estimator_

y_pred_1 = model_selected.predict(X_test_pca)

y_pred_1 = lab_enc.inverse_transform(y_pred_1)
### PREPARING OUTPUT DATAFRAME 1

column_names = ['ImageId','Celebrity']

output_df1 = pd.DataFrame(None,columns=column_names)

output_df1['ImageId'] = te_data[:,0]

output_df1['Celebrity'] = y_pred_1

output_df1.to_csv('SVM_PCA_2.csv',index=False)
### PREPARING DATA AS NUMPY ARRAYS

X_2 = []

for i in range(tr_data.shape[0]):

    X_2.append(tr_data[i][1].flatten())



X_te_2 = []

for i in range(te_data.shape[0]):

    X_te_2.append(te_data[i][1].flatten())

    

X_2 = np.array(X_2)

X_te_2 = np.array(X_te_2)
### SCALING THE DATA

mm_scaler = MinMaxScaler()

mm_scaler.fit(X_2)

mm_scaler.fit(X_te_2)

X_2_scal = mm_scaler.transform(X_2)

X_te_2_scal = mm_scaler.transform(X_te_2)

y_train = tr_data[:,0]
### USING SVM AND GRID SEARCH

param_grid_2 = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]

svc2 = SVC()

grid_clf = GridSearchCV(svc, param_grid,n_jobs=-1)

grid_clf.fit(X_2_scal, y_train)

y_pred_2 = grid_clf.predict(X_te_2_scal)
### PREPARING OUTPUT DATAFRAME 2

column_names2 = ['ImageId','Celebrity']

output_df2 = pd.DataFrame(None,columns=column_names2)

output_df2['ImageId'] = te_data[:,0]

output_df2['Celebrity'] = y_pred_2

output_df2.to_csv('SVM_GridSearch.csv',index=False)