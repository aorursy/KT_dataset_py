import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X_train=pd.read_csv('../input/data_set_ALL_AML_train.csv')
X_train.shape
y=pd.read_csv('../input/actual.csv')
y.shape
X_test=pd.read_csv('../input/data_set_ALL_AML_independent.csv')
X_test.shape
X_test.head()
# 1)  Remove "call" columns from training a test
train_keepers = [col for col in X_train.columns if "call" not in col]
test_keepers = [col for col in X_test.columns if "call" not in col]
X_train = X_train[train_keepers]
X_test = X_test[test_keepers]
X_train.shape,X_test.shape
X_train.head()
X_train.head()
X_test.head()
# 2) Transpose
X_train = X_train.T
X_test = X_test.T
X_train.head()
X_train.shape
# 3) Clean up the column names for training data
X_train.columns = X_train.iloc[1]
X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# Clean up the column names for Testing data
X_test.columns = X_test.iloc[1]
X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

X_train.head()
X_train.shape,X_test.shape
# 4) Split into train and test 
X_train = X_train.reset_index(drop=True)
print(X_train.shape)
y_train = y[y.patient <= 38].reset_index(drop=True)
# Subet the rest for testing
X_test = X_test.reset_index(drop=True)
y_test = y[y.patient > 38].reset_index(drop=True)
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.fit_transform(X_test)
X_train.head()
pca=PCA()
pca.fit_transform(X_train_scl)
total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1
k
pca = PCA(n_components=k)
X_train_pca=pca.fit_transform(X_train_scl)
X_test_pca=pca.transform(X_test_scl)
import matplotlib.pyplot as plt
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100
plt.bar(range(k), cum_sum)
plt.title("Around 90% of variance is explained by the 28 features");
X_train_pca.shape, X_test_pca.shape
y_train = y_train.replace({'ALL':0,'AML':1})
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]} 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(X_train_pca, y_train.iloc[:,1])

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
model = svm.SVC(kernel='rbf', C = 10.0, gamma=1e-05)
model.fit(X_train_pca,y_train.iloc[:,1])
pred=model.predict(X_test_pca)
pred
y_test = y_test.replace({'ALL':0,'AML':1})
pred=pred.tolist()
print('Accuracy: ', accuracy_score(y_test.iloc[:,1], pred))
import sklearn
sklearn.metrics.confusion_matrix(y_test.iloc[:,1], pred)
from sklearn.ensemble import GradientBoostingClassifier as XGB
model= XGB(max_depth=5, loss='exponential', n_estimators=50, learning_rate=0.8, random_state=2018)
model.fit(X_train_pca, y_train.iloc[:,1])
pred = model.predict(X_test_pca)
print('Accuracy: ', accuracy_score(y_test.iloc[:,1], pred))
sklearn.metrics.confusion_matrix(y_test.iloc[:,1], pred)
import xgboost
model=xgboost.XGBClassifier()
model.fit(X_train_pca, y_train.iloc[:,1])
pred = model.predict(X_test_pca)
print('Accuracy: ', accuracy_score(y_test.iloc[:,1], pred))
sklearn.metrics.confusion_matrix(y_test.iloc[:,1], pred)