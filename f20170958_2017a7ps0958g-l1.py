import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.formula.api as smf

import statsmodels.api as sm



from sklearn import linear_model

from sklearn import neighbors

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_selection import VarianceThreshold



from sklearn.preprocessing import StandardScaler



%matplotlib inline
train = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
train.describe()
X = train.drop('label',axis=1)

y = train['label']
correlated_features = set()

correlation_matrix = train.corr()



for i in range(len(correlation_matrix .columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)

            

X_train = X.drop(labels=correlated_features,axis=1)
constant_filter = VarianceThreshold(threshold=0)

constant_filter.fit(X_train)

constant_columns = [column for column in X_train.columns

                    if column not in X_train.columns[constant_filter.get_support()]]



X_train = X_train.drop(labels=constant_columns,axis=1)
qconstant_filter = VarianceThreshold(threshold=0.01)

qconstant_filter.fit(X_train)

qconstant_columns = [column for column in X_train.columns

                    if column not in X_train.columns[qconstant_filter.get_support()]]



X_train = X_train.drop(labels=qconstant_columns,axis=1)
X_Train=X_train.values

X_Train=np.asarray(X_Train)



# Finding normalised array of X_Train

X_std=StandardScaler().fit_transform(X_Train)
from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,16,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA

sklearn_pca=PCA(n_components=16)

X_Train=sklearn_pca.fit_transform(X_std)
number_of_samples = len(y)

np.random.seed(0)

random_indices = np.random.permutation(number_of_samples)

num_training_samples = int(number_of_samples*0.75)

x_train = X_Train[random_indices[:num_training_samples]]

y_train=y[random_indices[:num_training_samples]]

x_test=X_Train[random_indices[num_training_samples:]]

y_test=y[random_indices[num_training_samples:]]

y_Train=list(y_train)
model=linear_model.Ridge()

model.fit(x_train,y_train)

y_predict=model.predict(x_train)



error=0

for i in range(len(y_Train)):

    if(y_Train[i]!=0):

        error+=(abs(y_Train[i]-y_predict[i])/y_Train[i])

train_error_ridge=error/len(y_Train)*100

print("Train error = "'{}'.format(train_error_ridge)+" percent in Ridge Regression")



Y_test=model.predict(x_test)

y_Predict=list(y_test)



error=0

for i in range(len(y_test)):

    if(y_Predict[i]!=0):

        error+=(abs(y_Predict[i]-Y_test[i])/y_Predict[i])

test_error_ridge=error/len(Y_test)*100

print("Test error = "'{}'.format(test_error_ridge)+" percent in Ridge Regression")
#import matplotlib

#matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



#preds = pd.DataFrame({"preds":model.predict(x_train), "true":y_train})

#preds["residuals"] = preds["true"] - preds["preds"]

#preds.plot(x = "preds", y = "residuals",kind = "scatter")

#plt.title("Residual plot in Ridge Regression")
n_neighbors=5

knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')

knn.fit(x_train,y_train)

y1_knn=knn.predict(x_train)

y1_knn=list(y1_knn)



error=0

for i in range(len(y_train)):

    if(y_Train[i]!=0):

        error+=(abs(y1_knn[i]-y_Train[i])/y_Train[i])

train_error_knn=error/len(y_Train)*100

print("Train error = "+'{}'.format(train_error_knn)+" percent"+" in Knn algorithm")



y2_knn=knn.predict(x_test)

y2_knn=list(y2_knn)

error=0

for i in range(len(y_test)):

    error+=(abs(y2_knn[i]-Y_test[i])/Y_test[i])

test_error_knn=error/len(Y_test)*100

print("Test error = "'{}'.format(test_error_knn)+" percent"+" in knn algorithm")
#matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

#preds = pd.DataFrame({"preds":knn.predict(x_train), "true":y_train})

#preds["residuals"] = preds["true"] - preds["preds"]

#preds.plot(x = "preds", y = "residuals",kind = "scatter")

#plt.title("Residual plot in Knn")
reg = linear_model.BayesianRidge()

reg.fit(x_train,y_train)

y1_reg=reg.predict(x_train)

y1_reg=list(y1_reg)

y2_reg=reg.predict(x_test)

y2_reg=list(y2_reg)



error=0

for i in range(len(y_train)):

    if(y_Train[i]!=0):

        error+=(abs(y1_reg[i]-y_Train[i])/y_Train[i])

train_error_bay=error/len(y_Train)*100

print("Train error = "+'{}'.format(train_error_bay)+" percent"+" in Bayesian Regression")



error=0

for i in range(len(y_test)):

    error+=(abs(y2_reg[i]-Y_test[i])/Y_test[i])

test_error_bay=(error/len(Y_test))*100

print("Test error = "+'{}'.format(test_error_bay)+" percent"+" in Bayesian Regression")
#matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

#preds = pd.DataFrame({"preds":reg.predict(x_train), "true":y_train})

#preds["residuals"] = preds["true"] - preds["preds"]

#preds.plot(x = "preds", y = "residuals",kind = "scatter")

#plt.title("Residual plot in Bayesian Regression")
from sklearn import tree



dec = tree.DecisionTreeRegressor(max_depth=1)

dec.fit(x_train,y_train)

y1_dec=dec.predict(x_train)

y1_dec=list(y1_dec)

y2_dec=dec.predict(x_test)

y2_dec=list(y2_dec)



error=0

for i in range(len(y_train)):

    if(y_Train[i]!=0):

        error+=(abs(y1_dec[i]-y_Train[i])/y_Train[i])

train_error_tree=error/len(y_Train)*100

print("Train error = "+'{}'.format(train_error_tree)+" percent"+" in Decision Tree Regressor")



error=0

for i in range(len(y_test)):

    error+=(abs(y1_dec[i]-Y_test[i])/Y_test[i])

test_error_tree=error/len(Y_test)*100

print("Test error = "'{}'.format(test_error_tree)+" percent in Decision Tree Regressor")
#matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

#preds = pd.DataFrame({"preds":dec.predict(x_train), "true":y_train})

#preds["residuals"] = preds["true"] - preds["preds"]

#preds.plot(x = "preds", y = "residuals",kind = "scatter")

#plt.title("Residual plot in Decision Tree")
'''

from sklearn.model_selection import GridSearchCV



grid_params = {

    'n_neighbors': [5,11,17],

    'weights': ['uniform','distance'],

    'metric': ['euclidean']

}



gs = GridSearchCV(neighbors.KNeighborsRegressor(), grid_params)



gs_results = gs.fit(x_train,y_train)



knn_best = gs_results.best_estimator_ 



y1_knn=knn_best.predict(x_train)

y1_knn=list(y1_knn)



error=0

for i in range(len(y_train)):

    if(y_Train[i]!=0):

        error+=(abs(y1_knn[i]-y_Train[i])/y_Train[i])

train_error_knn=error/len(y_Train)*100

print("Train error = "+'{}'.format(train_error_knn)+" percent"+" in Knn algorithm")



y2_knn=knn_best.predict(x_test)

y2_knn=list(y2_knn)

error=0

for i in range(len(y_test)):

    error+=(abs(y2_knn[i]-Y_test[i])/Y_test[i])

test_error_knn=error/len(Y_test)*100

print("Test error = "'{}'.format(test_error_knn)+" percent"+" in knn algorithm")



'''
X_test = test.drop(labels=correlated_features,axis=1)

X_test = X_test.drop(labels=constant_columns,axis=1)

X_test = X_test.drop(labels=qconstant_columns,axis=1)



X_test_std=StandardScaler().fit_transform(X_test)

X_Test=sklearn_pca.fit_transform(X_test_std)



y_pred = knn.predict(X_Test)
submission = pd.DataFrame({'id':test['id'],'label':y_pred})

submission.describe()
submission.to_csv('Sub1.csv',index=False)