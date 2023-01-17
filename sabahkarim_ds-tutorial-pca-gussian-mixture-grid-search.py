import numpy as np 

import pandas as pd

from sklearn.metrics import accuracy_score 

from sklearn.model_selection import cross_val_score



import os

print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv',header = None)

train_labels = pd.read_csv('../input/trainLabels.csv',header = None)

test_data =  pd.read_csv('../input/test.csv',header = None)
train_data.head()
train_data.shape,test_data.shape,train_labels.shape
train_data.describe()
datasetHasNan = False

if train_data.count().min() == train_data.shape[0] and test_data.count().min() == test_data.shape[0] :

    print('There are no missing values.') 

else:

    datasetHasNan = True

    print('Yes, we have missing values')



# now list items    

if datasetHasNan == True:

    nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 

    print('--'*40)

    print('Nan in the data sets')

    print(nas[nas.sum(axis=1) > 0])
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(x_train,y_train.values.ravel())

predicted= model.predict(x_test)

print('Naive Bayes',accuracy_score(y_test, predicted))



#KNN

from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier()

knn_model.fit(x_train,y_train.values.ravel())

predicted= knn_model.predict(x_test)

print('KNN',accuracy_score(y_test, predicted))



#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 22)

rfc_model.fit(x_train,y_train.values.ravel())

predicted = rfc_model.predict(x_test)

print('Random Forest',accuracy_score(y_test,predicted))



#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression(random_state=0, solver='sag')

lr_model.fit(x_train,y_train.values.ravel())

lr_predicted = lr_model.predict(x_test)

print('Logistic Regression',accuracy_score(y_test, lr_predicted))



#SVM

from sklearn.svm import SVC



svc_model = SVC(gamma = 'auto')

svc_model.fit(x_train,y_train.values.ravel())

svc_predicted = svc_model.predict(x_test)

print('SVM',accuracy_score(y_test, svc_predicted))



#DECISON TREE

from sklearn.tree import DecisionTreeClassifier



dtree_model = DecisionTreeClassifier()

dtree_model.fit(x_train,y_train.values.ravel())

dtree_predicted = dtree_model.predict(x_test)

print('Decision Tree',accuracy_score(y_test, dtree_predicted))



#XGBOOST

from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train,y_train.values.ravel())

xgb_predicted = xgb.predict(x_test)

print('XGBoost',accuracy_score(y_test, xgb_predicted))

from sklearn.preprocessing import StandardScaler, Normalizer



std = StandardScaler()

std_train_data = std.fit_transform(train_data)



norm = Normalizer()

norm_train_data = norm.fit_transform(train_data)
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()

#nb_model.fit(x_norm_train,y_train.values.ravel())

#nb_predicted= nb_model.predict(x_norm_test)

#print('Naive Bayes',accuracy_score(y_test, nb_predicted))

print('Naive Bayes',cross_val_score(nb_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#KNN

from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors = 5)

#knn_model.fit(x_norm_train,y_train.values.ravel())

#knn_predicted= knn_model.predict(x_norm_test)

#print('KNN',accuracy_score(y_test, knn_predicted))

print('KNN',cross_val_score(knn_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)

#rfc_model.fit(x_norm_train,y_train.values.ravel())

#rfc_predicted = rfc_model.predict(x_norm_test)

#print('Random Forest',accuracy_score(y_test,rfc_predicted))

print('Random Forest',cross_val_score(rfc_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#SVM

from sklearn.svm import SVC



svc_model = SVC(gamma = 'auto')

#svc_model.fit(x_norm_train,y_train.values.ravel())

#svc_predicted = svc_model.predict(x_norm_test)

#print('SVM',accuracy_score(y_test, svc_predicted))

print('SVM',cross_val_score(svc_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#DECISION TREE

from sklearn.tree import DecisionTreeClassifier



dtree_model = DecisionTreeClassifier()

#dtree_model.fit(x_norm_train,y_train.values.ravel())

#dtree_predicted = dtree_model.predict(x_norm_test)

#print('Decision Tree',accuracy_score(y_test, dtree_predicted))

print('Decision Tree',cross_val_score(dtree_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#XGBOOST

from xgboost import XGBClassifier



xgb = XGBClassifier()

#xgb.fit(x_norm_train,y_train.values.ravel())

#xgb_predicted = xgb.predict(x_norm_test)

#print('XGBoost',accuracy_score(y_test, xgb_predicted))

print('XGBoost',cross_val_score(xgb,norm_train_data, train_labels.values.ravel(), cv=10).mean())
from sklearn.decomposition import PCA



pca = PCA(0.85, whiten=True)

pca_train_data = pca.fit_transform(train_data)

print(pca_train_data.shape,'\n')



explained_variance = pca.explained_variance_ratio_ 

print(explained_variance)
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()

print('Naive Bayes',cross_val_score(nb_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#KNN

from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors = 5)

#knn_model.fit(pca_train_data,y_train.values.ravel())

#knn_predicted= knn_model.predict(x_norm_test)

#print('KNN',accuracy_score(y_test, knn_predicted))

print('KNN',cross_val_score(knn_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)

#rfc_model.fit(pca_train_data,y_train.values.ravel())

#rfc_predicted = rfc_model.predict(x_norm_test)

#print('Random Forest',accuracy_score(y_test,rfc_predicted))

print('Random Forest',cross_val_score(rfc_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#SVM

from sklearn.svm import SVC



svc_model = SVC(gamma = 'auto')

#svc_model.fit(x_norm_train,y_train.values.ravel())

#svc_predicted = svc_model.predict(x_norm_test)

#print('SVM',accuracy_score(y_test, svc_predicted))

print('SVM',cross_val_score(svc_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#DECISION TREE

from sklearn.tree import DecisionTreeClassifier



dtree_model = DecisionTreeClassifier()

#dtree_model.fit(x_norm_train,y_train.values.ravel())

#dtree_predicted = dtree_model.predict(x_norm_test)

#print('Decision Tree',accuracy_score(y_test, dtree_predicted))

print('Decision Tree',cross_val_score(dtree_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#XGBOOST

from xgboost import XGBClassifier



xgb = XGBClassifier()

#xgb.fit(x_norm_train,y_train.values.ravel())

#xgb_predicted = xgb.predict(x_norm_test)

#print('XGBoost',accuracy_score(y_test, xgb_predicted))

print('XGBoost',cross_val_score(xgb,pca_train_data, train_labels.values.ravel(), cv=10).mean())
# Importing libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture

from sklearn.svm import SVC



X = np.r_[train_data,test_data]

print('X shape :',X.shape)

print('\n')



# USING THE GAUSSIAN MIXTURE MODEL 



#The Bayesian information criterion (BIC) can be used to select the number of components in a Gaussian Mixture in an efficient way. 

#In theory, it recovers the true number of components only in the asymptotic regime

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)



#The GaussianMixture comes with different options to constrain the covariance of the difference classes estimated: 

# spherical, diagonal, tied or full covariance.

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(X)

        bic.append(gmm.aic(X))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(X)

gmm_train = best_gmm.predict_proba(train_data)

gmm_test = best_gmm.predict_proba(test_data)
##### ------------------------ Random Forest Classifier -------------------------- #####

rfc = RandomForestClassifier(random_state=99)



#USING GRID SEARCH

#The first step you need to perform is to create a dictionary of all the parameters and their corresponding set of values that you want to test for best performance. 

n_estimators = [10, 50, 100, 200,400]

max_depth = [3, 10, 20, 40]

param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)

#in the above script we want to find which value (out of 10, 50, 100, 200,400) provides the highest accuracy.

#The Grid Search algorithm basically tries all possible combinations of parameter values and returns the combination with the highest accuracy.



#create an instance of the GridSearchCV class

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 10,scoring='accuracy',n_jobs=-1).fit(gmm_train, train_labels.values.ravel())

#Using 10-folds CV whereas a value of -1 for n_jobs parameter means that use all available computing power.



rfc_best = grid_search_rfc.best_estimator_

print('Random Forest Best Score',grid_search_rfc.best_score_)

print('Random Forest Best Parmas',grid_search_rfc.best_params_)

print('Random Forest Accuracy',cross_val_score(rfc_best,gmm_train, train_labels.values.ravel(), cv=10).mean())

print('--'*40,'\n')



##### -------------------------- KNN -------------------------- ##### 

knn = KNeighborsClassifier()



#USING GRID SEARCH

#First off, the n_neighbors should always be an odd number. You can choose an even number, but in the case of a tie vote, 

#the decision on which class to assign will be done randomly when weights is set to uniform. By choosing an odd number, there are no ties.

n_neighbors=[3,5,6,7,8,9,10]

param_grid = dict(n_neighbors=n_neighbors)

# Another important thing to note is that when you do a GridSearch, you’re running many more models than when you simply fit and score. It’s important to set 

# verbose so you’ll get feedback on the model and know how long it may take to finish. 

# kNN can take a long time to complete as it measures the individual distances for each point in the test set.



#create an instance of the GridSearchCV class

grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train,train_labels.values.ravel())

knn_best = grid_search_knn.best_estimator_

print('KNN Best Score', grid_search_knn.best_score_)

print('KNN Best Params',grid_search_knn.best_params_)

print('KNN Accuracy',cross_val_score(knn_best,gmm_train, train_labels.values.ravel(), cv=10).mean())

print('--'*40,'\n')



##### -------------------------- SVM -------------------------- ##### 

svc = SVC()



#USING GRID SEARCH

#setup a parameter grid (using multiples of 10’s is a good place to start) and then pass the algorithm, parameter grid and number of cross validations to the GridSearchCV method.

parameters = [{'kernel':['linear'],'C':[1,10,100]},

              {'kernel':['rbf'],'C':[1,10,100],'gamma':[0.05,0.0001,0.01,0.001]}]



#create an instance of the GridSearchCV class

grid_search_svm = GridSearchCV(estimator=svc, param_grid=parameters, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train, train_labels.values.ravel())

svm_best = grid_search_svm.best_estimator_

print('SVM Best Score',grid_search_svm.best_score_)

print('SVM Best Params',grid_search_svm.best_params_)

print('SVM Accuracy',cross_val_score(svm_best,gmm_train, train_labels.values.ravel(), cv=10).mean())
# Fitting our model

rfc_best.fit(gmm_train,train_labels.values.ravel())

pred  = rfc_best.predict(gmm_test)

rfc_best_pred = pd.DataFrame(pred)



rfc_best_pred.index += 1



# FRAMING OUR SOLUTION

rfc_best_pred.columns = ['Solution']

rfc_best_pred['Id'] = np.arange(1,rfc_best_pred.shape[0]+1)

rfc_best_pred = rfc_best_pred[['Id', 'Solution']]



rfc_best_pred.to_csv('Submission.csv',index=False)