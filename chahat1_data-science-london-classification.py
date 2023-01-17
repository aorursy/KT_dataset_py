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
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
# NAIBE BAYES

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



rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)

rfc_model.fit(x_train,y_train.values.ravel())

predicted = rfc_model.predict(x_test)

print('Random Forest',accuracy_score(y_test,predicted))



#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression(solver = 'saga')

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



norm = Normalizer()

#x_norm_train = norm.fit_transform(x_train)

#x_norm_test = norm.transform(x_test)

norm_train_data = norm.fit_transform(train_data)
# NAIBE BAYES

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



#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression(solver = 'saga')

#lr_model.fit(x_norm_train,y_train.values.ravel())

#lr_predicted = lr_model.predict(x_norm_test)

#print('Logistic Regression',accuracy_score(y_test, lr_predicted))

print('Logistic Regression',cross_val_score(lr_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#SVM

from sklearn.svm import SVC



svc_model = SVC(gamma = 'auto')

#svc_model.fit(x_norm_train,y_train.values.ravel())

#svc_predicted = svc_model.predict(x_norm_test)

#print('SVM',accuracy_score(y_test, svc_predicted))

print('SVM',cross_val_score(svc_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())



#DECISON TREE

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



pca  = PCA(n_components=12)

#x_train = pca.fit_transform(x_train)

#x_test = pca.transform(x_test)

pca_train_data = pca.fit_transform(train_data)

explained_variance = pca.explained_variance_ratio_ 

print(explained_variance)
pca_train_data.shape
# NAIBE BAYES

from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()

#nb_model.fit(pca_train_data,y_train.values.ravel())

#nb_predicted= nb_model.predict(x_norm_test)

#print('Naive Bayes',accuracy_score(y_test, nb_predicted))

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



#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression(solver = 'saga')

#lr_model.fit(pca_train_data,y_train.values.ravel())

#lr_predicted = lr_model.predict(x_norm_test)

#print('Logistic Regression',accuracy_score(y_test, lr_predicted))

print('Logistic Regression',cross_val_score(lr_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#SVM

from sklearn.svm import SVC



svc_model = SVC(gamma = 'auto')

#svc_model.fit(x_norm_train,y_train.values.ravel())

#svc_predicted = svc_model.predict(x_norm_test)

#print('SVM',accuracy_score(y_test, svc_predicted))

print('SVM',cross_val_score(svc_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())



#DECISON TREE



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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture

from sklearn.svm import SVC



x_all = np.r_[train_data,test_data]

print('x_all shape :',x_all.shape)



# USING THE GAUSSIAN MIXTURE MODEL 

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(x_all)

        bic.append(gmm.aic(x_all))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(x_all)

gmm_train = best_gmm.predict_proba(train_data)

gmm_test = best_gmm.predict_proba(test_data)





#Random Forest Classifier

rfc = RandomForestClassifier(random_state=99)



#USING GRID SEARCH

n_estimators = [10, 50, 100, 200,400]

max_depth = [3, 10, 20, 40]

param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)



grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 10,scoring='accuracy',n_jobs=-1).fit(gmm_train, train_labels.values.ravel())

rfc_best = grid_search_rfc.best_estimator_

print('Random Forest Best Score',grid_search_rfc.best_score_)

print('Random Forest Best Parmas',grid_search_rfc.best_params_)

print('Random Forest Accuracy',cross_val_score(rfc_best,gmm_train, train_labels.values.ravel(), cv=10).mean())



#KNN 

knn = KNeighborsClassifier()



#USING GRID SEARCH

n_neighbors=[3,5,6,7,8,9,10]

param_grid = dict(n_neighbors=n_neighbors)



grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train,train_labels.values.ravel())

knn_best = grid_search_knn.best_estimator_

print('KNN Best Score', grid_search_knn.best_score_)

print('KNN Best Params',grid_search_knn.best_params_)

print('KNN Accuracy',cross_val_score(knn_best,gmm_train, train_labels.values.ravel(), cv=10).mean())



#SVM

svc = SVC()



#USING GRID SEARCH

parameters = [{'kernel':['linear'],'C':[1,10,100]},

              {'kernel':['rbf'],'C':[1,10,100],'gamma':[0.05,0.0001,0.01,0.001]}]

grid_search_svm = GridSearchCV(estimator=svc, param_grid=parameters, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train, train_labels.values.ravel())

svm_best = grid_search_svm.best_estimator_

print('SVM Best Score',grid_search_svm.best_score_)

print('SVM Best Params',grid_search_svm.best_params_)

print('SVM Accuracy',cross_val_score(svm_best,gmm_train, train_labels.values.ravel(), cv=10).mean())

rfc_best.fit(gmm_train,train_labels.values.ravel())

pred  = rfc_best.predict(gmm_test)

rfc_best_pred = pd.DataFrame(pred)



rfc_best_pred.index += 1



rfc_best_pred.columns = ['Solution']

rfc_best_pred['Id'] = np.arange(1,rfc_best_pred.shape[0]+1)

rfc_best_pred = rfc_best_pred[['Id', 'Solution']]



rfc_best_pred.to_csv('Submission_GMM_RFC.csv',index=False)