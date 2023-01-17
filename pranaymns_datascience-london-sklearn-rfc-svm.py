import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



seed = 42

np.random.seed(seed)
train_data = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header=None)

train_labels = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header=None, names = ['Label'])

test_data = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header=None)
train_data.head()
train_labels.head()
train_data.shape, train_labels.shape
train_data.describe()
train_labels.hist()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
X, y = train_data, np.ravel(train_labels)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=seed)
def evaluate_model(model):

    

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print('Cross validation score - ', scores.mean()*100)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)



    accuracy = accuracy_score(y_valid, y_pred) 

    print('Validation accuracy - ',accuracy*100)

    

    #Return trained model

    return model
knn = KNeighborsClassifier()



print('\nKNN Classifier ')

knn = evaluate_model(knn)



dtc = DecisionTreeClassifier(random_state=seed)

print('\nDecision Tree Classifier')

dtc = evaluate_model(dtc)



rfc = RandomForestClassifier(n_estimators=10, random_state=seed)

print('\nRandom Forest Classifier')

rfc = evaluate_model(rfc)



svc = SVC(gamma='auto', random_state=seed)

print('\nSVM Classifier')

svc = evaluate_model(svc)





gbc = GradientBoostingClassifier(n_estimators=20, random_state=seed)

print('\nGradient Boosting Classifier')

gbc = evaluate_model(gbc)



adc = AdaBoostClassifier(base_estimator=rfc, n_estimators=30, random_state=seed)

print('\nAdaBoost classifier with Random Forest Classifier')

adc = evaluate_model(adc)

from sklearn.decomposition import PCA



pca = PCA(random_state=seed)



pca.fit(X)



features = range(pca.n_components_)

plt.figure(figsize=(13,6))

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.title('PCA Explained Variance')

plt.show()
pca = PCA(n_components=13)



X_reduced = pca.fit_transform(X)



X_train, X_valid, y_train, y_valid = train_test_split(X_reduced, y, test_size=0.3, random_state = seed)
knn = KNeighborsClassifier()



print('\nKNN Classifier ')

knn = evaluate_model(knn)



dtc = DecisionTreeClassifier(random_state=seed)

print('\nDecision Tree Classifier')

dtc = evaluate_model(dtc)



rfc = RandomForestClassifier(n_estimators=10, random_state=seed)

print('\nRandom Forest Classifier')

rfc = evaluate_model(rfc)



svc = SVC(gamma='auto', random_state=seed)

print('\nSVM Classifier')

svc = evaluate_model(svc)





gbc = GradientBoostingClassifier(n_estimators=20, random_state=seed)

print('\nGradient Boosting Classifier')

gbc = evaluate_model(gbc)



adc = AdaBoostClassifier(base_estimator=rfc, n_estimators=30, random_state=seed)

print('\nAdaBoost classifier with Random Forest Classifier')

adc = evaluate_model(adc)

def perform_grid_search(model, param_grid, cv = 10, scoring='accuracy'):

    

    grid_search_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = cv,scoring=scoring,n_jobs=-1, iid=False)

    grid_search_model.fit(X_train, y_train)





    best_model = grid_search_model.best_estimator_

    print('Best Accuracy :',grid_search_model.best_score_ * 100)

    print('Best Parmas',grid_search_model.best_params_)

    

    y_pred = best_model.predict(X_valid)

    print('Validation Accuracy',accuracy_score(y_valid, y_pred)*100)

    

    return best_model
knn = KNeighborsClassifier()



n_neighbors = [3,4,5,6,7,8,9,10]

param_grid_knn = dict(n_neighbors=n_neighbors)



print('\nKNN Classifier')

knn_best = perform_grid_search(knn, param_grid_knn)



rfc = RandomForestClassifier(random_state=seed)



n_estimators = [10, 50, 100, 200]

max_depth = [3, 10, 15, 30]

param_grid_rfc = dict(n_estimators=n_estimators,max_depth=max_depth)



print('\nRandom Forest Classifier')

rfc_best = perform_grid_search(rfc, param_grid_rfc)





svc = SVC(random_state=seed)



param_grid_svc = [{'kernel':['linear'],'C':[1,10, 50,100]},

              {'kernel':['rbf'],'C':[1,10, 50,100],'gamma':[0.05,0.0001,0.01,0.001]}]



print('\nSVM Classifier')

svc_best = perform_grid_search(svc, param_grid_svc)

from sklearn.mixture import GaussianMixture



X = np.r_[train_data,test_data]



lowest_bic = np.infty

bic = []

n_components_range = range(1, 12)



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
X_train, X_valid, y_train, y_valid = train_test_split(gmm_train, y, test_size=0.3, random_state = seed)
knn = KNeighborsClassifier()



n_neighbors = [3,4,5,6,7,8,9,10]

param_grid_knn = dict(n_neighbors=n_neighbors)



print('\nKNN Classifier')

knn_best = perform_grid_search(knn, param_grid_knn)



rfc_best = RandomForestClassifier(random_state=seed)



n_estimators = [10, 50, 100, 200]

max_depth =  [3, 10, 15, 30]

param_grid_rfc = dict(n_estimators=n_estimators,max_depth=max_depth)



print('\nRandom Forest Classifier')

rfc = perform_grid_search(rfc, param_grid_rfc)





svc = SVC(random_state=seed)



param_grid_svc = [{'kernel':['linear'],'C':[1,10, 50,100]},

              {'kernel':['rbf'],'C':[1,10, 50,100],'gamma':[0.05,0.0001,0.01,0.001]}]



print('\nSVM Classifier')

svc_best = perform_grid_search(svc, param_grid_svc)

pred  = svc_best.predict(gmm_test)

best_pred = pd.DataFrame(pred)



best_pred.index += 1



best_pred.columns = ['Solution']

best_pred['Id'] = np.arange(1, best_pred.shape[0]+1)

best_pred = best_pred[['Id', 'Solution']]



best_pred.head()
best_pred.to_csv('submission.csv',index=False)