import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.info()
df.shape
df.columns
df.isna().sum()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot('Class', data=df)
plt.title('Distribuição da variável target')
plt.xlabel('Class')
plt.ylabel('Frequencia')
df['Class'].value_counts()
plt.figure(figsize=(14,9))
sns.heatmap(df.corr(), linewidths=0.5)
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

features_all = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

label = ['Class']
from sklearn.model_selection import train_test_split
X, y = df[features], df[label]
X_train, X_test, y_train, y_test =\
     train_test_split(X, y,
                      test_size=0.94,
                      random_state=0,
                      stratify=y)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
num_folds = 10
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier()))
results = []
names = []
# fiz um pipeline
# aplicação do Kfold e dos modelos
# olhar o "model"
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
fig = pyplot.figure()
fig.suptitle('Comparação entre as acurácias')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
pipelines.append(('LDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('NB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
pipelines.append(('SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
fig = pyplot.figure()
fig.suptitle('Escala de Algoritmos - Comparação')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# Random Forest Classifier
from yellowbrick.classifier import ClassificationReport
model_RF =  RandomForestClassifier()
visualizer = ClassificationReport(model, size=(800, 533))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
# Regressão Logística 
model_LR =  LogisticRegression(solver='liblinear')
visualizer = ClassificationReport(model, size=(800, 533))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
# LDA
model_LDA =  LinearDiscriminantAnalysis()
visualizer = ClassificationReport(model, size=(800, 533))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
# Começar por Regressão logística
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)


C = [0.2, 0.4, 0.6, 0.8, 1.0]
fit_intercept = ['True','False']
class_weight = ['dic', 'balanced']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

param_grid = dict(C=C, fit_intercept=fit_intercept, class_weight=class_weight, solver=solver)

model = LogisticRegression()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Random Forest Classifier
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)


criterion = ['gini','entropy']
min_samples_split = [2,3,4]
max_features = ['auto','sqrt','log2']
oob_score = ['True','False']

param_grid = dict(criterion=criterion, min_samples_split=min_samples_split, max_features=max_features, oob_score=oob_score)

model = RandomForestClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# LinearDiscriminantAnalysis
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)


solver = ['svd', 'lsqr', 'eigen']
store_covariance = ['True','False']
tol = [0.0001,0.0002, 0.0003, 0.0004]

param_grid = dict(solver=solver, store_covariance=store_covariance, tol=tol)

model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_LR = LogisticRegression(C= 1.0, class_weight= 'balanced', fit_intercept= True, solver= 'liblinear')
model_LR.fit(rescaledX, y_train)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_RF = RandomForestClassifier(criterion ='entropy', max_features = 'auto', min_samples_split = 4, oob_score = True)
model_RF.fit(rescaledX, y_train)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_LDA = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
model_LDA.fit(rescaledX, y_train)
# Regressão Logistica
rescaledValidationX = scaler.transform(X_test)
predictions = model_LR.predict(rescaledValidationX)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# Random Forest
rescaledValidationX = scaler.transform(X_test)
predictions = model_RF.predict(rescaledValidationX)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# LDA
rescaledValidationX = scaler.transform(X_test)
predictions = model_LDA.predict(rescaledValidationX)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
from imblearn.over_sampling import (RandomOverSampler,ADASYN,BorderlineSMOTE,
                                    KMeansSMOTE,SMOTE,SVMSMOTE)

from imblearn.under_sampling import (RandomUnderSampler,CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                    RepeatedEditedNearestNeighbours,
                                    NeighbourhoodCleaningRule,AllKNN,TomekLinks)
from imblearn.pipeline import Pipeline
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_rus, y_rus)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
cnn = CondensedNearestNeighbour()
X_cnn, y_cnn = cnn.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_cnn, y_cnn)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_enn, y_enn)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
renn = RepeatedEditedNearestNeighbours()
X_renn, y_renn = renn.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_renn, y_renn)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
ncr = NeighbourhoodCleaningRule()
X_ncr, y_ncr = ncr.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_ncr, y_ncr)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
akn = AllKNN()
X_akn, y_akn = akn.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_akn, y_akn)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_tl, y_tl)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_ros, y_ros)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
adn = ADASYN()
X_adn, y_adn = adn.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_adn, y_adn)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
bsm = BorderlineSMOTE()
X_bsm, y_bsm = bsm.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_bsm, y_bsm)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X_train,y_train)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_sm, y_sm)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
smo = SVMSMOTE()
X_smo, y_smo = smo.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_smo, y_smo.values.ravel())
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
kms = KMeansSMOTE()
X_kms, y_kms = kms.fit_resample(X,y)
lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True, tol = 0.0001)
lda.fit(X_kms, y_kms)
pred_test = lda.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
from xgboost import XGBClassifier
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X,y)
xgb = XGBClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.94,random_state=0, stratify=y)
xgb.fit(X_train, y_train.values.ravel())
pred = xgb.predict(X_test)
print('Acurácia: ', accuracy_score(y_test,pred))
print('Classification report:\n', classification_report(y_test, pred))
print('Confusion matrix:\n', confusion_matrix(y_test, pred))