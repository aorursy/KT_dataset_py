# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.decomposition import PCA
train = pd.read_csv("../input/detecting-anomalies-in-water-manufacturing/Train.csv")
test = pd.read_csv("../input/detecting-anomalies-in-water-manufacturing/Test.csv")
print('Train data shape:', train.shape)
print('Test data shape:',test.shape)
train.head()
X= train.drop(["Class"], axis=1)
y= train["Class"]
ss=StandardScaler()
X2=ss.fit_transform(X)
X2
pca=PCA(svd_solver="arpack",random_state=42,tol=0.5)
pca.fit(X2)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=7 )
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('pca_lr', PCA(n_components=20)), ('LR',LogisticRegression())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('pca_nb', PCA(n_components=20)),('NB',GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('pca_knn', PCA(n_components=20)),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('pca_dtc', PCA(n_components=20)),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('pca_gbc', PCA(n_components=20)),('GBM', GradientBoostingClassifier())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('pca_rf', PCA(n_components=20)),('RF', RandomForestClassifier())])))
pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('pca_ada', PCA(n_components=20)),('Ada', AdaBoostClassifier())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('pca_etc', PCA(n_components=20)),('ET', ExtraTreesClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('pca_svc', PCA(n_components=20)),('SVM', SVC())])))
pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('pca_xgb', PCA(n_components=20)),('XGB', XGBClassifier())])))
pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('pca_nn', PCA(n_components=20)),('MLP', MLPClassifier())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Algorithm comparison
fig = plt.figure(figsize=(18,5))
fig.suptitle('Model Selection by comparision')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
steps=[("norm",MinMaxScaler()),("pca",PCA(n_components=300)),("r",RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10,max_features="log2"))]
rf=Pipeline(steps=steps)
rf.fit(X_train, y_train)
print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))
preds= rf.predict(X_test)
print('Classification report: \n', classification_report(y_test, preds))
plot_roc_curve(rf, X, y)
y.value_counts()
from sklearn.utils import resample 
maj_class=train[train["Class"]==0]
min_class=train[train["Class"]==1]
resamp_minclass=resample(min_class,n_samples=1620,replace=True,random_state=42)
print('Majority class shape:', maj_class.shape)
print('Minority shape:', resamp_minclass.shape)
train_new= pd.concat([maj_class,resamp_minclass])
train_new.head()
X_res= train_new.drop(["Class"], axis=1)
y_res= train_new["Class"]
X_train_res, X_test_res, y_train_res, y_test_res= train_test_split(X_res, y_res, test_size=0.3, random_state=2)
steps=[("norm",MinMaxScaler()),("pca",PCA(n_components=300)),("r",RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10,max_features="log2"))]
rf_model=Pipeline(steps=steps)
rf_model.fit(X_train_res, y_train_res)
print(rf_model.score(X_train_res,y_train_res))
print(rf_model.score(X_test_res,y_test_res))
rf_preds= rf_model.predict(X_test_res)
print('Classification report: \n', classification_report(y_test_res, rf_preds))
plot_roc_curve(rf_model, X_res, y_res)
stepz = [("scal",MinMaxScaler())]
pipe=Pipeline(steps=stepz)
Test=pipe.fit_transform(test)
test_predictions= rf_model.predict(Test)
test_predictions
submission_df= pd.DataFrame(test_predictions, columns=['Class'])
submission_df['Class']= submission_df['Class'].astype('float64')
submission_df.head()
submission_df.info()
submission_df.to_csv('/kaggle/working/RF_predictions.csv', index=False)
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.models import Sequential,load_model
Test.shape
model=Sequential()
model.add(Dense(500,input_shape=(1558,)))
model.add(Activation("relu"))
model.add(Dense(750))
model.add(Activation("relu"))
model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dense(400))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', metrics=['AUC'], optimizer='adam')
history= model.fit(X_train_res, y_train_res,
          batch_size=150, epochs=100,
          verbose=2,
          validation_data=(X_test_res, y_test_res))
model_metrics= model.evaluate(X_test_res, y_test_res)
print('Model Loss:', model_metrics[0])
print("Model AUC Score: {:.2%}".format(model_metrics[1]))
