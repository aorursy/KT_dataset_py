import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from IPython.display import display

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

import warnings

warnings.filterwarnings('ignore')
#import data into pandas dataframe

data = pd.read_csv("../input/winequality-red.csv")



#display first 5 lines

data.head()



#print data properties

print('Data Shape: {}'.format(data.shape))



display(data.describe())

display(data.info())

sns.countplot(data['quality'],label="Count")
plt.figure(figsize=(10,10))

corr_mat=sns.heatmap(data.corr(method='spearman'),annot=True,cbar=True,

            cmap='viridis', vmax=1,vmin=-1,

            xticklabels=data.columns,yticklabels=data.columns)

corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)
bins = (1, 6.5, 8.5)

quality_level = ['low', 'high']



data['quality'] = pd.cut(data['quality'], bins = bins, labels = quality_level)

sns.countplot(data['quality'])
#Extract data and label target

X = data.iloc[:,0:11]

le = LabelEncoder().fit(data['quality'])

y = le.transform(data['quality'])



X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)



#Scale data

scaler = MinMaxScaler()

scaler.fit(X_train,y_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
sns.pairplot(data, hue = "quality", diag_kind='kde')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

param_grid = {'xgb__n_estimators': [500], 

              'xgb__max_depth': [2,3,4,5], 

              'xgb__alpha': [0.001,0.01,0.1,1],

              'xgb__min_samples_leaf': [1,2,3]}

xgb = XGBClassifier(random_state=0)

pipe = Pipeline([("scaler",MinMaxScaler()), ("xgb",xgb)])

grid_xgb = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)

grid_xgb.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.3f}".format(grid_xgb.best_score_))

print("Test set score: {:.3f}".format(grid_xgb.score(X_test,y_test)))

print("Best parameters: {}".format(grid_xgb.best_params_))



conf_mat_xgb = confusion_matrix(y_test, grid_xgb.predict(X_test))

sns.heatmap(conf_mat_xgb, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=le.classes_, xticklabels=le.classes_)



# Feature importance

xgb = XGBClassifier(random_state=0, max_depth=grid_xgb.best_params_['xgb__max_depth'],

                                 n_estimators=grid_xgb.best_params_['xgb__n_estimators'],

                                 alpha=grid_xgb.best_params_['xgb__alpha'],

                                 min_samples_leaf = grid_xgb.best_params_['xgb__min_samples_leaf'])

xgb.fit(X_train_scaled,y_train)

plt.figure()

plt.bar(np.arange(X.shape[1]), xgb.feature_importances_)

plt.xticks(np.arange(X.shape[1]), X.columns, rotation=90)

plt.title('Feature Importance')



# Classification Report

print(classification_report(y_test, xgb.predict(X_test_scaled), target_names=le.classes_))

pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",SVC(random_state=0))])

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10],

              'svm__gamma': [0.001, 0.01, 0.1, 1, 10],

              'svm__kernel': ['linear', 'rbf']}

grid_svm = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy',n_jobs=-1)

grid_svm.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.3f}".format(grid_svm.best_score_))

print("Test set score: {:.3f}".format(grid_svm.score(X_test,y_test)))

print("Best parameters: {}".format(grid_svm.best_params_))





#SVM Cofusion matrix

conf_mat_svm = confusion_matrix(y_test, grid_svm.predict(X_test_scaled))

sns.heatmap(conf_mat_svm, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=le.classes_, xticklabels=le.classes_)



# Classification Report

print(classification_report(y_test, grid_svm.predict(X_test_scaled), target_names=le.classes_))

param = grid_svm.best_params_

svm = SVC(gamma = param["svm__gamma"], C = param["svm__C"], kernel=param["svm__kernel"], probability=True, random_state=99)



xgb = XGBClassifier(random_state=0, max_depth=grid_xgb.best_params_['xgb__max_depth'],

                                 n_estimators=grid_xgb.best_params_['xgb__n_estimators'],

                                 alpha=grid_xgb.best_params_['xgb__alpha'],

                                 min_samples_leaf = grid_xgb.best_params_['xgb__min_samples_leaf'])



ensemble = VotingClassifier(estimators=[('clf1',svm), ('clf2',xgb)], voting='soft', weights=[1,1])

ensemble.fit(X_train_scaled, y_train)

print("Ensemble test score: {:.3f}".format(ensemble.score(X_test_scaled, y_test)))



conf_mat_ens = confusion_matrix(y_test, ensemble.predict(X_test_scaled))

sns.heatmap(conf_mat_ens, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=le.classes_, xticklabels=le.classes_)



cv_score=np.mean(cross_val_score(ensemble,X_test,y_test,cv=kfold))

print("Cross Validation score: {:.3f}".format(cv_score))