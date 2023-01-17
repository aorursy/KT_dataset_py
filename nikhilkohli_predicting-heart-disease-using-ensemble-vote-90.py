import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings("ignore")
df_heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df_heart.head()
df_heart.shape
df_heart = df_heart.rename(columns= {'cp':'chest_pain_type','trestbps':'resting_BP','chol':'serum_cholestoral','fbs':'fasting_blood_sugar','restecg':'resting_ECG',
                                     'thalach':'max_heart_rate','exang':'exercise_induced_angina',
                                     'ca':'major_vessels_count','thal':'thalium_stress'})
df_heart.columns
df_heart.info()

df_heart.isnull().sum()
categorical_cols = []
continous_cols = []

for column in df_heart.columns:
    if(len(df_heart[column].unique()) <= 10):
        categorical_cols.append(column)
    else:
        continous_cols.append(column)
categorical_cols
continous_cols
df_heart_tmp = df_heart.copy()
for cols in categorical_cols:
    if(cols != 'target'):
        df_heart_tmp[cols] = df_heart_tmp[cols].astype('object')
df_heart_tmp.dtypes

df_heart_tmp.describe()

df_heart_tmp.describe(include='object')

df_heart.target.value_counts()

sns.countplot(df_heart['target'])

df_heart.hist(column=continous_cols, figsize=(12,12))

for index, column in enumerate(continous_cols):
    plt.figure(index)
    sns.distplot(df_heart[column])
df_heart[continous_cols].skew()
### Distribution of Categorical Values
df_heart.hist(column=categorical_cols, figsize=(10,10))

df_heart_tmp.describe().columns

sns.pairplot(df_heart_tmp[df_heart_tmp.describe().columns], hue='target')
df_heart.corr()

plt.figure(figsize=(15,8))
sns.heatmap(df_heart.corr(), annot=True, linewidths=1, linecolor='white', fmt=".2f",
                 cmap="YlGnBu")
df_heart.drop('target', axis=1).corrwith(df_heart.target).plot(kind='bar', grid=True, figsize=(10, 7), color='darkgreen')

for index,column in enumerate(continous_cols):
    plt.figure(index, figsize=(8,5))
    sns.boxplot(x=df_heart.target, y=column, data=df_heart, palette='rainbow',linewidth=1)
    plt.title('Relation of {} with target'.format(column), fontsize = 10)
for index,column in enumerate(continous_cols):
    plt.figure(index,figsize=(7,5))
    sns.catplot(x='target', y=column, hue='sex', kind='swarm', data=df_heart, palette='husl')
    plt.title('Relationship of {} with target for each sex'.format(column), fontsize = 10)
categorical_cols.pop(8)

for var in categorical_cols:
    print('Cardinality of {1} is {0}'.format(len(df_heart[var].unique()), var))
for index,column in enumerate(categorical_cols):
    plt.figure(index,figsize=(7,5))
    sns.countplot(x=column, hue='target', data=df_heart, palette='rainbow')
    plt.title('Relation of {} with target'.format(column), fontsize = 10)
categorical_dummy = [
 'chest_pain_type',
 'resting_ECG',
 'slope',
 'major_vessels_count',
 'thalium_stress']
df_heart = pd.get_dummies(df_heart, columns=categorical_dummy, drop_first=True )
df_heart.columns
len(df_heart.columns)

target = df_heart.target
features = df_heart.drop(columns=['target'])
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
features_SS = scaler.fit_transform(features)
features_SS = pd.DataFrame(features_SS, columns=features.columns)
X_train, X_test, Y_train, Y_test = train_test_split(features_SS, target, test_size=0.2, random_state=42)
Y_train.value_counts()

X_train.head()

X_train.shape, X_test.shape

# for each feature, calculate the VIF score
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['features'] = X_train.columns
vif.round(1)
X = X_train.drop(columns=['thalium_stress_3'])
# for each feature, calculate the VIF score
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['features'] = X.columns
vif.round(1)
from sklearn.feature_selection import RFECV, SelectFromModel
from xgboost import XGBClassifier
import time
start = time.time()

rf = RandomForestClassifier(n_estimators=10, random_state=40)
rfe_rf = RFECV(estimator=rf, step=1, cv=5, n_jobs=-1)
rfe_rf.fit_transform(X_train, Y_train)

end = time.time()
print('Time Taken - {}'.format(str(end - start)))
rfe_rf

rfe_rf.support_

rfe_rf_ranks = rfe_rf.ranking_
rfe_rf_ranks
params = {'axes.labelsize': 280,'axes.titlesize':40, 'legend.fontsize': 18, 'xtick.labelsize': 40, 'ytick.labelsize': 50}
plt.figure(figsize=(50,25))
plt.rcParams.update(params)
ax = plt.bar(range(X_train.shape[1]), rfe_rf_ranks, color='green', align = 'center')
ax = plt.title('Feature importance')
ax = plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.show()
feature_idx = rfe_rf.support_
feature_names = X_train.columns[feature_idx]
feature_names
start = time.time()

logit = LogisticRegression()
rfe_logit = RFECV(estimator=logit, step=1, cv=5, n_jobs=-1)
rfe_logit.fit_transform(X_train, Y_train)

end = time.time()

print('Time Taken - {}'.format(str(end - start)))
rfe_logit
rfe_logit.support_
rfe_logit_ranks = rfe_logit.ranking_
rfe_logit_ranks
params = {'axes.labelsize': 280,'axes.titlesize':40, 'legend.fontsize': 18, 'xtick.labelsize': 40, 'ytick.labelsize': 50}
plt.figure(figsize=(50,25))
plt.rcParams.update(params)
ax = plt.bar(range(X_train.shape[1]), rfe_logit_ranks, color='green', align = 'center')
ax = plt.title('Feature importance')
ax = plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.show()
feature_idx2 = rfe_logit.support_
feature_names2 = X_train.columns[feature_idx2]
feature_names2
xgb = XGBClassifier()
select_xg = SelectFromModel(estimator=xgb, threshold='median')
select_xg
select_xg.fit_transform(X_train, Y_train)
feature_idx3 = select_xg.get_support()
feature_names3 = X_train.columns[feature_idx3]
feature_names3
X_train_2 = X_train[feature_names]
X_train_3 = X_train[feature_names2]
X_train_4 = X_train[feature_names3]

X_test_2 = X_test[feature_names]
X_test_3 = X_test[feature_names2]
X_test_4 = X_test[feature_names3]
len(X_train.columns), len(X_train_2.columns), len(X_train_3.columns), len(X_train_4.columns)

logit_clf = LogisticRegression()
logit_clf.fit(X_train, Y_train)

y_pred = logit_clf.predict(X_test)
print('Accuracy Score: ', str(accuracy_score(Y_test, y_pred)))
print('Classification Report: ')
print(classification_report(Y_test, y_pred))
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, Y_train)

y_pred = knn_clf.predict(X_test)
print('Accuracy Score: ', str(accuracy_score(Y_test, y_pred)))
print('Classification Report: ')
print(classification_report(Y_test, y_pred))
svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_clf.fit(X_train, Y_train)

y_pred = svm_clf.predict(X_test)
print('Accuracy Score: ', str(accuracy_score(Y_test, y_pred)))
print('Classification Report: ')
print(classification_report(Y_test, y_pred))
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)

y_pred = dt_clf.predict(X_test)
print('Accuracy Score: ', str(accuracy_score(Y_test, y_pred)))
print('Classification Report: ')
print(classification_report(Y_test, y_pred))
def fit_model(X_train, Y_train, X_test, Y_test, classifier_name, classifier, gridSearchParam, cv, save_model=False):
    #setting the seed for reproducability
    #np.random.seed(100)
    print('Training {} algorithm.........'.format(classifier_name))
    grid_clf = GridSearchCV(estimator=classifier,
                            param_grid=gridSearchParam, 
                            cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_res = grid_clf.fit(X_train, Y_train)
    best_params = grid_res.best_params_
    Y_pred = grid_res.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    
    
    print(Y_pred)
    print("=====================================================================")
    print('Training Accuracy Score: ' + str(accuracy_score(Y_train, grid_res.predict(X_train))))
    print("---------------------------------------------------------------------")
    print('Test Accuracy Score: ' + str(accuracy_score(Y_test, Y_pred)))
    print("---------------------------------------------------------------------")
    print('Best HyperParameters: ', best_params)
    print("---------------------------------------------------------------------")
    print('Classification Report: ')
    print(classification_report(Y_test, Y_pred))
    print("---------------------------------------------------------------------")
    
    #fig, ax = plt.subplots(figsize=(7,7))
    ax= plt.subplot()
    #plt.figure(figsize=(6,6))
    sns.set(font_scale=1.0) # Adjust to fit
    label_font = {'size':'5'}
    plt.rcParams.update({'font.size': 14})
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(['No Heart Disease', 'Heart Disease'])
    ax.yaxis.set_ticklabels(['No Heart Disease', 'Heart Disease'])
    print("=====================================================================")
    
    if save_model:
        file_name = classifier_name + '.pkl'
        pickle.dump(grid_res, open(file_name, 'wb'))
        #joblib.dump(grid_res, file_name)
        print('Model is saved successfully!')
cv = 5 
hyper_params = {'C': [0.0001, 0.001, 0.1, 1, 10, 20],   #np.logspace(0, 4, 10),
               'penalty': ['l1','l2'],
               'solver': ['liblinear', 'saga']}
#Feature Set 1 
fit_model(X_train, Y_train, X_test, Y_test, 'Logistic Regression', LogisticRegression(), hyper_params, cv)
#Feature Set 2 
fit_model(X_train_2, Y_train, X_test_2, Y_test, 'Logistic Regression', LogisticRegression(), hyper_params, cv)
#Feature Set 3 
fit_model(X_train_3, Y_train, X_test_3, Y_test, 'Logistic Regression', LogisticRegression(), hyper_params, cv)
#Feature Set 4 
fit_model(X_train_4, Y_train, X_test_4, Y_test, 'Logistic Regression', LogisticRegression(), hyper_params, cv)
cv = 5 
hyper_params = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.1, 1, 3],
                'kernel': ['linear', 'rbf']}
#Feature Set 1 
fit_model(X_train, Y_train, X_test, Y_test, 'SVM Classifier', SVC(), hyper_params, cv)
#Feature Set 2
fit_model(X_train_2, Y_train, X_test_2, Y_test, 'SVM Classifier', SVC(), hyper_params, cv)
#Feature Set 3
fit_model(X_train_3, Y_train, X_test_3, Y_test, 'SVM Classifier', SVC(), hyper_params, cv)

#Feature Set 4
fit_model(X_train_4, Y_train, X_test_4, Y_test, 'SVM Classifier', SVC(), hyper_params, cv)
cv = 5 
hyper_params = {'n_neighbors': list(range(1,20)),
                'leaf_size': list(range(1,15)),
                'p': [1,2]}
#Feature Set 1 
fit_model(X_train, Y_train, X_test, Y_test, 'KNN Classifier', KNeighborsClassifier(), hyper_params, cv)
#Feature Set 2
fit_model(X_train_2, Y_train, X_test_2, Y_test, 'KNN Classifier', KNeighborsClassifier(), hyper_params, cv)
#Feature Set 3 
fit_model(X_train_3, Y_train, X_test_3, Y_test, 'KNN Classifier', KNeighborsClassifier(), hyper_params, cv)
#Feature Set 4
fit_model(X_train_4, Y_train, X_test_4, Y_test, 'KNN Classifier', KNeighborsClassifier(), hyper_params, cv)
cv = 5 
hyper_params = {'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [2, 4, 6, 10, 15, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10]}
#Feature Set 1 
fit_model(X_train, Y_train, X_test, Y_test, 'Random Forest', RandomForestClassifier(), hyper_params, cv)
#Feature Set 2
fit_model(X_train_2, Y_train, X_test_2, Y_test, 'Random Forest', RandomForestClassifier(), hyper_params, cv)
#Feature Set 3 
fit_model(X_train_3, Y_train, X_test_3, Y_test, 'Random Forest', RandomForestClassifier(), hyper_params, cv)
import pickle

cv = 5 
hyper_params = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.1, 1, 3],
                'kernel': ['linear', 'rbf']}


#Saving SVM Best Model using feature Set 4
fit_model(X_train_4, Y_train, X_test_4, Y_test, 'SVMClassifier', SVC(), hyper_params, cv, save_model=True)
cv = 5 
hyper_params = {'n_neighbors': list(range(1,20)),
                'leaf_size': list(range(1,15)),
                'p': [1,2]}

#Saving KNN Best Model using feature Set 1
fit_model(X_train, Y_train, X_test, Y_test, 'KNN Classifier', KNeighborsClassifier(), hyper_params, cv, save_model=True)
cv = 5 
hyper_params = {'C': [0.0001, 0.001, 0.1, 1, 10, 20],   #np.logspace(0, 4, 10),
               'penalty': ['l1','l2'],
               'solver': ['liblinear', 'saga']}

#Saving Logistic Regression Best Model using feature Set 4
fit_model(X_train_4, Y_train, X_test_4, Y_test, 'Logistic Regression', LogisticRegression(), hyper_params, cv, save_model=True)
svm = pickle.load(open('./SVMClassifier.pkl', 'rb'))
logit = pickle.load(open('./Logistic Regression.pkl', 'rb'))
knn = pickle.load(open('./KNN Classifier.pkl', 'rb'))
feature_set1 = ['age', 'sex', 'resting_BP', 'serum_cholestoral', 'fasting_blood_sugar',
       'max_heart_rate', 'exercise_induced_angina', 'oldpeak',
       'chest_pain_type_2', 'chest_pain_type_3', 'resting_ECG_1', 'slope_1',
       'slope_2', 'major_vessels_count_1', 'major_vessels_count_2',
       'thalium_stress_2', 'thalium_stress_3']

feature_set4 = ['sex', 'exercise_induced_angina', 'oldpeak', 'chest_pain_type_2',
       'chest_pain_type_3', 'slope_1', 'major_vessels_count_1',
       'major_vessels_count_2', 'major_vessels_count_3', 'thalium_stress_1',
       'thalium_stress_2']
pred_knn = knn.predict(X_test)
pred_logit = logit.predict(X_test_4)
pred_svm = svm.predict(X_test_4)
import statistics

df_ensemble = pd.DataFrame()
df_ensemble['KNN'] = pred_knn
df_ensemble['Logistic'] = pred_logit
df_ensemble['SVM'] = pred_svm
df_ensemble.head(10)
def max_vote(x):
    vote = statistics.mode([int(x['KNN']), int(x['Logistic']), int(x['SVM'])])
    return vote
df_ensemble['Ensemble'] = df_ensemble.apply(max_vote, axis=1)
df_ensemble.head(10)
print("---------------------------------------------------------------------")
print('Test Accuracy Score: ' + str(accuracy_score(Y_test, df_ensemble.Ensemble.values)))
print("---------------------------------------------------------------------")
print('Classification Report: ')
print(classification_report(Y_test, df_ensemble.Ensemble.values))
print("---------------------------------------------------------------------")

cm = confusion_matrix(Y_test, df_ensemble.Ensemble.values)

ax= plt.subplot()
#plt.figure(figsize=(6,6))
sns.set(font_scale=1.0) # Adjust to fit
label_font = {'size':'5'}
plt.rcParams.update({'font.size': 14})
sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix') 
ax.xaxis.set_ticklabels(['No Heart Disease', 'Heart Disease'])
ax.yaxis.set_ticklabels(['No Heart Disease', 'Heart Disease'])
print("=====================================================================")