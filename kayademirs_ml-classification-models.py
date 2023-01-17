import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
df = pd.read_csv('/kaggle/input/diabetes.csv')
df.head()
df['Outcome'].value_counts()
df.describe().T
y = df['Outcome'] 

X = df.drop(['Outcome'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='liblinear').fit(X_train,y_train)
# Sabit değeri verir. (bias)

log_model.intercept_
# Ağırlık değerlerini verir. (weights)

log_model.coef_
y_pred = log_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(log_model, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
logit_roc_auc = roc_auc_score(y_test, log_model.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, log_model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_params = {"n_neighbors": np.arange(1,50)}
knn_cv_model = GridSearchCV(knn, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_
knn_cv_model.best_score_
knn = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(knn, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
knn_roc_auc = roc_auc_score(y_test, knn.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % knn_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.svm import SVC
svm = SVC()
svm_params = {"C": np.arange(1,10), "kernel":["linear", "rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_
svm = SVC(C = 2, kernel='linear', probability=True).fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(svm, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
svm_roc_auc = roc_auc_score(y_test, svm.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, svm.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % svm_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaler = scaler.transform(X_train)



scaler.fit(X_test)

X_test_scaler = scaler.transform(X_test)
mlpc = MLPClassifier()
mlpc_params = {'alpha':[1, 5, 0.1, 0.01, 0.03, 0.005, 0.0001 ],

              'hidden_layer_sizes': [(10,10), (100,100,100), (100,100), (3,5)]}
mlpc = MLPClassifier(solver='lbfgs', activation='logistic')
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, cv=10, n_jobs=-1, verbose=2).fit(X_train_scaler, y_train)
mlpc_cv_model
mlpc_cv_model.best_params_
mlpc = MLPClassifier(solver = 'lbfgs', alpha=1 , hidden_layer_sizes=(3,5), activation='logistic').fit(X_train_scaler, y_train)
y_pred = mlpc.predict(X_test_scaler)
accuracy_score(y_test, y_pred)
cross_val_score(mlpc, X_test_scaler, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
mlpc_roc_auc = roc_auc_score(y_test, mlpc.predict(X_test_scaler))

fpr , tpr, thresholds = roc_curve(y_test, mlpc.predict_proba(X_test_scaler)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % mlpc_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart_params = {'max_depth': [1,3,5,8,10],

              'min_samples_split': [2,3,5,10,20,50]}
cart_cv_model = GridSearchCV(cart, cart_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
cart_cv_model
cart_cv_model.best_params_
cart = DecisionTreeClassifier(max_depth=5, min_samples_split=20).fit(X_train, y_train)
y_pred = cart.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(cart, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
cart_roc_auc = roc_auc_score(y_test, cart.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, cart.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % cart_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100,200,500,1000],

            'max_features': [3,5,7,8],

            'min_samples_split':[2,5,10,20]}
rf_cv_model = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
rf_cv_model
rf_cv_model.best_params_
rf = RandomForestClassifier(max_features=5, min_samples_split=10, n_estimators=200).fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)
rf.feature_importances_
feature_imp = pd.Series(rf.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
cross_val_score(rf, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % rf_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
gbm_params = {'learning_rate': [0.1, 0.01, 0.001, 0.05],

            'n_estimators': [100,200,500,1000],

            'max_depth':[2,3,5,8]}
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
gbm_cv_model
gbm_cv_model.best_params_
gbm = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=500).fit(X_train, y_train)
y_pred = gbm.predict(X_test)
accuracy_score(y_test, y_pred)
gbm.feature_importances_
feature_imp = pd.Series(gbm.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
cross_val_score(gbm, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
gbm_roc_auc = roc_auc_score(y_test, gbm.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, gbm.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % gbm_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from xgboost import XGBClassifier
xgboost = XGBClassifier()
xgboost_params = {'learning_rate': [0.1, 0.01, 0.001],

            'subsample':[0.6, 0.8, 1],

            'n_estimators': [100,500,1000,2000],

            'max_depth':[3,5,7]}
xgboost_cv_model = GridSearchCV(xgboost, xgboost_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgboost_cv_model
xgboost_cv_model.best_params_
xgboost = XGBClassifier(learning_rate=0.001, max_depth=5, n_estimators=2000, subsample=1).fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
accuracy_score(y_test, y_pred)
xgboost.feature_importances_
feature_imp = pd.Series(xgboost.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
cross_val_score(xgboost, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
xgboost_roc_auc = roc_auc_score(y_test, xgboost.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, xgboost.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % xgboost_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from lightgbm import LGBMClassifier
lightgbm = LGBMClassifier()
lightgbm_params = {'learning_rate': [0.1, 0.01, 0.001],

            'n_estimators': [200,500,100],

            'max_depth':[1,2,35,8]}
lightgbm_cv_model = GridSearchCV(lightgbm, lightgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
lightgbm_cv_model
lightgbm_cv_model.best_params_
lightgbm = LGBMClassifier(learning_rate=0.01, max_depth=1, n_estimators=500).fit(X_train, y_train)
y_pred = lightgbm.predict(X_test)
accuracy_score(y_test, y_pred)
lightgbm.feature_importances_
feature_imp = pd.Series(lightgbm.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
cross_val_score(lightgbm, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
lightgbm_roc_auc = roc_auc_score(y_test, lightgbm.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, lightgbm.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % lightgbm_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from catboost import CatBoostClassifier
catboost = CatBoostClassifier()
catboost_params = {'learning_rate': [0.1, 0.01, 0.003],

            'iterations': [200,500,1000],

            'depth':[4,5,8]}
catboost_cv_model = GridSearchCV(catboost, catboost_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train, verbose=False)
catboost_cv_model
catboost_cv_model.best_params_
catboost = CatBoostClassifier(iterations=1000, learning_rate=0.003, depth=8).fit(X_train, y_train, verbose=False)
y_pred = catboost.predict(X_test)
accuracy_score(y_test, y_pred)
catboost.feature_importances_
feature_imp = pd.Series(catboost.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.show()
catboost_roc_auc = roc_auc_score(y_test, catboost.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, catboost.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % catboost_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
models = [log_model, knn, svm, mlpc, cart, rf, gbm, xgboost, lightgbm, catboost]

result = []

results = pd.DataFrame(columns=['Models', "Accuracy"])



for model in models:

    names = model.__class__.__name__

    print(names)

    if names == 'MLPClassifier':

        y_pred = model.predict(X_test_scaler)

    else:

        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    result = pd.DataFrame([[names, acc*100]], columns=['Models', 'Accuracy'])

    results = results.append(result)
sns.barplot(x='Accuracy', y='Models', data=results, color='r')

plt.xlabel('Accuracy %')

plt.title('Modellerin Doğruluk Oranları');
results