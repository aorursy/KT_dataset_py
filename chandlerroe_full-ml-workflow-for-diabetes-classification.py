import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

%matplotlib inline

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning) 
df = pd.read_csv('D:/Data/diabetes.csv')

df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
df.describe()
print((df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum())
df.iloc[:, 1:6] = df.iloc[:, 1:6].replace(0, np.NaN)

print((df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].isnull()).sum())
df.describe()
from sklearn.preprocessing import Imputer
imputer_median = Imputer(strategy='median', axis=0)
null_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df_set_to_impute = df[null_columns] #create data frame of numerical columns to be imputed

imputer_median.fit(df_set_to_impute)

print('The median values are:', imputer_median.statistics_)
X_imputed_medians = imputer_median.transform(df_set_to_impute)

df_num_imputed_median = pd.DataFrame(X_imputed_medians, index=df.index, columns=df_set_to_impute.columns)

df_imputed = df.copy()

df_imputed[null_columns] = df_num_imputed_median[null_columns]
df.head() #see the NaNs in SkinThickness and Insulin
df_imputed.head() #No more NaNs
df.isnull().sum() #pre-imputed total NaNs
df_imputed.isnull().sum() #No more NaNs
sns.set_style('darkgrid')

features=df.columns.drop('Outcome')

df[features].hist(figsize=(10, 8))

plt.tight_layout()

plt.show()
sns.set_style(None)

sns.pairplot(df.dropna(), vars=features, hue='Outcome')

plt.show()
corr = df[features].corr()

plt.subplots(figsize=(9,6))

sns.set_context("notebook", font_scale=1.5)

sns.heatmap(corr.dropna(), annot=True, cmap='Reds', annot_kws={'size': 14})

plt.show()
X = df_imputed.iloc[:, 0:8] #iloc not inclusive on end index

y = df_imputed.iloc[:, -1]
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)

X_scaled = transformer.transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled.head()
X_scaled.hist(figsize=(10, 8))

plt.tight_layout()

plt.show()
y.value_counts(normalize=True) #naive rule
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=1, stratify=y) 
from sklearn import metrics

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1, 20)

scores = []

for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_val)

    scores.append(metrics.accuracy_score(y_val, y_pred))
plt.plot(k_range, scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')

plt.xticks(np.arange(0, 21, 2))

plt.yticks(np.arange(0.64, 0.73, 0.02))

plt.show()
k_range2 = range(1, 25)

knn_cv_loop_scores = []

knn_cv_loop_std = []

for k in k_range2:

    knn_cv_loop = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn_cv_loop, X_scaled, y, cv=10, scoring='accuracy')

    knn_cv_loop_scores.append(scores.mean())

    knn_cv_loop_std.append(scores.std())
plt.plot(k_range2, knn_cv_loop_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Mean Testing Accuracy')

plt.yticks(np.arange(0.68, 0.76, 0.02))

plt.xticks(np.arange(0, 26, 2))

plt.title("Mean Accuracy by K")

plt.show()
knn_13 = KNeighborsClassifier(n_neighbors=13)

knn_13_scores_acc = cross_val_score(knn_13, X_scaled, y, cv=10, scoring='accuracy')

print('Accuracy:', knn_13_scores_acc.mean())

print('Std:', knn_13_scores_acc.std())
knn_13_scores_roc = cross_val_score(knn_13, X_scaled, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', knn_13_scores_roc.mean())

print('Std:', knn_13_scores_roc.std())
knn_13_scores_f1 = cross_val_score(knn_13, X_scaled, y, cv=10, scoring='f1')

print('F1 Score:', knn_13_scores_f1.mean())

print('Std:', knn_13_scores_f1.std())
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

log_scores_acc = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')

print('Accuracy:', log_scores_acc.mean())

print('Std:', log_scores_acc.std())
log_scores_roc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', log_scores_roc.mean())

print('Std:', log_scores_roc.std())
log_scores_f1 = cross_val_score(logreg, X, y, cv=10, scoring='f1')

print('F1 Score:', log_scores_f1.mean())

print('Std:', log_scores_f1.std())
log_scores_acc = cross_val_score(logreg, X_scaled, y, cv=10, scoring='accuracy')

print('Accuracy:', log_scores_acc.mean())

print('Std:', log_scores_acc.std())
log_scores_roc = cross_val_score(logreg, X_scaled, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', log_scores_roc.mean())

print('Std:', log_scores_roc.std())
log_scores_f1 = cross_val_score(logreg, X_scaled, y, cv=10, scoring='f1')

print('F1 Score:', log_scores_f1.mean())

print('Std:', log_scores_f1.std())
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)

tree_scores_acc = cross_val_score(tree, X_scaled, y, cv=10, scoring='accuracy')

print('Accuracy:', tree_scores_acc.mean())

print('Std:', tree_scores_acc.std())
tree_scores_roc = cross_val_score(tree, X_scaled, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', tree_scores_roc.mean())

print('Std:', tree_scores_roc.std())
tree_scores_f1 = cross_val_score(tree, X_scaled, y, cv=10, scoring='f1')

print('F1 Score:', tree_scores_f1.mean())

print('Std:', tree_scores_f1.std())
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42)

forest_scores_acc = cross_val_score(forest, X_scaled, y, cv=10, scoring='accuracy')

print('Accuracy:', forest_scores_acc.mean())

print('Std:', forest_scores_acc.std())
forest_scores_roc = cross_val_score(forest, X_scaled, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', forest_scores_roc.mean())

print('Std:', forest_scores_roc.std())
forest_scores_f1 = cross_val_score(forest, X_scaled, y, cv=10, scoring='f1')

print('F1 Score:', forest_scores_f1.mean())

print('Std:', forest_scores_f1.std())
from sklearn.svm import SVC
svm = SVC(kernel='linear')

svm_scores_acc = cross_val_score(svm, X_scaled, y, cv=10, scoring='accuracy')

print('Accuracy:', svm_scores_acc.mean())

print('Std:', svm_scores_acc.std())
svm_scores_auc = cross_val_score(svm, X_scaled, y, cv=10, scoring='roc_auc')

print('Area Under ROC:', svm_scores_auc.mean())

print('Std:', svm_scores_auc.std())
svm_scores_f1 = cross_val_score(svm, X_scaled, y, cv=10, scoring='f1')

print('F1 Score:', svm_scores_f1.mean())

print('Std:', svm_scores_f1.std())
from sklearn.model_selection import GridSearchCV
param_grid_logreg = {

    'penalty': ['l1', 'l2'], 

    'C': [.01,0.1,1,10,100]

}
scoring = ['accuracy', 'roc_auc', 'f1']
grid_search_logreg = GridSearchCV(logreg, param_grid_logreg, cv=10, scoring=scoring, refit='roc_auc')
grid_search_logreg.fit(X_scaled, y)
grid_search_logreg.best_params_
print('Best Accuracy:', grid_search_logreg.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_logreg.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_logreg.cv_results_['mean_test_f1'].max())
logreg_tuned = grid_search_logreg.best_estimator_
param_grid_linear_svm = {

    'kernel': ['linear'],

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}
grid_search_linear_svm = GridSearchCV(svm, param_grid_linear_svm, scoring=scoring, refit='roc_auc')
grid_search_linear_svm.fit(X_scaled, y)
grid_search_linear_svm.best_params_
print('Best Accuracy:', grid_search_linear_svm.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_linear_svm.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_linear_svm.cv_results_['mean_test_f1'].max())
param_grid_poly_svm = {

    'kernel': ['poly'],

    'C': [0.1, 1, 10, 100],

    'degree': [1, 2],

}
grid_search_poly_svm = GridSearchCV(svm, param_grid_poly_svm, cv=3, scoring=scoring, refit='roc_auc')
warnings.filterwarnings('ignore')

grid_search_poly_svm.fit(X_scaled, y)
grid_search_poly_svm.best_params_
print('Best Accuracy:', grid_search_poly_svm.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_poly_svm.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_poly_svm.cv_results_['mean_test_f1'].max())
param_grid_rbf_svm = {

    'kernel': ['rbf'],

    'C': [0.1, 1, 10, 100, 1000],

    'gamma': [0, 0.1, 1, 10, 100],

}
grid_search_rbf_svm = GridSearchCV(svm, param_grid_rbf_svm, scoring=scoring, refit='roc_auc')
grid_search_rbf_svm.fit(X_scaled, y)
grid_search_rbf_svm.best_params_
print('Best Accuracy:', grid_search_rbf_svm.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_rbf_svm.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_rbf_svm.cv_results_['mean_test_f1'].max())
from sklearn.model_selection import RandomizedSearchCV
param_grid_rf = {

    'n_estimators': [50, 100, 200, 300],

    'criterion': ['gini', 'entropy'],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth': [int(x) for x in np.linspace(1, 45, num = 3)],

    'min_samples_leaf': [1, 2, 4],

    'min_samples_split': [5, 10]    

}
random_search_rf = RandomizedSearchCV(forest, param_grid_rf, cv=10, n_iter=100, scoring=scoring, 

                                      refit='roc_auc', random_state=42)
random_search_rf.fit(X_scaled, y)
random_search_rf.best_params_
print('Best Accuracy:', random_search_rf.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', random_search_rf.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', random_search_rf.cv_results_['mean_test_f1'].max())
rf_tuned = random_search_rf.best_estimator_
feature_importances = pd.DataFrame(rf_tuned.feature_importances_, index = X_scaled.columns, 

                                   columns=['Importance']).sort_values('Importance',ascending=False)

feature_importances
print('Best Accuracy:', grid_search_logreg.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_logreg.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_logreg.cv_results_['mean_test_f1'].max())
print('Best Accuracy:', grid_search_linear_svm.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', grid_search_linear_svm.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', grid_search_linear_svm.cv_results_['mean_test_f1'].max())
print('Best Accuracy:', random_search_rf.cv_results_['mean_test_accuracy'].max())

print('Best ROC:', random_search_rf.cv_results_['mean_test_roc_auc'].max())

print('Best F1 Score:', random_search_rf.cv_results_['mean_test_f1'].max())