import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%config InlineBackend.figure_format='svg'

%matplotlib inline 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split#, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score
clinical_data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

clinical_data.head()
clinical_data.info()
clinical_data[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']].describe()
plt.figure()

corr_death = clinical_data.corr('pearson')['DEATH_EVENT'].drop('DEATH_EVENT')

sorted_idx_corr_death = corr_death.argsort()

plt.barh(clinical_data.columns[sorted_idx_corr_death], corr_death[sorted_idx_corr_death])

plt.title('Pearson correlation with DEATH_EVENT')

plt.xlabel('corr_value')

plt.ylabel('variable')

plt.show()
X = clinical_data.drop(['DEATH_EVENT'], axis = 1)

y = clinical_data['DEATH_EVENT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)



print(X_train.shape)

print(X_test.shape)
xgb = GradientBoostingClassifier(max_depth=2, min_samples_split=0.5, n_estimators=50,random_state=1)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)



def create_confusion_graph(cm, title='Confusion matrix'):

    fig, ax = plt.subplots()

    im = ax.imshow(cm, cmap='Blues')

    

    ax.set_xticks([0,1])

    ax.set_yticks([0,1])

    ax.set_xticklabels(['True','False'])

    ax.set_yticklabels(['True','False'])



    plt.xlabel('Predicted')

    plt.ylabel('Real')



    for i in range(len(cm)):

        for j in range(len(cm[0])):

            text = ax.text(j, i, cm[i, j],

                           ha="center", va="center", color="black")



    plt.title(title)

    

    return fig



create_confusion_graph(cm, 'Confusion matrix for the first gradient boost classifier')

plt.show()
sorted_idx = xgb.feature_importances_.argsort()



plt.barh(y=X.columns[sorted_idx], width=xgb.feature_importances_[sorted_idx])



plt.title('Gini importance')

plt.xlabel('Gini importance')

plt.ylabel('feature')

plt.show()
from sklearn.inspection import permutation_importance

result = permutation_importance(xgb, X, y, scoring='accuracy', random_state=1)

sorted_idx = result.importances_mean.argsort()



fig, ax = plt.subplots()

ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=X_test.columns[sorted_idx])

ax.set_title("Permutation Importances (test set)")

fig.set_size_inches((7,7))

plt.show()
from sklearn.inspection import plot_partial_dependence

plot_partial_dependence(xgb, X, features=['time','ejection_fraction','serum_creatinine'], n_cols=3, response_method='predict_proba', method='brute')

plt.title('Partial dependece plot for the first classifier')

plt.show()
clinical_data[(clinical_data['time'] >= 150) & (clinical_data['time'] <= 170)]['DEATH_EVENT'].value_counts()
X_test[xgb.predict(X_test) != y_test][['time', 'ejection_fraction', 'serum_creatinine']]
X_no_time = X.drop(['time'], axis = 1)

X_train_no_time, X_test_no_time, y_train, y_test = train_test_split(X_no_time, y, test_size = 0.2, random_state = 2)

print(X_train_no_time.shape)

print(y_train.shape)
xgb_no_time = GradientBoostingClassifier(learning_rate=0.01, max_depth=2,

                           min_samples_leaf=0.1, min_samples_split=0.5,

                           random_state=1)

xgb_no_time.fit(X_train_no_time, y_train)
y_pred = xgb_no_time.predict(X_test_no_time)

print('Accuracy: ', accuracy_score(y_test, y_pred))

create_confusion_graph(confusion_matrix(y_pred, y_test))

plt.show()
sorted_idx_no_time = xgb_no_time.feature_importances_.argsort()



plt.barh(y=X.columns[sorted_idx_no_time], width=xgb_no_time.feature_importances_[sorted_idx_no_time])

plt.title('Gini importance')

plt.xlabel('Gini importance')

plt.ylabel('feature')

plt.show()
result = permutation_importance(xgb_no_time, X_no_time, y, scoring='accuracy', random_state=1)

sorted_idx_no_time = result.importances_mean.argsort()



fig, ax = plt.subplots()

ax.boxplot(result.importances[sorted_idx_no_time].T,

           vert=False, labels=X_test_no_time.columns[sorted_idx_no_time])

ax.set_title("Permutation Importances (test set)")

fig.set_size_inches((7,7))

plt.show()
plot_partial_dependence(xgb_no_time, X_no_time, ['ejection_fraction','age', 'serum_creatinine'], grid_resolution=500, response_method='predict_proba', method='brute')

plt.title('Partial dependence plot for the second classifier')

plt.show()
'''param_grid = {

    'learning_rate': [0.01, 0.1, 0.2],

    'n_estimators': [50, 100, 150],

    'max_depth': [2,3,5],

    'min_samples_split': [0.01, 0.1, 0.5],    

}



grid_search = GridSearchCV(GradientBoostingClassifier(random_state=1), cv=5, param_grid=param_grid, scoring='roc_auc', verbose = 1, n_jobs = 2)

grid_search.fit(X_train, y_train)



print('Best params found: \n\t', grid_search.best_params_)

print('Best ROC_AUC score found: \n\t', grid_search.best_score_)

print('Best estimator: \n\t', grid_search.best_estimator_)

'''



print('grid search - first model')
'''param_grid = {

    'learning_rate': [0.01, 0.1, 0.2],

    'n_estimators': [50, 100, 150],

    'max_depth': [2,3,5,10],

    'min_samples_split': [0.0, 0.1, 0.5],

    'min_samples_leaf': [0.0, 0.1, 0.5],

    'min_impurity_decrease': [0.0, 0.1,0.5]

}



grid_search = GridSearchCV(GradientBoostingClassifier(random_state=1), cv=5, param_grid=param_grid, scoring='roc_auc', verbose = 1, n_jobs = -1)

grid_search.fit(X_train, y_train)



print('Best params found: \n\t', grid_search.best_params_)

print('Best ROC_AUC score found: \n\t', grid_search.best_score_)

print('Best estimator: \n\t', grid_search.best_estimator_)'''



print('grid search - second model')