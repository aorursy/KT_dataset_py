# module imports

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix



from scipy import stats



# read data into dataframe

df = pd.read_csv("/kaggle/input/heart-disease-cleveland-uci/heart_cleveland_upload.csv")



print('count(*): ' + str(len(df.index)) + '\n')



sns.countplot(x = 'condition', data = df)

plt.title('Counts by Condition')

plt.show()



for col in df.columns:

    print(str(col) + ' null count: ' + str(df[col].isnull().sum()))
df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].corr()
cont_flds = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



for i in cont_flds:

    a = stats.pointbiserialr(df['condition'].values, df[i].values)

    print(i + ": " + str(a[0]))
sns.boxplot(x = 'condition', y = 'thalach', data = df)
fig, axes = plt.subplots(1,2, figsize = (12,6))





sns.countplot(x = 'condition', data = df.loc[df['thalach'] <= 150], ax = axes[0])

sns.countplot(x = 'condition', data = df.loc[df['thalach'] > 150], ax = axes[1])
fig, axes = plt.subplots(1,3, figsize = (18,6))



# chest pain types by sex

sns.countplot(x = 'cp', hue = 'slope', data = df.loc[df['slope'] != 2], ax=axes[0])

sns.countplot(x = 'thal', hue='slope', data = df.loc[df['slope'] != 2], ax = axes[1])

sns.countplot(x = 'condition', hue='slope', data = df.loc[df['slope'] != 2], ax = axes[-1])
fig, axes = plt.subplots(1,4, figsize = (12,4))



sns.distplot(a = df['age'].values, ax = axes[0])



# resting blood pressure

sns.distplot(a = df['trestbps'].values, ax = axes[1])

sns.distplot(a = df['chol'].values, ax = axes[2])

sns.distplot(a = df['thalach'].values, ax = axes[3])

# convert categorical variables into dummy indicators

cat_flds = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

train = pd.get_dummies(df, columns = cat_flds, dummy_na=False)



# restructure dataframe so labels are at the end

train = train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0',

       'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0',

       'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1',

       'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'thal_0', 'thal_1',

       'thal_2', 'condition']]



# create label array

y = train['condition'].values

# create x array

x = train.drop(labels = 'condition', axis = 1)



# 

seed = 7

np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
# Random Forest

rf_clf = RandomForestClassifier(random_state=42)

rf_y_train_pred = cross_val_predict(rf_clf, x_train, y_train, cv = 3)



# Stochastic Gradient Descent

sgd_clf = SGDClassifier(random_state = 42)

sgd_y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv = 3)
print('------- Random Forest -------')

print('Confusion Matrix: \n', confusion_matrix(y_train, rf_y_train_pred))

print('Precision:', precision_score(y_train, rf_y_train_pred))

print('Recall:', recall_score(y_train, rf_y_train_pred))

print('\n')

print('------- SGD -------')

print('Confusion Matrix: \n', confusion_matrix(y_train, sgd_y_train_pred))

print('Precision:', precision_score(y_train, sgd_y_train_pred))

print('Recall:', recall_score(y_train, sgd_y_train_pred))
rf_clf = RandomForestClassifier(random_state=42)



param_grid = [{'n_estimators': [3, 10, 15], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 5, 7]}]



# refit by default is true but specifying because we want the estimator to be retrained on the whole train set

grid_search = GridSearchCV(rf_clf, param_grid, cv = 3, scoring = 'f1', return_train_score=True, refit=True)



grid_search.fit(x_train, y_train)



best_params = grid_search.best_params_

print('Best parameters for RF: ', best_params)
# get best estimator and save it to a new object

new_rf_model = grid_search.best_estimator_



# obtain predictions on test set and check f1 score

y_test_pred = new_rf_model.predict(x_test)

print('f1_score:', f1_score(y_test, y_test_pred))
# probabilities example

y_test_probabilites = new_rf_model.predict_proba(x_test)

y_test_probabilites



# probabilities of 1 class

probs_of_heart_disease = y_test_probabilites[:, 1]

probs_of_heart_disease