import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, recall_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler



import lightgbm as lgb



seed_value = 7

sns.set_palette('coolwarm')



data_csv = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data_csv['normAmount'] = StandardScaler().fit_transform(np.array(data_csv['Amount']).reshape(-1, 1))

data_csv = data_csv.drop(['Time', 'Amount'], axis=1)



data_csv.head().T
fig, axs = plt.subplots(1, 2, figsize=(20, 3))

sns.countplot(data_csv['Class'], ax=axs[0]);

sns.distplot(data_csv['normAmount'], ax=axs[1]);
# Undersample so that we have equal (50/50) distribution between classes

minority_samples = data_csv[data_csv.Class == 1]

majority_samples = data_csv[data_csv.Class == 0].sample(n=minority_samples.shape[0], random_state=seed_value)



df_sampled = pd.concat([majority_samples, minority_samples])

df_sampled = df_sampled.sample(frac=1, random_state=seed_value).reset_index(drop=True)



# Plot distribution again

fig, axs = plt.subplots(1, 2, figsize=(20, 3))

sns.countplot(df_sampled['Class'], ax=axs[0]);

sns.distplot(df_sampled['normAmount'], ax=axs[1]);
X = df_sampled.drop('Class', axis=1).copy()

y = df_sampled.Class.copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)



del X

del y
parameters = {

    'C':[1],

    'solver': ['liblinear'],

    'max_iter': [100],

    'random_state': [seed_value]

}



clf = GridSearchCV(LogisticRegression(), parameters, scoring='f1', cv=5)

clf.fit(X_train, y_train)



print(f'Best score {clf.best_score_:.4f} with parameters {clf.best_params_}')
y_pred = clf.best_estimator_.predict(X_test)

print(f'F1 score for test set: {f1_score(y_test, y_pred):.4f}')

print(f'Recall score for test set: {recall_score(y_test, y_pred):.4f}')
parameters = {

    'learning_rate': [0.01, 0.1, 1],

    'n_estimators': [50, 100, 200],

    'random_state': [seed_value],

    'objective': ['binary']

}



clf = GridSearchCV(lgb.LGBMClassifier(), parameters, scoring='f1', cv=5)

clf.fit(X_train, y_train)



print(f'Best score {clf.best_score_:.4f} with parameters {clf.best_params_}')
y_pred = clf.best_estimator_.predict(X_test)

print(f'F1 score for test set: {f1_score(y_test, y_pred):.4f}')

print(f'Recall score for test set: {recall_score(y_test, y_pred):.4f}')