# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Gaphics

import matplotlib.pyplot as plt

import seaborn as sns

import graphviz 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_valid = pd.read_csv('../input/valid.csv')

df_exemple = pd.read_csv('../input/exemplo_resultado.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.describe()
df_train.columns
def correlation_plot(dataframe, method_='spearman'):

    correlations = dataframe.corr(method=method_)



    fig, ax = plt.subplots(figsize=(25,25))

    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',

                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .90})

    plt.show();
y = df_train['default payment next month'].copy()

# y
all_df = pd.concat([df_train, df_valid, df_test], sort=False)
all_df
all_df.info()
all_df.isna().sum()
correlation_plot(all_df)
genre_ohe = pd.get_dummies(all_df['SEX'], prefix='SEX')

genre_ohe.info()
all_df = pd.concat([all_df, genre_ohe], axis=1)



all_df = all_df.drop(['SEX'], axis=1)
education_ohe = pd.get_dummies(all_df['EDUCATION'], prefix='EDUCATION')

education_ohe.info()
all_df = pd.concat([all_df, education_ohe], axis=1)



all_df = all_df.drop(['EDUCATION'], axis=1)
marital_status_ohe = pd.get_dummies(all_df['MARRIAGE'], prefix='MARRIAGE')

marital_status_ohe.info()
all_df = pd.concat([all_df, marital_status_ohe], axis=1)



all_df = all_df.drop(['MARRIAGE'], axis=1)
all_df.info()
# all_df.isna().sum()
# all_df.head()
raw = all_df.copy()
correlation_plot(all_df)
# all_df = all_df.drop(['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis=1)

# all_df = all_df.drop(['LIMIT_BAL','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis=1)
# correlation_plot(all_df)
all_df = all_df.drop(['default payment next month'], axis=1)
all_df.info()
raw.info()
df_train_clean = all_df.iloc[:21000]

df_valid_clean = all_df.iloc[21000:25500]

df_test_clean = all_df.iloc[25500:30000]
# df_train_clean.isna().sum()

# df_test_clean.isna().sum()

# df_valid_clean.isna().sum()
# df_test_clean.info()
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
x = df_train_clean
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
param_grid = {

    'n_estimators': [25, 40, 50],

    'max_features': ['auto', 'sqrt', 'log2', 0.5],

    'criterion' :['gini', 'entropy'],

}



# rfc = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

rfc_model = RandomForestClassifier(min_samples_leaf=3, n_jobs=-1)
# rfc.fit(x, y)

# rfc.fit(X_train, y_train)

# CV_rfc = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv= 10)

# CV_rfc.fit(X_train, y_train)
# best_parameters = CV_rfc.best_params_

# print("Best Parameters: {}".format(best_parameters))
# rf_classifer = RandomForestClassifier(max_features = 0.5, n_estimators = 50, min_samples_leaf=3, n_jobs=-1, 

#                                       oob_score=True, random_state = 0, criterion = 'gini')

# rf_classifer.fit(X_train, y_train)
# predict_rf = rf_classifer.predict(X_test)



# print(predict)
# print(roc_auc_score(y_test, predict_rf))
# param_grid_dtc = {

#     'max_leaf_nodes': [25, 50, 75, 100, 150],

# }



# dtc_model = DecisionTreeClassifier(random_state = 0)



# CV_dtc = GridSearchCV(estimator=dtc_model, param_grid=param_grid_dtc, cv= 10)

# CV_dtc.fit(X_train, y_train)
# best_parameters_dtc = CV_dtc.best_params_

# print("Best Parameters: {}".format(best_parameters_dtc))
# dt_class = DecisionTreeClassifier(max_leaf_nodes=25)

# dt_class.fit(X_train, y_train)
# predict_dt = dt_class.predict(X_test)

# print(roc_auc_score(y_test, predict_dt))
# final_model = RandomForestClassifier(max_features = 0.5, n_estimators = 50, min_samples_leaf=3, n_jobs=-1, 

#                                      oob_score=True, random_state = 0)

final_model = DecisionTreeClassifier(max_leaf_nodes=25)

final_model.fit(x, y)
df_valid_test_clean = pd.concat([df_valid_clean, df_test_clean], sort=False)

df_valid_test_clean.info()
predicao = final_model.predict(df_valid_test_clean)



# print(predicao)
# result = pd.concat([submission, df_exemple[4500:]], sort=False)

submission = pd.DataFrame({'ID': df_valid_test_clean.ID, 'Default': predicao})

submission.to_csv('submission.csv', index=False)
submission