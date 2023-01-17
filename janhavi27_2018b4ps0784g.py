import numpy as np 

import pandas as pd 

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns



'''

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
#from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler
#over_sample = RandomOverSampler(sampling_strategy=0.5)
#x_over, y_over = over_sample.fit_resample(X,y)
undersample = RandomUnderSampler()
#X_under, y_under = undersample.fit_resample(X,y)
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, random_state = 42)
x_train, y_train = undersample.fit_resample(x_train,y_train)
x_train = x_train.drop(labels = 'id', axis = 1)
x_val = x_val.drop(labels = 'id', axis = 1)
import xgboost
model = xgboost.XGBClassifier()

model.fit(X,y)

print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
selected_columns = feat_importances.nlargest(20).index
drop_columns = [x for x in x_train.columns if x not in selected_columns]
x_train = x_train.drop(labels = drop_columns, axis = 1)

x_val = x_val.drop(labels = drop_columns, axis = 1)
corr = x_train.corr()

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)

plt.show()
#clf = RandomForestClassifier()
#param_grid = { 

    #'n_estimators': [200, 500],

    #'max_features': ['sqrt', 'log2'],

    #'max_depth' : [4,5],

    #'criterion' :['gini', 'entropy']

#}

#from sklearn.model_selection import GridSearchCV

#CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, verbose = 2, scoring='roc_auc', n_jobs=-1)
#CV_rfc.fit(x_train, y_train)
#CV_rfc.best_params_
#CV_rfc.best_score_
#y_pred = CV_rfc.predict(x_val)
#roc_auc_score(y_val, y_pred)
xgb = xgboost.XGBClassifier()
xgparams = {

    "learning_rate": [0.01,0.05],

    "max_depth":[3,5],

    "min_child_weight": [3,4,5],

    "gamma": [0.1, 0.3, 0.5]

}
from sklearn.model_selection import GridSearchCV

CV_xgb = GridSearchCV(estimator=xgb, param_grid=xgparams, cv= 5, verbose = 2, scoring='roc_auc')
CV_xgb.fit(x_train, y_train)
CV_xgb.best_params_
CV_xgb.best_score_
xg_pred = CV_xgb.predict(x_val)
roc_auc_score(y_val, xg_pred)
test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
drop_columns.append('id')
df_test = test.drop(labels = drop_columns, axis = 1)
#rfc_prediction = CV_rfc.predict(df_test)
#my_submission = pd.DataFrame({'id': test.id, 'target': rfc_prediction})

#my_submission.to_csv('rfc_prediction.csv', index=False)
xgb_prediction = CV_xgb.predict(df_test)
my_submission = pd.DataFrame({'id': test.id, 'target': xgb_prediction})

my_submission.to_csv('xgb_prediction.csv', index=False)