#################################

### Import requered libraries ###

#################################
import pandas

import pathlib

import IPython

import sklearn.model_selection

import sklearn.ensemble

import tensorflow

import numpy

import sklearn.metrics

import pandas

import numpy

import lightgbm

import sklearn

import pathlib

import sklearn.model_selection
###############################################

### Jupyter notebook and libraries settings ###

###############################################
pandas.options.display.max_rows = 1000

pandas.options.display.max_columns = 1000

IPython.core.interactiveshell.InteractiveShell.ast_node_interactivity = "all"
#################

### Read data ###

#################
data_path = '/kaggle/input'

data_extention = '.csv'



for path in pathlib.Path(data_path).rglob('*{}'.format(data_extention)):

    variable_name = str(path).split('/')[-1].replace(data_extention,'')

    globals()[variable_name] = pandas.read_csv(path)

    print(variable_name)
#####################

### Describe Data ###

#####################
# Display dataframes
train_targets_scored.head(1)

train_targets_nonscored.head(1)

train_features.head(1)

sample_submission.head(1)

test_features.head(1)
##################

### Base Model ###

##################
X = pandas.get_dummies(train_features.sort_values('sig_id').copy().drop('sig_id',axis=1),columns=['cp_type','cp_time','cp_dose'])
y = train_targets_scored.sort_values('sig_id').copy().drop('sig_id',axis=1)
X_train, X_valid , y_train, y_valid = sklearn.model_selection.train_test_split(X,y)
models = []

scr = 0

for i in range(y.shape[1]):

    model = lightgbm.LGBMClassifier(n_estimators=100,early_stopping_rounds=5)

    model.fit(X_train,y_train.iloc[:,i],eval_set=(X_valid,y_valid.iloc[:,i]),verbose=False)

    scr += model.best_score_['valid_0']['binary_logloss'] / y.shape[1]

    print('**{}**{}/{}'.format(scr,i,y.shape[1]))

    models.append(model)
X_test = pandas.get_dummies(test_features.sort_values('sig_id').copy().drop('sig_id',axis=1),columns=['cp_type','cp_time','cp_dose'])
results = []

for i in models:

    results.append(i.predict_proba(X_test)[:,1])
u = pandas.DataFrame(results).T
submission=pandas.DataFrame(u.values ,columns=sample_submission.sort_values('sig_id').drop('sig_id',1).columns,index=sample_submission.sort_values('sig_id')['sig_id'].values).reset_index()
submission = submission.rename(columns={'index':'sig_id'})
submission.head(2)
submission.to_csv('submission.csv',index=False)