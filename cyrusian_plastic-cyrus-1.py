# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import norm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = '../input'
training_set = pd.read_csv(os.path.join(PATH, 'training_set.csv'))
testing_set = pd.read_csv(os.path.join(PATH, 'test_set.csv'), nrows=100)
complete_set = training_set.append(testing_set)
#training_set = training_set.set_index(training_set['object_id'])
#training_set = training_set.drop(columns=['object_id'])
FEATURES = list(complete_set)
FEATURES
complete_set.head()
pd.set_option('display.float_format', lambda x : '%.0f' % x)
complete_set.describe()
PREDICTORS = ['mjd_a59k', 'passband', 'flux']
OUTCOME = ['detected']
def preprocessing(df):
    '''
    return a preprocessed df
    '''
    df_new = df.copy()
    ### SAND BOX
    df_new['mjd_a59k'] = df_new['mjd'] - df_new['mjd'].min()
#     MJD to Unix time conversion: (MJD - 40587) * 86400 + seconds past UTC midnight
#     https://wiki.polaire.nl/doku.php?id=mjd_convert_modified_julian_date
    df_new['unix'] = (df_new['mjd'] - 40587) * 86400
    df_new['unix'] = df_new['unix'] - df_new['unix'].min()
    ### SAND BOX END
    return df_new

df_current = preprocessing(training_set)
#### SAND BOX ####
sns.distplot(df_current[df_current.detected == 0]['flux'] , fit=norm)
df_current.describe()
# df_current[df_current.detected == 1].describe() - df_current[df_current.detected == 0].describe()
df_current[df_current.detected == 0].isna().sum()
df_current[df_current.detected == 0].describe()
# df_current[df_current.detected == 1].describe()[]






#### SAND BOX ENDED ####
def x_y(df):
    '''
    determine the x and y by global variable predictors and outcome
    '''
    X = df[PREDICTORS]
    y = df[OUTCOME]
    return (X, y)
training_final = preprocessing(training_set)
testing_final = preprocessing(pd.read_csv(os.path.join(PATH, 'test_set_sample.csv')))
X_tr, y_tr = x_y(training_final)
X_te, y_te = x_y(testing_final)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
models = []
### SAND BOX ####
# models.append(('Logistic Regression', LogisticRegression()))
# models.append(('Extreme Gradient Booster', XGBClassifier()))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.05)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.02)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.01)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.2)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.25)))

class ModelResult():
    def __init__(self, name, model) :
        self.name = name
        self.model = model
        self.metrics = []
    def set_metric(self, name, metric):
        self.metrics.append((name, metric))
    def __str__(self):
        returner = "###### " + self.name + " ######\n"
        for met_name, metric in self.metrics :
            returner += met_name + ":\n"
            returner += str(metric) + "\n"
        return returner
    
# mres = ModelResult('lr', LogisticRegression())
# print(mres)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import time
def run_models(models):
    results = []
    for name, model in models :
        start = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mr = ModelResult(name, model)
        mr.set_metric(('accuracy'), accuracy_score(y_te, y_pred))        
        mr.set_metric(('confusion_matrix'), confusion_matrix(y_te, y_pred))        
        mr.set_metric(('roc_curve'), roc_curve(y_te, y_pred))
        mr.set_metric(('time_elapsed'), time.time() - start)
        results.append(mr)
    return results
#uncomment this if ready
results = run_models(models)
for res in results :
    print(res)
for res in results :
    print(res)
results_3 = run_models(models_3)
for r in results_3 :
    print(r)









