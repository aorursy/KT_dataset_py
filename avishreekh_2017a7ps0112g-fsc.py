# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import os



from sklearn.ensemble import RandomForestClassifier



import warnings

warnings.filterwarnings('ignore')



np.random.seed(42)
train_path = os.path.join("/kaggle/input/data-mining-assignment-2/", 'train.csv')

test_path = os.path.join("/kaggle/input/data-mining-assignment-2/", 'test.csv')

submission_path = os.path.join("/kaggle/input/data-mining-assignment-2/", 'Sample Submission.csv')
df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

print(df_train.shape, df_test.shape)

df_train.head()
df_train['col11'] = df_train['col11'].map({'No': 0, 'Yes': 1})

df_train['col37'] = df_train['col37'].map({'Male': 0, 'Female': 1})

df_train['col44'] = df_train['col44'].map({'No': 0, 'Yes': 1})

df_train_onehot = pd.get_dummies(df_train, columns=['col2', 'col56'])



df_test['col11'] = df_test['col11'].map({'No': 0, 'Yes': 1})

df_test['col37'] = df_test['col37'].map({'Male': 0, 'Female': 1})

df_test['col44'] = df_test['col44'].map({'No': 0, 'Yes': 1})

df_test_onehot = pd.get_dummies(df_test, columns=['col2', 'col56'])



print(df_train_onehot.shape, df_test_onehot.shape)

df_train_onehot.head()
from sklearn.utils import resample



df_train_onehot['Class'].value_counts()
df_major = df_train_onehot[df_train_onehot['Class'] != 1]

df_minor = df_train_onehot[df_train_onehot['Class'] == 1]

df_minor.head()
df_minor_upsampled = resample(df_minor, replace=True, n_samples=200, random_state=42)

df_train_upsampled = pd.concat([df_major, df_minor_upsampled], axis=0)

df_train_upsampled['Class'].value_counts()
X = df_train_upsampled.drop(['Class'], axis=1)

y = df_train_upsampled['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=df_train_upsampled['Class'], test_size=0.2)



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i, oob_score=True, random_state=42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)

    print('Depth: {} | train score: {} | test score: {}'.format(i, sc_train, sc_test))
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 100, random_state=42, oob_score=True, bootstrap=True)        #Initialize the classifier object



parameters = {'max_depth':[15, 16, 19],'min_samples_split':[2, 3], 'max_features':[None, 'sqrt']}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
best_rf = RandomForestClassifier(n_estimators=100, max_depth = 19, min_samples_split=2, random_state=42, oob_score=True, bootstrap=True, max_features='sqrt')

best_rf.fit(X_train, y_train)



y_pred = best_rf.predict(X_test)



print(classification_report(y_test, y_pred))
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(df_train_onehot.drop('Class', axis=1), df_train_onehot['Class'])



y_pred = rfc.predict(df_test_onehot)

y_pred
df_sub = pd.read_csv(submission_path)

df_sub['Class'] = y_pred

df_sub.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(df_sub)