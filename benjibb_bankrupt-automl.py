import os # accessing directory structure

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy.io import arff

from sklearn.model_selection import StratifiedKFold,train_test_split
print(os.listdir('../input/'))
data_5 = arff.loadarff('../input/5year.arff')

data_4 = arff.loadarff('../input/4year.arff')

data_3 = arff.loadarff('../input/3year.arff')

data_2 = arff.loadarff('../input/2year.arff')

data_1 = arff.loadarff('../input/1year.arff')
df_5 = pd.DataFrame(data_5[0])

df_4 = pd.DataFrame(data_4[0])

df_3 = pd.DataFrame(data_3[0])

df_2 = pd.DataFrame(data_2[0])

df_1 = pd.DataFrame(data_1[0])
df = pd.concat([df_5, df_4, df_3, df_2, df_1], 0)
df.shape
df.head()
df['class'].isnull().sum()
cont_names = df.columns[:-1]

cont_names
df['class'].value_counts()
df.rename(columns={'class':'target'}, inplace=True)
df.replace({'target': b'1'}, int(1), inplace=True)

df.replace({'target': b'0'}, int(0), inplace=True)
df['Attr55'] = None
df.head()
X_train, X_test, y_train, y_test = train_test_split(df[cont_names], df['target'], test_size=0.2, random_state=42)

train = pd.concat([X_train, y_train], 1)

test = pd.concat([X_test, y_test], 1)
import h2o

from h2o.automl import H2OAutoML

h2o.init()
df = h2o.H2OFrame(train)
df['target'] = df['target'].asfactor()

y = "target"

cont_names = cont_names.tolist()

x = cont_names
#max_runtime_secs= 3600, sort_metric='AUC'

aml = H2OAutoML(max_runtime_secs= 3600*6, max_models=60, sort_metric='AUC')

aml.train(x = x, y = y, training_frame = df)
lb = aml.leaderboard

lb.head(rows=lb.nrows) # Entire leaderboard
hf = h2o.H2OFrame(test)

preds = aml.predict(hf)

preds = preds.as_data_frame()

preds['p_p0'] = np.exp(preds['p0'])

preds['p_p1'] = np.exp(preds['p1'])

preds['sm'] = preds['p_p1'] / (preds['p_p0'] + preds['p_p1'])
from sklearn.metrics import roc_auc_score



roc_auc_score(y_test, preds['sm'])
aml.leader.summary()