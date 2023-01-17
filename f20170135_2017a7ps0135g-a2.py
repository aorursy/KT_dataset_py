import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as pl

%matplotlib inline
DATA_DIR =  '../input/data-mining-assignment-2' # CHANGE TO THIS TO RUN IN KAGGLE

DATA_FILE = 'train.csv'

TEST_FILE = 'test.csv'
data_train = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE))
data_train.isnull().any().any()
data_train.head()
non_num_cols = data_train.select_dtypes('object').columns

non_num_cols
for col in non_num_cols:

    print(col, data_train[col].unique())
data_train_num = data_train.replace({'col2':{'Silver':1, 'Gold':2, 'Diamond':3, 'Platinum':4},

                                     'col11':{'No':0, 'Yes':1},

                                     'col44':{'No':0, 'Yes':1},

                                     'col56':{'Low':1, 'Medium':2, 'High':3}

                                    })

data_train_num = pd.get_dummies(data_train_num, columns = ['col37'])
(data_train_num.dtypes == 'object').any()
x = data_train_num.drop(labels = ['ID', 'Class'], axis = 1, inplace = False) # [features] 

y = data_train_num['Class']
x.columns
x.head()
print(x.values.shape, y.values.shape)
from sklearn.preprocessing import StandardScaler, MinMaxScaler



ss = StandardScaler()

x_norm = ss.fit_transform(x.values)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



x_train_norm, x_test_norm, y_train, y_test = train_test_split(x_norm, y, test_size = 0.1, random_state = 20)
print((y == 0).sum())

print((y == 1).sum())

print((y == 2).sum())

print((y == 3).sum())
print((y_test == 0).sum())

print((y_test == 1).sum())

print((y_test == 2).sum())

print((y_test == 3).sum())
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



# rf2 = RandomForestClassifier(n_estimators = 50)

param = {'max_depth' : [i for i in range(5, 35)], 

         'class_weight' : [{0:1, 1:1, 2:1, 3:1}, {0:1, 1:5, 2:1, 3:1}, {0:1, 1:10, 2:1, 3:1}],

          'bootstrap': [True, False]}



gs_rf = GridSearchCV(RandomForestClassifier(n_estimators = 100), param)



gs_rf.fit(x_norm, y)
gs_rf.best_params_
print(gs_rf.predict(x_test_norm))

print(y_test.values)
print(f1_score(gs_rf.predict(x_test_norm), y_test, average = 'weighted'))
data_test = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))



data_test_num = data_test.replace({'col2':{'Silver':1, 'Gold':2, 'Diamond':3, 'Platinum':4},

                                     'col11':{'No':0, 'Yes':1},

                                     'col44':{'No':0, 'Yes':1},

                                     'col56':{'Low':1, 'Medium':2, 'High':3}

                                    })

data_test_num = pd.get_dummies(data_test_num, columns = ['col37'])



id_eval = data_test_num['ID']

x_eval = data_test_num.drop(labels = ['ID'], axis = 1, inplace = False)

# x_eval.drop(labels = ['col19', 'col45', 'col51', 'col4', 'col53', 'col63'], axis = 1, inplace = True)  # , col63'

x_eval_norm = ss.fit_transform(x_eval)



y_eval_pred = gs_rf.predict(x_eval_norm)
pred_df = pd.DataFrame(y_eval_pred, index = id_eval, columns = ['Class'], dtype = int)

pred_df.to_csv('sub.csv')
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



create_download_link(pred_df)