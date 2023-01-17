# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv("../input/zimnat-insurance-recommendation-challenge/Test.csv")

submission = pd.read_csv("../input/zimnat-insurance-recommendation-challenge/SampleSubmission.csv")

train = pd.read_csv("../input/zimnat-insurance-recommendation-challenge/Train.csv")
train
set(train.columns) - set(test.columns)
test
dataset = pd.concat([train,test]).reset_index()

dataset
dataset['join_date'] = pd.to_datetime(dataset['join_date'], format='%d/%m/%Y').dt.year

dataset['age'] = dataset['join_date'] - dataset['birth_year']

dataset
dataset_final = dataset[['age','sex','marital_status','branch_code','occupation_code','occupation_category_code']]

dataset_final
dataset_final['sex'].value_counts()
dataset_final['marital_status'].value_counts()
dataset_final['branch_code'].value_counts()
dataset_final['occupation_category_code'].value_counts()
train['occupation_code'].value_counts()
len(train['occupation_code'].value_counts())
cutoff = 100

df_col = dataset_final['occupation_code'].value_counts()[dataset_final['occupation_code'].value_counts() > cutoff]

df_col
len(df_col.index),df_col.index
dict_map = {value:key+1 for key,value in enumerate(df_col.index)}

dict_map['etc'] = 37 

dict_map
df_error = dataset_final['occupation_code'].value_counts()[dataset_final['occupation_code'].value_counts() <= cutoff]

df_error
dataset_final
# occupation_code : cutoff below 100, rename as etc
df_error.index
for value in df_error.index:

    dataset_final.loc[dataset_final['occupation_code'] == value,'occupation_code'] = 'etc'
dataset_final['occupation_code'] = dataset_final['occupation_code'].map(dict_map)

dataset_final
from sklearn.preprocessing import LabelEncoder

dataset_final['sex'] = LabelEncoder().fit_transform(dataset_final['sex'])

dataset_final['marital_status'] = LabelEncoder().fit_transform(dataset_final['marital_status'])

dataset_final['branch_code'] = LabelEncoder().fit_transform(dataset_final['branch_code'])

dataset_final['occupation_category_code'] = LabelEncoder().fit_transform(dataset_final['occupation_category_code'])

dataset_final
dataset.columns
dataset
dataset_final.iloc[:len(train)*8//10]
def prediction_column(columns):

    x_train = dataset_final.iloc[:len(train)*8//10]

    x_val = dataset_final.iloc[len(train)*8//10:len(train)]

    x_test = dataset_final.iloc[len(train):]



    y_train = dataset[columns].iloc[:len(train)*8//10]

    y_val = dataset[columns].iloc[len(train)*8//10:len(train)]

    

    import time

    import xgboost as xgb

    ts = time.time()



    model = xgb.XGBClassifier(

        max_depth=10,

        n_estimators=1000,

        min_child_weight=0.5, 

        colsample_bytree=0.8, 

        subsample=0.8, 

        eta=0.1,

    #     tree_method='gpu_hist',

        seed=42)



    model.fit(

        x_train, 

        y_train, 

        eval_metric="rmse", 

        eval_set=[(x_train, y_train), (x_val, y_val)], 

        verbose=True, 

        early_stopping_rounds = 20)



    print(time.time() - ts)



    Y_test_class = model.predict(x_test)

    

    df = pd.DataFrame(data=Y_test_class, columns=["column1"])

    df['column1'].value_counts().sort_values().plot(kind = 'barh')

    

    return Y_test_class
dataset.columns
Y1 = prediction_column('RIBP')
x_train = dataset_final.iloc[:len(train)*8//10]

x_val = dataset_final.iloc[len(train)*8//10:len(train)]

x_test = dataset_final.iloc[len(train):]



y_train = dataset['P5DA'].iloc[:len(train)*8//10]

y_val = dataset['P5DA'].iloc[len(train)*8//10:len(train)]



import time

import xgboost as xgb

ts = time.time()



model = xgb.XGBClassifier(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



print(time.time() - ts)



Y_test_class = model.predict(x_test)

Y_test_class
df = pd.DataFrame(data=Y_test_class, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
test
Y1 = prediction_column('RIBP')
submission_draft = pd.DataFrame()

submission_draft['index'] = test['ID']

submission_draft['P5DA'] = Y_test_class

submission_draft['RIBP'] = Y1

for column in dataset.columns[-20:-1]:

    submission_draft[column] = prediction_column(column)

submission_draft
submission_draft
submission_draft.index = submission_draft['index']

submission_draft = submission_draft.drop(['index'],axis=1)

submission_draft
submission_draft
output = []

for i in range(len(submission_draft)):

    output += list(submission_draft.iloc[0].values)

output
submission_sample = submission.copy()

submission_sample['Label'] = output

submission_sample
submission_sample.to_csv('submission_XGBClassifier.csv',index=False)
submission_sample['Label'].value_counts().sort_values().plot(kind = 'barh')
submission
dict_map
for key in dict_map:

    print(key, dict_map[key])
dataset_final['occupation_code'].value_counts()
len(dataset_final['occupation_code'].value_counts().index),dataset_final['occupation_code'].value_counts().index
dict_map
list(dict_map.values())
dict_map.values()
dict_map
df_error.index
len(df_error), sum(df_error.values)
df_col.plot(kind = 'barh')
df_col
df_error
df_error.sort_values().plot(kind = 'barh')
train['occupation_code'].value_counts()[train['occupation_code'].value_counts() == 5].value_counts().sort_values().plot(kind = 'barh')
df_error.plot(kind = 'barh')
df_error.value_counts().sort_values().plot(kind = 'barh')
df_error.value_counts().sort_values().plot(kind = 'barh')
df_error.value_counts().values
sum(np.array(df_error.value_counts().index)*np.array(df_error.value_counts().values))
df_error.value_counts()
df_col.plot(kind = 'barh')
df_col.value_counts().plot(kind = 'barh')
df_col.value_counts().sort_values().plot(kind = 'barh')
df_col.value_counts()
test.columns
test.columns[:8]
test.columns[8:]
len(test.columns[8:])
submission