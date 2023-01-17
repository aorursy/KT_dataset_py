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
#1

df_hearth_failure = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df_hearth_failure.head()
df_hf = df_hearth_failure[['DEATH_EVENT', 'time', 'age', 'high_blood_pressure', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']].copy()

df_hf.head()
age_text = lambda x: 'low' if(x<=56) else ('high' if(x>=73) else 'medium')

df_hf['age_text'] = df_hf['age'].apply(age_text)

df_hf.head()
df_hf = df_hf.drop(['age'], axis=1)

df_hf.head()
from sklearn.preprocessing import OneHotEncoder

import numpy as np



encoder = OneHotEncoder(handle_unknown='ignore')



encoder.fit(np.c_[df_hf['age_text']])

encoder.categories
transformed = encoder.transform(np.c_[df_hf['age_text']])

df_oh = pd.DataFrame(transformed.toarray())

df_oh.columns = ['high', 'low', 'medium']

df_oh.head()
df_hf[['age_text0', 'age_text1', 'age_text2']] = df_oh[['high', 'low', 'medium']]

df_hf = df_hf.drop(['age_text'], axis=1)

df_hf.head()
df_hf[['ejection_fraction', 'serum_creatinine', 'serum_sodium']].describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_hf[['ejection_fraction', 'serum_creatinine', 'serum_sodium']])

print(scaler.mean_)
df_hf[['ejection_fraction_sc', 'serum_creatinine_sc', 'serum_sodium_sc']] = scaler.transform(df_hf[['ejection_fraction', 'serum_creatinine', 'serum_sodium']])

df_hf.head()
askbjd = df_hf[['ejection_fraction', 'serum_creatinine', 'serum_sodium']]#saving just in case

df_hf = df_hf.drop(['ejection_fraction', 'serum_creatinine', 'serum_sodium'], axis=1)
df_hf['serum_creatinine_sc_log'] = askbjd['serum_creatinine'].apply(np.log)

df_hf['serum_creatinine_sc_log'].hist()
df_hf['serum_creatinine_sc'].hist()

askbjd['serum_creatinine_sc'] = df_hf['serum_creatinine_sc']

df_hf = df_hf.drop(['serum_creatinine_sc'], axis=1)
df_hf.head()
df_hf.head()

y = df_hf['DEATH_EVENT']

X = np.c_[df_hf[['age_text0', 'age_text1', 'age_text2', 'high_blood_pressure', 'ejection_fraction_sc', 'serum_creatinine_sc_log', 'serum_sodium_sc']]]

print('success')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('success')

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)
y_train_pred = log_reg.predict(X_train)

y_test_pred = log_reg.predict(X_test)

y_test_pred_naive = pd.Series([int(round(y.sum()/len(y))) for _ in range(len(y_test))])

print('predicted')
len(y_test_pred_naive) == len(y_test)
from sklearn.metrics import accuracy_score

print('TRAIN SCORE')

print(accuracy_score(y_train ,y_train_pred))



print('TEST SCORE')

print(accuracy_score(y_test ,y_test_pred))



print('NAIVE SCORE')

print(accuracy_score(y_test ,y_test_pred_naive))

from sklearn.metrics import confusion_matrix

print('Format')

print('TN, FP')

print('FN, TP')

print('TRAIN CM SCORE')

print(confusion_matrix(y_train ,y_train_pred))

print('TEST CM SCORE')

print(confusion_matrix(y_test ,y_test_pred))



print('NAIVE CM SCORE')

print(confusion_matrix(y_test ,y_test_pred_naive))
import xgboost as xgb



X_trainxg, X_valid, y_trainxg, y_valid = train_test_split(X_train, y_train, test_size=0.05, random_state=42)





model=xgb.XGBClassifier(learning_rate=0.05, n_estimators=1000)

model.fit(X_trainxg, y_trainxg, eval_set=[(X_valid, y_valid)], early_stopping_rounds=2500, verbose=False)
xg_pred = model.predict(X_test)

print(accuracy_score(xg_pred ,y_test))
i_see_dead_people= df_hf[df_hf['DEATH_EVENT'] == 1].copy()

i_see_dead_people.head()
y = i_see_dead_people['time']

X = i_see_dead_people[['age_text0', 'age_text1', 'age_text2', 'high_blood_pressure', 'ejection_fraction_sc', 'serum_creatinine_sc_log', 'serum_sodium_sc']]

X.head()
X = np.c_[X]
from sklearn.model_selection import train_test_split

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)

print('success')
from sklearn.linear_model import LinearRegression



lr_model = LinearRegression()



lr_model.fit(X_train_r, y_train_r)
y_train_pred_r = lr_model.predict(X_train_r)

y_test_pred_r = lr_model.predict(X_test_r)

y_test_pred_naive_r = pd.Series([y_train_r.sum()/len(y_train_r) for _ in range(len(y_test_pred_r))])

print('Done')
from sklearn.metrics import mean_squared_error

print('TRAIN MSE')

print(mean_squared_error(y_train_r, y_train_pred_r))

print('TEST MSE')

print(mean_squared_error(y_test_r, y_test_pred_r))

print('NAIVE MSE')

print(mean_squared_error(y_test_r, y_test_pred_naive_r))

from sklearn.metrics import mean_absolute_error

print('TRAIN MAE')

print(mean_absolute_error(y_train_r, y_train_pred_r))

print('TEST MAE')

print(mean_absolute_error(y_test_r, y_test_pred_r))

print('NAIVE MAE')

print(mean_absolute_error(y_test_r, y_test_pred_naive_r))

from sklearn.neighbors import KNeighborsClassifier



NN = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')

NN.fit(X_train_r, y_train_r)

nny_train_pred_r = NN.predict(X_train_r)

nny_test_pred_r = NN.predict(X_test_r)

print('Done')
from sklearn.metrics import mean_absolute_error

print('TRAIN MAE')

print(mean_absolute_error(y_train_r, nny_train_pred_r))

print('TEST MAE')

print(mean_absolute_error(y_test_r, nny_test_pred_r))

print('NAIVE MAE')

print(mean_absolute_error(y_test_r, y_test_pred_naive_r))