import pandas as pd

import numpy as np
df = pd.read_csv("../input/train_jqd04QH.csv")
df.head(2)
df.info()
def fill_all_na(df):

  df['gender'] = df["gender"].fillna("undesclosed")

  df['enrolled_university'] = df['enrolled_university'].fillna('no_school')

  df['education_level'] = df['education_level'].fillna("no_education")

  df['major_discipline'] = df['major_discipline'].fillna("no_major")

  df['experience']=df['experience'].fillna(df['experience'].mean()) ##This is the most important feature eng area

  df['company_size'] = df['company_size'].fillna('no_size')

  return df

def change_exp(df):

  lst = []

  for i in df['experience']:

    if isinstance(i, str):

      i = str(i)

      i = i.replace('>','')

      i = i.replace('<','')

      lst.append(int(i))

    else:

      lst.append(i)

  df['experience'] = lst

  return df

df = change_exp(df)

df = fill_all_na(df)
df = change_exp(df)

df.head()
df.company_type.value_counts()
df['company_type'] = df['company_type'].fillna('Pvt Ltd')
df['last_new_job'] = df['last_new_job'].fillna('unknown_job')
df.info()
df.isna().sum()*100/len(df)
df_test = pd.read_csv('../input/test_KaymcHn.csv')

df_test.head()
df_train = df
for col in df_train.columns:

    if col not in df_test.columns:

        print(col)
df_train.dtypes
for col in df_train.columns:

  print(col, ' ', str(df_train[col].nunique()))
for col in df_test.columns:

  print(col, ' ', str(df_test[col].nunique()))
df_train.drop(columns=['enrollee_id'], axis=1, inplace=True)

df_train.dtypes
df_test = change_exp(df_test)

df_test.head()
for exp in df_train.experience:

  if exp not in df_test.experience:

    print(exp)
for training_hours in df_train.training_hours:

  if training_hours not in df_test.training_hours:

    print(training_hours)
df_train = pd.get_dummies(df_train, columns=df_train.columns)

df_train.head()
df_train.drop(columns=['target_0'], axis=1, inplace=True)

df_train.head()
X = df_train.drop(columns=['target_1'], axis=1).values

Y = df_train['target_1'].values

Y = Y.reshape(-1,1)

print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.8)

print(X_train.shape)

print(X_test.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_test.shape)

print(Y_val.shape)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(300, input_dim=X.shape[1], activation='relu'))

model.add(Dense(150, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs=3, validation_data=(X_val, Y_val))

print(model.metrics_names)

print(model.evaluate(X_test, Y_test))
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score

Y_predict = xgb_clf.predict(X_test)

accuracy_score(Y_test, Y_predict)
df_test = change_exp(df_test)

df_test = fill_all_na(df_test)

df_test.head()
df_test.dtypes
df_test.drop(columns=['enrollee_id'], axis=1, inplace=True)

df_test.dtypes
df_test = pd.get_dummies(df_test, columns=df_test.columns)

df_test.shape
df_train.shape ## This includes target values
for col in df_train.columns:

    if col not in df_test.columns:

        print(col)

        df_test[col] = 0
df_test.shape ## This includes target values
for col in df_test.columns:

    if col not in df_train.columns:

        print(col)
df_test['experience_10.41792349726776'] = df_test['experience_10.383988782800294']

df_test['experience_10.41792349726776'].value_counts()
df_test.drop(columns=['experience_10.383988782800294'],axis=1, inplace=True)

df_test.shape
X_orignal_test = df_test.drop(columns=['target_1'], axis=1).values

X_orignal_test.shape
Y_final_predict = xgb_clf.predict(X_orignal_test)
df_sample = pd.read_csv('/content/sample_data/sample_submission_sxfcbdx.csv')

df_sample.head()
df_sample.target = Y_final_predict

df_sample.head()
df_sample.target.value_counts()
df_sample.to_csv('/content/sample_data/prediction_xgb_without_fet_eng.csv', index=False)