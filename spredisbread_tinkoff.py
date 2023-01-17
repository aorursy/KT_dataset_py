import argparse
import json
import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import pandas
import pandas as pd
import pandas as pd
import tqdm
from catboost import (
    CatBoost, CatBoostClassifier, CatBoostClassifier, CatBoostRegressor, CatBoostRegressor,
    FeaturesData, Pool,
)
from hyperopt import hp, space_eval, STATUS_OK, tpe, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_submission_path():
    return datetime.now().strftime('./submission_%d.%m.%Y_%H.%M.%S.csv')


df_train = pd.read_csv('../input/tinkoff-exam/train.txt', index_col='id')
df_test = pd.read_csv('../input/tinkoff-exam/test.txt', index_col='id')


def processing(df):
    genders = ['муж.', 'жен.']
    degrees = ['высшее', 'начальное', 'неоконч. высшее',
        'ср. общ.', 'ср. специальное', 'среднее', 'уч. степень']
    height = ['низкий', 'средний', 'высокий']

    df["x5"] = df["x5"].replace({element: index for index, element in enumerate(genders)})
    df['x6'] = df['x6'].replace({degree: degrees.index(degree) for degree in degrees})
    df['x11'] = df['x11'].replace({element: index for index, element in enumerate(height)})
    
    return df


df_train = processing(df_train)
df_test = processing(df_test)


X = df_train.drop(columns=['y'])
X_test = df_test
y = df_train['y']

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_validation.shape)

model = CatBoostClassifier(
    iterations=20000, thread_count=8, use_best_model=True,
    eval_metric='AUC'
)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation))
print('Train accuracy:', accuracy_score(y, model.predict(X)))

y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
submission_path = get_submission_path()
print(f'Saving to {submission_path} ...')
df_submission = pandas.DataFrame({'proba': y_pred}, index=df_test.index)
df_submission.to_csv(submission_path)
df_submission.to_csv(submission_path)
