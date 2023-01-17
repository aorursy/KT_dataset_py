# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score, accuracy_score
from h2o.automl import H2OAutoML
import h2o
h2o.init()
%%time
import os
import json
from tqdm import tqdm_notebook as tqdm

def load_desc_sentiment(path):
    all_desc_sentiment_files = os.listdir(path)
    count_file = len(all_desc_sentiment_files)
    desc_sentiment_df = pd.DataFrame(columns=['PetID','desc_senti_magnitude','desc_senti_score'])
    current_file_index = 1
    for filename in tqdm(all_desc_sentiment_files):
        with open(path+filename, 'r') as f:
            sentiment_json = json.load(f)
            petID = filename.split('.')[0]
            magnitude = sentiment_json['documentSentiment']['magnitude']
            score = sentiment_json['documentSentiment']['score']
            desc_sentiment_df = desc_sentiment_df.append({'PetID': petID, 'desc_sentiment_df':magnitude,'desc_senti_score':score}, \
                                                         ignore_index=True)
    
    desc_sentiment_df['desc_sent_mult'] = desc_sentiment_df['desc_sentiment_df'] * desc_sentiment_df['desc_senti_score']
    return desc_sentiment_df

df_train = pd.read_csv("../input/train/train.csv")
df_train_sent = load_desc_sentiment("../input/train_sentiment/")

df_test = pd.read_csv("../input/test/test.csv")
df_test_sent = load_desc_sentiment("../input/test_sentiment/")
def build_total_df(df_data, df_desc_sent):
    df_total = pd.merge(df_data, df_desc_sent, on='PetID', how='outer')
    df_total['words_count'] = df_train['Description'].fillna('').apply(lambda x: len(x.split()))
    df_total['desc_len'] = df_train['Description'].fillna('').apply(lambda x: len(x))
    df_total.fillna(0, inplace=True)
    return df_total

df_total_train = build_total_df(df_train, df_train_sent)
df_total_test = build_total_df(df_test, df_test_sent)
%%time
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def regression_h2o(X_train, X_test, y_name, project_name):
    aml = H2OAutoML(max_runtime_secs=180, seed=1, project_name=project_name)
    aml.train(y=y_name, training_frame=X_train, leaderboard_frame=X_test)
    return aml

number_cols = df_total_train.describe().columns
data = df_total_train[number_cols]
X_train, X_test = train_test_split(data)
df_total_train = df_total_train.fillna(0)

print("use cols:", number_cols)
X_train = h2o.H2OFrame(X_train.fillna(0))
X_test = h2o.H2OFrame(X_test.fillna(0))
y_name = 'AdoptionSpeed'

model = regression_h2o(X_train, X_test, y_name, "automl_h2o")
print(model.leaderboard.head())
y_pred = model.predict(X_test).as_data_frame()['predict'].values
y_test = X_test.as_data_frame()[y_name].values
# 0.23752238131720194
# 0.522885 mse
y_pred_int = y_pred.round()
print("new kappa:", kappa(y_test, y_pred_int), "mse:", mean_squared_error(y_test, y_pred))
# print("kappa:", quadratic_weighted_kappa(y_test, y_pred), "mse:", mean_squared_error(y_test, y_pred))
# print("kappa int:", quadratic_weighted_kappa(y_test, y_pred_int), "mse:", mean_squared_error(y_test, y_pred_int), "acc:", accuracy_score(y_test, y_pred_int))
print()

# 'Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
#        'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
#        'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt',
#        'PhotoAmt', 'AdoptionSpeed', 'desc_senti_magnitude', 'words_count',
#        'desc_len'

# kappa: 0.6080146031303728 mse: 0.5448913813566739
# kappa int: 0.6080146031303728 mse: 0.8823686316351027 acc: 0.40037343291544414
# kappa int: 0.6192112374874424 mse: 0.8570285409442518 acc: 0.40037343291544414
number_cols
df_test = df_total_test
df_test.fillna(0, inplace=True)

X_submission = df_test[list(set(number_cols) - {'AdoptionSpeed'})]
y_submission =  model.predict(h2o.H2OFrame(X_submission)).as_data_frame()['predict'].values
y_submission_int = [int(x) for x in y_submission]
df_sample_submisson = pd.read_csv("../input/test/sample_submission.csv")
df_sample_submisson['AdoptionSpeed'] = y_submission_int
df_sample_submisson.to_csv("submission.csv", index=False)
