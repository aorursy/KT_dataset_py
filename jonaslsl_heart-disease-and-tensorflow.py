# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # charts

import pandas_profiling as pp # pandas reporting tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading and check loaded data

url = '/kaggle/input/heart-disease-uci/heart.csv'

data = pd.read_csv(url)

data.head(10)
# Verifying dataset size

data.shape
data.describe()
# Seaching for null values

data.isnull().any()
pp.ProfileReport(data)
import tensorflow as tf

from sklearn.model_selection import train_test_split
target_name = 'target'

data_target = data[target_name]

data = data.drop([target_name], axis=1)



train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

data.head(10)
#age, ca, chol, oldpeak, thalach, trestbps,              cp, exang, fbs, retecg, sex, slope, thal,



CATEGORICAL_COLUMNS = ['cp', 'exang', 'fbs', 'restecg', 'sex', 'slope', 'thal']

NUMERIC_COLUMNS = ['age', 'ca', 'chol', 'oldpeak', 'thalach', 'trestbps']



feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:

    vocabulary = train[feature_name].unique()

    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    

for feature_name in NUMERIC_COLUMNS:

    feature_columns.append(tf.feature_column.numeric_column(feature_name))

    

    
# Criando um data frame em TF para passagem dos dados como streamming

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

  def input_function():

    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if shuffle:

      ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds

  return input_function



train_input_fn = make_input_fn(train, target)

eval_input_fn = make_input_fn(test, target_test, num_epochs=5, shuffle=False)
# treinando e validando o modelo

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn)



#clear_output()

print(result)