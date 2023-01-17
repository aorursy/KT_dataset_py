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
salary = pd.read_csv('/kaggle/input/salary-slab-classification/Salary_data.csv')
salary.head()
salary.income_bracket.unique()

# salary['income_bracket'].unique()
def label_fix(label):

    if label == ' <=50K':

        return 0

    else:

        return 1
salary['income_bracket'] = salary['income_bracket'].apply(label_fix)
salary['income_bracket'].unique()
salary.head()
from sklearn.model_selection import train_test_split
X = salary.drop('income_bracket', axis = 1)

y = salary['income_bracket']
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
import tensorflow as tf
gender = tf.compat.v1.feature_column.categorical_column_with_vocabulary_list('gender',["Female", "Male"])
occupation = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size = 500)
marital_status = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size = 500)

relationship = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size = 500)

education = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size = 500)

workclass = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size = 500)

native_country = tf.compat.v1.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size = 500)
age                 = tf.compat.v1.feature_column.numeric_column('age')

education_num       = tf.compat.v1.feature_column.numeric_column('education_num')

capital_gain        = tf.compat.v1.feature_column.numeric_column('capital_gain')

capital_loss        = tf.compat.v1.feature_column.numeric_column('capital_loss')

hours_per_week      = tf.compat.v1.feature_column.numeric_column('hours_per_week')
feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country, age, education_num, capital_gain, capital_loss, hours_per_week]
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10, num_epochs = 1000, shuffle = True)
model = tf.compat.v1.estimator.LinearClassifier(feature_columns = feat_cols)
model.train(input_fn = input_func, steps = 10000)
predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)
prediction = list(model.predict(input_fn = predict_input_func))
prediction[0]
final_pred = []



for pred in prediction:

    final_pred.append(pred['class_ids'][0])
final_pred[:10]
from sklearn.metrics import classification_report
print(classification_report(y_test, final_pred))