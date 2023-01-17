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
input_dir = '../input/'

train_csv = pd.read_csv(input_dir + 'train.csv')
train_csv.sample(10)
train_csv.info()
train_csv.describe()
for col in train_csv.columns:
    print(col)
unique_count_df = pd.Series()
for col in train_csv.columns:
    unique_count_df[col] = len(train_csv[col].unique())
unique_count_df
train_csv.isnull().sum()
train_csv['Cabin'].str[0]
import tensorflow as tf
def split_data(df: pd.DataFrame):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    num_rows = df.shape[0]
    num_rows_train_data = int(num_rows * 0.8)

    train_data = df_shuffled[:num_rows_train_data]
    val_data = df_shuffled[num_rows_train_data:]
    
    return (train_data, val_data)

def input_fn(df: pd.DataFrame, labels, batch_size: int, num_epochs: int):
    if labels is None:
        input = df.to_dict(orient='series')
    else:
        input = (df.to_dict(orient='series'), labels)
    dataset = tf.data.Dataset.from_tensor_slices(input)
    return dataset.shuffle(buffer_size=10000).repeat(count=num_epochs).batch(batch_size)

def eval_input_fn(df: pd.DataFrame, labels):
    if labels is None:
        input = df.to_dict(orient='series')
    else:
        input = (df.to_dict(orient='series'), labels)
    dataset = tf.data.Dataset.from_tensor_slices(input)
    return dataset.batch(128)

def embedding_dimension(unique_count):
    return min(50, unique_count // 2)

def define_feature_columns(df: pd.DataFrame, numeric_columns, categorical_columns):
    feature_columns = []
    for col in df.columns:
        if col in categorical_columns:
            sorted_unique_values = sorted(set(list(df[col].unique()) + ['']))
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=sorted_unique_values)
            embedding_dim = embedding_dimension(len(sorted_unique_values))
            if embedding_dim <= 2:
                feature_columns.append((col, tf.feature_column.indicator_column(cat_col)))
            else:
                feature_columns.append((col, tf.feature_column.embedding_column(cat_col, embedding_dim)))
        elif col in numeric_columns:
            feature_columns.append((col, tf.feature_column.numeric_column(key=col)))
            isnull_col_name = col + 'IsNull'
            feature_columns.append((isnull_col_name, tf.feature_column.numeric_column(key=isnull_col_name)))
    return dict(feature_columns)

def feature_preprocess(df: pd.DataFrame, numeric_columns, categorical_columns, col_mean, col_stddev):
    processed = pd.DataFrame()
    processed['PassengerId'] = df['PassengerId']
    
    for col_name in numeric_columns:
        processed[col_name] = (df[col_name].astype(float) - col_mean[col_name]) / col_stddev[col_name]
        isnull_col_name = col_name + 'IsNull'
        processed[isnull_col_name] = df[col_name].isnull()
        processed.loc[processed[isnull_col_name], col_name] = 0
        processed[isnull_col_name] = processed[isnull_col_name].astype(np.int8)
        
    for col_name in categorical_columns:
        processed[col_name] = df[col_name].copy().astype(str)
        processed.loc[df[col_name].isnull(), col_name] = ''
    
    return processed
input_dir = '../input/'

train_csv = pd.read_csv(input_dir + 'train.csv')
test_csv = pd.read_csv(input_dir + 'test.csv')

train_csv['Deck'] = train_csv['Cabin'].str[0]
test_csv['Deck'] = test_csv['Cabin'].str[0]

train_data, val_data = split_data(train_csv)
print(train_data.shape, val_data.shape)
numeric_columns = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_columns = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Deck']

col_mean = train_data[numeric_columns].mean()
col_stddev = train_data[numeric_columns].std()

def feature_preprocess_1(df: pd.DataFrame):
    return feature_preprocess(df, numeric_columns=numeric_columns, categorical_columns=categorical_columns, col_mean=col_mean, col_stddev=col_stddev)

train_features = feature_preprocess_1(train_data)
val_features = feature_preprocess_1(val_data)

print(train_features.head())

feature_columns = define_feature_columns(
    train_features,
    numeric_columns=numeric_columns,
    categorical_columns=categorical_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=list(feature_columns.values()),
    hidden_units=[64, 32, 16],
    n_classes=2,
    dropout=0.5)
classifier.train(input_fn=lambda: input_fn(train_features, train_data['Survived'], batch_size=64, num_epochs=500))
classifier.evaluate(input_fn=lambda: eval_input_fn(val_features, val_data['Survived']))
submit_data = feature_preprocess_1(test_csv)

predictions = classifier.predict(input_fn=lambda: eval_input_fn(submit_data, None))
predictions = list(predictions)
submit_output_csv = pd.DataFrame({'PassengerId': submit_data['PassengerId'], 'Survived': [x['class_ids'][0] for x in predictions]})
submit_output_csv.head()
submit_output_csv.to_csv('submit_output_1.csv', index=False)
submit_output = pd.read_csv('submit_output_1.csv')
submit_output.head()
