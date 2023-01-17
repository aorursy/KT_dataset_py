!pip install -qU pip

!pip install -qU tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
!head -n5 ../input/train.csv
column_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

select_columns = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

label_name = 'Survived'
train_dataset = tf.data.experimental.make_csv_dataset(

    file_pattern = '../input/train.csv',

    batch_size = 32,

    column_names = column_names,

    select_columns = select_columns,

    label_name = label_name,

    num_epochs = 1)
features, labels = next(iter(train_dataset))

print(features)
from tensorflow import feature_column

from tensorflow.keras import layers
feature_columns = []



raw = feature_column.categorical_column_with_identity('Pclass', num_buckets=4)

feature_columns.append(feature_column.indicator_column(raw))

raw = feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])

feature_columns.append(feature_column.indicator_column(raw))

feature_columns.append(feature_column.numeric_column('Age'))

raw = feature_column.categorical_column_with_identity('SibSp', num_buckets=9)

feature_columns.append(feature_column.indicator_column(raw))

raw = feature_column.categorical_column_with_identity('Parch', num_buckets=12)

feature_columns.append(feature_column.indicator_column(raw))

feature_columns.append(feature_column.numeric_column('Fare'))

raw = feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])

feature_columns.append(feature_column.indicator_column(raw))



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.models.Sequential([

    feature_layer,

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.5), 

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')

])
model.compile(

    optimizer = 'adam',

    loss = 'binary_crossentropy',

    metrics = ['accuracy']

)
model.fit(train_dataset, epochs=100)
test_column_names = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

test_select_columns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']



test_dataset = tf.data.experimental.make_csv_dataset(

    shuffle = False,

    file_pattern = '../input/test.csv',

    batch_size = 64,

    column_names = test_column_names,

    select_columns = test_select_columns,

    num_epochs = 1)
features = next(iter(test_dataset))

print(features)
predictions = model.predict(test_dataset) > 0.5
predictions[:, 0]
import pandas as pd
test_df = pd.read_csv('../input/test.csv')
final_df = pd.DataFrame()
final_df['PassengerId'] = test_df['PassengerId']
final_df['Survived'] = predictions[:, 0]*1
final_df.head()
final_df.to_csv('submission.csv', index=False)
!head -n5 submission.csv