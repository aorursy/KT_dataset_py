import pandas as pd

import tensorflow as tf

import csv

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
base = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
base.head()
test.head()
base['Pclass'].unique()
base['Sex'].unique()
base['SibSp'].unique()
base['Parch'].unique()
base['Ticket'].unique()
base['Cabin'].unique()
base['Embarked'].unique()
base['Embarked'] = base['Embarked'].fillna('miss')
test['Embarked'] = test['Embarked'].fillna('miss')
base['Embarked'].unique()
base['Cabin'] = base['Cabin'].fillna('miss')
test['Cabin'] = test['Cabin'].fillna('miss')
base['Cabin'].unique()
# X values are all columns except for Survived

X = base.drop('Survived', axis=1)
X.head()
### y values are only the Survived column

y = base['Survived']
y.head()
# create feature columns list

feature_columns = []
# Graph shows how many people have by age ranges

age_hist = X.Age.hist()
# saves age ranges in the variable

age_boundaries = age_hist.get_xticks()
# convert numpy array to list

age_boundaries = age_boundaries.tolist()
age_boundaries
# remove negative value from list

age_boundaries.pop(0)
age_boundaries
# create age feature column

age_fc = tf.feature_column.numeric_column('Age')
# create bucketized age column

age_categorical = tf.feature_column.bucketized_column(

    age_fc,

    boundaries=age_boundaries

)
feature_columns.append(age_categorical)
list_columns_vocabulary = ['Sex', 'Embarked', 'Ticket', 'Cabin']
# create vocabulary list columns

vocabulary_columns = [

    tf.feature_column.categorical_column_with_vocabulary_list(

        key=c,

        vocabulary_list=X[c].unique()

    ) for c in list_columns_vocabulary

]
# add categorical to embedding columns

for column in vocabulary_columns:

    feature_columns.append(

        tf.feature_column.embedding_column(column, dimension=base.shape[0])

    )
list_numeric_columns = ['PassengerId', 'Pclass', 'SibSp', 'Parch']
for c in list_numeric_columns:

    feature_columns.append(tf.feature_column.numeric_column(key=c))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape
X_test.shape
def train_input_fn(features, labels, batch_size=32):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)
def eval_input_fn(features, labels, batch_size=32):

    features = dict(features)

    if labels is None:

        inputs = features

    else:

        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    dataset = dataset.batch(batch_size)

    return dataset
classifier = tf.estimator.DNNClassifier(

    hidden_units=[8,8,8,8],

    feature_columns=feature_columns,

    n_classes=2,

    activation_fn=tf.nn.relu,

    optimizer='Adam'

)
batch_size = 32

train_steps = 10000
classifier.train(

        input_fn=lambda:train_input_fn(X_train, y_train, batch_size),

        steps=train_steps

    )
eval_result = classifier.evaluate(

    input_fn=lambda:eval_input_fn(X_test, y_test, batch_size)

)
eval_result
predictions = []

for p in classifier.predict(input_fn=lambda:eval_input_fn(test, labels=None, batch_size=batch_size)):

    predictions.append(p['class_ids'])
predictions
passengers = {}

_id = 892

for results in predictions:

    passengers[_id] = int(results[0])

    _id+=1
passengers
csvfile = 'submission.csv'

with open(csvfile, 'w') as f:

    outcsv = csv.writer(f, delimiter= ',')

    header = ['PassengerId', 'Survived']

    outcsv.writerow(header)

    for k, v in passengers.items():

        outcsv.writerow([k, v])
submission = pd.read_csv(csvfile)
submission.head()