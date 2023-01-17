# accuracy score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
train = pd.concat([train, pd.get_dummies(train['isTruthful'], prefix='isTruthful')],axis=1)

print (train.describe())
y = train.isTruthful

columns=['try0', 'try1', 'try2', 'try3', 'try4', 'try5', 'try6', 'try7', 'try8', 'try9', 'try10', 'try11']

X = train[columns]
#X = train.drop(columns=['Id', 'isTruthful'])

model = DecisionTreeClassifier()
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# train model on training subset
model.fit(train_X, train_y)

# get predicted result for validation subset
predicted_isTruthful = model.predict(val_X)

# show mean error for accuracy
print(accuracy_score(val_y, predicted_isTruthful))

model2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(100, 2), random_state=1)

model2.fit(train_X, train_y)

predicted_isTruthful2 = model2.predict(val_X)

# show mean error for accuracy
print(accuracy_score(val_y, predicted_isTruthful2))


tf.logging.set_verbosity(tf.logging.ERROR)
print('TensorFlow version: ', tf.__version__)

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.


isTruthful_col = tf.feature_column.numeric_column('isTruthful')
try0 = tf.feature_column.numeric_column('try0')
try1 = tf.feature_column.numeric_column('try1')
try2 = tf.feature_column.numeric_column('try2')
try3 = tf.feature_column.numeric_column('try3')
try4 = tf.feature_column.numeric_column('try4')
try5 = tf.feature_column.numeric_column('try5')

features_tf = [try0, try1, try2, try3, try4, try5]
integer_encoded=[0, 1]

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform([[0], [1]], [[0, 1], [1, 0]])
onehot_encoded = onehot_encoder.transform([[0], [1], [1], [1], [1], [1], [0]])
onehot_encoded = onehot_encoded.reshape(-1, 2)
#print(onehot_encoded)


# 1. INSTANTIATE
def indices_to_one_hot(data):
    print(data.describe())
    targets = data.values.reshape(-1, 1)
    targets = onehot_encoder.transform(targets) 
    targets.reshape(-1, 2)
    print(targets)    
    data.isTruthful = targets    
    print(data)
    return data

input_func = tf.estimator.inputs.pandas_input_fn(x=train_X,y=train_y,batch_size=100, num_epochs=1000,shuffle=True)

model = tf.estimator.DNNClassifier(feature_columns=features_tf, hidden_units=[10, 5])
 
model.train(input_fn=input_func,steps=1000)

pred_input_func= tf.estimator.inputs.pandas_input_fn(x=val_X, y=val_y, batch_size=100, num_epochs=1, shuffle=False)
result_tf=model.evaluate(pred_input_func)
# y_pred= [d['logits'] for d in result_tf]

print(result_tf['accuracy'])
result_tf2=model.predict(pred_input_func)
print(result_tf2)
y_pred2= [np.argmax(d['probabilities']) for d in result_tf2]
print(y_pred2)



