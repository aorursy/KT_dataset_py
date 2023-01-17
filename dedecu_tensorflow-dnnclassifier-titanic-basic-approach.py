import pandas as pd
import numpy as np
train = pd.read_csv('../input/titanic-test-ready/train-ready.csv')
test = pd.read_csv('../input/titanic-test-ready/test-ready.csv')
train.head(5)
def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset
from sklearn.model_selection import train_test_split 
y = train.pop('Survived')
X = train
# 20% for evaluate
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23) 
import tensorflow as tf

feature_columns = []

for key in X_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=2)
batch_size = 100
train_steps = 400

for i in range(0,100):
    
    classifier.train(
        input_fn=lambda:train_input_fn(X_train, y_train,
                                                 batch_size),
        steps=train_steps)
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(X_tmp, y_tmp,batch_size)
)
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(test,labels=None,
    batch_size=batch_size))
results = list(predictions)

def x(res,j):
    class_id = res[j]['class_ids'][0]
    probability = int(results[j]['probabilities'][class_id] *100)

    if int(class_id) == 0:
        return ('%s%% probalitity to %s' % (probability,'Not survive'))
    else:
        return ('%s%% probalitity to %s' % (probability,'Survive!'))

print ('Predictions for 10 first records on test(dataset):')

for i in range(0,10):    
    print (x(results,i))
len(results)
len(train)
passengers = {}
i = 892
for x in results:
    passengers[i] = int(x['class_ids'][0])
    i+=1
import csv
csvfile = 'submissions.csv'
with open(csvfile, 'w') as f:
    outcsv = csv.writer(f, delimiter=',')
    header = ['PassengerId','Survived']
    outcsv.writerow(header)
    for k,v in passengers.items():
        outcsv.writerow([k,v])
submissions = pd.read_csv(csvfile)
submissions.head(5)