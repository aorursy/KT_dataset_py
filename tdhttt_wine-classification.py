import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NAMES = ['Types','Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315_of_diluted_wines','Proline']
BATCH = 100
STEPS = 1000

def load_data():
    whole = pd.read_csv("../input/wine.data",names=NAMES,header=None)
    # important to randomize
    rwhole = whole.sample(frac=1)
    train = rwhole[:100]
    test = rwhole[100:]

    train_x, train_y = train, train.pop('Types')
    test_x, test_y = test, test.pop('Types')

    return (train_x, train_y), (test_x, test_y)
def train_input_fn(features,labels,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features,labels,batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset
def make_DNNClassifier(fc):
    classifier = tf.estimator.DNNClassifier(
        feature_columns = fc,
        hidden_units = [10,10],
        n_classes = 4)

    return classifier
def make_LinearClassifier(fc):
    classifier = tf.estimator.LinearClassifier(
        feature_columns = fc,
        n_classes = 4)

    return classifier
def make_BaseModel(s):
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,
                           input_shape=(s,)),
        keras.layers.Dense(4,activation=tf.nn.softmax)
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model
def make_BiggerModel(s):
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,
                           input_shape=(s,)),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dense(4,activation=tf.nn.softmax)
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    return model
def make_BaseModelO(s):
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,
                           input_shape=(s,)),
        keras.layers.Dense(4,activation=tf.nn.softmax)
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    
    return model
def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()
    my_feature_columns = []
    for k in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=k))

    DNNclassifier = make_DNNClassifier(my_feature_columns)
    DNNclassifier.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH),steps=STEPS)
    DNNresult = DNNclassifier.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH))
    print("\nDNN normal test accuracy: {accuracy:0.3f}\n".format(**DNNresult))

    Lclassifier = make_LinearClassifier(my_feature_columns)
    Lclassifier.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH),steps=STEPS)
    Lresult = Lclassifier.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH))

    print("\nLinear test accuracy: {accuracy:0.3f}\n".format(**Lresult))

    DNNclassifier1 = make_DNNClassifier(my_feature_columns)
    DNNclassifier1.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH-90),steps=STEPS)
    DNNresult1 = DNNclassifier1.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH-90))
    print("\nDNN small batch(10) test accuracy: {accuracy:0.3f}\n".format(**DNNresult1))

    DNNclassifier2 = make_DNNClassifier(my_feature_columns)
    DNNclassifier2.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH+900),steps=STEPS)
    DNNresult2 = DNNclassifier2.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH+900))
    print("\nDNN large batch(1000) test accuracy: {accuracy:0.3f}\n".format(**DNNresult2))

    DNNclassifier3 = make_DNNClassifier(my_feature_columns)
    DNNclassifier3.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH),steps=STEPS-500)
    DNNresult3 = DNNclassifier3.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH))
    print("\nDNN small step(500) test accuracy: {accuracy:0.3f}\n".format(**DNNresult3))

    DNNclassifier4 = make_DNNClassifier(my_feature_columns)
    DNNclassifier4.train(input_fn=lambda:train_input_fn(train_x,train_y,BATCH),steps=STEPS+500)
    DNNresult4 = DNNclassifier4.evaluate(input_fn=lambda:eval_input_fn(test_x,test_y,BATCH))
    print("\nDNN large step(1500) test accuracy: {accuracy:0.3f}\n".format(**DNNresult4))

    Bmodel = make_BaseModel(train_x.shape[1])
    print(Bmodel.summary())
    Bmodel.fit(train_x, train_y, epochs=5)
    Btest_loss, Btest_acc = Bmodel.evaluate(test_x, test_y)
    print("\nBase Keras test accuracy: {:0.3f}\n".format(float(Btest_acc)))

    Wmodel = make_BiggerModel(train_x.shape[1])
    print(Wmodel.summary())
    Wmodel.fit(train_x, train_y, epochs=5)
    Wtest_loss, Wtest_acc = Wmodel.evaluate(test_x, test_y)
    print("\Larger Keras test accuracy: {:0.3f}\n".format(float(Btest_acc)))

    Omodel = make_BaseModel(train_x.shape[1])
    print(Omodel.summary())
    Omodel.fit(train_x, train_y, epochs=5)
    Otest_loss, Otest_acc = Bmodel.evaluate(test_x, test_y)
    print("\nBase with alternative loss accuracy: {:0.3f}\n".format(float(Otest_acc)))

    Bmodel = make_BaseModel(train_x.shape[1])
    print(Bmodel.summary())
    Bmodel.fit(train_x, train_y, epochs=10)
    Btest_loss, Btest_acc = Bmodel.evaluate(test_x, test_y)
    print("\nBase Keras with more epochs test accuracy: {:0.3f}\n".format(float(Btest_acc)))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)