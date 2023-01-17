import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(tf.__version__)

import os
print(os.listdir("../input"))
iris = pd.read_csv('../input/Iris.csv').set_index('Id')
iris.head(5)
iris.info()
iris.describe()
iris.groupby('Species').count()
sn.boxplot(x='Species', y='SepalLengthCm', data=iris)
sn.boxplot(x='Species', y='SepalWidthCm', data=iris)
sn.boxplot(x='Species', y='PetalLengthCm', data=iris)
sn.boxplot(x='Species', y='PetalWidthCm', data=iris)
sn.pairplot(iris,hue='Species')
plt.show()
iris.corr()
sn.heatmap(iris.corr(), annot=True)
X = iris.drop(['Species'],axis=1)
y = iris['Species'].replace({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape, X_test.shape)
# data is shuffled
X_train.head(5)
y_train.head(5)
# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.int)
y_test  = np.array(y_test).astype(np.int)

print(X_train)
print(y_train)
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,shape=[4])]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                          n_classes=3,
                                          model_dir='/tmp/iris_model')
def input_fn(X,y):
    def _fn():
        features = {feature_name: tf.constant(X)}
        label = tf.constant(y)
        return features, label
    return _fn

print(input_fn(X_train,y_train)())
        
classifier.train(input_fn=input_fn(X_train,y_train), steps=1000)
print('fit done')
accuracy = classifier.evaluate(input_fn=input_fn(X_test,y_test),steps=100)['accuracy']
print('\nAccuracy: {0:f}'.format(accuracy))