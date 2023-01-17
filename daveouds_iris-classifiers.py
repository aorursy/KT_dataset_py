import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load dataset(s).
csv_data = pd.read_csv("../input/Iris.csv")

## Check Dataset
sns.set(style="ticks")
sns.set_palette("husl")
sns.pairplot(csv_data.iloc[:,1:6],hue="Species")

# Prepare the datasets
## Shuffle data
csv_data = csv_data.sample(frac=1).reset_index(drop=True)

## Split data
data = csv_data.iloc[:,1:5].values
labels = csv_data.iloc[:,5].values

## Encode data into numerical notation
encoded_labels = preprocessing.LabelEncoder().fit_transform(labels)

## One-hot encoding (0 = [0,0,0], 1 = [0,1,0], 2 = [1,0,0])
Y = pd.get_dummies(encoded_labels).values

## Initialize train and test variables
x_train, x_test, y_train, y_test = train_test_split(data,Y,test_size=0.2,random_state=0) 

# Neural Network Model using Estimators
def input_fn(dataset, labels):
    def _fn():
        features = {feature_name: tf.constant(dataset)}
        label = tf.constant(np.argmax(labels, axis=1))
        return features, label
    return _fn

feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                    shape=[4])]

# Define the model
classifier = tf.estimator.LinearClassifier(
                feature_columns=feature_columns, 
                n_classes=3,
                model_dir="./models/linear1"
                )

# Train the model.
classifier.train(input_fn=input_fn(x_train, y_train),
                steps=1000)
print('fit done')

## Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(x_test, y_test), 
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))
# Deep Neural Network Model using Estimators
def input_fn(dataset, labels):
    def _fn():
        features = {feature_name: tf.constant(dataset)}
        label = tf.constant(np.argmax(labels, axis=1))
        return features, label
    return _fn

feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                    shape=[4])]
# Define Model
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="/tmp/iris_model",
    hidden_units=[100, 70, 50, 25])

# Train the model.
classifier.train(input_fn=input_fn(x_train, y_train),
                steps=1000)
print('fit done')

## Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(x_test, y_test), 
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

# Deep Neural Network Model using Keras
## Define model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(10,input_shape=(4,),activation='tanh'))
model.add(tf.keras.layers.Dense(8,activation='tanh'))
model.add(tf.keras.layers.Dense(6,activation='tanh'))
model.add(tf.keras.layers.Dense(3,activation='softmax'))

## Compile the model
model.compile(tf.keras.optimizers.Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])

## Train the model
model.fit(x_train,y_train,epochs=100)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('test_acc:', test_acc)
print('test_loss:', test_loss)