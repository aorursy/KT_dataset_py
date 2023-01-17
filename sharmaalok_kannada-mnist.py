import tensorflow as tf

import numpy as np

import pandas as pd

from tensorflow import keras
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

train_data.describe()
print(train_data.head())



print(np.shape(train_data))
#train_data[0]

train_data.columns
set(train_data['label'])
train_feature = train_data.drop(columns=['label'])

train_feature = train_feature/255

train_target = train_data['label']
model = keras.Sequential([

    tf.keras.layers.Dense(128, activation='relu', input_shape=[len(train_feature.keys())]),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer='sgd',

    metrics=['accuracy']

)
#Now we train our model by model.fit
print(set(train_data['pixel500']))
model.fit(train_feature, train_target, epochs=10)
!ls

#predictions = model.predict()
evaluate_data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
evaluate_data.columns
evaluate_feature = evaluate_data.drop(columns=['label'])

evaluate_feature = evaluate_feature/255

evaluate_target = evaluate_data['label']
model.evaluate(evaluate_feature, evaluate_target)
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
test_data.columns

#predictions = model.predict(test_data)
len(set(test_data['id']))
test_data['id'][:10]
test_data = test_data.drop(columns=['id'])
predictions = model.predict(test_data)
predictions[0]
np.argmax(predictions[0])
fo = open('/kaggle/working/Submission1.csv', 'w')

fo.write('id' + ',' + 'label' + '\n')



for i in range(5000):

    fo.write(str(i) + ',' + str(np.argmax(predictions[i])) + '\n')

    

fo.close()
import matplotlib.pyplot as plt
predict_data = pd.read_csv('Submission1.csv')



predict_data.head()
plt.scatter(predict_data['id'], predict_data['label'])

plt.show()
#The End

#Please ignore submission1.csv file