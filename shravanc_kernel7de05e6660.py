import pandas as pd
train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')



test.head(3)

test=test.drop('id',axis=1)
print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

print('The Test  dataset has {} rows and {} columns'.format(test.shape[0],test.shape[1]))
import numpy as np



training_images = np.array(train.drop('label',axis=1))

training_labels = np.array(train.label)



testing_images  = np.array(test)

#testing_labels  = np.array(test.label)

print("hello")
print(training_images.shape)

training_images = training_images.reshape(60000, 28, 28, 1)

training_images = training_images / 255.0





print(testing_images.shape)

testing_images  = testing_images.reshape(5000, 28, 28, 1)

testing_images  = testing_images / 255.0
import tensorflow as tf
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax"),

])



model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



num_epochs = 10

model.fit(training_images, training_labels,epochs=num_epochs)


test=pd.read_csv('../input/Kannada-MNIST/test.csv')

test_id=test.id



test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)

test.shape
y_pre=model.predict_classes(test)

sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)

sample_sub.head()