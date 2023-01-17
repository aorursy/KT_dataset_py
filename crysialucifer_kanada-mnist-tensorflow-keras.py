import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")



print(f"Train data shape {train.shape}")

print(f"Test data shape {test.shape}")

print(f"Dig shape {Dig_MNIST.shape}")
train.head(3)
test.head(3)
X = train.iloc[:,1:].values

y = train.iloc[:,0].values

val = test.iloc[:,1:].values
y = tf.keras.utils.to_categorical(y)

y.shape
X_flattened = X.reshape(X.shape[0],28,28,1)

val_flattened = val.reshape(val.shape[0],28,28,1)



print(f"Train Flattened image shape {X_flattened.shape}")

print(f"Validation Flattened image shape {val_flattened.shape}")
X_rescaled = X_flattened/255

val_rescaled = val_flattened/255
x_train,x_test,y_train,y_test = train_test_split(X_rescaled,y,train_size=0.85,random_state=10)
plt.imshow(x_train[0][:,:,0])
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),    

    

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

 

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

model.summary()
history = model.fit(x_train,y_train,batch_size=512,epochs=20,validation_data=[x_test,y_test])
predictions = model.predict_classes(val_rescaled)
sample_submission['label'] = pd.Series(predictions)

sample_submission.head()
sample_submission.to_csv("submission.csv",index=False)