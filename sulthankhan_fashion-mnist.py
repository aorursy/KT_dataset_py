import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# Build the Sequential convolutional neural network model
model = Sequential([
    Conv2D(16,kernel_size=3,padding='SAME',strides=2,activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(pool_size=3),
    Flatten(),
    Dense(10,activation='softmax')
])

# Print the model summary

model.summary()
# Define the model optimizer, loss function and metrics

opt=tf.keras.optimizers.Adam(learning_rate=0.005)
acc=tf.keras.metrics.CategoricalAccuracy()
mae=tf.keras.metrics.MeanAbsoluteError()
model.compile(optimizer=opt,
             loss='sparse_categorical_crossentropy',
             metrics=[acc,mae])
# Print the resulting model attributes
print(model.loss)
print(model.optimizer,model.metrics,model.optimizer.lr)

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the Fashion-MNIST dataset

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()
# Print the shape of the training data

train_images.shape
# Define the labels

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
print(train_labels[0])
# Rescale the image values so that they lie in between 0 and 1.
train_images=train_images/255.
test_images=test_images/255.

# Display one of the images
img=train_images[0,:,:]
plt.imshow(img)
# Fit the model

history=model.fit(train_images[...,np.newaxis],train_labels,epochs=20,batch_size=256)
# Load the history into a pandas Dataframe

df=pd.DataFrame(history.history)
df.head()
# Make a plot for the loss
loss_plot=df.plot(y='loss',title='Loss vs epochs',legend=False)

loss_plot.set(xlabel='Epochs')
# Make a plot for the accuracy

loss_plot=df.plot(y='categorical_accuracy',title='Accuracy vs epochs',legend=False)

loss_plot.set(xlabel='Epochs')
# Make a plot for the additional metric
loss_plot=df.plot(y='mean_absolute_error',title='Mae vs epochs',legend=False)

loss_plot.set(xlabel='Epochs')

import matplotlib.pyplot as plt
import numpy as np
# Evaluate the model
test_loss,test_acc,test_mae=model.evaluate(test_images[...,np.newaxis],test_labels,verbose=2)

# Choose a random test image

random_inx = np.random.choice(test_images.shape[0])

test_image = test_images[30]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[30]]}")
# Get the model predictions
pred=model.predict(test_image[np.newaxis,...,np.newaxis])
print(f'Model predictions:{labels[np.argmax(pred)]}' )
