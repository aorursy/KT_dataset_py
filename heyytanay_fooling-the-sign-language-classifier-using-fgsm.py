import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

from sklearn.preprocessing import LabelBinarizer
# Try to detect TPUs
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
# Load the data
train_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

train_data.head()
# Join both Training and Testing sets, shuffling them, splitting them, resizing and rescaling them.

data = pd.concat([train_data, test_data])
data = data.sample(frac=1).reset_index(drop=True)

val_split = 1200
valid = data[:val_split]
train = data[val_split:]

trainX = train.drop(['label'], axis=1).values
trainY = train['label']

validX = valid.drop(['label'], axis=1).values
validY = valid['label']

# Label Binarize
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
validY = lb.fit_transform(validY)

trainX = trainX / 255.
validX = validX / 255.

trainX = trainX.reshape(-1, 28, 28, 1)
validX = validX.reshape(-1, 28, 28, 1)
# Visualize a few Images
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(trainX[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()  
# Data augmentation to prevent overfitting
with strategy.scope():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(trainX)
# Train the Model on GPU
with strategy.scope():
    model = tf.keras.Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    model.summary()
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
# Plot the Model Architecture
tf.keras.utils.plot_model(model)
# Traing the Model
with strategy.scope():
    history = model.fit(datagen.flow(trainX, trainY, batch_size = 128) ,epochs = 10 , validation_data = (validX, validY) , callbacks = [learning_rate_reduction])
# Define a few functions for easy visualizations
def get_label(input_image, model=model, good=True):
    if not good:
        img = input_image.reshape(1, 28, 28, 1)
    clss = model.predict_classes([img])
    return clss[0]

def get_confidence(input_image, model=model, good=False):
    if not good:
        input_image = input_image.reshape(1, 28, 28, 1)
    clss = model.predict_classes([input_image])
    conf = model.predict_proba([input_image]).max()
    return clss[0], conf
# Visualize a predicted Image Label and it's Confidence value
# Original Label = Predicted Label = 9
clss, conf = get_confidence(validX[141], model)
plt.title(f"Predicted Label: {clss}\nConfidence: {conf * 100:.2f}%")
plt.imshow(validX[141].reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
# Create a Loss Object, and the adversarial Function

loss_obj = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_img, input_label):
    """
    Watch the Gradients of the Loss w.r.t to the input using TF Gradient Tape
    Then find the needed loss and return the specified gradient matrix signs(only)
    """
    
    with tf.GradientTape() as tape:
        tape.watch(input_img)
        preds = model(input_img)
        prediction = tf.reshape(preds, (24,))
        loss = loss_obj(input_label, prediction)
    # Gradient of loss wrt to input image
    gradient = tape.gradient(loss, input_img)
    
    # Sign of gradient to get the peturbation
    sign = tf.sign(gradient)
    
    return sign
# Now, test the function and see the peturbation for the image we visualized before. 
test_img = tf.convert_to_tensor(validX[141].reshape(1, 28, 28, 1))
test_label = validY[141]

peturbation = create_adversarial_pattern(test_img, test_label).numpy()
plt.imshow(peturbation[0].reshape(28,28)*0.5+0.5)
plt.title("Peturbation Matrix")
plt.show()
epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

for i, eps in enumerate(epsilons):
    adversarial_inp = test_img + eps * peturbation
    adversarial_inp = tf.clip_by_value(adversarial_inp, -1, 1)
    
    # Original Label = Predicted Label = 23
    clss, conf = get_confidence(adversarial_inp, model, good=True)
    plt.title(f"Predicted Label: {clss}\nConfidence: {conf * 100:.2f}%")
    plt.imshow(validX[141].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
