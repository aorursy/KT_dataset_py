import tensorflow as tf

print(tf.__version__)
# Import the CIFAR-10 dataset and rescale the pixel values



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255.0

x_test = x_test / 255.0



# Use smaller subset -- speeds things up

x_train = x_train[:10000]

y_train = y_train[:10000]

x_test = x_test[:1000]

y_test = y_test[:1000]
# Plot the first 10 CIFAR-10 images



import matplotlib.pyplot as plt



fig, ax = plt.subplots(1, 10, figsize=(10, 1))

for i in range(10):

    ax[i].set_axis_off()

    ax[i].imshow(x_train[i])
# Introduce function to test model accuracy



def get_test_accuracy(model, x_test, y_test):

    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)

    print('accuracy: {acc:0.3f}'.format(acc=test_acc))
# Introduce function that creates a new instance of a simple CNN



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D



def get_new_model():

    model = Sequential([

        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3), 

               activation='relu', name='conv_1'),

        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),

        MaxPooling2D(pool_size=(4, 4), name='pool_1'),

        Flatten(name='flatten'),

        Dense(units=32, activation='relu', name='dense_1'),

        Dense(units=10, activation='softmax', name='dense_2')

    ])

    model.compile(optimizer='adam',

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    return model
# Create an instance of the model and show model summary



model = get_new_model()

model.summary()
# Test accuracy of the untrained model, around 10% (random)



get_test_accuracy(model, x_test, y_test)
from tensorflow.keras.callbacks import ModelCheckpoint
# Create Tensorflow checkpoint object

checkpoint_path = "modelcheckpoints/checkpoint"



checkpoint = ModelCheckpoint(filepath = checkpoint_path,

                            frequency = 'epoch',

                            save_weights_only = True,

                            verbose = True)
# Fit model, with simple checkpoint which saves (and overwrites) model weights every epoch

model.fit(x = x_train, y = y_train,epochs = 3, callbacks = [checkpoint])

# Have a look at what the checkpoint creates

!ls -lh modelcheckpoints
# Evaluate the performance of the trained model

get_test_accuracy(model, x_test, y_test)

# Create a new instance of the (initialised) model, accuracy around 10% again



model = get_new_model()

get_test_accuracy(model, x_test, y_test)
# Load weights -- accuracy is the same as the trained model

model.load_weights(checkpoint_path)

get_test_accuracy(model, x_test, y_test)
! rm -r modelcheckpoints
from tensorflow.keras.callbacks import ModelCheckpoint
# Create Tensorflow checkpoint object with epoch and batch details



checkpoint_5000_path = "modelcheckpoints_5000/checkpoint_{epoch:02d}"



checkpoint_5000 = ModelCheckpoint(filepath = checkpoint_5000_path,

                            save_frequency = 5000,

                            save_weights_only = True,

                            verbose = True)
# Create and fit model with checkpoint

model = get_new_model()

model.fit(x = x_train, y = y_train,

          epochs = 3, 

          validation_data = (x_test,y_test),

          batch_size = 10,

          callbacks = [checkpoint_5000])
# Have a look at what the checkpoint creates

!ls -lh modelcheckpoints_5000
# Use tiny training and test set -- will overfit!



x_train = x_train[:100]

y_train = y_train[:100]

x_test = x_test[:100]

y_test = y_test[:100]
# Create a new instance of untrained model



model = get_new_model()
!rm -r modelcheckpoints_best
# Create Tensorflow checkpoint object which monitors the validation accuracy



checkpoint_best_path = "modelcheckpoints_best/checkpoint"



checkpoint_best = ModelCheckpoint(filepath = checkpoint_best_path,

                            save_frequency = 'epoch',

                            save_weights_only = True,

                            save_best_only = True,

                            monitor = 'val_accuracy',

                            verbose = 1)
# Fit the model and save only the weights with the highest validation accuracy



history = model.fit(x = x_train, y = y_train,

          epochs = 50, 

          validation_data = (x_test,y_test),

          callbacks = [checkpoint_best])
# Plot training and testing curves



import pandas as pd



df = pd.DataFrame(history.history)

df.plot(y=['accuracy', 'val_accuracy'])
# Inspect the checkpoint directory



!ls -lh modelcheckpoints_best
# Create a new model with the saved weights



new_model = get_new_model()

new_model.load_weights(checkpoint_best_path)

get_test_accuracy(model, x_test, y_test)
! rm -r  modelcheckpoints_best
from tensorflow.keras.callbacks import ModelCheckpoint
# Create Tensorflow checkpoint object

checkpoint_path = "modelcheckpoints/checkpoint"



checkpoint = ModelCheckpoint(filepath = checkpoint_path,

                            save_frequency = 'epoch',

                            save_weights_only = False,

                            monitor = 'val_accuracy',

                            verbose = 1)

# Create and fit model with checkpoint

model = get_new_model()



history = model.fit(x = x_train, y = y_train,

          epochs = 50, 

          validation_data = (x_test,y_test),

          callbacks = [checkpoint])

# Have a look at what the checkpoint creates

!ls modelcheckpoints/checkpoint
# Enter variables directory



!ls modelcheckpoints/checkpoint/variables
# Get the model's test accuracy



get_test_accuracy(model, x_test, y_test)
# Delete model



model
del model
from tensorflow.keras.models import load_model
# Reload model from scratch



model=load_model(checkpoint_path)



get_test_accuracy(model, x_test, y_test)
# Save the model in .h5 format



model.save('my_model.h5')
# Inspect .h5 file



!ls -lh my_model.h5
# Delete model



del model
# Reload model from scratch

model=load_model('my_model.h5')



get_test_accuracy(model, x_test, y_test)

!ls
! rm -r modelcheckpoints

! rm my_model.h5
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights='imagenet')
# Retrieve the image files



!wget -q -O lemon.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1JSgQ9qgi9nO9t2aGEk-zA6lzYNUT9vZJ"

!wget -q -O viaduct.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1sQzMKmyCR5Tur19lP3n1IIlEMG_o6Mct"

!wget -q -O water_tower.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1cPAQD1O6mAiMbg0fmG5HIk8OuO_BSC6J"
# Import 3 sample ImageNet images



from tensorflow.keras.preprocessing.image import load_img



lemon_img = load_img('lemon.jpg', target_size=(224, 224))

viaduct_img = load_img('viaduct.jpg', target_size=(224, 224))

water_tower_img = load_img('water_tower.jpg', target_size=(224, 224))
# Useful function: presents top 5 predictions and probabilities



from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np

import pandas as pd



def get_top_5_predictions(img):

    x = img_to_array(img)[np.newaxis, ...]

    x = preprocess_input(x)

    preds = decode_predictions(model.predict(x), top=5)

    top_preds = pd.DataFrame(columns=['prediction', 'probability'],

                             index=np.arange(5)+1)

    for i in range(5):

        top_preds.loc[i+1, 'prediction'] = preds[0][i][1]

        top_preds.loc[i+1, 'probability'] = preds[0][i][2] 

    return top_preds
# Display image

lemon_img
# Display top 5 predictions

get_top_5_predictions(lemon_img)

# Display image



viaduct_img
# Display top 5 predictions

get_top_5_predictions(viaduct_img)

# Display image



water_tower_img
# Display top 5 predictions



get_top_5_predictions(water_tower_img)
import tensorflow_hub as hub
# Build Google's Mobilenet v1 model



module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"

model = Sequential([hub.KerasLayer(module_url)])

model.build(input_shape=[None, 160, 160, 3])
# Retrieve the image files



!wget -q -O lemon.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1JSgQ9qgi9nO9t2aGEk-zA6lzYNUT9vZJ"

!wget -q -O viaduct.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1sQzMKmyCR5Tur19lP3n1IIlEMG_o6Mct"

!wget -q -O water_tower.jpg --no-check-certificate "https://docs.google.com/uc?export=download&id=1cPAQD1O6mAiMbg0fmG5HIk8OuO_BSC6J"
# Import and preprocess 3 sample ImageNet images



from tensorflow.keras.preprocessing.image import load_img



lemon_img = load_img("lemon.jpg", target_size=(160, 160))

viaduct_img = load_img("viaduct.jpg", target_size=(160, 160))

water_tower_img = load_img("water_tower.jpg", target_size=(160, 160))
# Read in categories text file



with open('data/imagenet_categories.txt') as txt_file:

    categories = txt_file.read().splitlines()
# Useful function: presents top 5 predictions



import pandas as pd



def get_top_5_predictions(img):

    x = img_to_array(img)[np.newaxis, ...] / 255.0

    preds = model.predict(x)

    top_preds = pd.DataFrame(columns=['prediction'],

                             index=np.arange(5)+1)

    sorted_index = np.argsort(-preds[0])

    for i in range(5):

        ith_pred = categories[sorted_index[i]]

        top_preds.loc[i+1, 'prediction'] = ith_pred

            

    return top_preds
lemon_img
viaduct_img
water_tower_img