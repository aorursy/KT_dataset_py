#import modules
import pandas as pd
from keras import applications
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, Input
from keras import Model
df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")
y_train = df_train["label"]
onehot_encoder = OneHotEncoder(sparse=False, n_values=10)
y_train = onehot_encoder.fit_transform(np.array(y_train).reshape(-1, 1))
#remove the label column
X_train = df_train.drop(labels = ["label"], axis = 1)
#Test dataset does not have label column
X_test = df_test
#free some space
del df_train 

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
#convert the grayscale image (single channel) to RGB (3 channel) so it becomes 28x28x3
X_train = tf.image.grayscale_to_rgb(X_train, name=None)
X_test = tf.image.grayscale_to_rgb(X_test, name=None)

#resize image to twice it size so we got an integer scale factor of 2, the image becomes 56x56x3
X_train = tf.keras.backend.resize_images(X_train, height_factor=2, width_factor=2, data_format='channels_last')
X_test = tf.keras.backend.resize_images(X_test, height_factor=2, width_factor=2,data_format='channels_last')
sess = tf.Session()
with sess.as_default():
    X_train = X_train.eval()
    X_test = X_test.eval()
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#I already uploaded the vgg16 model(without the top layer because we are going to replace it with our own) in this notebook so we are just going to load it
vgg_model = load_model('../input/vgg16-model/vgg16.h5')
#create dictionary of model layers so its easy to access them later
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
vgg_model.summary()
#from the output above, we saw that block5_pool is the last/top layer of the vgg16 we have so we will add our layers from that point
x = layer_dict['block5_pool'].output
#add flatten layer so we can add the fully connected layer later
x = Flatten()(x)
#Fully connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
#this is the final layer so the size of output in this layer is equal to the number of class in our problem
x = Dense(10, activation='softmax')(x)
#create the new model
custom_model = Model(input=vgg_model.input, output=x)
#freeze the vgg16 layers so they will not be retrained
for layer in custom_model.layers[:19]:
    layer.trainable = False
#compile the model
custom_model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
is_train = False #set this to True to train the model otherwise it will load the model I have already trained
if is_train is True:
    custom_model.fit(x=X_train, y=y_train, epochs=5, validation_data=(X_cv, y_cv), batch_size=512)
else:
    custom_model = load_model('../input/digitrec-model/digit_model.h5')
#predict the X_test using the model we created
predict_all = False #set this to true to predict the entire test set otherwise it will only predict 5 as sample
if predict_all is False:
    X_test = X_test[:5, :, :, :]
results = custom_model.predict(X_test, verbose=1)
#get the index of the maximum probability and use that as the representation of the class (digit)
results = np.argmax(results, axis = 1)
#create a series from results with label column
results = pd.Series(results,name="Label")
#combine the results with corresponding image id
submission = pd.concat([pd.Series(range(1, X_test.shape[0] + 1),name = "ImageId"),results],axis = 1)
#save dataframe to csv
submission.to_csv("cnn_mnist_datagen.csv",index=False)
