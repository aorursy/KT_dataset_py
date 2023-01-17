!pip install --upgrade pip

!pip install tensorflow==2.2.0
import numpy as np

import os

import zipfile

import pandas as pd

import random

import matplotlib.pyplot as plt

import cv2



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.python.framework import ops

from tensorflow.keras.applications import ResNet50V2

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras import optimizers

from tensorflow.keras.layers import Dense, Flatten, Activation

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K

print("Tensorflow version: ", tf.__version__)
# Path and parameters

IMAGE_DIR = "../working/train"

H = 224

W = 224

epochs = 5

batch_size = 100

SEED = 42
# Unzip data

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:

    z.extractall(".")
# Create dataframe

filenames = os.listdir(IMAGE_DIR)

labels = [x.split(".")[0] for x in filenames]

df = pd.DataFrame({"filename": filenames, "label": labels})

df.head()
df.label.value_counts()
train_df, val_df = train_test_split(df, test_size=0.2, random_state = SEED, stratify = df.label)

train_df.sample(frac=1, random_state=SEED)

train_df = train_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)
# Train data distribution

train_df.label.value_counts()
# Validation data distribution

val_df.label.value_counts()
dogs = list(df[df.label=="dog"].filename)

cats = list(df[df.label=="cat"].filename)
# Adapted with serveral modifications from https://www.kaggle.com/serkanpeldek/keras-cnn-transfer-learnings-on-cats-dogs-dataset



def get_side(img, side_type, n = 5):

    h, w, c = img.shape

    if side_type == "horizontal":

        return np.ones((h,n,c))

    return np.ones((n,w,c))



def show_gallery(im_ls,n=5, shuffle=True):

    images = []

    vertical_images = []

    if shuffle:

        random.shuffle(im_ls)

    vertical_images = []

    for i in range(n*n):

        img = load_img(os.path.join(IMAGE_DIR,im_ls[i]), target_size=(W,H))

        img = img_to_array(img)

        hside = get_side(img,side_type="horizontal")

        images.append(img)

        images.append(hside)

        

        if (i+1) % n == 0:

            himage=np.hstack((images))

            vside = get_side(himage, side_type="vertical")

            vertical_images.append(himage)

            vertical_images.append(vside)

            

            images = []

        

    gallery = np.vstack((vertical_images))

    plt.figure(figsize=(20,20))

    plt.axis("off")

    plt.imshow(gallery.astype(np.uint8))

    plt.show()
# Show dogs images

show_gallery(dogs, n=10)
# Show cat images

show_gallery(cats, n=10)
class GradCAM:

    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

    def __init__(self, model, layerName=None):

        """

        model: pre-softmax layer (logit layer)

        """

        self.model = model

        self.layerName = layerName

            

        if self.layerName == None:

            self.layerName = self.find_target_layer()

    

    def find_target_layer(self):

        for layer in reversed(self.model.layers):

            if len(layer.output_shape) == 4:

                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

            

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):

        gradModel = Model(

            inputs = [self.model.inputs],

            outputs = [self.model.get_layer(self.layerName).output, self.model.output]

        )

        # record operations for automatic differentiation

        

        with tf.GradientTape() as tape:

            inputs = tf.cast(image, tf.float32)

            (convOuts, preds) = gradModel(inputs) # preds after softmax

            loss = preds[:,classIdx]

        

        # compute gradients with automatic differentiation

        grads = tape.gradient(loss, convOuts)

        # discard batch

        convOuts = convOuts[0]

        grads = grads[0]

        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        

        # compute weights

        weights = tf.reduce_mean(norm_grads, axis=(0,1))

        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        

        # Apply reLU

        cam = np.maximum(cam, 0)

        cam = cam/np.max(cam)

        cam = cv2.resize(cam, upsample_size,interpolation=cv2.INTER_LINEAR)

        

        # convert to 3D

        cam3 = np.expand_dims(cam, axis=2)

        cam3 = np.tile(cam3, [1,1,3])

        

        return cam3

    

def overlay_gradCAM(img, cam3):

    cam3 = np.uint8(255*cam3)

    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    

    new_img = 0.3*cam3 + 0.5*img

    

    return (new_img*255.0/new_img.max()).astype("uint8")
@tf.custom_gradient

def guidedRelu(x):

    def grad(dy):

        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy

    return tf.nn.relu(x), grad



# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  

class GuidedBackprop:

    def __init__(self,model, layerName=None):

        self.model = model

        self.layerName = layerName

        self.gbModel = self.build_guided_model()

        

        if self.layerName == None:

            self.layerName = self.find_target_layer()



    def find_target_layer(self):

        for layer in reversed(self.model.layers):

            if len(layer.output_shape) == 4:

                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")



    def build_guided_model(self):

        gbModel = Model(

            inputs = [self.model.inputs],

            outputs = [self.model.get_layer(self.layerName).output]

        )

        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]

        for layer in layer_dict:

            if layer.activation == tf.keras.activations.relu:

                layer.activation = guidedRelu

        

        return gbModel

    

    def guided_backprop(self, images, upsample_size):

        """Guided Backpropagation method for visualizing input saliency."""

        with tf.GradientTape() as tape:

            inputs = tf.cast(images, tf.float32)

            tape.watch(inputs)

            outputs = self.gbModel(inputs)



        grads = tape.gradient(outputs, inputs)[0]



        saliency = cv2.resize(np.asarray(grads), upsample_size)



        return saliency



def deprocess_image(x):

    """Same normalization as in:

    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

    """

    # normalize tensor: center on 0., ensure std is 0.25

    x = x.copy()

    x -= x.mean()

    x /= (x.std() + K.epsilon())

    x *= 0.25



    # clip to [0, 1]

    x += 0.5

    x = np.clip(x, 0, 1)



    # convert to RGB array

    x *= 255

    if K.image_data_format() == 'channels_first':

        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')

    return x
def show_gradCAMs(model, gradCAM, GuidedBP, im_ls, n=3, decode={}):

    """

    model: softmax layer

    """

    random.shuffle(im_ls)

    plt.subplots(figsize=(30, 10*n))

    k=1

    for i in range(n):

        img = cv2.imread(os.path.join(IMAGE_DIR,im_ls[i]))

        upsample_size = (img.shape[1],img.shape[0])

        if (i+1) == len(df):

            break

        # Show original image

        plt.subplot(n,3,k)

        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        plt.title("Filename: {}".format(im_ls[i]), fontsize=20)

        plt.axis("off")

        # Show overlayed grad

        plt.subplot(n,3,k+1)

        im = img_to_array(load_img(os.path.join(IMAGE_DIR,im_ls[i]), target_size=(W,H)))

        x = np.expand_dims(im, axis=0)

        x = preprocess_input(x)

        preds = model.predict(x)

        idx = preds.argmax()

        if len(decode)==0:

            res = decode_predictions(preds)[0][0][1:]

        else:

            res = [decode[idx],preds.max()]

        cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)

        new_img = overlay_gradCAM(img, cam3)

        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        plt.imshow(new_img)

        plt.title("GradCAM - Pred: {}. Prob: {}".format(res[0],res[1]), fontsize=20)

        plt.axis("off")

        

        # Show guided GradCAM

        plt.subplot(n,3,k+2)

        gb = GuidedBP.guided_backprop(x, upsample_size)

        guided_gradcam = deprocess_image(gb*cam3)

        guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)

        plt.imshow(guided_gradcam)

        plt.title("Guided GradCAM", fontsize=20)

        plt.axis("off")

        

        k += 3

    plt.show()
resnet50_logit = ResNet50V2(include_top=True, weights='imagenet', classifier_activation=None)
resnet50 = ResNet50V2(include_top=True, weights='imagenet')
gradCAM = GradCAM(model=resnet50_logit, layerName="conv5_block3_out")

guidedBP = GuidedBackprop(model=resnet50,layerName="conv5_block3_out")
show_gradCAMs(resnet50, gradCAM,guidedBP,dogs, n=5)
show_gradCAMs(resnet50, gradCAM, guidedBP,cats, n=5)
# Train generator

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_dataframe(train_df, IMAGE_DIR,x_col="filename", y_col="label",

                                                    target_size=(W,H), class_mode="categorical",

                                                   batch_size=batch_size, shuffle=True, seed=SEED)
# Validation generator

val_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

val_generator = val_datagen.flow_from_dataframe(val_df, IMAGE_DIR, x_col="filename", y_col="label",

                                               target_size=(W,H), class_mode="categorical",

                                                batch_size=batch_size)
# Look at how data generator augment the data

ex_df = train_df.sample(n=15).reset_index(drop=True)

ex_gen = train_datagen.flow_from_dataframe(ex_df,IMAGE_DIR,x_col="filename", y_col="label",

                                           target_size=(W,H), class_mode="categorical")
plt.figure(figsize=(15,15))

for i in range(0, 9):

    plt.subplot(5,3,i+1)

    for x, y in ex_gen:

        im = x[0]

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        plt.axis("off")

        break

plt.tight_layout()

plt.show()
resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')

for layer in resnet.layers:

    layer.trainable=False



logits = Dense(2)(resnet.layers[-1].output)

output = Activation('softmax')(logits)

model = Model(resnet.input, output)
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=["accuracy"])
earlystoper = EarlyStopping(monitor="val_loss", patience=3)

checkpointer = ModelCheckpoint(filepath="../working/resnet50best.hdf5", monitor='val_loss', save_best_only=True, mode='auto')

callbacks = [earlystoper, checkpointer]
history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=val_generator,

    validation_steps=20,

    steps_per_epoch=20,

    callbacks=callbacks

)
model.load_weights("../working/resnet50best.hdf5")
plt.figure(1, figsize = (15,8)) 

    

plt.subplot(221)  

plt.plot(history.history['accuracy'])  

plt.plot(history.history['val_accuracy'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 

    

plt.subplot(222)  

plt.plot(history.history['loss'])  

plt.plot(history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.show()
model_logit = Model(model.input,model.layers[-2].output)
retrained_gradCAM = GradCAM(model=model_logit, layerName="conv5_block3_out")

retrained_guidedBP = GuidedBackprop(model=model, layerName="conv5_block3_out")
data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = data_gen.flow_from_dataframe(val_df, IMAGE_DIR, x_col="filename",

                                               target_size=(W,H), class_mode=None,

                                                batch_size=1, shuffle=False)
pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

pred_indices = np.argmax(pred,axis=1)
results = val_df.copy()

results["pred"] = pred_indices

true_dogs = list(results[(results.label == "dog") & (results.pred ==1)].filename)

true_cats = list(results[(results.label == "cat") & (results.pred ==0)].filename)

wrong_class = [x for x in results.filename if x not in (true_cats+true_dogs)]
show_gradCAMs(model, retrained_gradCAM,retrained_guidedBP,true_dogs, n=5, decode={0:"cat", 1:"dog"})
show_gradCAMs(model, retrained_gradCAM,retrained_guidedBP,true_cats, n=5, decode={0:"cat", 1:"dog"})
len(wrong_class)
show_gradCAMs(model, retrained_gradCAM,retrained_guidedBP,wrong_class, n=5, decode={0:"cat", 1:"dog"})
resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')

for layer in resnet.layers:

    layer.trainable=False

    

fc1 = Dense(100)(resnet.layers[-1].output)

fc2 = Dense(100)(fc1)

logits = Dense(2)(fc2)

output = Activation('softmax')(logits)

model_with_fc = Model(resnet.input, output)
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model_with_fc.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=["accuracy"])
earlystoper = EarlyStopping(monitor="val_loss", patience=3)

checkpointer = ModelCheckpoint(filepath="../working/resnet50fcbest.hdf5", monitor='val_loss', save_best_only=True, mode='auto')

callbacks = [earlystoper, checkpointer]
history = model_with_fc.fit_generator(

    train_generator, 

    epochs=5,

    validation_data=val_generator,

    validation_steps=20,

    steps_per_epoch=20,

    callbacks=callbacks

)
model_with_fc.load_weights("../working/resnet50fcbest.hdf5")
plt.figure(1, figsize = (15,8)) 

    

plt.subplot(221)  

plt.plot(history.history['accuracy'])  

plt.plot(history.history['val_accuracy'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 

    

plt.subplot(222)  

plt.plot(history.history['loss'])  

plt.plot(history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.show()
model_fc_logit = Model(model_with_fc.input,model_with_fc.layers[-2].output)

fctrained_gradCAM = GradCAM(model=model_fc_logit, layerName="conv5_block3_out")

fctrained_guidedBP = GuidedBackprop(model=model_with_fc, layerName="conv5_block3_out")
test_generator = data_gen.flow_from_dataframe(val_df, IMAGE_DIR, x_col="filename",

                                               target_size=(W,H), class_mode=None,

                                                batch_size=1, shuffle=False)

pred = model_with_fc.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

pred_indices = np.argmax(pred,axis=1)
results = val_df.copy()

results["pred"] = pred_indices

true_dogs = list(results[(results.label == "dog") & (results.pred ==1)].filename)

true_cats = list(results[(results.label == "cat") & (results.pred ==0)].filename)

wrong_class = [x for x in results.filename if x not in (true_cats+true_dogs)]
show_gradCAMs(model_with_fc, fctrained_gradCAM, fctrained_guidedBP,true_dogs, n=5, decode={0:"cat", 1:"dog"})
show_gradCAMs(model_with_fc, fctrained_gradCAM, fctrained_guidedBP, true_cats, n=5, decode={0:"cat", 1:"dog"})
len(wrong_class)
show_gradCAMs(model_with_fc, fctrained_gradCAM, fctrained_guidedBP, wrong_class, n=5, decode={0:"cat", 1:"dog"})