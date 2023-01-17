# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.layers import *

from keras.models import Model

# Visualize Layer Keras



from __future__ import print_function



import time

from PIL import Image as pil_image

from keras.preprocessing.image import save_img

from keras import layers

from keras import backend as K





def normalize(x):

    """utility function to normalize a tensor.



    # Arguments

        x: An input tensor.



    # Returns

        The normalized input tensor.

    """

    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())





def deprocess_image(x):

    """utility function to convert a float array into a valid uint8 image.



    # Arguments

        x: A numpy-array representing the generated image.



    # Returns

        A processed numpy-array, which could be used in e.g. imshow.

    """

    # normalize tensor: center on 0., ensure std is 0.25

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





def process_image(x, former):

    """utility function to convert a valid uint8 image back into a float array.

       Reverses `deprocess_image`.



    # Arguments

        x: A numpy-array, which could be used in e.g. imshow.

        former: The former numpy-array.

                Need to determine the former mean and variance.



    # Returns

        A processed numpy-array representing the generated image.

    """

    if K.image_data_format() == 'channels_first':

        x = x.transpose((2, 0, 1))

    return (x / 255 - 0.5) * 4 * former.std() + former.mean()





def visualize_layer(model,

                    layer_name,

                    step=1.,

                    epochs=15,

                    upscaling_steps=9,

                    upscaling_factor=1.2,

                    output_dim=(412, 412),

                    filter_range=(0, None)):

    """Visualizes the most relevant filters of one conv-layer in a certain model.



    # Arguments

        model: The model containing layer_name.

        layer_name: The name of the layer to be visualized.

                    Has to be a part of model.

        step: step size for gradient ascent.

        epochs: Number of iterations for gradient ascent.

        upscaling_steps: Number of upscaling steps.

                         Starting image is in this case (80, 80).

        upscaling_factor: Factor to which to slowly upgrade

                          the image towards output_dim.

        output_dim: [img_width, img_height] The output image dimensions.

        filter_range: Tupel[lower, upper]

                      Determines the to be computed filter numbers.

                      If the second value is `None`,

                      the last filter will be inferred as the upper boundary.

    """



    def _generate_filter_image(input_img,

                               layer_output,

                               filter_index):

        """Generates image for one particular filter.



        # Arguments

            input_img: The input-image Tensor.

            layer_output: The output-image Tensor.

            filter_index: The to be processed filter number.

                          Assumed to be valid.



        #Returns

            Either None if no image could be generated.

            or a tuple of the image (array) itself and the last loss.

        """

        s_time = time.time()



        # we build a loss function that maximizes the activation

        # of the nth filter of the layer considered

        if K.image_data_format() == 'channels_first':

            loss = K.mean(layer_output[:, filter_index, :, :])

        else:

            loss = K.mean(layer_output[:, :, :, filter_index])



        # we compute the gradient of the input picture wrt this loss

        grads = K.gradients(loss, input_img)[0]



        # normalization trick: we normalize the gradient

        grads = normalize(grads)



        # this function returns the loss and grads given the input picture

        iterate = K.function([input_img], [loss, grads])



        # we start from a gray image with some random noise

        intermediate_dim = tuple(

            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)

        if K.image_data_format() == 'channels_first':

            input_img_data = np.random.random(

                (1, 3, intermediate_dim[0], intermediate_dim[1]))

        else:

            input_img_data = np.random.random(

                (1, intermediate_dim[0], intermediate_dim[1], 3))

        input_img_data = (input_img_data - 0.5) * 20 + 128



        # Slowly upscaling towards the original size prevents

        # a dominating high-frequency of the to visualized structure

        # as it would occur if we directly compute the 412d-image.

        # Behaves as a better starting point for each following dimension

        # and therefore avoids poor local minima

        for up in reversed(range(upscaling_steps)):

            # we run gradient ascent for e.g. 20 steps

            for _ in range(epochs):

                loss_value, grads_value = iterate([input_img_data])

                input_img_data += grads_value * step



                # some filters get stuck to 0, we can skip them

                if loss_value <= K.epsilon():

                    return None



            # Calculate upscaled dimension

            intermediate_dim = tuple(

                int(x / (upscaling_factor ** up)) for x in output_dim)

            # Upscale

            img = deprocess_image(input_img_data[0])

            img = np.array(pil_image.fromarray(img).resize(intermediate_dim,

                                                           pil_image.BICUBIC))

            input_img_data = np.expand_dims(

                process_image(img, input_img_data[0]), 0)



        # decode the resulting input image

        img = deprocess_image(input_img_data[0])

        e_time = time.time()

        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,

                                                                  loss_value,

                                                                  e_time - s_time))

        return img, loss_value



    def _draw_filters(filters, n=None):

        """Draw the best filters in a nxn grid.



        # Arguments

            filters: A List of generated images and their corresponding losses

                     for each processed filter.

            n: dimension of the grid.

               If none, the largest possible square will be used

        """

        if n is None:

            n = int(np.floor(np.sqrt(len(filters))))



        # the filters that have the highest loss are assumed to be better-looking.

        # we will only keep the top n*n filters.

        filters.sort(key=lambda x: x[1], reverse=True)

        filters = filters[:n * n]



        # build a black picture with enough space for

        # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between

        MARGIN = 5

        width = n * output_dim[0] + (n - 1) * MARGIN

        height = n * output_dim[1] + (n - 1) * MARGIN

        stitched_filters = np.zeros((width, height, 3), dtype='uint8')



        # fill the picture with our saved filters

        for i in range(n):

            for j in range(n):

                img, _ = filters[i * n + j]

                width_margin = (output_dim[0] + MARGIN) * i

                height_margin = (output_dim[1] + MARGIN) * j

                stitched_filters[

                    width_margin: width_margin + output_dim[0],

                    height_margin: height_margin + output_dim[1], :] = img



        # save the result to disk

        save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)



    # this is the placeholder for the input images

    assert len(model.inputs) == 1

    input_img = model.inputs[0]



    # get the symbolic outputs of each "key" layer (we gave them unique names).

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])



    output_layer = layer_dict[layer_name]

    assert isinstance(output_layer, layers.Conv2D)



    # Compute to be processed filter range

    filter_lower = filter_range[0]

    filter_upper = (filter_range[1]

                    if filter_range[1] is not None

                    else len(output_layer.get_weights()[1]))

    assert(filter_lower >= 0

           and filter_upper <= len(output_layer.get_weights()[1])

           and filter_upper > filter_lower)

    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))



    # iterate through each filter and generate its corresponding image

    processed_filters = []

    for f in range(filter_lower, filter_upper):

        img_loss = _generate_filter_image(input_img, output_layer.output, f)



        if img_loss is not None:

            processed_filters.append(img_loss)



    print('{} filter processed.'.format(len(processed_filters)))

    # Finally draw and store the best filters to disk

    _draw_filters(processed_filters)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.metrics import confusion_matrix

%matplotlib inline

def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)

    #  square=True,

    plt.ylabel('Label')

    plt.xlabel('Prediction')
train_data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y=train_data["label"]

X=train_data.copy()

del X["label"]
# plt.imshow(X.values[0].reshape((28,28)))
# display(X,y)
SIZE=32
def reshape32(img):

    img=img.reshape((28,28))

    img=np.pad(img,((SIZE-28)//2,(SIZE-28)//2))

    img=img.reshape((SIZE,SIZE,1))

#     img=np.stack((img,)*3, axis=-1)

    return img
# plt.imshow(reshape32(X.values[12]))
new_X=[]

for i,img in enumerate(X.values):

    new_X.append(reshape32(img))

new_X=np.array(new_X)

new_X[new_X<50]=0

# plt.imshow(new_X[12])
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

train_X,val_X,train_y,val_y = train_test_split(new_X/255,y,test_size=0.1)
inp=Input(shape=(32,32,1))





model = Conv2D(filters=32, kernel_size=(2, 2), padding='SAME', activation='relu',name="conv32")(inp)

model = Conv2D(filters=32, kernel_size=(2, 2), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(2, 2), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu',name="conv64")(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.2)(model)



model = Conv2D(filters=128, kernel_size=(7, 7), padding='SAME', activation='relu',name="conv128")(model)

model = Conv2D(filters=128, kernel_size=(7, 7), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(7, 7), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=128, kernel_size=(7, 7), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.2)(model)







model = Conv2D(filters=256, kernel_size=(15, 15), padding='SAME', activation='relu',name="conv256")(model)

model = Conv2D(filters=256, kernel_size=(15, 15), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(15, 15), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

# model = Conv2D(filters=256, kernel_size=(2, 2), padding='SAME', activation='relu')(model)

# model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.2)(model)





my_x=Flatten()(model)

# my_x=Dense(1024,activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros')(my_x)

# my_x=Dropout(0.2)(my_x)



# my_x=Dense(512,activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros')(my_x)

# my_x=Dropout(0.2)(my_x)



my_x=Dense(256,activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros')(my_x)

my_x=Dropout(0.2)(my_x)



my_x=Dense(128,activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros')(my_x)

my_x=Dropout(0.2)(my_x)





my_x=Dense(64,activation='relu',name='my')(my_x)

my_x=Dropout(0.2)(my_x)



my_x=Dense(32,activation='relu',name='my2')(my_x)

my_x=Dropout(0.2)(my_x)







preds=Dense(10,activation='softmax',kernel_initializer='he_uniform',bias_initializer='zeros',name='output')(my_x)



my_model=Model(inputs=inp,outputs=preds)







# my_model=new_model
# for layer in my_model.layers:

#     if layer.name == 'output' or layer.name == 'my' :

#         layer.trainable=True

#     else:

#         layer.trainable=True
from keras.optimizers import Adadelta

my_model.compile(optimizer=Adadelta(),loss='categorical_crossentropy',metrics=['accuracy','mse'])
my_model.summary()
from keras.callbacks import ReduceLROnPlateau



rlrp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_delta=1E-30,verbose=1)

history=my_model.fit(x=train_X,y=pd.get_dummies(train_y),validation_data=(val_X,pd.get_dummies(val_y)),epochs=100, batch_size=1024,callbacks=[rlrp])

for layer in my_model.layers:

    print(layer.name,)

    if 'conv' not in layer.name:

        continue

    filters, biases = layer.get_weights()

    

    filters, biases = layer.get_weights()

    f_min, f_max = filters.min(), filters.max()

    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1

    for i in range(n_filters):

        f = filters[:, :, :, i]

        for j in range(1):

            ax = plt.subplot(n_filters, 3, ix)

            ax.set_xticks([])

            ax.set_yticks([])

            plt.imshow(f[:, :, j], cmap='gray')

            ix += 1

    plt.show()
preds=my_model.predict(train_X)

print("Training Accuracy : ",accuracy_score(train_y,np.argmax(preds,axis=1)))







preds=my_model.predict(val_X)

print("Validation Accuracy : ",accuracy_score(val_y,np.argmax(preds,axis=1)))

plot_confusion_matrix(val_y,np.argmax(preds,axis=1))



# #mis predicted



for i in range(len(val_X)):

    if np.argmax(my_model.predict(val_X[i].reshape(1,32,32,1)),axis=1)!=val_y.values[i]:

        (plt.imshow(  val_X[i].reshape(32,32),))

        plt.show()

        print("Label : ",val_y.values[i])

        print("Prediction : ",np.argmax(my_model.predict(val_X[i].reshape(1,32,32,1)),axis=1))

test_X=[]

for i,img in enumerate(test_data.values):

    z=reshape32(img)

    test_X.append(z)

test_X=np.array(test_X)

test_X[test_X<50]=0

test_X=test_X/255
sol=np.argmax(my_model.predict((test_X)),axis=1)

df=pd.DataFrame(sol)

df.index+=1

df.to_csv("/kaggle/working/sol_final.csv",index=True,header=["Label"],index_label=["ImageId"])