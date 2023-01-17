

import numpy as np

import cv2

import glob

import itertools





def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):



    try:

        img = cv2.imread(path, 1)



        if imgNorm == "sub_and_divide":

            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1

        elif imgNorm == "sub_mean":

            img = cv2.resize(img, (width, height))

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939

            img[:, :, 1] -= 116.779

            img[:, :, 2] -= 123.68

        elif imgNorm == "divide":

            img = cv2.resize(img, (width, height))

            img = img.astype(np.float32)

            img = img/255.0



        if odering == 'channels_first':

            img = np.rollaxis(img, 2, 0)

        return img

    except Exception as e:

        print(path, e)

        img = np.zeros((height, width, 3))

        if odering == 'channels_first':

            img = np.rollaxis(img, 2, 0)

        return img





def getSegmentationArr(path, nClasses,  width, height):

    seg_labels = np.zeros((height, width, nClasses))

    try:

        img = cv2.imread(path, 1)

        img = cv2.resize(img, (width, height))

        img = img[:, :, 0]



        for c in range(nClasses):

            seg_labels[:, :, c] = (img == c).astype(int)



    except Exception as e:

        print(e)



    seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels





def imageSegmentationGenerator(images_path, segs_path, batch_size,  n_classes, input_height, input_width, output_height, output_width):



    assert images_path[-1] == '/'

    assert segs_path[-1] == '/'



    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")

    images.sort()

    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")

    segmentations.sort()



    assert len(images) == len(segmentations)

    for im, seg in zip(images, segmentations):

        assert(im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])



    zipped = itertools.cycle(zip(images, segmentations))



    while True:

        X = []

        Y = []

        for _ in range(batch_size):

            im, seg = next(zipped)

            

            X.append(getImageArr(im, input_width, input_height))

            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))



        yield np.array(X), np.array(Y)





# import Models , LoadBatches

# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 



# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )

# m.fit_generator( G , 512  , nb_epoch=10 )





from keras.layers.core import Layer, Activation, Reshape, Permute

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import  UpSampling2D, ZeroPadding2D

from keras import models

from keras.layers import Conv2D, InputLayer, MaxPool2D

from keras import optimizers

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)





df = "channels_first"



def segnet(nClasses, input_height=360, input_width=480):

    kernel = 3

    filter_size = 64

    pad = 1

    pool_size = 2



    model = models.Sequential()

    model.add(InputLayer((3, input_height, input_width)))



    # encoder

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))



    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(128, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))



    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(256, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))



    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(512, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())

    model.add(Activation('relu'))





    # decoder

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(512, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())



    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(256, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())



    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(128, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())



    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))

    model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))

    model.add(BatchNormalization())



    model.add(Conv2D(nClasses, 1, padding='valid', data_format=df))



    # model.summary()

    # --------------------------------------------

    outputHeight = model.output_shape[2]

    outputWidth = model.output_shape[3]

    print("output hieight = ", outputHeight)

    print("output  Width = ", outputWidth)



    model.add(Reshape((nClasses,  outputHeight*outputWidth)))



    model.add(Permute((2, 1)))

    model.add(Activation('softmax'))

    model.outputWidth = outputWidth

    model.outputHeight = outputHeight





    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'] )

    model.summary()

    return model



input_height = 320

input_width = 640

epochs = 7

n_classes = 10

train_images_path = "../input/dataset1/dataset1/images_prepped_train/"

train_segs_path = "../input/dataset1/dataset1/annotations_prepped_train/"

train_batch_size = 8
m = segnet(n_classes,  input_height=input_height, input_width=input_width)
output_height = m.outputHeight

output_width = m.outputWidth



G = imageSegmentationGenerator(train_images_path,

                               train_segs_path,

                               train_batch_size,

                               n_classes,

                               input_height,

                               input_width,

                               output_height,

                               output_width)

input_height = 320

input_width = 640

epochs = 5

n_classes = 10

val_images_path = "../input/dataset1/dataset1/images_prepped_test/"

val_segs_path = "../input/dataset1/dataset1/annotations_prepped_test/"

val_batch_size = 8
G2 = imageSegmentationGenerator(val_images_path,

                                val_segs_path,

                                val_batch_size,

                                n_classes,

                                input_height,

                                input_width,

                                output_height,

                                output_width)
print(epochs)
for ep in range(epochs):

    m.fit_generator(G, 512, validation_data=G2, validation_steps=200,  epochs=10)

    m.save_weights("wight" + "." + str(ep))

    m.save("mo" + ".model." + str(ep))
images_test = "../input/dataset1/dataset1/images_prepped_test/"

seg_test = "../input/dataset1/dataset1/annotations_prepped_test/"



print("Done")
import random
import matplotlib.pyplot as plt

%matplotlib inline


images = glob.glob(images_test + "*.jpg") + glob.glob(images_test + "*.png") + glob.glob(images_test + "*.jpeg")

images.sort()

segs = glob.glob(seg_test + "*.jpg") + glob.glob(seg_test + "*.png") + glob.glob(seg_test + "*.jpeg")

segs.sort()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]



for imgName, seg in zip(images, segs):

    X = getImageArr(imgName, input_width, input_height)

    immm = cv2.imread(imgName)

    immm = cv2.resize(immm, (input_width, input_height))

    pr = m.predict(np.array([X]))[0]

    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    segg = cv2.imread(seg)

    seg_img1 = np.zeros_like(segg)

    for c in range(n_classes):

        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')

        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')

        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

        

        seg_img1[:, :, 0] += ((segg[:, :, 0] == c) * (colors[c][0])).astype('uint8')

        seg_img1[:, :, 1] += ((segg[:, :, 0] == c) * (colors[c][1])).astype('uint8')

        seg_img1[:, :, 2] += ((segg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (input_width, input_height))

    seg_img1 = cv2.resize(seg_img1, (input_width, input_height))

    immm = cv2.resize(immm, (input_width, input_height))

    

    plt.imshow(immm)

    plt.show()

    plt.imshow((seg_img * 255).astype(np.uint8))

    plt.show()

    plt.imshow((seg_img1 * 255).astype(np.uint8))

    plt.show()

#     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

#     ax1.set_title("orignal")

#     ax1.imshow(immm)

#     ax2.set_title("GT")

#     ax2.imshow((seg_img * 255).astype(np.uint8))

#     ax3.set_title("Predicted")

#     ax3.imshow((seg_img1 * 255).astype(np.uint8))

#     plt.show()

    

print("DOne")
m.save("model.h5")