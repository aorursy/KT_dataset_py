import random
import cv2
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ind_train = random.sample(list(range(x_train.shape[0])), 1000)
x_train = x_train[ind_train]
y_train = y_train[ind_train]
ind_test = random.sample(list(range(x_test.shape[0])), 1000)
x_test = x_test[ind_test]
y_test = y_test[ind_test]
def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 320, 320, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled
x_train_resized = resize_data(x_train)
x_test_resized = resize_data(x_test)
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_test)
def model(x_train, y_train, base_model):

    # get layers and add average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully-connected layer
    x = Dense(512, activation='relu')(x)

    # add output layer
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = False

    # update the weight that are added
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    # choose the layers which are updated by training
    layer_num = len(model.layers)
    for layer in model.layers[:int(layer_num * 0.9)]:
        layer.trainable = False

    for layer in model.layers[int(layer_num * 0.9):]:
        layer.trainable = True

    # update the weights
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5)
    return history
inception_model = InceptionV3(weights='imagenet', include_top=False)
res_50_model = ResNet50(weights='imagenet', include_top=False)
vgg_19_model = VGG19(weights='imagenet', include_top=False)
vgg_16_model = VGG16(weights='imagenet', include_top=False)
xception_model = Xception(weights='imagenet', include_top=False)
history_inception_v3 = model(x_train_resized, y_train_hot_encoded, inception_model)
history_res_50 = model(x_train_resized, y_train_hot_encoded, res_50_model)
history_vgg_19 = model(x_train_resized, y_train_hot_encoded, vgg_19_model)
history_vgg_16 = model(x_train_resized, y_train_hot_encoded, vgg_16_model)
history_xception = model(x_train_resized, y_train_hot_encoded, xception_model)
# check accuracy
evaluation_inception_v3 = history_inception_v3.model.evaluate(x_test_resized,y_test_hot_encoded)
evaluation_res_50 = history_res_50.model.evaluate(x_test_resized,y_test_hot_encoded)
evaluation_vgg_19 = history_vgg_19.model.evaluate(x_test_resized,y_test_hot_encoded)
evaluation_vgg_16 = history_vgg_16.model.evaluate(x_test_resized,y_test_hot_encoded)
evaluation_xception = history_xception.model.evaluate(x_test_resized,y_test_hot_encoded)
print("inception_v3:{}".format(evaluation_inception_v3))
print("res_50:{}".format(evaluation_res_50))
print("vgg_19:{}".format(evaluation_vgg_19))
print("vgg_16:{}".format(evaluation_vgg_16))
print("xception:{}".format(evaluation_xception))