from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Input



# this could also be the output a different Keras model or layer

input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

input_shape = (224,224,3)

model_224 = InceptionV3(input_tensor=input_tensor,input_shape = input_shape, weights='imagenet', include_top=False)

model_224.summary()


# this could also be the output a different Keras model or layer

input_tensor = Input(shape=(300, 300, 3))  # this assumes K.image_data_format() == 'channels_last'

input_shape = (300,300,3)



model_300 = InceptionV3(input_tensor=input_tensor, input_shape = input_shape,weights='imagenet', include_top=False)

model_300.summary()