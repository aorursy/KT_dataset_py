from tensorflow.keras import backend as K

from tensorflow.keras import layers as L

from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.inception_v3 import InceptionV3
K.clear_session()



input_tensor = L.Input(shape=(299,299,3))

inception = InceptionV3(include_top=False, # leaves out the classifier end

                        input_shape=(299,299,3),

                        pooling=None) # leaves out flattening/global pooling at end

output = inception(input_tensor)

model = Model(input_tensor,output) # as of this moment, this network's shapes are locked.

model.summary()
K.clear_session()



input_tensor = L.Input(shape=(96,96,3))

inception = InceptionV3(include_top=False,

                        input_shape=(96,96,3),

                        pooling=None)

output = inception(input_tensor)

model = Model(input_tensor,output)

model.summary()
K.clear_session()



input_tensor = L.Input(shape=(123,666,3))

inception = InceptionV3(include_top=False,

                        input_shape=(123,666,3),

                        pooling=None)

output = inception(input_tensor)

model = Model(input_tensor,output)

model.summary()
K.clear_session()



inception = InceptionV3(include_top=False,

                        input_shape=(299,299,3),

                        pooling=None)



plot_model(inception)
inception.summary()
K.clear_session()



inception = InceptionV3(include_top=False,

                        input_shape=(256,256,3),

                        pooling=None)

inception.summary()
K.clear_session()



input_tensor = L.Input(shape=(299,299,3))

inception = InceptionV3(include_top=False,

                        input_shape=(299,299,3),

                        pooling=None) # or make it 'avg' or 'max' and remove global pooling layer

x = inception(input_tensor)

x = L.GlobalMaxPooling2D()(x)

x = L.Dense(1024,activation='relu')(x)

x = L.Dense(69,activation='softmax',name='predictions_yayyy')(x)



model = Model(input_tensor,x)



display(model.summary())

display(plot_model(model))