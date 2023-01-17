from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory='../input/dogs-gone-sideways/images/train',

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

                                        directory='../input/dogs-gone-sideways/images/val',

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')





fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=22,

                                       validation_data=validation_generator,

                                       validation_steps=1)
