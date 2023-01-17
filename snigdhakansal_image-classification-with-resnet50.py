from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.applications.resnet50 import ResNet50
resnet = ResNet50(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(150,150,3),
                    pooling=None,classes=6)
CNN_Classifier = Sequential()
CNN_Classifier.add(resnet)
CNN_Classifier.add(Conv2D(64, 3, 3, input_shape=(150,150,3), activation='relu'))
CNN_Classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN_Classifier.add(Conv2D(128, 3, 3, activation='relu',padding='same'))
CNN_Classifier.add(MaxPooling2D(pool_size=(1, 1),padding='same'))
CNN_Classifier.add(Conv2D(256, 3, 3, activation='relu',padding='same'))
CNN_Classifier.add(MaxPooling2D(pool_size=(1, 1),padding='same'))
CNN_Classifier.add(Flatten())
CNN_Classifier.add(Dense(1024, activation='relu'))
CNN_Classifier.add(Dropout(0.5))
CNN_Classifier.add(Dense(6, activation='softmax'))
CNN_Classifier.compile(optimizer = optimizers.RMSprop(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip=True,
                           preprocessing_function= preprocess_input)

test_datagen = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip=True,
                              preprocessing_function= preprocess_input)

training_set = train_datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_train/seg_train',
                                                 target_size = (150, 150),
                                                 batch_size = 100,
                                                 class_mode = 'categorical',
                                                 seed=42,
                                                 shuffle=True)
testing_set = test_datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_test/seg_test',
                                                 target_size = (150, 150),
                                                 batch_size = 10,
                                                 class_mode = 'categorical',
                                                 seed=42,
                                                 shuffle=True)
CNN_Classifier.fit_generator(training_set,
                    steps_per_epoch = 140,
                    validation_data = testing_set,
                    validation_steps = 300,
                    epochs=10,verbose=1)
