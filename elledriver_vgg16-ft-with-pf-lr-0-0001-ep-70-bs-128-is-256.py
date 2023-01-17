# Дообучение (fine tuning) верхнего сверточного блока предварительно обученной сверточной нейросети VGG16.

# На этот раз с использованием preprocess_function для предобработки изображений таким образом,

# каким изначально обучали сеть на imagenet.

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import models

from keras.applications import VGG16, Xception, InceptionV3

from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt





# ----------------------------------------------------------------------------------------------------------------------

# Вспомогательные функции:

def draw_training_info(_history):

    """Нарисовать графики потерь на этапе обучения и проверки"""

    acc = _history.history['acc']

    val_acc = _history.history['val_acc']

    loss = _history.history['loss']

    val_loss = _history.history['val_loss']



    epochs_plot = range(1, len(acc) + 1)

    plt.plot(epochs_plot, acc, 'bo', label='Training acc')

    plt.plot(epochs_plot, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.legend()

    plt.figure()



    plt.plot(epochs_plot, loss, 'bo', label='Training loss')

    plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend()

    plt.show()

    return





def print_model_info(_conv_base, _model):

    """Напечатать информацию о сверточной основе и всей моделе"""

    print('CONV_BASE.SUMMARY:')

    _conv_base.summary()



    print('CONV_BASE LAYERS INFO:')

    for i, layer in enumerate(_conv_base.layers):

        print(i, ')', layer, layer.trainable)



    print('MODEL.SUMMARY:')

    _model.summary()

    return





# ----------------------------------------------------------------------------------------------------------------------

# Модели:

def get_vgg16_fine_tune_model(_input_shape, _num_classes):

    """Fine tuning верхнего сверточного блока VGG16 с дополнительным Dense-слоем"""

    _conv_base = VGG16(weights='imagenet', include_top=False, input_shape=_input_shape)

    _conv_base.trainable = True



    for _i, _layer in enumerate(_conv_base.layers):

        if _i < 15:

            _layer.trainable = False

        else:

            _layer.trainable = True



    _model = models.Sequential()

    _model.add(_conv_base)



    # add a global spatial average pooling layer

    _model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    _model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have 200 classes

    _model.add(layers.Dense(_num_classes, activation='softmax'))



    print_model_info(_conv_base, _model)

    return _model





def get_xception_fine_tune_model(_input_shape, _num_classes):

    """Fine tuning верхнего блока Xception"""

    _conv_base = Xception(weights='imagenet', include_top=False, input_shape=_input_shape)

    _conv_base.trainable = True



    for _i, _layer in enumerate(_conv_base.layers):

        if _i < 115:

            _layer.trainable = False

        else:

            _layer.trainable = True



    _model = models.Sequential()

    _model.add(_conv_base)



    _model.add(layers.GlobalAveragePooling2D())

    _model.add(layers.Dense(_num_classes, activation='softmax'))



    print_model_info(_conv_base, _model)

    return _model





def get_inceptionv3_fine_tune_model(_input_shape, _num_classes):

    """Fine tuning InceptionV3"""

    _conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=_input_shape)

    _conv_base.trainable = True



    # we chose to train the top 2 inception blocks, i.e. we will freeze

    # the first 249 layers and unfreeze the rest:

    for layer in _conv_base.layers[:249]:

        layer.trainable = False

    for layer in _conv_base.layers[249:]:

        layer.trainable = True



    _model = models.Sequential()

    _model.add(_conv_base)



    # add a global spatial average pooling layer

    _model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    _model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have 200 classes

    _model.add(layers.Dense(_num_classes, activation='softmax'))



    print_model_info(_conv_base, _model)

    return _model





# ----------------------------------------------------------------------------------------------------------------------

# Директории с данными:

# train_dir = 'C:/Users/james/Downloads/stanford-car-dataset-by-classes-folder/car_data/train'

# validation_dir = 'C:/Users/james/Downloads/stanford-car-dataset-by-classes-folder/car_data/test'

train_dir = '../input/car_data/car_data/train'

validation_dir = '../input/car_data/car_data/test'

# train_dir = 'C:/Users/james/Downloads/cat-and-dog/training_set'

# validation_dir = 'C:/Users/james/Downloads/cat-and-dog/test_set'

# train_dir = 'C:/Users/james/Downloads/chest_xray/train'

# validation_dir = 'C:/Users/james/Downloads/chest_xray/test'



# Гиперпараметры:

input_shape = (256, 256, 3)

target_size = (256, 256)

batch_size = 128

lr = 0.0001

epochs = 70

preprocess_function = preprocess_input



# ----------------------------------------------------------------------------------------------------------------------

# Создадим генераторы:

train_datagen = ImageDataGenerator(  # Расширяем тренировочные данные.

    preprocessing_function=preprocess_function,

    rescale=1. / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=target_size,

    batch_size=batch_size,

    class_mode='categorical'

)

num_classes = train_generator.num_classes



# Проверочные данные не расширяем.

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_function, rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(

    validation_dir,

    target_size=target_size,

    batch_size=batch_size,

    class_mode='categorical'

)



# Создадим модель:

model = get_vgg16_fine_tune_model(input_shape, num_classes)



# Компилируем модель:

model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(lr=lr),

              metrics=['accuracy'])



# Тренируем модель:

train_steps = len(train_generator.filenames) // batch_size

validation_steps = len(validation_generator.filenames) // batch_size

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_steps

)



# Сохраняем модель:

model.save('fine_tune_vgg16.h5')



# Выведем рейт на тестовых данных:

validation_score = model.evaluate_generator(validation_generator, steps=validation_steps)

print('Validation loss: ', validation_score[0])

print('Validation acc:  ', validation_score[1])



# Сформируем графики потерь на этапе обучения и проверки:

draw_training_info(history)
