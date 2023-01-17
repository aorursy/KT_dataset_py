import matplotlib.pyplot as plt





def draw_training_info_plots(_history):

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



    if 'lr' in _history.history:

        learning_rate = _history.history['lr']

        plt.plot(epochs_plot, learning_rate, 'b', label='Learning rate')

        plt.title('Learning rate')

        plt.xlabel('epoch')

        plt.ylabel('learning rate')

        plt.legend()

        plt.show()

    return





def print_model(model):

    """Напечатать информацию о моделе"""

    print('MODEL TRAINABLE LAYERS:')

    for i, layer in enumerate(model.layers):

        print(i, ')', layer, layer.trainable)



    print('MODEL.SUMMARY:')

    model.summary()

    return

from typing import Tuple, Callable



import keras

from keras import layers

from keras import models

from keras.applications import VGG16, Xception, ResNet50, NASNetMobile, MobileNetV2, MobileNet

from keras.engine.training import Model





def get_vgg16_fine_tune_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Fine tuning верхнего сверточного блока VGG16 с дополнительным Dense-слоем"""

    image_size = 224

    channels_count = 3

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, channels_count))

    conv_base.trainable = True



    for i, layer in enumerate(conv_base.layers):

        if i < 15:

            layer.trainable = False

        else:

            layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.vgg16.preprocess_input





def get_vgg16_full_tune_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Полное дообучение VGG16"""

    image_size = 224

    channels_count = 3

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, channels_count))



    # Дообучаем всю сеть.

    conv_base.trainable = True

    for i, layer in enumerate(conv_base.layers):

        layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.vgg16.preprocess_input





def get_vgg16_fine_tune_model_concatenated(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Fine tuning верхнего сверточного блока VGG16 с дополнительным Dense-слоем.

    Прямое включение слоев VGG16 в итоговую модель"""

    image_size = 224

    channels_count = 3

    initial_model: Model = VGG16(weights='imagenet', include_top=False,

                                 input_shape=(image_size, image_size, channels_count))



    # Дообучаем верхние сверточные слои.

    initial_model.trainable = True

    for i, layer in enumerate(initial_model.layers):

        if i < 15:

            layer.trainable = False

        else:

            layer.trainable = True



    # Включаем слои VGG16 напрямую в итоговую модель.

    initial_model_output = initial_model.output

    x = layers.GlobalAveragePooling2D()(initial_model_output)

    x = layers.Dense(1024, activation='relu')(x)

    predictions = layers.Dense(_num_classes, activation='softmax')(x)



    model = Model(initial_model.input, predictions)



    return model, initial_model, image_size, keras.applications.vgg16.preprocess_input





def get_xception_fine_tune_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Fine tuning верхнего блока Xception"""

    # It should have exactly 3 inputs channels, and width and height should be no smaller than 71.

    # E.g. (150, 150, 3) would be one valid value.

    image_size = 299

    channels_count = 3

    conv_base = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, channels_count))

    conv_base.trainable = True



    for i, layer in enumerate(conv_base.layers):

        if i < 115:

            layer.trainable = False

        else:

            layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.xception.preprocess_input





def get_resnet50_feature_extraction_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Feature extraction ResNet50"""

    image_size = 224

    channels_count = 3

    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, channels_count))



    # Обучаем всю сверточную основу.

    conv_base.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.resnet50.preprocess_input





def get_nasnetmobile_full_tune_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Fine tuning верхнего сверточного блока VGG16 с дополнительным Dense-слоем"""

    image_size = 224

    channels_count = 3

    conv_base = NASNetMobile(weights='imagenet', include_top=False,

                             input_shape=(image_size, image_size, channels_count))

    conv_base.trainable = True



    # for i, layer in enumerate(conv_base.layers):

    #     if i < 15:

    #         layer.trainable = False

    #     else:

    #         layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.nasnet.preprocess_input





def get_mobilenet_full_tune_model(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Дообучение всех слоев MobileNet"""

    image_size = 224

    channels_count = 3

    conv_base = MobileNet(weights='imagenet', alpha=1.0, include_top=False,

                          input_shape=(image_size, image_size, channels_count))



    # Дообучаем всю сеть.

    conv_base.trainable = True

    for i, layer in enumerate(conv_base.layers):

        layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.mobilenet_v2.preprocess_input





def get_mobilenetv2_full_tune_model_alpha_1(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Дообучение всех слоев MobileNetV2, alpha=1.0"""

    image_size = 224

    channels_count = 3

    conv_base = MobileNetV2(weights='imagenet', alpha=1.0, include_top=False,

                            input_shape=(image_size, image_size, channels_count))



    # Дообучаем всю сеть.

    conv_base.trainable = True

    for i, layer in enumerate(conv_base.layers):

        layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.mobilenet_v2.preprocess_input





def get_mobilenetv2_full_tune_model_alpha_1_4(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Дообучение всех слоев MobileNetV2, alpha=1.4"""

    # If imagenet weights are being loaded, alpha can be one of `0.35`, `0.50`, `0.75`, `1.0`, `1.3` or `1.4` only.

    image_size = 224

    channels_count = 3

    conv_base = MobileNetV2(weights='imagenet', alpha=1.4, include_top=False,

                            input_shape=(image_size, image_size, channels_count))



    # Дообучаем всю сеть.

    conv_base.trainable = True

    for i, layer in enumerate(conv_base.layers):

        layer.trainable = True



    model = models.Sequential()

    model.add(conv_base)



    # add a global spatial average pooling layer

    model.add(layers.GlobalAveragePooling2D())



    # let's add a fully-connected layer

    model.add(layers.Dense(1024, activation='relu'))



    # and a logistic layer -- let's say we have N classes

    model.add(layers.Dense(_num_classes, activation='softmax'))



    return model, conv_base, image_size, keras.applications.mobilenet_v2.preprocess_input





def get_mobilenetv2_full_tune_model_alpha_1_4_concatenated(_num_classes) -> Tuple[Model, Model, int, Callable]:

    """Дообучение всех слоев MobileNetV2, alpha=1.4. Слои MobileNetV2 включены напрямую."""

    # If imagenet weights are being loaded, alpha can be one of `0.35`, `0.50`, `0.75`, `1.0`, `1.3` or `1.4` only.

    image_size = 224

    channels_count = 3

    initial_model: Model = MobileNetV2(weights='imagenet', alpha=1.4, include_top=False,

                                       input_shape=(image_size, image_size, channels_count))



    # Дообучаем всю сеть.

    initial_model.trainable = True

    for i, layer in enumerate(initial_model.layers):

        layer.trainable = True



    # Включаем слои MobileNetV2 напрямую в итоговую модель.

    initial_model_output = initial_model.output

    x = layers.GlobalAveragePooling2D()(initial_model_output)

    x = layers.Dense(1024, activation='relu')(x)

    predictions = layers.Dense(_num_classes, activation='softmax')(x)



    model = Model(initial_model.input, predictions)



    return model, initial_model, image_size, keras.applications.mobilenet_v2.preprocess_input

import functools

import os



from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model



# ----------------------------------------------------------------------------------------------------------------------

# Вспомогательные функции:

def print_model_info(_conv_base, _model):

    """Напечатать информацию о сверточной основе и всей моделе"""

    print('CONV_BASE.SUMMARY:')

    _conv_base.summary()



    print('CONV_BASE TRAINABLE LAYERS:')

    for i, layer in enumerate(_conv_base.layers):

        print(i, ')', layer, layer.trainable)



    print('MODEL.SUMMARY:')

    _model.summary()

    return





# ----------------------------------------------------------------------------------------------------------------------

def get_callbacks_list(_early_stopping_patience, _reduce_lr_on_plateau_factor, _reduce_lr_on_plateau_patience):

    """Получить коллбеки для модели"""

    return [

        keras.callbacks.EarlyStopping(

            monitor='val_acc',

            patience=_early_stopping_patience

        ),

        keras.callbacks.ModelCheckpoint(

            verbose=1,

            filepath='best_model.h5',

            monitor='val_loss',

            save_best_only=True

        ),

        keras.callbacks.ReduceLROnPlateau(

            verbose=1,

            monitor='val_loss',

            factor=_reduce_lr_on_plateau_factor,

            patience=_reduce_lr_on_plateau_patience

        ),

    ]





# ----------------------------------------------------------------------------------------------------------------------

# Директории с данными:

# train_dir = 'D:/ML Datasets/stanford-car-dataset-by-classes-folder/car_data/train'

# validation_dir = 'D:/ML Datasets/stanford-car-dataset-by-classes-folder/car_data/test'

train_dir = '../input/car_data/car_data/train'

validation_dir = '../input/car_data/car_data/test'

# train_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'

# validation_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'



# ----------------------------------------------------------------------------------------------------------------------

# Гиперпараметры:

batch_size = 80

init_lr = 0.0001

momentum = 0.9

epochs = 45

# optimazer = keras.optimizers.SGD(lr=init_lr, momentum=momentum)

optimazer = keras.optimizers.Adam(lr=init_lr)



# Настройки коллбеков:

early_stopping_patience = 10

reduce_lr_on_plateau_factor = 0.2

reduce_lr_on_plateau_patience = 3



# Выбранная модель:

model_function = get_resnet50_feature_extraction_model



# ----------------------------------------------------------------------------------------------------------------------

# Узнаем число классов задачи по число директорий в тренировочной папке:

num_classes = len(os.listdir(train_dir))



# Получим полную модель, сверточную основу, размер изоображения и функцию препроцессинга входных изображений:

model, conv_base, image_size, preprocess_function = model_function(num_classes)



# Распечатаем характеристики модели:

print_model_info(conv_base, model)



# Нарисуем изображение с распечаткой структуры модели:

plot_model(conv_base, show_shapes=True, to_file='conv_base.png')

plot_model(model, show_shapes=True, to_file='model.png')



# ----------------------------------------------------------------------------------------------------------------------

# Создадим генераторы:

# Расширяем тренировочные данные.

# Используем только preprocess_function, без какой-либо иной предобработки значений пикселов и каналов.

train_image_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_function,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_generator = train_image_datagen.flow_from_directory(

    train_dir,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical'

)

train_images_count = len(train_generator.filenames)



# Проверочные данные не расширяем.

validation_image_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

validation_generator = validation_image_datagen.flow_from_directory(

    validation_dir,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical'

)

validation_images_count = len(validation_generator.filenames)



# ----------------------------------------------------------------------------------------------------------------------

# Метрика для получения топ-5 точности

top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)

top5_acc.__name__ = 'top5_acc'



# Компилируем модель:

model.compile(loss='categorical_crossentropy', optimizer=optimazer, metrics=['accuracy', top5_acc])



# ----------------------------------------------------------------------------------------------------------------------

# Тренируем модель:

train_steps = len(train_generator.filenames) // batch_size

validation_steps = len(validation_generator.filenames) // batch_size

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    callbacks=get_callbacks_list(early_stopping_patience, reduce_lr_on_plateau_factor, reduce_lr_on_plateau_patience)

)



# ----------------------------------------------------------------------------------------------------------------------



# Выведем рейт на тестовых данных:

validation_score = model.evaluate_generator(validation_generator, steps=validation_steps)

print('Validation loss: ', validation_score[0])

print('Validation acc:  ', validation_score[1])



# Сформируем графики потерь на этапе обучения и проверки:

draw_training_info_plots(history)
