# Дообучение (fine tuning) верхнего сверточного блока предварительно обученной сверточной нейросети Xception.

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import models

from keras.applications import VGG16, Xception

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

    for layer in _conv_base.layers:

        print(layer, layer.trainable)



    print('MODEL.SUMMARY:')

    _model.summary()

    return





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

batch_size = 256

lr = 0.0001

epochs = 50



# ----------------------------------------------------------------------------------------------------------------------

# Создадим генераторы:

train_datagen = ImageDataGenerator(  # Расширяем тренировочные данные.

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



validation_datagen = ImageDataGenerator(rescale=1. / 255)  # Проверочные данные не расширяем.

validation_generator = validation_datagen.flow_from_directory(

    validation_dir,

    target_size=target_size,

    batch_size=batch_size,

    class_mode='categorical'

)



# Создадим модель:

model = get_xception_fine_tune_model(input_shape, num_classes)



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

model.save('fine_tune_xception.h5')



# Выведем рейт на тестовых данных:

validation_score = model.evaluate_generator(validation_generator, steps=validation_steps)

print('Validation loss: ', validation_score[0])

print('Validation acc:  ', validation_score[1])



# Сформируем графики потерь на этапе обучения и проверки:

draw_training_info(history)
