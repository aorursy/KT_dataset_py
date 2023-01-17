from matplotlib import pyplot as plt
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
%matplotlib inline
def val_loss_acc_stat(result):
    val_loss = min(result.history.get('val_loss'))
    val_acc = max(result.history.get('val_acc'))

    print('val_loss: {} | val_acc: {}'.format(
        val_loss, val_acc
    ))
def test_loss_acc_stat(model, x, y=None):
    if y:
        loss, acc = model.evaluate(x, y)
    else:
        loss, acc = model.evaluate_generator(x)

    print('test_loss: {} | test_acc: {}'.format(
        loss, acc
    ))
def loss_acc_plot(result):
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(
        epochs, 
        loss, 
        color='green', 
        linestyle='', 
        marker='o',
        markersize=2, 
        label='Training'
    )
    plt.plot(
        epochs, 
        val_loss, 
        color='red', 
        linestyle='', 
        marker='o',
        markersize=2, 
        label='Validation'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(
        epochs, 
        acc, 
        color='green', 
        linestyle='', 
        marker='o',
        markersize=2, 
        label='Training'
    )
    plt.plot(
        epochs, 
        val_acc, 
        color='red', 
        linestyle='', 
        marker='o',
        markersize=2, 
        label='Validation'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
train_dir = '../input/training_set/training_set/'
test_dir = '../input/test_set/test_set/'
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    subset='validation',
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
)
train_generator.batch_size
train_generator.image_shape
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

result = model.fit_generator(
    train_generator,
    epochs=150,
    validation_data=validation_generator,
    callbacks=[
        EarlyStopping(
            monitor='val_loss', 
            mode='auto', 
            verbose=1, 
            patience=25
        ), 
        ModelCheckpoint(
            'best_model.h5', 
            monitor='val_loss', 
            mode='auto', 
            save_best_only=True, 
            verbose=1
        )
    ]
)
val_loss_acc_stat(result)
loss_acc_plot(result)
model = load_model('best_model.h5')
test_loss_acc_stat(model, test_generator)