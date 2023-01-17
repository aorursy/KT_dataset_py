!pip install git+https://github.com/keras-team/keras-tuner.git -q
import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    y = data["label"]
    x = data.drop(labels=["label"], axis=1).values.reshape(-1, 28, 28, 1)
    return x, y

x_train, y_train = load_data("../input/digit-recognizer/train.csv")
x_test, _ = load_data("../input/digit-recognizer/train.csv")
from tensorflow import keras
from tensorflow.keras import layers

def augment_images(x, hp):
    use_rotation = hp.Boolean('use_rotation')
    if use_rotation:
        x = layers.experimental.preprocessing.RandomRotation(
            hp.Float('rotation_factor', min_value=0.05, max_value=0.2)
        )(x)
    use_zoom = hp.Boolean('use_zoom')
    if use_zoom:
        x = layers.experimental.preprocessing.RandomZoom(
            hp.Float('use_zoom', min_value=0.05, max_value=0.2)
        )(x)
    return x

def make_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.experimental.preprocessing.Resizing(64, 64)(x)
    x = augment_images(x, hp)
    
    num_block = hp.Int('num_block', min_value=2, max_value=5, step=1)
    num_filters = hp.Int('num_filters', min_value=32, max_value=128, step=32)
    for i in range(num_block):
        x = layers.Conv2D(
            num_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)
        x = layers.Conv2D(
            num_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)
        x = layers.MaxPooling2D(2)(x)
    
    reduction_type = hp.Choice('reduction_type', ['flatten', 'avg'])
    if reduction_type == 'flatten':
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        units=hp.Int('num_dense_units', min_value=32, max_value=512, step=32),
        activation='relu'
    )(x)
    x = layers.Dropout(
        hp.Float('dense_dropout', min_value=0., max_value=0.7)
    )(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-3)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
    model.summary()
    return model
import kerastuner as kt

tuner = kt.tuners.RandomSearch(
    make_model,
    objective='val_acc',
    max_trials=100,
    overwrite=True)

callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=3, baseline=0.9)]
tuner.search(x_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=100)
best_hp = tuner.get_best_hyperparameters()[0]
model = make_model(best_hp)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50)
val_acc_per_epoch = history.history['val_acc']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
model = make_model(best_hp)
model.fit(x_train, y_train, epochs=best_epoch)
import numpy as np

predictions = model.predict(x_test)
submission = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                           "Label": np.argmax(predictions, axis=-1)})
submission.to_csv("submission.csv", index=False)