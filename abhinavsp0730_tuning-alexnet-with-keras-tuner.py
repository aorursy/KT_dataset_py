import tensorflow as tf
import IPython
import tensorflow_datasets as tfds
!pip install -q -U keras-tuner
import kerastuner as kt
# loding the datasets from tensorflow_datasets
train, train_info = tfds.load('fashion_mnist', split='train', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('fashion_mnist', 
                          split='test', 
                          as_supervised=True, 
                          with_info=True)
def resize(img, lbl):
  img_size = 96
  return (tf.image.resize(img, [img_size, img_size])/255.) , lbl

train = train.map(resize)
val = val.map(resize)

train = train.batch(32, drop_remainder=True)
val = val.batch(32, drop_remainder=True)     
def baseline_alexnet():
     return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
#training on gpu if available
def try_gpu(i=0): 
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
  model = baseline_alexnet()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    
                    callbacks = [callback])
# storing the result on baseline_val_acc so that we can use it to compare later.
baseline_val_acc = max(history.history['val_accuracy'])
baseline_val_acc
def modified_alexnet(hp):
  alexnet = tf.keras.models.Sequential()
  # filter size from 96-256
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_1',
                                                      min_value=96,
                                                       max_value=256,
                                                       default=96,
                                                       step=32),
                                      kernel_size=11, strides=4, activation='relu'))
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  # filter size from 256-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_2',
                                                       min_value=256,
                                                       max_value=512,
                                                       default=256,
                                                       step=32),
                                      kernel_size=5, padding='same', activation='relu'))
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  # filter size from 384-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_3',
                                                       min_value=384,
                                                       max_value=512,
                                                       default=384,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
  # filter size from 384-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_4',
                                                       min_value=384,
                                                       max_value=512,
                                                       default=384,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
   # filter size from 256-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_5',
                                                       min_value=256,
                                                       max_value=512,
                                                       default=256,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
  
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  alexnet.add(tf.keras.layers.Flatten())
  # dense unit from 4096 to 8192
  alexnet.add(tf.keras.layers.Dense(units = hp.Int(name='units_1',
                                                  min_value=4096,
                                                  max_value=8192,
                                                  default=4096,
                                                  step=256),
                                                  activation='relu'))
  # dropout value from 0-0.5
  alexnet.add(tf.keras.layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1, default=0.5)))
  alexnet.add(tf.keras.layers.Dense(units = hp.Int(name='units_2',
                                                   min_value=4096,
                                                    max_value=8192,
                                                    default=4096,
                                                    step=256),
                                                    activation='relu'))
  # dropout value from 0-0.5
  alexnet.add(tf.keras.layers.Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1, default=0.5)))
  alexnet.add(tf.keras.layers.Dense(10, activation='softmax'))
  # choice for the learning rate, i.e 0.01, 0.001, 0.0001 
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
  
  alexnet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                metrics = ['accuracy'])

  return alexnet
tuner = kt.Hyperband(modified_alexnet,
                     objective = 'val_accuracy', 
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                      distribution_strategy=tf.distribute.OneDeviceStrategy(device_name),
                     project_name ='intro_to_kt')    
#callback to clear the training output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)
tuner.search(train, epochs = 10, validation_data = (val), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
callback = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])
# storing the result on modified_alexnet_val_acc baseline_val_acc so that we can use it to compare baseline_val_acc.
modified_alexnet_val_acc = max(history.history['val_accuracy'])
modified_alexnet_val_acc
print("The validation accuracy of the baseline alexnet model is {} VS The validation accuracy of the baseline alexnet model is {}".format(baseline_val_acc, modified_alexnet_val_acc))