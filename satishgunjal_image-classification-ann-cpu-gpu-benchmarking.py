import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system names
        if name == "PIL":
            name = "Pillow"
        elif name == "sklearn":
            name = "scikit-learn"

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))
tf.config.experimental.list_physical_devices()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {y_test.shape}')
# We have uint8 arrays of RGB image data i.e the number is stored as an 8-bit integer giving a range of possible values from 0 to 255
X_train
# Images classes possible values from 0 to 9
y_train[:10]
# There are 10 image classes
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def plot_classes():    
    fig, ax = plt.subplots(nrows=2, ncols=5)
    index = 0
    for row in ax:
        for col in row:
            col.imshow(X_train[index])
            col.title.set_text(classes[y_train[index][0]])
            index += 1
    plt.show()

plot_classes()
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
y_train[:5]
y_train_categorical = keras.utils.to_categorical(y_train, num_classes= 10, dtype='float')
y_test_categorical = keras.utils.to_categorical(y_test, num_classes= 10, dtype='float')

y_train_categorical[:5]
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer= 'SGD',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
%%timeit -n1 -r1 # time required toexecute this cell once

model = create_model()
model.fit(X_train_scaled, y_train_categorical, epochs=50)
model.evaluate(X_test_scaled, y_test_categorical)
def model_predict(index):
    print(f'Index of the predicted label: { np.argmax(model.predict(X_test_scaled)[index])}')
    print(f'True value of the label: {y_test[index][0]} and class name: {classes[y_test[index][0]]}')
    plt.imshow(X_test[index])
    
model_predict(10)                                      
                                           
%%timeit -n1 -r1

# CPU benchmarking for 1 epoch
with tf.device('/CPU:0'):
    cpu_model = create_model()
    cpu_model.fit(X_train_scaled, y_train_categorical, epochs= 1)
%%timeit -n1 -r1

# GPU benchmarking for 1 epoch
with tf.device('/GPU:0'):
    gpu_model = create_model()
    gpu_model.fit(X_train_scaled, y_train_categorical, epochs= 1)
%%timeit -n1 -r1

# CPU benchmarking for 5 epoch
with tf.device('/CPU:0'):
    cpu_model = create_model()
    cpu_model.fit(X_train_scaled, y_train_categorical, epochs= 5)
%%timeit -n1 -r1

# GPU benchmarking for 5 epoch
with tf.device('/GPU:0'):
    gpu_model = create_model()
    gpu_model.fit(X_train_scaled, y_train_categorical, epochs= 5)