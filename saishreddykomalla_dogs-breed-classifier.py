import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# dict_labels={}
label_csv = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
X = label_csv.iloc[:, 0].values
Y = label_csv.iloc[:, 1].values
#____________________________________--
breeds = list(sorted(set(Y)))
os.mkdir("/kaggle/working/training/")
for i in breeds:
    os.mkdir("/kaggle/working/training/" + i)
label_paths = label_csv.assign(img_path = lambda x: "/kaggle/input/dog-breed-identification/train/" + x["id"] + ".jpg")
for i in range(10222):
    shutil.copyfile(label_paths['img_path'][i], "/kaggle/working/training/" + label_paths['breed'][i] + "/" + label_paths['id'][i] + ".jpg")

from tensorflow.keras.applications.inception_v3 import InceptionV3
pre_trained_model = InceptionV3(input_shape = (150,150, 3), include_top = False, weights = 'imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
print(last_output)
class mcb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['acc']>0.99:
            print("\nReached 99% accuracy\n")
            self.model.stop_training = True
callbacks = mcb()
from tensorflow import keras
import tensorflow as tf
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dense(120, activation = 'softmax')(x)
model = tf.keras.Model(pre_trained_model.input, x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.summary()
#_______________________________
print(tf.version)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale =1.0/ 255.0
)
train_generator = train_datagen.flow_from_directory(
    "/kaggle/working/training",
    target_size = (150,150),
    batch_size = 19,
    class_mode = 'categorical'
)
test_datagen = ImageDataGenerator(
    rescale = 1.0/ 255.0
)
fmodelt = model.fit_generator(train_generator, epochs =10, steps_per_epoch = len(train_generator), callbacks = [callbacks])
%matplotlib inline
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
acc = fmodelt.history['acc']
loss = fmodelt.history['loss']
epochs = range(len(acc))
plt.plot(epochs,acc, 'r', label = "Training_accuracy")
plt.legend(loc = "lower right")
plt.figure()
plt.plot(epochs, loss, 'b', label = "Training_loss")
plt.title("Training accuracy and loss")
plt.legend(loc=0)
os.mkdir("/kaggle/working/testing/")
os.mkdir("/kaggle/working/testing/test")
for i in os.listdir("/kaggle/input/dog-breed-identification/test/"):
    shutil.copyfile("/kaggle/input/dog-breed-identification/test/" + i, "/kaggle/working/testing/test/"+ i)
test_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
test_generator = test_datagen.flow_from_directory(
    "/kaggle/working/testing/",
    target_size = (150,150),
    class_mode = 'categorical',
    shuffle=False
)
preds = model.predict(test_generator, verbose=1)
preds[0].round(3)
ids=[]
for i in os.listdir("/kaggle/working/testing/test/"):
    ids.append(i[:-4])
sol = pd.DataFrame(preds.round(4))
sol.columns = breeds
sol.insert(0, 'id', ids)
sol.to_csv("Dog_breed_classifier.csv", index=False)