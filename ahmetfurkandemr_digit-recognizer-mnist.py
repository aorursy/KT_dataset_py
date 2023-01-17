# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Verileri ve etiketleri koda tanımlıyoruz.

# eğitim verileri
x_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# eğitim verileri etiketi
y_train = x_train['label']

# eğitim verilerinden etiket sütununu siliyoruz
x_train = x_train.drop(labels=['label'], axis=1)

# test verileri
x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# test verileri etiketleri
y_test = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

y_test = y_test['Label']



# eğitim verisi 42000 adet 784 pixel
print(x_train.shape)

# test verisi 28000 adet 784 pixel
print(x_test.shape)

# eğitim verilerinin etiketi
print(y_train.shape)

# test verilerinin etiketleri
print(y_test.shape)

# Eğitim verilerinin ilk beşi.

x_train.head()
# 100. indexdeki rakamın görsel hali.

plt.figure(figsize=(7,7))
row_index = 100
grid_data = np.array(x_train.iloc[row_index]).reshape(28,28)
plt.imshow(grid_data, interpolation = 'none', cmap= "gray")
plt.show()
from keras.utils import to_categorical

train_images = x_train.values.reshape((42000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = x_test.values.reshape((28000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)


from keras import layers
from keras import models
from keras.utils import to_categorical

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
# modelimiz.
model = models.Sequential()

# evrişimli sinir ağları
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
# enbüyükleri biriktirme
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))

# 3B çıktıları 1B vektörlere dönüştürmek.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# modeli derleme işlemi
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# eğitim
model.fit(train_images, train_labels, epochs=8, batch_size=128)

# Model doğruluğu 0.99
# Model kaybı 0.01