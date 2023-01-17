# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def input_files():
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
input_files()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot
output = {
    0:"T-Shirt/Top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle Boot",
}
train_df.tail()
train_arr = np.array(train_df, dtype='float32')
test_arr = np.array(test_df, dtype='float32')
# Individual random image
import random
i = random.randint(1,60000)
plt.imshow(train_arr[i,1:].reshape(28,28))
label = train_arr[i, 0]
output[label]
# Grid random images
W_grid = 10
L_grid = 10
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17,17))
axes = axes.ravel()
for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, len(train_arr))
    axes[i].imshow(train_arr[index, 1:].reshape(28,28))
    label = output[train_arr[index,0]]
    axes[i].set_title(label, fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)
X_train = train_arr[:,1:]/255
y_train = train_arr[:,0]
X_test = test_arr[:,1:]/255
y_test = test_arr[:,0]
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28,28,1))
print(X_train.shape)
print(X_test.shape)
print(X_validate.shape)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model = Sequential()
cnn_model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32, activation='relu'))
cnn_model.add(Dense(output_dim=10, activation='sigmoid'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
cnn_model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=50,
    verbose=1,
    validation_data=(X_validate, y_validate)
)
evaluation = cnn_model.evaluate(X_test, y_test)
print(evaluation)
print("Test Set Accuracy: {:.2f} %".format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test)
predicted_classes
L=5
W=5
fig, axes = plt.subplots(L, W, figsize=((15,15)))
axes = axes.ravel()
for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {} \nTest Class = {}".format(output[predicted_classes[i]], output[y_test[i]]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=True, fmt='.0f')
from sklearn.metrics import classification_report
num_classes = 10
target_names = ["{}". format(output[i]) for i in range(num_classes) ]
print(classification_report(y_test, predicted_classes, target_names = target_names))