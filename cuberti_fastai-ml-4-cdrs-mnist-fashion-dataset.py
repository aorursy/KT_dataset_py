%%capture
!pip install kaggle;
!pip install fastai==0.7.0;
!pip install torchtext==0.2.3;
!pip install pdpbox;
import pandas as pd
from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import *
from sklearn.model_selection import *
from IPython.display import display

from sklearn.metrics import *
from scipy.cluster import hierarchy as hc
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from pdpbox import pdp
from plotnine import *
import feather
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn
%load_ext autoreload
%autoreload 2

%matplotlib inline
train = pd.read_csv('../input/fashion-mnist_train.csv')
train_re = np.reshape(train.drop('label', axis=1).values,(-1, 28,28))

labs = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

x,x_test,y,y_test = train_test_split(train.drop('label',axis=1), train.label, random_state=42)
y_dum=pd.get_dummies(y); y_test_dum = pd.get_dummies(y_test)
x_normd = x/255
x_sq=np.reshape(x_normd.values, (-1,28,28,1))
def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)
        
def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')
plots(train_re[:8], titles=train.label[:8].map(labels))
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
model = Sequential()
# Add convolution 2D
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, 
                 kernel_size=(3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['categorical_accuracy'])
hist1=model.fit(x_sq, y_dum, epochs = 50, verbose = 1, batch_size=64)
model.summary()
x_test_2d_normd = np.reshape(x_test.values/255, (-1,28,28,1))
preds = model.predict(x_test_2d_normd)
preds[:3] #show results of first three rows of probability predictions
preds_val=preds.argmax(axis=1)
np.mean(preds_val == y_test)
plots(np.reshape(x_test.values, (-1,28,28))[:8], titles=pd.Series(preds_val)[:8].map(labels))
model2 = Sequential()
model2.add(Dense(100, activation = 'relu', input_shape=(784,)))
model2.add(Dense(100, activation = 'relu'))
model2.add(Dense(10, activation = 'softmax'))

model2.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['categorical_accuracy'])
hist2=model2.fit(x_normd, y_dum, epochs=50, verbose = 1)
preds2 = model2.predict(x_test/255)
preds2[:3] #show results of first three rows of probability predictions
preds_val2=preds2.argmax(axis=1)
np.mean(preds_val2 == y_test)
plt.plot(model2.history.history['categorical_accuracy'])
plt.plot(model.history.history['categorical_accuracy'])
plt.legend(['Simple Dense NN', 'NN w Conv Layers']); plt.xlabel('Epochs'); plt.ylabel('Categorical Accuracy')
plt.show()
test = pd.read_csv('../input/fashion-mnist_train.csv')
y_final = test.label
x_test_sq_norm = np.reshape(test.drop('label', axis = 1).values/255, (-1,28,28,1))
x_train_re = np.reshape(train.drop('label', axis=1).values/255, (-1,28,28,1))
y_trn_dum=pd.get_dummies(train.label)
y_fin_dum = pd.get_dummies(y_final)
model = Sequential()
# Add convolution 2D
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, 
                 kernel_size=(3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['categorical_accuracy'])
model.fit(x_train_re, y_trn_dum, epochs = 15, verbose = 1, batch_size=64)
pred_f=model.predict(x_test_sq_norm)
preds_f=pred_f.argmax(axis=1)
print("Final Model Accuracy on Test Set: " + str(np.mean(preds_f == y_final)))
plots(np.reshape(test.drop('label', axis = 1).values, (-1,28,28))[:8], titles=pd.Series(preds_f)[:8].map(labels))