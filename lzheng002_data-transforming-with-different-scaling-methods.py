# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import tensorflow.keras as ks
df = pd.read_csv("../input/train.csv", sep=',', header=0)
df.head()
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def drawExampleImage(X):
    # X is pd dataframe
    for k in range(16) :
        plt.subplot(4, 4, k+1)
        plt.imshow(np.reshape(X.iloc[k, :].values, [28, 28]), cmap=cm.get_cmap('binary'))
        #plt.text(df.iloc[k, 0], 3, 3)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
drawExampleImage(df.iloc[:, 1:])
# show keras version
ks.__version__
def DNN(hidden_layer_sizes=[128, 50, 50]) :
    model = ks.models.Sequential()

    model.add(ks.layers.Flatten())
    
    # define hidden layer sizes
    for ls in hidden_layer_sizes :
        model.add(ks.layers.Dense(ls, activation=tf.nn.relu))

    model.add(ks.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
from sklearn import preprocessing, cross_validation

X = df.iloc[:, 1:]
Y = df.label
# transform Y label to one-hot encoding
Ys = pd.get_dummies(Y).values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Ys, test_size=0.2)
# train model
model = DNN([28*28, 100, 50])
hist_null = model.fit(X_train.values, y_train, epochs=50)

# evaluate the model
model.save_weights("./DNN_normalize_null.model")
model.evaluate(x=X_test, y=y_test, )
from sklearn import preprocessing, cross_validation

X = df.iloc[:, 1:]
Y = df.label
# transform Y label to one-hot encoding
Ys = pd.get_dummies(Y).values

MEAN = X.mean(axis=0)
STD  = X.std(axis=0)
STD = pd.Series([ 0.0001 if x == 0 else x for x in STD ])
STD.index = X.columns

# define a scaled dataset
#scaler = preprocessing.StandardScaler()
Xs = (X - MEAN) / STD

X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xs, Ys, test_size=0.2)
# show some sample images
drawExampleImage(X=X_train)

# train the model
model = DNN([28*28, 100, 50])
hist_mean = model.fit(X_train.values, y_train, epochs=50)

# evaluate the model
model.save_weights("./DNN_normalize_meanstd.model")
model.evaluate(x=X_test, y=y_test )
X = df.iloc[:, 1:]
Y = df.label
# transform Y label to one-hot encoding
Ys = pd.get_dummies(Y).values

MAX = X.max(axis=0)
MIN = X.min(axis=0)
RANGE = MAX - MIN

# using 10000 is to avoid any divide by zero error
RANGE = pd.Series([ 10000 if x == 0 else x for x in RANGE ])
RANGE.index = X.columns

# define a scaled dataset
#scaler = preprocessing.StandardScaler()
Xs = (X - MIN) / RANGE

X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xs, Ys, test_size=0.2)
# show some sample images
drawExampleImage(X=X_train)

# train the model
model = DNN([28*28, 100, 50])
hist_min = model.fit(X_train.values, y_train, epochs=50)
# evaluate the model
model.save_weights("./DNN_normalize_maxmin.model")
model.evaluate(x=X_test, y=y_test, )
X = df.iloc[:, 1:]
Y = df.label
# transform Y label to one-hot encoding
Ys = pd.get_dummies(Y).values

# aviod log zero error
X = X + .5
X = np.log10(X.values)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Ys, test_size=0.2)
# show some sample images
drawExampleImage(X=pd.DataFrame(X_train))

# train the model
model = DNN([28*28, 100, 50])
hist_log = model.fit(X_train, y_train, epochs=50)
# evaluate the model
model.save_weights("./DNN_normalize_log.model")
model.evaluate(x=X_test, y=y_test, )

methods = ['Null', 'MaxMin', 'MeanStd', 'Log']
accs = [
    0.368,
    0.966,
    0.982,
    0.979   
]
# test accuracy
#plt.scatter(range(1,5), accs)

plt.bar(range(1,5), accs)
plt.ylabel("Accuracy", size=16)
plt.xlabel("Methods", size=16)
plt.xticks(range(1,5), methods)
# loss and accuracy along epochs
hists = [ hist_null, hist_min, hist_mean, hist_log]
for i in range(4):
    plt.plot(np.arange(1, 51), hists[i].history['loss'], label=methods[i])
plt.legend()
plt.xlabel("Epochs", size=16)
plt.ylabel("Loss", size=16)
plt.ylim([0, .3])
plt.show()

for i in range(4):
    plt.plot(np.arange(1, 51), hists[i].history['acc'], label=methods[i])
plt.legend()
plt.xlabel("Epochs", size=16)
plt.ylabel("Accuracy", size=16)
plt.ylim([0.92, 1.])
plt.show()
X_pred = pd.read_csv("../input/test.csv", header=0, sep=',')

X_pred =(X_pred - MIN) / RANGE

model = DNN([28*28, 100, 50])
model.load_weights("./DNN_normalize_maxmin.model")

y_pred = np.argmax( model.predict(X_pred.values), axis=0 )
y = pd.DataFrame()

y['ImageId'] = range(1, y_pred.shape[0]+1)
y['Label'] = y_pred

y.to_csv("submissions_dnn.csv", sep=",", header=True, index=False)