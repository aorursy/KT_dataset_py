import numpy as np
import pandas as pd
import keras as kr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

#references:
# https://machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6

def plot_multiclass_decision_boundary_2D(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(8, 8))

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def plot_multiclass_decision_boundary_4D(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    w_min, w_max = X[:, 2].min() - 0.1, X[:, 2].max() + 0.1
    t_min, t_max = X[:, 3].min() - 0.1, X[:, 3].max() + 0.1

    xx, yy, ww, tt = np.meshgrid(np.linspace(x_min, x_max, 20),
                                 np.linspace(y_min, y_max, 20),
                                 np.linspace(w_min, w_max, 20),
                                 np.linspace(t_min, t_max, 20))

    Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel(), ww.ravel(), tt.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    xx = np.reshape(xx,(400,400))
    yy = np.reshape(yy,(400,400))
    ZZ = np.reshape(Z ,(400,400))

    fig = plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, ZZ, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    

# steps to produce a MLBPB
#1. Load Training Data.
#2. Define Model.
#3. Compile Model.
#4. Fit Model.
#5. Evaluate Model.

#1. Load Training Data.
#load
datatrain = datasets.load_iris()
#change string value to numeric
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)
#change dataframe to array
datatrain_array = datatrain.as_matrix()
#split x and y (feature and target)
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]
ytrain = to_categorical(ytrain)


#2. Defining Model:
#Network parameters
n_input = 4
n_hidden1 = 10
n_output = 3
#Learning parameters
number_epochs = 10000
batch_size = 10

model = Sequential()
#model.add(Dense(10, input_dim=n_input, activation='relu'))
model.add(Dense(n_hidden1, activation='relu'))
model.add(Dense(n_output,  activation='softmax'))



#3. Compile Model.
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

#4. Fit Model.
earlystop = kr.callbacks.EarlyStopping(monitor='acc', min_delta=0.02, patience=1000, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(xtrain, ytrain, validation_split=0.33, verbose=2, epochs=number_epochs,callbacks=[earlystop])

#5. Evaluate Model.#5. Evaluate Model.
#load
datatest = pd.read_csv('D:/Users/elton/Downloads/Datasets/iris/iris_test.csv')

#change string value to numeric
datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)
#change dataframe to array
datatest_array = datatest.as_matrix()

#split x and y (feature and target)
xtest = datatest_array[:,:4]
ytest = datatest_array[:,4]
#5. Evaluate Model.
predict = model.predict_classes(xtest, batch_size=batch_size, verbose=0)
accuration = np.sum(predict == ytest)/30.0 * 100

print("Test Accuration : " + str(accuration) + '%')
print("Prediction :")
print(predict)
print("Target :")
print(np.asarray(ytest,dtype="int32"))


predict = model.predict_classes(xtest, batch_size=batch_size, verbose=0)
accuration = np.sum(predict == ytest)/30.0 * 100

print("Test Accuration : " + str(accuration) + '%')
print("Prediction :")
print(predict)
print("Target :")
print(np.asarray(ytest,dtype="int32"))

# summarize history for accuracy
plt.figure(figsize=(8,10))
plt.title('model accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training Accuracy', 'test Accuracy'], loc='upper left')
plt.show()
plt.figure(figsize=(8,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training Loss', 'test loss'], loc='upper left')
plt.show()

plot_multiclass_decision_boundary_4D(model, xtest, ytest)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


a = plt.figure(figsize=(7,6))
fig = a
ax = fig.add_subplot(111, projection='3d')


x = (xtest[:,0])
y = (xtest[:,1])
z = (xtest[:,2])
c = (xtest[:,3])

ax.scatter(x, y, z, c=c, cmap=plt.hot())
plt.show()
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

from ann_visualizer.visualize import ann_viz
ann_viz(model, title="MyNeural Network")
from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))