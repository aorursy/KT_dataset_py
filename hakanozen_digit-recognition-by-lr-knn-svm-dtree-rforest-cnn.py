import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from numpy import argmax
import warnings
warnings.filterwarnings('ignore')
#data_sample = pd.read_csv("../input/sample_submission.csv")
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
data_train.head()
y_train = data_train.label.values
x_train = data_train.drop(["label"], axis = 1)
x_train = x_train/255 # normalization

x_test = data_test/255
print("y_train: ", y_train.shape)
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
image1 = x_train.values[0].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(image1)
image2 = x_train.values[300].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(image2)
image3 = x_train.values[600].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(image3)
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=46)
print("x_train: ",x_train.shape)
print("x_val: ",x_val.shape)
print("y_train: ",y_train.shape)
print("y_val: ",y_val.shape)
# Store accuracies of the machine learning methods for comparison at the end
list_names = []
list_accuracy = []
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
LR_accuracy = lr.score(x_val, y_val)*100
LR_accuracy = round(LR_accuracy, 2)

print("LR_accuracy is %", LR_accuracy)
list_names.append("Logistic Regression")
list_accuracy.append(LR_accuracy)
y_pred_LR = lr.predict(x_test)
N = 210
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_LR[N])
N = 27000
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_LR[N])
N = 4000
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_LR[N])
from sklearn.neighbors import KNeighborsClassifier

Knn_accuracies = []
number_of_neighbors = []
for neighbors in range(1,20):
    Knn = KNeighborsClassifier(n_neighbors = neighbors)
    Knn.fit(x_train, y_train)
    Knn_accuracy = round(Knn.score(x_val, y_val)*100,2)
    Knn_accuracies.append(Knn_accuracy)
    number_of_neighbors.append(neighbors)
    
## Visualization
# Accuracy vs n_estimators
trace1 = go.Scatter(
                    y = Knn_accuracies,
                    x = number_of_neighbors,
                    mode = "lines",
                    name = "K-NN Classifier",
                   )

data = [trace1]
layout = dict(title = 'KNN Accuracy',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Validation Accuracy (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Number of Neighbors',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)    
    
Knn_accuracy = max(Knn_accuracies)
print("Knn_accuracy is %", Knn_accuracy)
list_names.append("K-nn")
list_accuracy.append(Knn_accuracy)
y_pred_KNN = Knn.predict(x_test)
N = 4500
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_KNN[N])
N = 7700
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_KNN[N])
N = 21851
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_KNN[N])
from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train, y_train)
SVM_accuracy = svm.score(x_val, y_val)*100
SVM_accuracy = round(SVM_accuracy, 2)

print("SVM_accuracy is %", SVM_accuracy)

list_names.append("SVM")
list_accuracy.append(SVM_accuracy)
y_pred_SVM = svm.predict(x_test)
N = 8
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_SVM[N])
N = 1142
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_SVM[N])
N = 6548
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_SVM[N])
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
DecisionTree_accuracy = dt.score(x_val, y_val)*100
DecisionTree_accuracy = round(DecisionTree_accuracy,2)

print("DecisionTree_accuracy is %", DecisionTree_accuracy)

list_names.append("Decision Tree")
list_accuracy.append(DecisionTree_accuracy)
y_pred_DT = dt.predict(x_test)
N = 180
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_DT[N])
N = 17520
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_DT[N])
N = 23150
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_DT[N])
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RF_accuracies = []
number_of_estimators = []
for num_of_estimators in range(5,405,20):
    rf = RandomForestClassifier(n_estimators = num_of_estimators, random_state = 1)
    rf.fit(x_train, y_train)
    RandomForest_accuracy = rf.score(x_val, y_val)*100
    RF_accuracies.append(RandomForest_accuracy)
    number_of_estimators.append(num_of_estimators)

# Accuracy vs n_estimators
trace1 = go.Scatter(
                    y = RF_accuracies,
                    x = number_of_estimators,
                    mode = "lines",
                    name = "Random Forest Classifier",
                   )

data = [trace1]
layout = dict(title = 'Random Forest Accuracy',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Validation Accuracy (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Number of Estimators',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)

RandomForest_accuracy = round(max(RF_accuracies),2)
print("Random Forest accuracy is ", RandomForest_accuracy)
list_names.append("Random Forest")
list_accuracy.append(RandomForest_accuracy)
y_pred_RF = rf.predict(x_test)
N = 7000
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_RF[N])
N = 24000
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_RF[N])
N = 18000
img = x_test.values[N].reshape(28,28) # reshape 1*784 raw data to 28*28 image
imgplot = plt.imshow(img)
print("The digit in the following image is ",y_pred_RF[N])
# Reshape
x_train = x_train.values.reshape(-1,28,28,1)
x_val = x_val.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
print("x_train shape: ",x_train.shape)
print("x_val shape: ",x_val.shape)
print("x_test shape: ",x_test.shape)
# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
y_train = to_categorical(y_train, num_classes = 10)
y_val = to_categorical(y_val, num_classes = 10)
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 12, kernel_size = (5,5),padding = 'Same', 
                 activation ='tanh', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 20, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 201  # for better result increase the epochs
batch_size = 250
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), 
                              steps_per_epoch=x_train.shape[0] // batch_size)

val_accuracy = history.history['val_acc']
CNN_accuracy = round(max(val_accuracy)*100,2)
print("CNN accuracy is ", CNN_accuracy)
list_names.append("CNN")
list_accuracy.append(CNN_accuracy)
# Plot the loss and accuracy curves for training and validation 
accuracy = []
num_of_epochs = []
for i in range(1,201,10):
    accuracy.append(round(100*val_accuracy[i],3))
    num_of_epochs.append(i)

trace1 = go.Scatter(y = accuracy, x = num_of_epochs, mode = "lines")
data = [trace1]
layout = dict(title = 'CNN Accuracy',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Accuracy (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Number of Epochs',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
df = pd.DataFrame({'METHOD': list_names, 'ACCURACY (%)': list_accuracy})
df = df.sort_values(by=['ACCURACY (%)'])
df = df.reset_index(drop=True)
df.head()
trace1 = go.Bar(x = df.iloc[:,0].tolist(), y = df.iloc[:,1].tolist())

data1 = [trace1]
layout1 = go.Layout(
    margin=dict(b=150),
    title='Comparison of the Learning Methods',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='ACCURACY (%)',gridwidth=2, gridcolor='#bdbdbd', range=[80, 100]),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

fig = go.Figure(data=data1, layout=layout1)
py.iplot(fig, filename='grouped-bar')