# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/digit-recognizer/train.csv')  # importing data in csv format with pandas library
test = pd.read_csv('../input/boratest/test.csv')
data.head()                           #  
data.info()
x = data.drop(['label'], axis=1).values/255 # make input values numpy array, then normalize by dividing with 255.
for i in range(9):   
    
    plt.subplot(3,3,i+1)
    plt.imshow(x[i].reshape(28,28), cmap='gray')
    plt.axis('off') 


### seperating label (y) values and One-Hot encoding for multi-label classification ###

# It doesen't matter in what format you'll form y, but after one-hot encoding, it must be converted to array by .toarray()
#y = pd.DataFrame(data.label)
y = data.label.values.reshape(-1,1)
x_test = test.values/255
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
y = ohe.fit_transform(y).toarray()

# Now every column in y corresponds to a class.

y.shape
### train test split ###

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 23)
### building ANN function
# importing libraries
from keras.models import Sequential # initializing neural network library
from keras.layers import Dense, Dropout # building layers

# feed-forward neural network classifier is assigned as "model".
model = Sequential()  
# we use dropout in the ratio of 0.25 to prevent overfitting.
model.add(Dropout(0.25)) 
# 8 units for the first layer, also the input shape must be given in this line. 
# ReLU activation function is more useful than tanh function due to vanishing gradient problem.
# weights are initialized as "random uniform".
model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', input_dim = x_train.shape[1])) 
# 16 nodes for the second layer
model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
# since we have 10 outputs, in the last layer we need to enter 10 nodes. The output of the softmax function can be used to represent a categorical distribution. 
model.add(Dense(10, activation='softmax', kernel_initializer='random_uniform'))

# we compile our model by using "adadelta" optimizer. 
# since we have categorical outputs, loss function must be the cross entropy. if you use grid search, you need to use "sparse_categoricalentropy".
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model with below batch size and number of epochs.
# verbose integers 0,1,2 sets the appearance of progress bar. "2" shows just a line.
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 10, batch_size = 155, verbose = 2)
# a look on test data

# since we don't have any labels on test data that helps to find accuracy, we take a look at our first 9 predictions.

predicted_classes = pd.DataFrame(model.predict(test)) # make a dataframe from prediction values because their index will be needed.
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(predicted_classes.iloc[i].idxmax(axis=1)) # idmax gives us the column name(which are our outputs) of the maximum value in a row
    plt.axis('off') # don't show the axis
### Test Loss Visualization ###
loss = go.Scatter(y= history.history['val_loss'], x=np.arange(0,10), mode = "lines+markers", name='Test Loss') 
accuracy = go.Scatter(y= history.history['val_acc'], x=np.arange(0,10), mode = "lines+markers", name='Test Accuracy') 
layout = dict(title = 'Test Loss & Accuracy Visualization',
              xaxis= dict(title= 'Epochs',ticklen= 5,zeroline= True),
              yaxis= dict(title= 'Loss & Accuracy',ticklen= 5,zeroline= True))
data = [loss, accuracy]
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# Importing confusion matrix
from sklearn.metrics import confusion_matrix

# Since we don't have the labels for "test" data like in real life, we will only create a confusion matrix of validation values.

# Predict test values.
y_predicted = model.predict(x_valid)

# Find the column indices of maximum values which corresponds to predicted digits.
# An alternative method to do this, as it's done in subplots above, to convert the matrix to dataframe first, then find maximum column indices with "idxmax".
y_predicted = np.argmax(y_predicted, axis = 1) 
y_true = np.argmax(y_valid, axis = 1) 

# Create the confusion matrix.
confusion__matrix = confusion_matrix(y_true, y_predicted) 

# Plot it!
plt.figure(figsize=(10,10))
sns.heatmap(confusion__matrix, annot=True, linewidths=0.2, cmap="Blues",linecolor="black",  fmt= '.1f')
plt.xlabel("Predicted Labels", fontsize=15)
plt.ylabel("True Labels", fontsize=15)
plt.title("Confusion Matrix", color = 'red', fontsize = 20)
plt.show()
from sklearn.metrics import precision_recall_curve
classes = y.shape[1]

precision = dict()
recall = dict()
y_predict = model.predict(x_valid)
for i in range(classes):
    precision[i], recall[i], _ = precision_recall_curve(y_valid[:, i], y_predict[:, i])
colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []
plt.figure(figsize=(10,10))

for i in range(classes):
    plt.plot(recall[i], precision[i], color=colors[i])
    labels.append('Precision-recall for class {0}'.format(i+1))
    
plt.ylim([0.0, 1.03])
plt.xlim([0.0, 1.03])
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision', fontsize = 15)
plt.title('Recall vs Precision',fontsize = 20)
plt.legend(labels, loc=(.3, .3), prop={'size':12})
plt.show()
colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,30))

for i in range(classes):
    plt.subplot(5,2,i+1)
    labels.append('Precision-recall for class {}'.format(i+1))
    plt.plot(recall[i], precision[i], color=colors[i], label=labels[i])
    plt.legend(loc=(.1, .3), prop={'size':12})
    plt.title('Class {}'.format(i+1),fontsize = 15)
    plt.xlabel('Recall',fontsize=10)
    plt.ylabel('Precision', fontsize = 10)

    
plt.show()

from sklearn.metrics import f1_score
print('F1 Score: {}'.format(f1_score(y_true, y_predicted, average='macro')))
y_predicted = y_predicted.T
y_true = y_true.T

from sklearn.preprocessing import label_binarize
y_true_roc = label_binarize(y_true,classes=[0,1,2,3,4,5,6,7,8,9])
y_pred_roc= label_binarize(y_predicted, classes=[0,1,2,3,4,5,6,7,8,9])

fpr = {} # false positive rate
tpr = {} #  true positive rate
roc_auc = {}
from sklearn.metrics import roc_curve, auc
for i in range(y_true_roc.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_pred_roc[:, i], y_true_roc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,10))
for i in range(y_true_roc.shape[1]):
    labels.append('ROC curve for class {} & Area = {:f}'.format(i+1, roc_auc[i])) 
    plt.plot(fpr[i], tpr[i], color = colors[i],label=labels[i])
    plt.legend(loc=(.2, .3), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
plt.show()
colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,30))

for i in range(classes):
    plt.subplot(5,2,i+1)
    labels.append('Precision-recall for class {}'.format(i+1))
    plt.plot(recall[i], precision[i], color=colors[i], label=labels[i])
    plt.legend(loc=(.1, .3), prop={'size':12})
    plt.title('Class {}'.format(i+1),fontsize = 15)
    plt.xlabel('Recall',fontsize=10)
    plt.ylabel('Precision', fontsize = 10)

    
plt.show()