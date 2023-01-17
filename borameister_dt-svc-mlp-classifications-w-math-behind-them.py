# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
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

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# Importing data
train = pd.read_csv('../input/sign_mnist_train.csv')
test = pd.read_csv('../input/sign_mnist_test.csv')
#train.label = train.label.apply(lambda x: str(x))
#train.label = train.label.apply(lambda x: str(x).replace('0','A') if 0 in train.label else str(x))
#train.label = train.label.apply(lambda x: str(x).replace('1','B') if 1 in train.label else str(x))
#train.label = train.label.apply(lambda x: str(x).replace('2','C') if 2 in train.label else str(x))

plt.figure(figsize=(10,10))
for i in range(9):   
    
    plt.subplot(3,3,i+1)
    plt.imshow(train.drop(['label'], axis=1).values[i].reshape(28,28), cmap='gray')
    plt.axis('off')
a = train[train.label==0]
b = train[train.label==1]
c = train[train.label==2]
new_train = pd.concat([a,b,c],axis=0, ignore_index=True)
print('shape & labels of train: {}, {}'.format(new_train.shape,new_train.label.unique()))

a_test = test[test.label==0]
b_test = test[test.label==1]
c_test = test[test.label==2]
new_test = pd.concat([a_test,b_test,c_test],axis=0, ignore_index=True)
print('shape & labels of test : {}, {}'.format(new_test.shape,new_test.label.unique()))
x_train = new_train.drop(['label'],axis=1).values/255
y_train = new_train.label.values.reshape(-1,1)
x_test = new_test.drop(['label'], axis=1).values/255
y_test = new_test.label.values.reshape(-1,1)
bar = go.Bar(x=new_train.label.value_counts().index,
       y=new_train.label.value_counts().values,
       marker = dict(color = 'rgba(15, 100, 111)'))
iplot([bar])
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=9, random_state=42)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print('score:',dtc.score(x_test,y_test))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, linecolor='black', linewidths=1, cmap='Greens', fmt='.1f')
plt.xticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15)
plt.yticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15, rotation=0)
plt.show()

from sklearn.preprocessing import label_binarize
y_true_roc = label_binarize(y_test,classes=[0,1,2])
y_pred_roc = label_binarize(y_pred, classes=[0,1,2])
fpr = {} #  false positive rate
tpr = {} #  true positive rate
roc_auc = {}
from sklearn.metrics import roc_curve, auc
for i in range(y_true_roc.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_pred_roc[:, i], y_true_roc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = ['red','orange','blue']
labels = []
liste = ['A','B','C']
plt.figure(figsize=(10,30))
for i in range(y_true_roc.shape[1]):
    plt.subplot(3,1,i+1)
    labels.append('ROC curve for sign {} & Area = {:f}'.format(liste[i], roc_auc[i])) 
    plt.plot(fpr[i], tpr[i], color = colors[i], label=labels[i])
    plt.legend(loc=(.17, .45), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
    
plt.show()
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)
y_pred2 = svc.predict(x_test)
print('score:',svc.score(x_test,y_test))
y_pred2 = pd.DataFrame(y_pred2.T)
ytest = pd.DataFrame(y_test)
conf = pd.concat([ytest,y_pred2],ignore_index=True,axis=1)
sum(conf[0]-conf[1])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, linecolor='black', linewidths=1, cmap='Greens', fmt='.1f')
plt.xticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15)
plt.yticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15, rotation=0)
plt.show()
y_true_roc2 = label_binarize(y_test,classes=[0,1,2])
y_pred_roc2 = label_binarize(y_pred2, classes=[0,1,2])
fpr2 = {} # false positive rate
tpr2 = {} #  true positive rate
roc_auc2 = {}
from sklearn.metrics import roc_curve, auc
for i in range(y_true_roc2.shape[1]):
    fpr2[i], tpr2[i], _ = roc_curve(y_pred_roc2[:, i], y_true_roc2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
colors = ['red','orange','blue']
labels = []
liste = ['A','B','C']
plt.figure(figsize=(10,30))
for i in range(y_true_roc2.shape[1]):
    plt.subplot(3,1,i+1)
    labels.append('ROC curve for class {} & Area = {:f}'.format(liste[i], roc_auc2[i])) 
    plt.plot(fpr2[i], tpr2[i], color = colors[i],label=labels[i])
    plt.legend(loc=(.2, .45), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
plt.show()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
y_test_encoded = ohe.fit_transform(y_test).toarray()
y_train_encoded = ohe.fit_transform(y_train).toarray()

### building ANN function
# importing libraries
from keras.models import Sequential     # initializing neural network library
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
model.add(Dense(3, activation='softmax', kernel_initializer='random_uniform'))

# we compile our model by using "adadelta" optimizer. 
# since we have categorical outputs, loss function must be the cross entropy. if you use grid search, you need to use "sparse_categoricalentropy".
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model with below batch size and number of epochs.
# verbose integers 0,1,2 sets the appearance of progress bar. "2" shows just a line.
history = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), epochs = 20, batch_size = 100, verbose = 2)

loss = go.Scatter(y= history.history['val_loss'], x=np.arange(0,20), mode = "lines+markers", name='Test Loss') 
accuracy = go.Scatter(y= history.history['val_acc'], x=np.arange(0,20), mode = "lines+markers", name='Test Accuracy') 
layout = dict(title = 'Test Loss & Accuracy Visualization',
              xaxis= dict(title= 'Epochs',ticklen= 5,zeroline= True),
              yaxis= dict(title= 'Loss & Accuracy',ticklen= 5,zeroline= True))
data = [loss, accuracy]
fig = go.Figure(data = data, layout = layout)
iplot(fig)
y_pred3 = model.predict(x_test)

# Find the column indices of maximum values which corresponds to predicted digits.
# An alternative method to do this is to convert the matrix into a dataframe first, then find maximum column indices with "idxmax".
y_pred3 = np.argmax(y_pred3, axis = 1) 

# Create the confusion matrix.
confusion__matrix = confusion_matrix(y_test, y_pred3) 

# Plot
plt.figure(figsize=(12,12))
sns.heatmap(confusion__matrix, annot=True, linewidths=0.2, cmap="Blues",linecolor="black",  fmt= '.1f')
plt.xlabel("Predicted Labels", fontsize=15)
plt.xticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15)
plt.yticks(np.arange(3), ('A', 'B', 'C'),fontsize = 15, rotation=0)
plt.ylabel("True Labels", fontsize=15)
plt.title("Confusion Matrix", color = 'red', fontsize = 20)
plt.show()
y_true_roc3 = label_binarize(y_test,classes=[0,1,2])
y_pred_roc3 = label_binarize(y_pred3, classes=[0,1,2])
fpr3 = {} # false positive rate
tpr3 = {} #  true positive rate
roc_auc3 = {}
from sklearn.metrics import roc_curve, auc
for i in range(y_true_roc3.shape[1]):
    fpr3[i], tpr3[i], _ = roc_curve(y_pred_roc3[:, i], y_true_roc3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])
colors = ['red','orange','blue']
labels = []
liste = ['A','B','C']
plt.figure(figsize=(10,30))
for i in range(y_true_roc3.shape[1]):
    plt.subplot(3,1,i+1)
    labels.append('ROC curve for class {} & Area = {}'.format(liste[i], roc_auc3[i])) 
    plt.plot(fpr3[i], tpr3[i], color = colors[i],label=labels[i])
    plt.legend(loc=(.2, .45), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
plt.show()
