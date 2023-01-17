# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
import h5py

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')
iris.drop(columns=['Id'],inplace=True)
iris['target']=0
iris.head()
iris.loc[iris['Species']=='Iris-setosa','target']=0
iris.loc[iris['Species']=='Iris-versicolor','target']=1
iris.loc[iris['Species']=='Iris-virginica','target']=2
classes = iris['Species'].unique()
classes
iris.drop('Species',axis=1,inplace=True)
iris.head()
from sklearn.model_selection import train_test_split
X = iris.drop('target',axis=1)
y = iris['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
model = Sequential([
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(Adam(lr=0.005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train.values,y_train.values,epochs=5000,validation_split=0.1)
prediction = model.predict_classes(X_test)
prediction
from sklearn.metrics import confusion_matrix,accuracy_score
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
cm = confusion_matrix(y_test,prediction)
plot_confusion_matrix(cm,classes=classes,title='Iris Prediction Confusion Matrix')
print('Accuracy :',accuracy_score(y_test,prediction,))
from tensorflow.keras.models import save_model
save_model(model,'iris_ann.h5')