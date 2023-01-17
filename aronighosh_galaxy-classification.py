# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
def split_data(data,split_size):

    np.random.seed(0)

    np.random.shuffle(data)

    

    split=int(len(data)*split_size)

    train=data[:split]

    test=data[split:]

    

    return train,test
def generate_features_labels(data):

    features=np.zeros((data.shape[0],13))

    labels=data['class']

    

    features[:,0]=data['u-g']

    features[:,1]=data['g-r']

    features[:,2]=data['r-i']

    features[:,3]=data['i-z']

    features[:,4]=data['ecc']

    features[:,5]=data['m4_u']

    features[:,6]=data['m4_g']

    features[:,7]=data['m4_r']

    features[:,8]=data['m4_i']

    features[:,9]=data['m4_z']

    

    features[:,10]=data['petroR50_u']/data['petroR90_u']

    features[:,11]=data['petroR50_r']/data['petroR90_r']

    features[:,12]=data['petroR50_z']/data['petroR90_z']



    

    return features,labels

    
def decision_tree_predict_actual(data,split_size):

    train_data,test_data=split_data(data,split_size)

    train_features,train_labels=generate_features_labels(train_data)

    test_features,test_labels=generate_features_labels(test_data)

    

    dtr=DecisionTreeClassifier()

    dtr.fit(train_features,train_labels)

    predictions=dtr.predict(test_features)

    

    return predictions,test_labels

    

    
def calculate_accuracy(pred,actual):

    n=len(actual)

    acc=sum([1 for i,j in zip(pred,actual) if i==j])/n

    return acc
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

 

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, "{}".format(cm[i, j]),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True Class')

    plt.xlabel('\nPredicted Class')
data=np.load('/kaggle/input/galaxy_catalogue.npy')

split_size=0.7

pred_label,test_label=decision_tree_predict_actual(data,split_size)

print('some results of galaxy classifications')

for i in range(10):

    print("{}.{},{}".format(i,pred_label[i],test_label[i]))
model_acc=calculate_accuracy(pred_label,test_label)

print('Model Accuracy:',model_acc)
class_labels = list(set(test_label))

model_cm=confusion_matrix(y_pred=pred_label,y_true=test_label,labels=class_labels)

print(model_cm)

plt.figure()

plot_confusion_matrix(model_cm, classes=class_labels)

plt.show()
def random_forest_actual(data,estimators):

    features,labels=generate_features_labels(data)

    rfc=RandomForestClassifier(n_estimators=estimators)

    predictions=cross_val_predict(rfc,features,labels,cv=10)

    return predictions,labels
data=np.load('/kaggle/input/galaxy_catalogue.npy')

np.random.seed(0)

np.random.shuffle(data)

estimators=50  #number of trees

pred_label,test_label=random_forest_actual(data,estimators)

print('some results of galaxy classifications')

for i in range(10):

    print("{}.{},{}".format(i,pred_label[i],test_label[i]))
model_acc=calculate_accuracy(pred_label,test_label)

print('Model Accuracy:',model_acc)
class_labels = list(set(test_label))

model_cm=confusion_matrix(y_pred=pred_label,y_true=test_label,labels=class_labels)

print(model_cm)

plt.figure()

plot_confusion_matrix(model_cm, classes=class_labels)

plt.show()