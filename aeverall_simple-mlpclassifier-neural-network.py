import os, sys
import itertools, time
import numpy as np 
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

# preprocessing
from sklearn.model_selection import train_test_split

# postprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
# Test set
test = pd.read_csv('../input/test.csv')
# Training set
data = pd.read_csv('../input/train.csv')
# Target features
y = data[['label']]
X = data.drop('label', axis=1)
# Train test split - 20% testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=15)

# Train cross-validation split - overall 20% cross-validation
Xtrain, Xcv, ytrain, ycv = train_test_split(Xtrain, ytrain, test_size=(0.2/0.8), random_state=15)

# So we have a 60-20-20 train-cv-test split
len(Xtrain), len(Xcv), len(Xtest), len(test)

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

    #print(cm)

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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(200), random_state=0, max_iter=500)
start = time.time()

cnn.fit(Xtrain, np.array(ytrain)[:,0])

end = time.time()
print("Time taken: %f" % (end-start))
predict = cnn.predict(Xcv)
acc = accuracy_score(ycv, predict)
print("Accuracy: %f" % acc)

confusion = confusion_matrix(np.array(ycv)[:,0], predict)

plt.figure(figsize=(10,10))
plot_confusion_matrix(confusion, classes=np.arange(10), normalize=True)

predict = cnn.predict(Xtest)
acc = accuracy_score(ytest, predict)
print("Accuracy: %f" % acc)

confusion = confusion_matrix(np.array(ytest)[:,0], predict)

plt.figure(figsize=(10,10))
plot_confusion_matrix(confusion, classes=np.arange(10), normalize=True)

