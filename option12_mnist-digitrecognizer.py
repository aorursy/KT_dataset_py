# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import sys
import time
import random
import tarfile
from IPython.display import display, Image

from scipy import ndimage
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# Config the matlotlib backend as plotting inline in IPython
%matplotlib inline
def loadData(filename):
    # Load the wholesale customers dataset
    #data = pd.read_csv(filename)
    try:
        data = pd.read_csv(filename, parse_dates=True)
        #data.drop(['Region', 'Channel'], axis = 1, inplace = True)
        print ("Dataset has {} samples with {} features each.".format(*data.shape))
    except Exception as e:
        print ("Dataset could not be loaded. Is the dataset missing?")
        print(e)
    return data
def writeData(data,filename):
    # Load the wholesale customers dataset
    try:
        data.to_csv(filename, index=False)
    except Exception as e:
        print ("Dataset could not be written.")
        print(e)
    verify=[]
    try:
        with open(filename, 'r') as f:
            for line in f:
                verify.append(line)
        f.closed
        return verify[:5]
    except IOError:
        sys.std

def dispImage(image):
        plt.imshow(image, cmap='binary')
        plt.show()
    
def runPredict(clf,Data, display=True):
    index=random.randrange(len(Data))
    y_pred = clf.predict(Data[index].reshape(1, -1))[0]
    if display==True:
        dispImage(np.reshape(Data[index],(28,28)))
    return y_pred

def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return end - start
    #print "Done!\nTraining time (secs): {:.3f}".format(end - start)
    
# Predict on training set and compute F1 score
def predict_labels(clf, features, target):
    #print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    #print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target, y_pred,average='micro'),end - start #(None, 'micro', 'macro', 'weighted', 'samples')

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):

    timeTrain=train_classifier(clf, X_train, y_train)
    predict_train,trainDelta=predict_labels(clf, X_train, y_train)
    predict_test,testDelta=predict_labels(clf, X_test, y_test)
    return predict_test,testDelta,predict_train,trainDelta,timeTrain # let's return the scores, so we can use them for comparisons

#for each data set size run and plot a train/test
def runTests(test_sizes, train_dataset,train_labels,test_dataset,test_labels, clf=""):
    test_f1=[]
    train_f1=[]

    for test_size in test_sizes:
        # Set up the train set for the test size
        X_train=train_dataset[:test_size]
        y_train=train_labels[:test_size]
        # Same for test
        X_test=test_dataset[-test_size:]
        y_test=test_labels[-test_size:]

        # logistic regresion needs some data massaging to work
      #  X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
      #  X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

        if clf == "":
            clf=LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42,  max_iter=1000,C=1e-5)

        # Fit model to training data
        test,testDelta,train,trainDelta,timeTrain = train_predict(clf, X_train, y_train, X_test, y_test)
        test_f1.append(test)
        train_f1.append(train)
        print ("------------------------------------------")
        print ("Training set size: {},".format(len(X_train)),"Train time (secs): {:.3f}".format(timeTrain))
        print ("F1 score for training set: {},".format(train),"Prediction time (secs): {:.3f}".format(trainDelta))
        print ("F1 score for test set: {},".format(test),"Prediction time (secs): {:.3f}".format(testDelta))

    
    print ("\n",clf)
    print("Test F1:{}".format(test_f1))
    display("Train F1:{}".format(train_f1))
    plt.plot(test_f1,label="Test F1")
    plt.plot(train_f1,label="Train F1")
    plt.legend(loc=2)
    plt.title("F1 Score per run")
    plt.show()
    
    return clf    
#Load up the train data
trainData=loadData("../input/train.csv")
print (trainData.head(1))
y = trainData["label"]
x = trainData.drop("label", axis=1)
#print (y.head(2))
#print (x.head(2))

#print (x.values[5])
print ("size of each entry",len(x.values[5]))

index=random.randrange(len(x))
print("for index",index,"label is:",y.values[index])
dispImage(np.reshape(x.values[index],(28,28)))
#  train/validation split
X_train, X_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.25, random_state=42)

dataSize=X_train.shape[0]
print ("size of train data",dataSize)
test_sizes=[50]
for i in range(5):
    test_sizes.append(int(round(dataSize*(i+1)*.2)))

test_sizes=[63,630,6300,31500]
#test_sizes=[50,500,5001]
print ("run tests of size",test_sizes)
clf=runTests(test_sizes, X_train,y_train,X_test,y_test)

print("Validation Prediction is:",runPredict(clf,X_test))
#loadup the  test data
print ("Test Data:")
testData=loadData("../input/test.csv") # no need to load this yet!print (testData)

#testData = np.array(testData).reshape((len(testData), -1))

print("Test Set Prediction is:",runPredict(clf,testData.values))

submission =[]
for index in range(len(testData.values)):
    submission.append((index+1,clf.predict(testData.values[index].reshape(1, -1))[0]))
    if index%5000 == 0:
        print("run:",index,"entry#:",submission[index][0], "predicted:",submission[index][1])
        dispImage(np.reshape(testData.values[index],(28,28)))
        
print ("size of submission",len(submission))


#Write our the data for submission
verify=writeData(pd.DataFrame(submission,columns=["ImageId","Label"]),'submission.csv')
print(verify)