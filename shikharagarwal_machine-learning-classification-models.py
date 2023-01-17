import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.utils import shuffle

# reading data from csv files and converting to matrix 

test  = pd.read_csv("../input/test.csv")  

train = pd.read_csv("../input/train.csv") 



print("load input data successfull!!")
#shuffling data 

test  = shuffle(test)

train = shuffle(train)



train.head()
#seperating class label from the dataset



trainLabels= train.Activity.values

trainData=train.drop("Activity",axis=1).values



testLabels= test.Activity.values

testData=test.drop("Activity",axis=1).values



print("Class labels striped off the dataset")
#transforming non-numerical labels to numerical labels using sklearn.preprocessing.LabelEncoder



from sklearn import preprocessing

labelEncoder= preprocessing.LabelEncoder()



labelEncoder.fit(trainLabels)

trainLabelsE=labelEncoder.transform(trainLabels)



labelEncoder.fit(testLabels)

testLabelsE=labelEncoder.transform(testLabels)



print("Labels Transformed and Encoded")
#seperating class label from the dataset



trainLabels= train.Activity.values

trainData=train.drop("Activity",axis=1).values



testLabels= test.Activity.values

testData=test.drop("Activity",axis=1).values



print("Class labels striped off the dataset")
#transforming non-numerical labels to numerical labels using sklearn.preprocessing.LabelEncoder



from sklearn import preprocessing

labelEncoder= preprocessing.LabelEncoder()



labelEncoder.fit(trainLabels)

trainLabelsE=labelEncoder.transform(trainLabels)



labelEncoder.fit(testLabels)

testLabelsE=labelEncoder.transform(testLabels)



print("Labels Transformed and Encoded")
#applying k-nearest neighbours



from sklearn.neighbors import KNeighborsClassifier as knn

import numpy as np



knnScoreDistance=np.zeros(51)

knnScoreUniform=np.zeros(51)



for num in range(5,51):

    knnclf = knn(n_neighbors=num, weights='distance')

    knnModel = knnclf.fit(trainData , trainLabelsE)

    knnScoreDistance[num]=knnModel.score(testData  , testLabelsE )

    print("Testing  set score for KNN_Distance(k=%d): %f" %(num,knnScoreDistance[num]))

    

for num in range(5,51):

    knnclf = knn(n_neighbors=num, weights='uniform')

    knnModel = knnclf.fit(trainData , trainLabelsE)

    knnScoreUniform[num]=knnModel.score(testData  , testLabelsE )

    print("Testing  set score for KNN_Uniform(k=%d): %f" %(num,knnScoreUniform[num]))
import matplotlib.pyplot as plt



x=np.array(range(5,51))



plt.plot(x,knnScoreDistance[5:])

plt.plot(x,knnScoreUniform[5:])

plt.xlabel("No of neighbors (K)")

plt.ylabel("Test Data Mean Accuracy")

plt.legend(['KNN_Distance','KNN_Uniform'])

plt.show
import itertools

import numpy as np

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix



decsnTreeClf= DecisionTreeClassifier(criterion='entropy')

tree=decsnTreeClf.fit(trainData,trainLabelsE)

testPred=tree.predict(testData)



acc= accuracy_score(testLabelsE,testPred)

cfs = confusion_matrix(testLabelsE, testPred)



print("Accuracy: %f" %acc)



plt.figure()

class_names = labelEncoder.classes_

plot_confusion_matrix(cfs, classes=class_names,

                      title='DecisionTree Confusion Matrix, without normalization')
decsnTreeClf= DecisionTreeClassifier()

tree=decsnTreeClf.fit(trainData,trainLabelsE)

testPred=tree.predict(testData)



acc= accuracy_score(testLabelsE,testPred)

cfs = confusion_matrix(testLabelsE, testPred)



print("Accuracy: %f" %acc)



plt.figure()

class_names = labelEncoder.classes_

plot_confusion_matrix(cfs, classes=class_names,

                      title='DecisionTree Confusion Matrix, without normalization')