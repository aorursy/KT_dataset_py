import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/fashion-mnist_train.csv').sample(frac=1).set_index('label')
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
def beautifyAxes(axes):
    """Strip away the frame and ticks from a matplotlib.axes.Axes object
    """
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_yticklabels([])
    axes.set_xticklabels([])

def generateSampleImageGrid(df, labels, examplesPerClothingType=10):
    """Display a grid of Fashion MNIST images to allow users to become acquainted with the dataset

    Keyword arguments:
    df -- FMNIST DataFrame, reindexed by the 'label'
    labels -- A list of strings, with each string corresponding to the clothing article label of its index
    examplesPerClothingType -- Number of times each clothing type should be printed on grid (default 10)
    """
    figure = plt.figure(figsize=(examplesPerClothingType * 5, len(labels) * 5))
    figureIndex = 1
    for label in range(0, len(labels)):
        dfFilteredByLabel = df.loc[[label]]
        for exampleIndex in range (0, examplesPerClothingType):
            singleItem = dfFilteredByLabel.iloc[[exampleIndex]]
            image = singleItem.values.reshape(28,28)
            axes = figure.add_subplot(len(labels), examplesPerClothingType, figureIndex)
            figureIndex += 1

            # Put the name of the clothing article on the rightmost image.
            if (exampleIndex == 0):
                axes.set_ylabel(labels[label], fontsize=40)
            beautifyAxes(axes)

            plt.imshow(image, cmap=plt.cm.binary)

    plt.show()

generateSampleImageGrid(df, labels=labels)
from sklearn import metrics, svm

def getConfusionMatrixForTrainingDataSize(size):
    training_data = df.head(size).values
    training_values = df.head(size).index.values

    classifier = svm.SVC(C=100000,gamma=.000001)
    classifier.fit(training_data, training_values)
    expected = df.tail(4000).index.values
    predicted = classifier.predict(df.tail(4000).values)
    return metrics.confusion_matrix(expected, predicted)


figure = plt.figure(figsize=(15,15))
for matrix in range (0,10):
    beautifyAxes(figure.add_subplot(5, 2, matrix+1))
    plt.imshow(getConfusionMatrixForTrainingDataSize((matrix+5)*10),cmap=plt.cm.binary)
plt.show()
    



