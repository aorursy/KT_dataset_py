#Code help taken from: https://www.back-prop.com/deep_learning/knn/mnist/2019/05/16/knn_classifier/
import mnist #Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #Graph
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.datasets import load_digits
from IPython.display import display, HTML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
train_images = mnist.train_images() #training data images
train_labels = mnist.train_labels() #training data labels
test_images = mnist.test_images() #testing data images
test_labels = mnist.test_labels() #testing data labels

#validation dataset
train_images,val_images,train_labels,val_labels = train_test_split(train_images,
                                                                   train_labels,
                                                                   test_size=0.1,#10% validation data
                                                                   random_state=84)
print("training data points: {}".format(len(train_labels)))
print("validation data points: {}".format(len(val_labels)))
print("testing data points: {}".format(len(test_labels)))

train_images = train_images / 255
test_images = test_images / 255
val_images = val_images / 255


train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
val_images = val_images.reshape((-1, 784))


print(test_images.shape) 
print(train_images.shape)
print(val_images.shape) 
kVals = np.arange(1,30,2)

for k in kVals:
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_images,train_labels)
    
    # evaluate the model and update the accuracies list
    score = model.score(val_images, val_labels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#For best value of accuracy, k turned out to be 3
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_images, train_labels)
predictions = model.predict(test_images)

print(classification_report(test_labels, predictions))

#predict on the first five test images
predictions = model.predict(test_images[:6])
print('labels: ' , test_labels[:6])
print('predictions by our model: ' , predictions)
#ref https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

predictions = model.predict(test_images)
plot_confusion_matrix(confusion_matrix(test_labels, predictions), 
                      normalize    = True,
                      target_names = ['0', '1', '2',  '3',  '4', '5',  '6', '7',  '8', '9'],
                      title        = "Confusion Matrix, Normalized")


sub = pd.DataFrame(predictions)
sub.index.name = 'ImageId'
sub.index += 1
sub.columns = ['Label']
sub.to_csv('submission_kNN.csv', header = True)