import  numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import itertools

%matplotlib inline
def plot_confusion_matrix(cm,

                          classes,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions

    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix 

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        # title of graph



   

    """



  



    accuracy = (np.trace(cm) / float(np.sum(cm)))*100

    misclass = 100 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if classes is not None:

        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes, rotation=45)

        plt.yticks(tick_marks, classes)



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
plot_confusion_matrix(cm           = np.array([[ 5284.0,  62.0, 167.0, 26.0],

                                              [  60.0,  1553.0,  11.0, 76.0],

                                              [  193.0,  21.0, 980.0,  118.0],

                                              [  27.0,  108.0,  82.0, 3754.0]]), 

                      normalize    = False,

                      classes = ['CNV', 'DME', 'DRUSEN','NORMAL'],

                      title        = "Confusion Matrix")
