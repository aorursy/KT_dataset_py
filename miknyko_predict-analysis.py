import os

import keras

import matplotlib.pyplot as plt

import numpy as np

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import seaborn as sns

import time

from sklearn.metrics import roc_curve, auc



%matplotlib inline

plt.rcParams['figure.figsize'] = [200,9]
valid_dir = '../input/fcc-data-0718/fcc_data_0718/valid/'

train_dir = '../input/fcc-data-0718/fcc_data_0718/train/'

inputsize = (299,299)

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

       

    Arguments

    ---------

    confusion_matrix: numpy.ndarray

        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 

        Similarly constructed ndarrays can also be used.

    class_names: list

        An ordered list of class names, in the order they index the given confusion matrix.

    figsize: tuple

        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,

        the second determining the vertical size. Defaults to (10,7).

    fontsize: int

        Font size for axes labels. Defaults to 14.

        

    Returns

    -------

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.title('Confusion Matrix')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



def roc_auc(y,prob):

    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率

    roc_auc = auc(fpr,tpr) ###计算auc的值

 

    plt.figure()

    lw = 2

    plt.figure(figsize=(10,10))

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

 

    plt.show()
model = load_model('../input/fcc-model-0714/fcc_model_0717_2.h5')
predict_generator = ImageDataGenerator(rescale = 1./255)

valid_generator = predict_generator.flow_from_directory(valid_dir,

                                                       target_size = inputsize,

                                                       batch_size = 500,

                                                       class_mode = 'binary',

                                                       shuffle = True)

train_generator = predict_generator.flow_from_directory(train_dir,

                                                       target_size = inputsize,

                                                       batch_size = 500,

                                                       class_mode = 'binary',

                                                       shuffle = True)
batch_num = 0

start = time.time()

predict = model.predict(valid_generator[batch_num][0])

predict_train = model.predict(train_generator[batch_num][0])

end = time.time()
y_pred = (predict > 0.5)

y_pred_train = (predict_train > 0.5)

labels = valid_generator[batch_num][1].reshape(-1,1)

labels_train = train_generator[batch_num][1].reshape(-1,1)
predict[y_pred != labels]
j = 0

for i in range(len(labels)):

    if j >= 10:

        break

    

    if int(labels[i] != y_pred[i]):       

        plt.subplots(figsize=(9,6))

        plt.imshow(valid_generator[batch_num][0][i])

        plt.axis('off')

        plt.title(f'pred{y_pred[i]}Label{labels[i]}Score{predict[i]}',fontsize = 30)

#         plt.show()

        j += 1

    

            

j = 0

for i in range(len(labels)):

    if j >= 20:

        break

    

    if int(labels[i] == y_pred[i]):       

        plt.subplots(figsize=(9,6))

        plt.imshow(valid_generator[batch_num][0][i])

        plt.axis('off')

        plt.title(f'pred{y_pred[i]}Label{labels[i]}Score{predict[i]}',fontsize = 30)

#         plt.show()

        j += 1
from sklearn.metrics import confusion_matrix

class_name = ['bad','good']

cnf_matrix = confusion_matrix(labels,y_pred)

_ = print_confusion_matrix(cnf_matrix, class_name)
from sklearn.metrics import classification_report



report = classification_report(labels, y_pred, target_names=class_name)

print('='*100)

print(report)

print('='*100)
report_train = classification_report(labels_train,y_pred_train,target_names = class_name)

print('='*100)

print(report_train)

print('='*100)
print(f'predict time per pic is {(end - start)/len(labels)}s')
roc_auc(labels_train,predict_train)
roc_auc(labels,predict)
batch_num = 0

model.evaluate(valid_generator[batch_num][0],valid_generator[batch_num][1])