!pip install --upgrade efficientnet-pytorch
!pip install pretrainedmodels
from efficientnet_pytorch import EfficientNet



import torch

from torch import nn

from torch.nn import functional as F



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.callbacks import *



import pretrainedmodels



from torchvision import models as md



from sklearn.metrics import *



import os
SIZE = 320



BATCH_SIZE = 32
def plot_confusion_matrix(y_true, 

                          y_pred, 

                          classes,

                          normalize = False,

                          title = None,

                          cmap = plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
def model_interpretation(learn, is_test = False, normalize_cm = True):

    

    print('Test Interpretation' if is_test else 'Validation Interpretation')

    

    interp = ClassificationInterpretation.from_learner(learn)

    

    title = 'Normalized Confusion matrix' if normalize_cm else 'Confusion matrix'

    

    interp.plot_confusion_matrix(normalize = normalize_cm, title = title)

    

    interp.plot_top_losses(9, figsize = (12, 12))

    

    interp.most_confused()
model_b4 = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 7)
model_b5 = EfficientNet.from_pretrained('efficientnet-b5', num_classes = 7)
model_b6 = EfficientNet.from_pretrained('efficientnet-b6', num_classes = 7)
transforms = [*zoom_crop(scale = (0.6, 1.5), do_rand = False), 

              contrast(scale = (0.7, 1.15), p = 0.35),

              #cutout(n_holes = (1, 8), length = (10, 30)),

              cutout(n_holes = (3, 8), length = (30, 40), p = .4),

              brightness(change = (0.4, 0.6)),

              pad(mode = 'zeros')]



tfms = get_transforms(do_flip = True, 

                      flip_vert = True,

                      max_rotate = 360.,

                      max_zoom = 1.6, 

                      max_lighting = 0.5, 

                      max_warp = 0.2, 

                      p_affine = 0.7, 

                      p_lighting = 0.5,

                      xtra_tfms = transforms)
data_test = ImageDataBunch.from_folder('../input/skin-cancer/dataset/dataset',

                                       size = SIZE,

                                       bs = BATCH_SIZE,

                                       train = 'train',

                                       valid = 'valid',

                                       ds_tfms = tfms)



data_test.normalize(imagenet_stats)
def top_2_accuracy(y_true, y_pred): return top_k_accuracy(y_true, y_pred, k = 2)



def top_3_accuracy(y_true, y_pred): return top_k_accuracy(y_true, y_pred, k = 3)



def balanced_accuracy(y_true, y_pred): return balanced_accuracy_score(y_true, y_pred)

kappa = KappaScore()

kappa.weights = 'quadratic'



b5 = Learner(data_test, 

                     model_b5, 

                     metrics = [accuracy, 

                                Precision(), 

                                MatthewsCorreff(), 

                                kappa,

                                top_2_accuracy, 

                                top_3_accuracy],

                     path = './',

                     model_dir = 'models').to_fp16()



b5.model_dir = '../input/effnetb503/'



b5.load('effnet-b5-0-3')
kappa = KappaScore()

kappa.weights = 'quadratic'



b6 = Learner(data_test, 

                     model_b6, 

                     metrics = [accuracy, 

                                Precision(), 

                                MatthewsCorreff(), 

                                kappa,

                                top_2_accuracy, 

                                top_3_accuracy],

                     path = './',

                     model_dir = 'models').to_fp16()



b6.model_dir = '../input/effnetb601/'



b6.load('effnet-b6-0-1')
b4 = Learner(data_test, 

                     model_b4, 

                     metrics = [accuracy, 

                                AUROC(), 

                                Precision(), 

                                MatthewsCorreff(), 

                                kappa,

                                top_2_accuracy, 

                                top_3_accuracy],

                     path = './',

                     model_dir = 'models').to_fp16()



b4.model_dir = '../input/effnetb43rdstage/'



b4.load('bestmodel_effnet_b4-3rd-stage')
resnet101 = cnn_learner(data_test, 

                     models.resnet101, 

                     metrics = [accuracy, 

                                AUROC(), 

                                Precision(), 

                                MatthewsCorreff(), 

                                kappa,

                                top_2_accuracy, 

                                top_3_accuracy],

                     path = './',

                     model_dir = 'models').to_fp16()



resnet101.model_dir = '../input/resnet101ham10000001/'



resnet101.load('resnet-101-ham10000-0-0-1')
preds_b5, y = b5.get_preds()

preds_b4, y = b4.get_preds()

preds_b6, y = b6.get_preds()

preds_resnet101, y = resnet101.get_preds()
preds = (preds_b5 * 0.55  + preds_b4 * 0.85 + preds_resnet101 * 0.38 + preds_b6 * 0.7)
balanced_accuracy(y, preds.argmax(axis = 1))
print('ENSEMBLE')

print('Sensitivity: ', recall_score(preds.argmax(axis = 1), y, average = 'weighted'))

print('Balanced Accuracy: ', balanced_accuracy(y, preds.argmax(axis = 1)))

print('Matthews Corref: ', matthews_corrcoef(y, preds.argmax(axis = 1)))

print('Kappa: ', cohen_kappa_score(y, preds.argmax(axis = 1)))
print('EFFICIENTNET B5')

print('Sensitivity: ', recall_score(preds_b5.argmax(axis = 1), y, average = 'weighted'))

print('Balanced Accuracy: ', balanced_accuracy(y, preds_b5.argmax(axis = 1)))

print('Matthews Corref: ', matthews_corrcoef(y, preds_b5.argmax(axis = 1)))

print('Kappa: ', cohen_kappa_score(y, preds_b5.argmax(axis = 1)))
print('EFFICIENTNET B6')

print('Sensitivity: ', recall_score(preds_b6.argmax(axis = 1), y, average = 'weighted'))

print('Balanced Accuracy: ', balanced_accuracy(y, preds_b6.argmax(axis = 1)))

print('Matthews Corref: ', matthews_corrcoef(y, preds_b6.argmax(axis = 1)))

print('Kappa: ', cohen_kappa_score(y, preds_b6.argmax(axis = 1)))
print('RESNET 101')

print('Sensitivity: ', recall_score(preds_resnet101.argmax(axis = 1), y, average = 'weighted'))

print('Balanced Accuracy: ', balanced_accuracy(y, preds_resnet101.argmax(axis = 1)))

print('Matthews Corref: ', matthews_corrcoef(y, preds_resnet101.argmax(axis = 1)))

print('Kappa: ', cohen_kappa_score(y, preds_resnet101.argmax(axis = 1)))
cm = confusion_matrix(y, preds.argmax(axis = 1))
print(classification_report(preds.argmax(axis = 1), y, target_names = data_test.classes))
plot_confusion_matrix(y, preds.argmax(axis = 1), data_test.classes, normalize = True)