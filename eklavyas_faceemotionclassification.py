!nvidia-smi



import os

import sys

import cv2

import time

import random

import logging

import collections

import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf

from scipy import interp

import albumentations as A

from itertools import cycle

from sklearn import metrics

from IPython import display

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import classification_report

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import confusion_matrix as sk_cm

%matplotlib inline
class_to_num = {

    'Angry': 0,

    'Disgust': 1,

    'Fear': 2,

    'Happy': 3,

    'Sad': 4,

    'Surprise': 5,

    'Neutral': 6

}



num_to_class = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



NUM_CLASSES = len(num_to_class)
from myutilitymethods import MyMethods

from mycnn import MyCNN

from mydeepcnn import MyDeepCNN
mm = MyMethods(NUM_CLASSES, num_to_class, class_to_num)
def plot_all_confusion_matrices(y_true, y_pred, y_true_val, y_pred_val, y_true_test, y_pred_test, 

                                classes, save_title, normalize=False, title=None, cmap='GnBu', dpi=150):

    '''Plot train, validation, and test confusion matrices'''

    if not title:

        if normalize:

            title = 'Normalized Confusion Matrices'

        else:

            title = 'Non-Normalized Confusion Matrices'

    # Compute confusion matrix

    cm_train = sk_cm(y_true, y_pred)

    cm_val = sk_cm(y_true_val, y_pred_val)

    cm_test = sk_cm(y_true_test, y_pred_test)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize: 

        cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

        cm_val   = cm_val.astype('float')   / cm_val.sum(axis=1)  [:, np.newaxis]

        cm_test  = cm_test.astype('float')  / cm_test.sum(axis=1) [:, np.newaxis]

    # Lists  

    cms = [cm_train, cm_val, cm_test]

    titles = ['Train', 'Validation', 'Test']

    fig, axes = plt.subplots(nrows=1, ncols=3, dpi=dpi, figsize=(15, 8))

    # Loop

    for i, ax in enumerate(axes):

        im = ax.imshow(cms[i], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

        # Label

        ax.set(xticks=np.arange(cms[i].shape[1]),

               yticks=np.arange(cms[i].shape[0]),

               xticklabels=classes, 

               yticklabels=classes,

               title=titles[i]

               )

        # Rotate the tick labels and set their alignment.

        plt.setp(ax.get_yticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Loop

    for c, cm in enumerate(cms):

        # Loop over data dimensions and create text annotations.

        fmt = '.2f' if normalize else 'd'

        thresh = cm.max() / 2.

        # Loop

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                axes[c].text(j, i, format(cm[i, j], fmt), 

                             ha="center", va="center", 

                             color="white" if cm[i, j] > thresh else "black")

    # For only one ax

    axes[0].set(ylabel='True label',)

    axes[1].set(xlabel='Predicted label')

    # Adjust

    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.83, 0.315, 0.025, 0.375]) # [left, bottom, width, height]

    fig.colorbar(im, cax=cbar_ax)

    

    # Save

    fig.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)

    

    # Plot

    plt.show()
def make_fpr_tpr_auc_dicts(y, probs_list):

    '''Compute and return the ROC curve and ROC area for each class in dictionaries'''

    # Dicts

    fpr = dict()

    tpr = dict()

    thresholds = dict()

    roc_auc = dict()

    

    # For test

    for i in range(NUM_CLASSES):

        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y[:, i], probs_list[:, i])

        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        

    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), probs_list.ravel())

    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    

    # Compute macro-average ROC curve and ROC area

    

    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

    

    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(NUM_CLASSES):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    

    # Finally average it and compute AUC

    mean_tpr /= NUM_CLASSES

    

    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    

    return fpr, tpr, thresholds, roc_auc
def plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(-0.0025, 0.03), ylim=(0.99, 1.001), seed=0, save_title=None):

    '''Plot ROC AUC Curves'''

    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(10,5))

    

    lw = 2

    axes[0].set_xlabel('False Positive Rate')

    axes[1].set_xlabel('False Positive Rate')

    axes[0].set_ylabel('True Positive Rate')

    

    if NUM_CLASSES!=4:

        class_colors = randomColorGenerator(NUM_CLASSES, seed)

    

    for i in range(NUM_CLASSES):

        axes[0].plot(fpr[i], tpr[i], color=class_colors[i], label='{0} ({1:0.2f}%)' ''.format(num_to_class[i], roc_auc[i]*100))

        axes[1].plot(fpr[i], tpr[i], color=class_colors[i], lw=lw, label='{0} ({1:0.2f}%)' ''.format(num_to_class[i], roc_auc[i]*100))

    

    axes[0].plot(fpr['micro'], tpr['micro'], label='Micro avg ({:0.2f}%)' ''.format(roc_auc['micro']*100), linestyle=':', color='deeppink')

    axes[0].plot(fpr['macro'], tpr['macro'], label='Macro avg ({:0.2f}%)' ''.format(roc_auc['macro']*100), linestyle=':', color='navy')

    axes[0].plot([0, 1], [0, 1], color='k', linestyle='--', lw=0.5)

    axes[0].scatter(0,1, label='Ideal', s=2)

    

    axes[1].plot(fpr['micro'], tpr['micro'], lw=lw, label='Micro avg ({:0.2f}%)'.format(roc_auc['micro']*100), linestyle=':', color='deeppink')

    axes[1].plot(fpr['macro'], tpr['macro'], lw=lw, label='Macro avg ({:0.2f}%)'.format(roc_auc['macro']*100), linestyle=':', color='navy')

    axes[1].plot([0, 1], [0, 1], color='k', linestyle='--', lw=0.5)

    axes[1].scatter(0,1, label='Ideal', s=50)

    

    axes[1].set_xlim(xlim)

    axes[1].set_ylim(ylim)

    

    axes[0].grid(True, linestyle='dotted', alpha=1)

    axes[1].grid(True, linestyle='dotted', alpha=1)

    

    axes[0].legend(loc=4)

    axes[1].legend(loc=4)

    

    plt.legend(loc="lower right")

    plt.tight_layout()

    fig.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)

    plt.show()
def randomColorGenerator(number_of_colors=1, seed=0):

    '''Generate list of random colors'''

    np.random.seed(seed)

    return ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(number_of_colors)]
raw_data = pd.read_csv('../input/fer2013/fer2013.csv', header=0)
data = raw_data.values
y = data[:, 0]

pixels = data[:, 1]
%%time



x = np.zeros((pixels.shape[0], 48*48))



for x_i in range(x.shape[0]):

    p = pixels[x_i].split(' ')

    

    for y_i in range(x.shape[1]):

        x[x_i, y_i] = int(p[y_i])
x = x.reshape(x.shape[0], 48, 48)
print('x shape:', x.shape)

print('y shape:', y.shape)
indices_to_remove = []



for i,x_i in enumerate(x):

    if x_i.std() == 0:

        indices_to_remove.append(i)

        

x = np.delete(x, indices_to_remove, axis=0)

y = np.delete(y, indices_to_remove, axis=0)
# Convert to 3 channels

imgs = []

for i in range(len(x)):

    imgs.append(cv2.cvtColor(x[i].astype('uint8'), cv2.COLOR_GRAY2RGB))
x = np.concatenate(imgs)

x = x.reshape(x.shape[0]//48, x.shape[1], x.shape[1], x.shape[2])
# Standardise data

x = mm.standardise_images(x)



# Reshape

x = np.concatenate(x)

x = x.reshape(x.shape[0]//x.shape[1], x.shape[1], x.shape[1], x.shape[2])
# Resize data

imgs = []

for i in range(len(x)):

    imgs.append(mm.resize_image(x[i]))
# Reshape

x = np.concatenate(imgs)

x = x.reshape(x.shape[0]//x.shape[1], x.shape[1], x.shape[1], x.shape[2])
# Shuffle once so order of emotions is random

x, y = shuffle(x, y, random_state=0)
# Split

x_train, y_train, x_test, y_test = mm.split_train_test(x, y, split=0.95)
sets = ((x,y), (x_train, y_train), (x_test, y_test))
for my_set in sets:

    mm.plot_pca(my_set[0], my_set[1], dpi=75)
# Entire dataset

counts = collections.Counter(y)

nbs = []

for k in counts:

    nbs.append(counts[k])

    

plt.figure(dpi=75)

plt.bar(num_to_class, nbs)

plt.tight_layout()

plt.savefig('DataBarPlot.pdf', bbox_inches='tight', format='pdf', dpi=200)

plt.show()
# Train set

counts = collections.Counter(y_train)

train_nbs = []

for k in counts:

    train_nbs.append(counts[k])

    

plt.figure(dpi=75)

plt.bar(num_to_class, train_nbs)

plt.tight_layout()

plt.savefig('TrainBarPlot.pdf', bbox_inches='tight', format='pdf', dpi=200)

plt.show()
# Test set

counts = collections.Counter(y_test)

test_nbs = []

for k in counts:

    test_nbs.append(counts[k])

    

plt.figure(dpi=75)

plt.bar(num_to_class, test_nbs)

plt.tight_layout()

plt.savefig('TestBarPlot.pdf', bbox_inches='tight', format='pdf', dpi=200)

plt.show()
plt.figure(dpi=75)

plt.bar(num_to_class, train_nbs, label='train')

plt.bar(num_to_class, test_nbs, label='test', bottom=train_nbs)

plt.legend()

plt.tight_layout()

plt.savefig('TrainTestBarPlot.pdf', bbox_inches='tight', format='pdf', dpi=200)

plt.show()
# One-hot encode

if y_train.ndim == 1:

    y_train = mm.one_hot_encode(list(y_train))



if y_test.ndim == 1:

    y_test = mm.one_hot_encode(list(y_test))
# Assert shapes

assert(y_train.shape[1]==NUM_CLASSES)

assert(y_test.shape[1]==NUM_CLASSES)
# Assert std devs

assert(np.isclose(np.round(x_train[0].std(), 3), 1, atol=0.5))

assert(np.isclose(np.round(x_test[0].std(), 3), 1, atol=0.5))
# Assert means

assert(np.isclose(np.round(x_train[0].mean(), 3), 0, atol=0.5))

assert(np.isclose(np.round(x_test[0].mean(), 3), 0, atol=0.5))
# Assert 4D

assert(len(x_train.shape)==4)

assert(len(x_test.shape)==4)
# Check

print('x_train :', x_train.shape)

print('y_train :', y_train.shape)

print('')

print('x_test  :', x_test.shape)

print('y_test  :', y_test.shape)
try:

  model.sess

except NameError:

    pass

else:

    model.sess.close()

tf.reset_default_graph()



# Setup model

model = MyCNN(x_train, 

              y_train, 

              x_test,

              y_test,

              output_dir='./FER_logdir/',

              num_to_class=num_to_class, 

              class_to_num=class_to_num,

              lr=5e-5,

              nb_epochs=50, 

              batch_size_train=30,

              seed=0,

              final_activation='softmax')



# Initialise model

model.create_model()

model.compute_loss()

model.optimizer()

model.set_up_saver()

tf.initialize_all_variables().run(session=model.sess)



# Make path if necessary

if not os.path.exists(model.output_dir):

    os.makedirs(model.output_dir)
model.model_variables()
model.model_summary()
# Train model w/o k-fold cross validation

model.train(verbose=False, cross_k_fold_validation=False)
# Test model w/o k-fold cross validation

model.test()
# Plot variables over training and validation w/o k-fold cross validation

mm.plot_metrics(model.accuracy_list, 

                model.losses_list, 

                model.val_accuracy_list, 

                model.val_losses_list,

                save_title='wo_kCV_metrics')
plot_all_confusion_matrices(np.argmax(model.y_train, axis=1), model.preds_list,

                            np.argmax(model.y_val, axis=1), model.preds_list_val,

                            np.argmax(model.y_test, axis=1), model.preds_list_test,

                            np.array(num_to_class), 'CM_wo_kCV', normalize=True)
# Get validation metrics report

report = classification_report(np.argmax(model.y_train, axis=1), 

                               model.preds_list, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get validation metrics report

report = classification_report(np.argmax(model.y_val, axis=1), 

                               model.preds_list_val, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get testing metrics report

report = classification_report(np.argmax(model.y_test, axis=1), 

                               model.preds_list_test, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Train ROC/AUC w/o-cross val

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(model.y_train, model.probs_list)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='TrainROC_wo_kCV')
# Train ROC/AUC w/o-cross val

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(model.y_test, model.probs_list_test)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='TestROC_wo_kCV')
model.sess.close()
try:

  sess

except NameError:

    pass

else:

    sess.close()

tf.reset_default_graph()



# Setup model_2

model_2 = MyCNN(x_train[:33150], 

                y_train[:33150], 

                x_test,

                y_test,

                output_dir='./FER_logdir/',

                num_to_class=num_to_class, 

                class_to_num=class_to_num,

                lr=5e-5,

                nb_epochs=50, 

                batch_size_train=30,

                seed=0,

                final_activation='softmax')



# Initialise model_2

model_2.create_model()

model_2.compute_loss()

model_2.optimizer()

model_2.set_up_saver()

tf.initialize_all_variables().run(session=model_2.sess)



# Make path if necessary

if not os.path.exists(model_2.output_dir):

    os.makedirs(model_2.output_dir)
# Train model w/k-CV

model_2.train(verbose=False, cross_k_fold_validation=True)
# Plot variables over training and validation w/k-CV

mm.plot_metrics(model_2.accuracy_list, 

                model_2.losses_list, 

                model_2.val_accuracy_list, 

                model_2.val_losses_list,

                save_title='w_kCV_metrics')
# Test model w/k-CV

model_2.test()
# Plot CMs w/k-CV

plot_all_confusion_matrices(np.argmax(model_2.y_train, axis=1), model_2.preds_list,

                            np.argmax(model_2.y_val, axis=1),   model_2.preds_list_val,

                            np.argmax(model_2.y_test, axis=1),  model_2.preds_list_test,

                            np.array(num_to_class), 'CM_w_kCV', normalize=True)
# Get validation metrics report

report = classification_report(np.argmax(model_2.y_train, axis=1), 

                               model_2.preds_list, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get validation metrics report

report = classification_report(np.argmax(model_2.y_val, axis=1), 

                               model_2.preds_list_val, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get testing metrics report

report = classification_report(np.argmax(model_2.y_test, axis=1), 

                               model_2.preds_list_test, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Train ROC/AUC w/k-cross val

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(model_2.y_train, model_2.probs_list)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='TrainROC_w_kCV')
# Test ROC/AUC w/k-cross val

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(model_2.y_test, model_2.probs_list_test)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 1), ylim=(0, 1), seed=5, save_title='TestROC_w_kCV')
model_2.sess.close()
# (Re-)Split

x_train, y_train, x_val, y_val, x_test, y_test = mm.split_train_val_test(x, y)
# One-hot encode

if y_train.ndim == 1:

    y_train = mm.one_hot_encode(list(y_train))

    

if y_val.ndim == 1:

    y_val = mm.one_hot_encode(list(y_val))



if y_test.ndim == 1:

    y_test = mm.one_hot_encode(list(y_test))
# Assert shapes

assert(y_train.shape[1]==NUM_CLASSES)

assert(y_val.shape[1]==NUM_CLASSES)

assert(y_test.shape[1]==NUM_CLASSES)
try:

  sess

except NameError:

    pass

else:

    sess.close()

tf.reset_default_graph()



# Setup model_3

model_3 = MyDeepCNN(x_train,

                    y_train,

                    x_val,

                    y_val,

                    x_test,

                    y_test,

                    output_dir='./FER_logdir/',

                    lr=1e-3,

                    beta_1=0.9,

                    beta_2=0.999,

                    nb_epochs=50,

                    epsilon=1e-7,

                    batch_size=64,

                    seed=0,

                    num_features=64)



# Initialise model_3

model_3.create_model()



# Make path if necessary

if not os.path.exists(model_3.output_dir):

    os.makedirs(model_3.output_dir)
model_3.model_variables()
model_3.model_summary()
# Train

model_3.train(verbose=False)
# Test

model_3.test()
# Plot variables over training and validation - 5e-6 - 100 epochs

mm.plot_metrics(model_3.history.history['acc'], 

                model_3.history.history['loss'], 

                model_3.history.history['val_acc'], 

                model_3.history.history['val_loss'],

                save_title='deepcnn_metrics')

probs_list_train = model_3.model.predict(x_train, batch_size=64)

probs_list_val = model_3.model.predict(x_val, batch_size=64)

probs_list_test = model_3.model.predict(x_test, batch_size=64)



y_hat_train = np.argmax(probs_list_train, axis=1)

y_hat_val = np.argmax(probs_list_val, axis=1)

y_hat_test = np.argmax(probs_list_test, axis=1)



y_real_train = np.argmax(y_train, axis=1)

y_real_val = np.argmax(y_val, axis=1)

y_real_test = np.argmax(y_test, axis=1)
# Get validation metrics report

report = classification_report(y_real_train, 

                               y_hat_train, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get validation metrics report

report = classification_report(y_real_val, 

                               y_hat_val, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
# Get validation metrics report

report = classification_report(y_real_test, 

                               y_hat_test, 

                               target_names=class_to_num, 

                               output_dict=True)

my_df = pd.DataFrame.from_dict(report).T.round(2)

my_df
print(my_df.to_latex())
plot_all_confusion_matrices(y_real_train, y_hat_train,

                            y_real_val, y_hat_val,

                            y_real_test, y_hat_test,

                            np.array(num_to_class), 'CM_deepcnn', normalize=True)
# Train ROC/AUC

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_train, probs_list_train)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.6), ylim=(0.6, 1), seed=5, save_title='TrainROC_deepcnn')
# Val ROC/AUC

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_val, probs_list_val)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 1), ylim=(0, 1), seed=5, save_title='ValROC_deepcnn')
# Test ROC/AUC

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_test, probs_list_test)

plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 1), ylim=(0, 1), seed=5, save_title='TestROC_deepcnn')
model_3.sess.close()