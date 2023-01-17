import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from sklearn import metrics

from pathlib import Path

from matplotlib import pyplot as plt #Ploting charts

from os import listdir



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

print(listdir('../input'))



# Note the directory structure is /kaggle/input/chest_xray/chest_xray/*
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *
base_dir = Path('../input/chest_xray/chest_xray')

print([x for x in (base_dir/'train').iterdir() if x.is_dir()])



# Two ways to get number of images (they are all .jpeg)

def get_summary_info(path):

    num_negatives = len(list(path.glob('NORMAL/*.jpeg')))

    num_positives = len(list(path.glob('PNEUMONIA/*.jpeg')))

    num_total = num_negatives + num_positives

    return([num_positives, num_negatives, num_total, num_positives/num_total])

    



def print_summary_info(path):

    num_1, num_0, num_T, incid = get_summary_info(path)

    return('%d positives, %d negatives, %d total, %0.3f incidence' % (num_1, num_0, num_T, incid))



train_sizes = get_summary_info(base_dir/'train')

print('Train set:', print_summary_info(base_dir/'train'))

print('Validation set:', print_summary_info(base_dir/'val'))

print('Test set:', print_summary_info(base_dir/'test'))
# Get image sizes for training set

im_sizes = [PIL.Image.open(im).size for im in (base_dir/'train').glob('**/*.jpeg')]
df = pd.DataFrame(im_sizes, columns = ['Width', 'Height'])

df['Class'] = np.array(['NORMAL', 'PNEUMONIA']).repeat([train_sizes[1], train_sizes[0]])

df['AspRatio'] = df['Width']/df['Height']

df = df.sample(frac=1).reset_index(drop=True)



for cl, group in df.groupby(['Class']):

    plt.scatter(group['Width'], group['AspRatio'], c= 'green' if cl=='NORMAL' else 'red', alpha=0.05, label=cl)



plt.legend()

plt.xlabel('Width')

plt.ylabel('Height/Width (aspect ratio)')

plt.title('Image size for each class');
## Set levers for data:

data_size = 256

batch_size = 32

tfms = get_transforms(do_flip = True, flip_vert = False, max_zoom = 1.2)

do_use_default_val = False

print('Data options:\nData size:', data_size ,'\nBatch size:', batch_size, '\nUse default validation dir:', do_use_default_val)
## Basic way to read in data. Works b/c it is already formatted similarly to ImageNet

#data = ImageDataBunch.from_folder(base_dir, train = 'train', valid = 'val', seed = 1107, valid_pct = 0.20, size = 224

## Alternate, more expressive, read-in:

## In this case we have resized the data

## Question: What about the fact that our data is in greyscale and ImageNet expects color?

## Because the val dir is really small, we can also sample val set from train dir

random.seed(1107)

if do_use_default_val:

    data = (ImageList.from_folder(base_dir, convert_mode = 'L')

       .split_by_folder(train = 'train', valid = 'val')

       .label_from_folder()

       .transform(tfms, size = data_size)

       .databunch(bs = batch_size)).normalize(imagenet_stats)

else:

    data = (ImageList # What is the data class? 

        # [*Opt: Should we filter this data somehow]

        .from_folder(base_dir/'train', convert_mode = 'L') # Where is it located? convert_mode = 'L' for greyscale (vs 'RGB')

        # [*Opt: Where is the test set?]

       .split_by_rand_pct(0.10) # How should we split into train/val? 

       .label_from_folder() # Where are the labels?

       .transform(tfms, size = data_size) # [Opt: How should data be transformed?]

       .databunch(bs = batch_size).normalize(imagenet_stats)) # Finally, convert to bunch and normalize



print(data)
data.show_batch(rows = 3, figsize = (6,6))

print('Classes:', data.classes, 'Num train:', len(data.train_ds), 'Num valid', len(data.valid_ds))
# Define model architecture

learn = cnn_learner(data, models.resnet34, metrics = error_rate, model_dir = '/kaggle/working')

#print(learn.summary())

# Manually finding starting (max) LR -- uses cyclical LR scheme

lr_find(learn)

learn.recorder.plot(suggestion = True)
#print(learn.summary())
lr_base = 5e-3

print('Using base LR:', lr_base)

# Using CLR: lr_init = max_lr/div_factor, pct_start = % of iterations with increasing LR

learn.fit_one_cycle(5, div_factor = 25.0, pct_start = 0.3,

                    callbacks = callbacks.EarlyStoppingCallback(learn, monitor = 'valid_loss', min_delta = 0.01, patience = 3))
# Some visualizations of the process:

learn.recorder.plot_lr(show_moms=True)

learn.recorder.plot_losses()

learn.recorder.plot_metrics()

learn.show_results(rows=3, figsize=(8,9))
# Save initial results, and unfreeze:

learn.save('stage_1')

learn.unfreeze()
# Need to now find new set of LR:

learn.lr_find()

learn.recorder.plot()
# Re-fit entire model, using geometric (slice(a,b)) LR

lr_stage2 = slice(1e-5, lr_base/10)

learn.fit_one_cycle(4, lr_stage2)

# Visualize

learn.recorder.plot_lr(show_moms=True)

learn.recorder.plot_losses()

#learn.recorder.plot_metrics()

learn.show_results(rows=3, figsize=(8,9))

# Save

learn.save('stage_2')
# Use this second model to evaluate predictions on the test set

data_test = (ImageList.from_folder(base_dir, convert_mode = 'RGB')

       .split_by_folder(train = 'train', valid = 'test')

       .label_from_folder()

       .transform(tfms, size = data_size)

       .databunch(bs = batch_size)).normalize(imagenet_stats)

learner_test = cnn_learner(data_test, models.resnet34, metrics = accuracy, model_dir = '/kaggle/working')

learner_test.load('stage_2')



# Get predictions

y_pred, y_true = learner_test.get_preds()



# Plot confusion matrix

interp = ClassificationInterpretation.from_learner(learner_test)

interp.plot_confusion_matrix()

interp.plot_top_losses(6)
# Plot curves

fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:,1])

pr, rec, _ = metrics.precision_recall_curve(y_true, y_pred[:,1])



plt.subplot(1,2,1)

plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC curve (area = %0.2f)' % metrics.roc_auc_score(y_true, y_pred[:,1]))

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.legend(loc="lower right")



plt.subplot(1,2,2)

plt.plot(rec, pr, color = 'darkgreen', label = 'PR curve (area = %0.2f)' % metrics.auc(rec, pr))

test_incidence = to_np(y_true).mean()

plt.plot([0, 1], [test_incidence, test_incidence], color='navy', linestyle='--')

plt.xlim([0, 1.0])

plt.ylim([test_incidence - 0.10, 1.05])

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('PR curve')

plt.legend(loc="lower right")



# Report additional metrics from sklearn

prediction_threshold = 0.5

y_hat = y_pred[:,1] > prediction_threshold # Pr(y = 1) > thres



print(' Accuracy: %0.3f \n Balanced accuracy: %0.3f \n F1 score: %0.3f \n F0.5 score: %0.3f \n F2 score: %0.3f' % \

      (metrics.accuracy_score(y_true, y_hat), metrics.balanced_accuracy_score(y_true, y_hat),

      metrics.f1_score(y_true, y_hat), metrics.fbeta_score(y_true, y_hat, beta = 0.5),

      metrics.fbeta_score(y_true, y_hat, beta = 2.)))
