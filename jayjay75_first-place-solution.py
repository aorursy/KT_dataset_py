import fastai

from fastai.vision import *

import os

import cv2



work_dir = Path('/kaggle/working/')

path = Path('../input/')



train = path/'train_images/train_images'

test =  path/'leaderboard_test_data/leaderboard_test_data'

holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'

sample_sub = path/'SampleSubmission.csv'

labels = path/'traininglabels.csv'



df_sample = pd.read_csv(sample_sub)
df_sample = pd.read_csv(sample_sub)
df = pd.read_csv(labels)

df.head()
test_names = [f for f in test.iterdir()]

holdout_names = [f for f in holdout.iterdir()]

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
!pip install pretrainedmodels
#these are the 2 pretrainedmodels we used , for the purpose of the example we only used  se_resnext101_32x4d

import pretrainedmodels



#def model_f(pretrained=True, **kwargs):

#    return pretrainedmodels.senet154(num_classes=1000,pretrained='imagenet')



def model_f(pretrained=True,**kwargs):

    return pretrainedmodels.se_resnext101_32x4d(num_classes=1000,pretrained='imagenet')
folds = StratifiedKFold(n_splits=11, random_state=2019)



y = df['has_oilpalm'].values



FOLD=5 # just one fold for the kaggle kernel



for fold_, (trn_, val_) in enumerate(folds.split(df, y)):

    if fold_==FOLD:

      train_x, trn_y = df.loc[trn_], y[trn_]

      val_x, val_y   = df.loc[val_], y[val_]

      filename = "resnext101_32x4d_tfms_" + str(fold_)

      filename_sub = filename+"_sub.csv"



      print("fold : ", fold_, train_x.shape,val_x.shape,trn_y.mean(),val_y.mean())



      src = (ImageItemList.from_df(df, path, folder=train)

          .split_by_idxs(train_x.index, val_x.index)

          .label_from_df('has_oilpalm')

          .add_test(test_names+holdout_names))



      data =  (src.transform(get_transforms(), size=256)

             .databunch(bs=16)

             .normalize(imagenet_stats))



      learn = create_cnn(data=data,arch= model_f,cut=-2, 

                   metrics=[accuracy], 

                   model_dir='/kaggle/working/models')

  

      lr = 1e-2

      learn.fit_one_cycle(5, lr)

      lr = 1e-3

      learn.fit_one_cycle(5, lr)



      learn.unfreeze()

      learn.fit_one_cycle(4, slice(1e-5, 1e-4))

      learn.save(filename)

      p,t = learn.TTA()

 

      p = to_np(p); p.shape

      sub = val_x

      #We only recover the probs of having palmoil (column 1)

      sub['preds'] =  p[:,1]

      sub.to_csv("val_"+filename_sub, index=False)



      p,t = learn.TTA(ds_type=DatasetType.Test)

      p = to_np(p); p.shape

      ids = np.array([f.name for f in (test_names+holdout_names)]);ids.shape

      #We only recover the probs of having palmoil (column 1)

      sub = pd.DataFrame(np.stack([ids, p[:,1]], axis=1), columns=df_sample.columns)

      sub.to_csv(filename_sub, index=False)

      

      # lets take a second look at the confusion matrix. See if how much we improved.

      interp = ClassificationInterpretation.from_learner(learn)

      interp.plot_confusion_matrix(title='Confusion matrix')

  
# this come from this kernel https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai

from random import randint



def plot_overview(interp:ClassificationInterpretation, classes=['Negative','Positive']):

    # top losses will return all validation losses and indexes sorted by the largest first

    tl_val,tl_idx = interp.top_losses()

    #classes = interp.data.classes

    fig, ax = plt.subplots(3,6, figsize=(24,12))

    fig.suptitle('Predicted / Actual / Loss / Probability',fontsize=20)

    # Random

    for i in range(6):

        random_index = randint(0,len(tl_idx))

        idx = tl_idx[random_index]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        im = image2np(im.data)

        cl = int(cl)

        ax[0,i].imshow(im)

        ax[0,i].set_xticks([])

        ax[0,i].set_yticks([])

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)

    # Most incorrect or top losses

    for i in range(6):

        idx = tl_idx[i]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        im = image2np(im.data)

        ax[1,i].imshow(im)

        ax[1,i].set_xticks([])

        ax[1,i].set_yticks([])

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)

    # Most correct or least losses

    for i in range(6):

        idx = tl_idx[len(tl_idx) - i - 1]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        im = image2np(im.data)

        ax[2,i].imshow(im)

        ax[2,i].set_xticks([])

        ax[2,i].set_yticks([])

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)
plot_overview(interp, ['Negative','Positive'])
from fastai.callbacks.hooks import *

# hook into forward pass

def hooked_backward(m, oneBatch, cat):

    # we hook into the convolutional part = m[0] of the model

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(oneBatch)

            preds[0,int(cat)].backward()

    return hook_a,hook_g
# We can create a utility function for getting a validation image with an activation map

def getHeatmap(val_index):

    """Returns the validation set image and the activation map"""

    # this gets the model

    m = learn.model.eval()

    tensorImg,cl = data.valid_ds[val_index]

    # create a batch from the one image

    oneBatch,_ = data.one_item(tensorImg)

    oneBatch_im = vision.Image(data.denorm(oneBatch)[0])

    # convert batch tensor image to grayscale image with opencv

    cvIm = cv2.cvtColor(image2np(oneBatch_im.data), cv2.COLOR_RGB2GRAY)

    # attach hooks

    hook_a,hook_g = hooked_backward(m, oneBatch, cl)

    # get convolutional activations and average from channels

    acts = hook_a.stored[0].cpu()

    #avg_acts = acts.mean(0)



    # Grad-CAM

    grad = hook_g.stored[0][0].cpu()

    grad_chan = grad.mean(1).mean(1)

    grad.shape,grad_chan.shape

    mult = (acts*grad_chan[...,None,None]).mean(0)

    return mult, cvIm
# Then, modify our plotting func a bit

def plot_heatmap_overview(interp:ClassificationInterpretation, classes=['Negative','Positive']):

    # top losses will return all validation losses and indexes sorted by the largest first

    tl_val,tl_idx = interp.top_losses()

    #classes = interp.data.classes

    fig, ax = plt.subplots(3,6, figsize=(24,12))

    fig.suptitle('Grad-CAM\nPredicted / Actual / Loss / Probability',fontsize=20)

    # Random

    for i in range(6):

        random_index = randint(0,len(tl_idx))

        idx = tl_idx[random_index]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[0,i].imshow(im)

        ax[0,i].imshow(im, cmap=plt.cm.gray)

        ax[0,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[0,i].set_xticks([])

        ax[0,i].set_yticks([])

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)

    # Most incorrect or top losses

    for i in range(6):

        idx = tl_idx[i]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[1,i].imshow(im)

        ax[1,i].imshow(im, cmap=plt.cm.gray)

        ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[1,i].set_xticks([])

        ax[1,i].set_yticks([])

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)

    # Most correct or least losses

    for i in range(6):

        idx = tl_idx[len(tl_idx) - i - 1]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[2,i].imshow(im)

        ax[2,i].imshow(im, cmap=plt.cm.gray)

        ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[2,i].set_xticks([])

        ax[2,i].set_yticks([])

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)
plot_heatmap_overview(interp, ['Negative','Positive'])