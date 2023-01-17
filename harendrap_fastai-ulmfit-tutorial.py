from fastai import *

from fastai.text import *

from fastai.callbacks import *

from pathlib import Path

import pandas as pd

import numpy as np

import re



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score



import warnings

warnings.filterwarnings('ignore')



import torch

print("Cuda available" if torch.cuda.is_available() is True else "CPU")

print("PyTorch version: ", torch.__version__)
train_df = pd.read_csv('../input/train.csv', nrows=10000)

test_df = pd.read_csv('../input/test.csv', nrows=2000)
train_df.head()
test_df.head()
len(train_df), len(test_df)
train_df['label'] = (train_df['target'] >= 0.4)

train_df['label'].value_counts()
train_df[['target','comment_text','label']].sample(10)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'])
len(train_df), len(val_df)
data_lm = TextLMDataBunch.from_df(

    path='',

    train_df=train_df,

    valid_df=val_df,

    test_df=test_df,

    text_cols=['comment_text'],

    label_cols=['label'],

    #label_cols=['target_better'],

    #classes=['target_better'],

    min_freq=3

)
data_lm.show_batch()
x,y = next(iter(data_lm.train_dl))

example = x[:15,:15].cpu()

texts = pd.DataFrame([data_lm.train_ds.vocab.textify(l).split(' ') for l in example])

texts
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.6)
learn.lr_find(start_lr=1e-6, end_lr=1e2)

learn.recorder.plot()
learn.fit_one_cycle(cyc_len=3, max_lr=1e-01)
learn.unfreeze()

learn.fit_one_cycle(cyc_len=5, max_lr=1e-3, moms=(0.8, 0.7))
learn.save_encoder('ft_enc')
learn.predict("Thank",n_words=5)
data_class = TextClasDataBunch.from_df(

    path='',

    train_df=train_df,

    valid_df=val_df,

    test_df=test_df,

    text_cols=['comment_text'],

    label_cols=['label'],  

    min_freq=3,

    vocab=data_lm.train_ds.vocab,

    #label_delim=' '

)
iter_dl = iter(data_class.train_dl)

_ = next(iter_dl)

x,y = next(iter_dl)

x[-10:,:10]
learn = text_classifier_learner(data_class, arch=AWD_LSTM, drop_mult=0.6)

learn.load_encoder('ft_enc')

learn.freeze()
learn.lr_find(start_lr=1e-8, end_lr=1e2)

learn.recorder.plot()
learn.fit_one_cycle(cyc_len=2, max_lr=1e-2)

#learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(3, slice(1e-4,1e-2))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)


learn.fit_one_cycle(10, slice(1e-5,1e-3),callbacks=[SaveModelCallback(learn, name="best_lm")])
learn.recorder.plot_losses()
preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)


p = preds[0][:,1]

test_df['prediction'] = p

test_df.sort_values('prediction', inplace=True)

test_df.reset_index(drop=True, inplace=True)

ii = 100

print(test_df['comment_text'][ii])

print(test_df['prediction'][ii])

learn.show_results()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
preds = learn.get_preds(ds_type=DatasetType.Valid, ordered=True)

p = preds[0][:,1]
preds,y, loss = learn.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
from sklearn.metrics import roc_curve, auc

# probs from log preds

probs = np.exp(preds[:,1])

# Compute ROC curve

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)



# Compute ROC area

roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))
plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
interp2 = TextClassificationInterpretation.from_learner(learn) 

interp2.show_top_losses(10)
print(interp2.show_intrinsic_attention("Thank you, for your comments"))