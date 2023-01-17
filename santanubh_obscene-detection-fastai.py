from fastai.text import *

from fastai.callbacks import *

from torch import nn



# To get consistent results

import numpy as np

import pandas as pd

from sklearn.metrics import classification_report

np.random.seed(seed=1)
# import resource



# resource.setrlimit(resource.RLIMIT_NOFILE, (160000, 160000))
train_dataset = pd.read_csv('../input/train.csv')

train_dataset
bs=32

cwd = Path('/kaggle/working')



data_lm = (TextList.from_df(train_dataset, cols='comment_text')

           .split_by_rand_pct(valid_pct=0.15, seed=1)

           .label_for_lm()

           .databunch(bs=bs))

data_lm.save(cwd/'data_lm.pkl')
len(data_lm.vocab.itos)
learn = language_model_learner(data_lm, AWD_LSTM)
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(1, 2.29E-02)
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
learn.freeze_to(-2)

min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(1, 1.00E-05)
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
learn.unfreeze()

min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(2, 4e-04)
data_lm.save(cwd/'data_lm.pkl')
train_dataset = pd.concat([train_dataset.loc[(train_dataset['toxic'] == 0)

                                             & (train_dataset['severe_toxic'] == 0)

                                             & (train_dataset['threat'] == 0)

                                             & (train_dataset['insult'] == 0)

                                             & (train_dataset['identity_hate'] == 0)

                                             & (train_dataset['obscene'] == 0)

                                            ], train_dataset[train_dataset['obscene'] == 1]], ignore_index=True, sort=False)
train_dataset.shape
train_dataset.drop(columns=['id', 'toxic', 'severe_toxic', 'threat', 'insult', 'identity_hate'], inplace=True)

train_dataset = train_dataset.sample(frac=1, random_state=20)

train_dataset.head()
train_dataset[train_dataset['obscene'] == 0].shape, train_dataset[train_dataset['obscene'] == 1].shape
# train_dataset = train_dataset[train_dataset['obscene'] == 1].append(train_dataset[train_dataset['obscene'] == 0].sample(n=20000, random_state=1), ignore_index=True)

# train_dataset = train_dataset.sample(frac=1, random_state=20)

# train_dataset
data_class = (TextList.from_df(train_dataset, cols='comment_text', vocab=data_lm.vocab)

             .random_split_by_pct(valid_pct=0.2, seed=1)

             .label_from_df(cols='obscene')

             .databunch(bs=bs))
learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)



weights = [1, 19]

class_weights=torch.FloatTensor(weights).cuda()

learn.crit = nn.CrossEntropyLoss(weight=class_weights)
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(1, 2.51E-01)
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
learn.freeze_to(-2)

min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(1, 1e-02)
learn.save('last_two_layers_tuned')
learn.lr_find()

learn.recorder.plot(skip_end=10, suggestion=True)
learn.unfreeze()

min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(3, 1.5E-04, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
learn.load('bestmodel_2')

learn.save(cwd/'obscene_classifier')
v_pred, v_y, losses = learn.get_preds(DatasetType.Valid, ordered=True, with_loss=True)
interp = ClassificationInterpretation(learn, v_pred, v_y, losses)

interp.plot_confusion_matrix()
print(classification_report(v_y, v_pred.argmax(dim=1)))
interp = TextClassificationInterpretation.from_learner(learn)

interp.show_intrinsic_attention("this is a fucking bad movie")
learn.predict("this is a fucking bad movie")
t_pred, t_y, losses = learn.get_preds(DatasetType.Fix, ordered=True, with_loss=True)
interp = ClassificationInterpretation(learn, t_pred, t_y, losses)

interp.plot_confusion_matrix()
print(classification_report(t_y, t_pred.argmax(dim=1)))