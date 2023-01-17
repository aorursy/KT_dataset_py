import numpy as np

import pandas as pd

from fastai.text import *

import os
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
os.listdir('../input/contradictory-my-dear-watson/')
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

test  = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
train = train.sample(frac=1, random_state=1).reset_index(drop=True)
val_count = 2000

trn_count = train.shape[0] - val_count



df_val = train[:val_count]

df_trn = train[val_count:val_count+trn_count]
df_trn.head()
os.listdir('../input/awd-lstm-model/')
# !cp ../input/awd-lstm-model/datasets_425796_811126_itos_wt103.pkl ~/.fastai/models/wt103-fwd/itos_wt103.pkl
!mkdir -p ~/.fastai/models/wt103-fwd



!cp ../input/awd-lstm-model/lstm_fwd.pth ~/.fastai/models/wt103-fwd/

!cp ../input/awd-lstm-model/datasets_425796_811126_itos_wt103.pkl ~/.fastai/models/wt103-fwd/itos_wt103.pkl
data_lm = TextLMDataBunch.from_df('.', df_trn, df_val, test,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['premise', 'hypothesis'],

                  label_cols='label',

                  bs=128,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(5, 1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.freeze()

learn.fit_one_cycle(5, 1e-3)
learn.unfreeze()

learn.fit_one_cycle(3, 1e-3, moms = [0.8,0.7])
learn.save_encoder('ft_enc')
data_cls = TextClasDataBunch.from_df('.', df_trn, df_val, test,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['premise', 'hypothesis'],

                  label_cols='label',

                  bs=128,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
data_cls.show_batch()
learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc');
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(3, 1e-5*3, moms = [0.8,0.7])
learn.fit_one_cycle(8, 1e-4, moms = [0.8,0.7])
def get_ordered_preds(learn, ds_type, preds):

  np.random.seed(42)

  sampler = [i for i in learn.data.dl(ds_type).sampler]

  reverse_sampler = np.argsort(sampler)

  preds = [p[reverse_sampler] for p in preds]

  return preds
val_raw_preds = learn.get_preds(ds_type=DatasetType.Valid)

val_preds = get_ordered_preds(learn, DatasetType.Valid, val_raw_preds)
print(val_preds[0].shape)

torch.argmax(val_preds[0], dim=1)
### Accuracy on val

df_val['prediction'] = torch.argmax(val_preds[0], dim=1)
df_val['accuracy'] = np.where(df_val.label == df_val.prediction, 1, 0)

df_val['accuracy'].mean()
test_raw_preds = learn.get_preds(ds_type=DatasetType.Test)

test_preds = get_ordered_preds(learn, DatasetType.Test, test_raw_preds)

test_preds = torch.argmax(test_preds[0], dim=1)
test.shape, len(test_preds)
sample_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')

sample_submission.head()
sample_submission['prediction'] = test_preds

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head();
sample_submission.prediction.value_counts()