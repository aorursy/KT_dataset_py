from fastai import *

from fastai.text import *



from scipy.stats import spearmanr
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
data_dir = '../input/google-quest-challenge/'

!ls {data_dir}
train_raw = pd.read_csv(f'{data_dir}train.csv')

train_raw.head()
test_raw = pd.read_csv(f'{data_dir}test.csv')

test_raw.head()
train_raw.shape, test_raw.shape
# randomly shuffle our data

train_raw = train_raw.sample(frac=1, random_state=1).reset_index(drop=True)
val_count = 1000

trn_count = train_raw.shape[0] - val_count



df_val = train_raw[:val_count]

df_trn = train_raw[val_count:val_count+trn_count]
!mkdir -p ~/.fastai/models/wt103-fwd
!cp ../input/wt103-fastai-nlp/lstm_fwd.pth ~/.fastai/models/wt103-fwd/

!cp ../input/wt103-fastai-nlp/itos_wt103.pkl ~/.fastai/models/wt103-fwd/
target_cols = train_raw.columns.tolist()[-30:]
data_lm = TextLMDataBunch.from_df('.', df_trn, df_val, test_raw,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['question_body', 'answer'],

                  label_cols=target_cols,

                  bs=32,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(5, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(5, 1e-3)
learn.save_encoder('ft_enc')
data_cls = TextClasDataBunch.from_df('.', df_trn, df_val, test_raw,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['question_body', 'answer'],

                  label_cols=target_cols,

                  bs=32,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
data_cls.show_batch()
learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc');
learn.fit_one_cycle(7, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(3, 1e-3)
learn.freeze_to(-3)

learn.fit_one_cycle(5, 1e-3)
learn.unfreeze()

learn.fit_one_cycle(3, 1e-3)
def get_ordered_preds(learn, ds_type, preds):

  np.random.seed(42)

  sampler = [i for i in learn.data.dl(ds_type).sampler]

  reverse_sampler = np.argsort(sampler)

  preds = [p[reverse_sampler] for p in preds]

  return preds
val_raw_preds = learn.get_preds(ds_type=DatasetType.Valid)

val_preds = get_ordered_preds(learn, DatasetType.Valid, val_raw_preds)
score = 0

for i in range(30):

    score += np.nan_to_num(spearmanr(df_val[target_cols].values[:, i], val_preds[0][:, i]).correlation) / 30

score
test_raw_preds = learn.get_preds(ds_type=DatasetType.Test)

test_preds = get_ordered_preds(learn, DatasetType.Test, test_raw_preds)
sample_submission = pd.read_csv(f'{data_dir}sample_submission.csv')

sample_submission.head()



sample_submission.iloc[:, 1:] = test_preds[0].numpy()

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()