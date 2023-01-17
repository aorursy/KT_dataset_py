import gc

from functools import partial

from pathlib import Path



from fastai.text import *

from fastai.callbacks import *

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



# pd.set_option('display.max_colwidth', 200)

# pd.set_option('display.max_columns', None)

# pd.set_option('display.min_rows', 100)

# pd.set_option('display.max_rows', 100)



home = Path(".")

input_dir = Path("/kaggle/input/google-quest-challenge/")



!mkdir -p ~/.fastai/models/

!cp -R /kaggle/input/fastai-wt103-models/* ~/.fastai/models/
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



# seed_everything(42)
raw_train = pd.read_csv(input_dir/"train.csv")

raw_test = pd.read_csv(input_dir/"test.csv")

subm = pd.read_csv(input_dir/"sample_submission.csv")



q_labels = subm.columns[subm.columns.str.startswith("question_")].to_list()

assert len(q_labels) == 21



a_labels = subm.columns[subm.columns.str.startswith("answer_")].to_list()

assert len(a_labels) == 9
train_df = raw_train.iloc[np.random.permutation(len(raw_train))]

cut = int(0.2 * len(train_df)) + 1

train_df, valid_df = train_df[cut:], train_df[:cut]

train_lm_df = train_df.append(raw_test, ignore_index=True, sort=False)
data_lm = TextLMDataBunch.from_df(home, train_lm_df, valid_df,

                                  text_cols=["question_title", "question_body", "answer"],

                                  mark_fields=True,

                                  bs=128)



q_data_clas = TextClasDataBunch.from_df(home, train_df, valid_df, raw_test,

                                      vocab=data_lm.train_ds.vocab,

                                      text_cols=["question_title", "question_body"],

                                      label_cols=q_labels,

                                      mark_fields=True,

                                      bs=64)



a_data_clas = TextClasDataBunch.from_df(home, train_df, valid_df, raw_test,

                                      vocab=data_lm.train_ds.vocab,

                                      text_cols=["question_title", "question_body", "answer"],

                                      label_cols=a_labels,

                                      bs=64)



data_lm.save('./data_lm_export.pkl')

q_data_clas.save('./q_data_clas.pkl')

a_data_clas.save('./a_data_clas.pkl')
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5,

                               metrics=[accuracy, Perplexity()],

                               callback_fns=[partial(SaveModelCallback, monitor="perplexity", mode="min", name="best_model"),

                                             partial(EarlyStoppingCallback, monitor="perplexity", mode="min", patience=10)])

learn = learn.to_fp16()

lr = 5e-02

moms = (0.8, 0.7)

wd=0.1
learn.fit_one_cycle(5, slice(lr), moms=moms, wd=wd)
learn.unfreeze()

learn.fit_one_cycle(10, slice(lr/2), moms=moms, wd=wd)
learn.save_encoder('ft_enc')

learn.save('lm_model')
del learn

del data_lm

gc.collect()
# q_data_clas.show_batch()
# a_data_clas.show_batch()
from scipy.stats import spearmanr



class AvgSpearman(Callback):

    def on_epoch_begin(self, **kwargs):

        self.preds = None

        self.target = None

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        if self.preds is None or self.target is None:

            self.preds = last_output

            self.target = last_target

        else:

            self.preds = np.append(self.preds, last_output, axis=0)

            self.target = np.append(self.target, last_target, axis=0)

    

    def on_epoch_end(self, last_metrics, **kwargs):

        spearsum = 0

        for col in range(self.preds.shape[1]):

            spearsum += spearmanr(self.preds[:,col], self.target[:,col]).correlation

        res = spearsum / (self.preds.shape[1] + 1)

        return add_metrics(last_metrics, res)
q_learn = text_classifier_learner(q_data_clas, AWD_LSTM,

                                metrics=[AvgSpearman()],

                                callback_fns=[partial(EarlyStoppingCallback, monitor='avg_spearman', mode="max", min_delta=0.01, patience=7),

                                              partial(SaveModelCallback, monitor="avg_spearman", mode="max", name="best_model"),]).to_fp16()

q_learn.load_encoder("ft_enc");
lr = 5e-02

moms = (0.8, 0.7)

wd=0.1
def fit(learn, name):

    learn.fit_one_cycle(4, lr, moms=moms, wd=wd)

    learn.freeze_to(-2)

    learn.fit_one_cycle(2, slice(lr/2/(2.6**4),lr), moms=moms, wd=wd)

    learn.freeze_to(-3)

    learn.fit_one_cycle(2, slice(lr/4/(2.6**4),lr/2), moms=moms, wd=wd)

    learn.unfreeze()

    learn.save(f'{name}-stage3-clas')

    learn.fit_one_cycle(20, slice(lr/20/(2.6**4),lr), moms=moms, wd=wd)
fit(q_learn, "q")
q_test_preds, _ = q_learn.get_preds(DatasetType.Test, ordered=True)
del q_learn

del q_data_clas

gc.collect()
a_learn = text_classifier_learner(a_data_clas, AWD_LSTM,

                                  metrics=[AvgSpearman()],

                                  callback_fns=[partial(EarlyStoppingCallback, monitor='avg_spearman', mode="max", min_delta=0.01, patience=5),

                                                partial(SaveModelCallback, monitor="avg_spearman", mode="max", name="best_model"),]).to_fp16()

a_learn.load_encoder("ft_enc");
fit(a_learn, "a")
a_test_preds, _ = a_learn.get_preds(DatasetType.Test, ordered=True)
sample_submission = pd.DataFrame(columns=["qa_id"]+q_labels+a_labels)

sample_submission.loc[:, "qa_id"] = raw_test["qa_id"]

sample_submission.loc[:, q_labels] = q_test_preds

sample_submission.loc[:, a_labels] = a_test_preds

# sample_submission.loc[:, 1:] = np.clip(sample_submission.loc[:, 1:], 0.00001, 0.999999)



sample_submission.to_csv("submission.csv", index=False)
sample_submission.tail()