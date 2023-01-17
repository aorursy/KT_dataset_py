%reload_ext autoreload

%autoreload 2
# Repo with all the fastai wrappers and helpers



! git clone https://github.com/deepklarity/fastai-bert-finetuning.git fastai_bert_finetuning

! mv fastai_bert_finetuning/* .

! rm -fr fastai_bert_finetuning
!ls
! pip install -r requirements.txt
import csv

import pandas as pd

from pathlib import Path

import matplotlib.cm as cm

from fastai import *

from fastai.text import *

from fastai.callbacks import *

from fastai.metrics import *

import utils, bert_fastai, bert_helper
# Seed random generators so all model runs are reproducible



utils.seed_everything()
# https://www.microsoft.com/en-us/download/details.aspx?id=52398

# Microsoft Research Paraphrase Corpus

TASK='MRPC'

DATA_ROOT = Path(".")

label_col = "Quality"

text_cols = ["#1 String", "#2 String"]

# Execute script to download MRPC data and create train.csv and test.csv



! python download_glue_data.py --data_dir='glue_data' --tasks=$TASK --test_labels=True
train_df = pd.read_csv(DATA_ROOT / "glue_data" / "MRPC" / "train.tsv", sep = '\t', quoting=csv.QUOTE_NONE)

train_df.head()
test_df = pd.read_csv(DATA_ROOT / "glue_data" / "MRPC" / "test.tsv", sep = '\t', quoting=csv.QUOTE_NONE)

test_df.head()
print(f"Number of Training records={len(train_df)}")

print(f"Number of Test records={len(test_df)}")
def sample_sentences(quality, n=5):

    ctr = 0

    for row in train_df.query(f'Quality=={quality}').itertuples():

        print(f"1. {row[4]}\n2. {row[5]}")

        print("="*100)

        ctr += 1

        if n==ctr:

            break

# Different sentences samples            

sample_sentences(0)
# Similar sentences samples            

sample_sentences(1)
# Specify BERT configs



config = utils.Config(

    bert_model_name="bert-base-uncased",

    num_labels=2, # 0 or 1

    max_lr=2e-5,

    epochs=3,

    batch_size=32,

    max_seq_len=128

)
fastai_tokenizer = bert_fastai.FastAITokenizer(model_name=config.bert_model_name, max_seq_len=config.max_seq_len)
databunch = TextDataBunch.from_df(".", train_df=train_df, valid_df=test_df,

                  tokenizer=fastai_tokenizer.bert_tokenizer(),

                  vocab=fastai_tokenizer.fastai_bert_vocab(),

                  include_bos=False,

                  include_eos=False,

                  text_cols=text_cols,

                  label_cols=label_col,

                  bs=config.batch_size,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )

# Show wordpiece tokenized data



for i in range(5): 

    print(f"Original==> {train_df.loc[i][text_cols[0]]},{train_df.loc[i][text_cols[1]]}\n\nTokenized==>. {databunch.x[i]}")

    print("="*100)
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification





bert_model = BertForSequenceClassification.from_pretrained(

    config.bert_model_name, num_labels=config.num_labels)



learner = bert_fastai.BertLearner(databunch,

                                  bert_model,

                                  metrics=[accuracy])



learner.callbacks.append(ShowGraph(learner))

preds, pred_values, true_labels = learner.get_predictions()

learner.print_metrics(preds, pred_values, true_labels)
txt_ci = TextClassificationInterpretation.from_learner(learner)
utils.custom_show_top_losses(txt_ci, test_df, text_cols, 5)
learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
preds, pred_values, true_labels = learner.get_predictions()

learner.print_metrics(preds, pred_values, true_labels)
txt_ci = TextClassificationInterpretation.from_learner(learner)
utils.custom_show_top_losses(txt_ci, test_df, text_cols, 10)