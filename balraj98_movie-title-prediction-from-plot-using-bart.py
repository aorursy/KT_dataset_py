import numpy as np
import pandas as pd
import os, json, gc, re, random
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
%%time

!pip uninstall -q torch -y > /dev/null
!pip install -q torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null
!pip install -q -U transformers > /dev/null
!pip install -q -U simpletransformers > /dev/null
import torch, transformers, tokenizers
torch.__version__, transformers.__version__, tokenizers.__version__
movies_df = pd.read_csv("../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv")
movies_df.head()
movies_df = movies_df[(movies_df["Origin/Ethnicity"]=="American") | (movies_df["Origin/Ethnicity"]=="British")]
movies_df = movies_df[["Plot", "Title"]]
movies_df.columns = ['input_text', 'target_text']
movies_df
%%time

from simpletransformers.seq2seq import Seq2SeqModel

eval_df = movies_df.sample(frac=0.1, random_state=42)
train_df = movies_df.drop(eval_df.index)

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 8,
    "num_train_epochs": 2,
}

# Create a Bart-base model
model = Seq2SeqModel(encoder_decoder_type="bart",
                    encoder_decoder_name="facebook/bart-base",
                    args=model_args)
train_df
%%time

# Train the model
model.train_model(train_df)

# Evaluate the model
result = model.eval_model(eval_df)
print(result)
test_df = eval_df.sample(n=200)

for idx, row in test_df.iterrows():

    plot = row['input_text']
    true_title = row['target_text']

    # Predict with trained BART model
    predicted_title = model.predict([plot])[0]

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Plot: {plot}\n\n\n')