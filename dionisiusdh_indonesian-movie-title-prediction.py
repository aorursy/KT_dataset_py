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
!pip uninstall -q torch -y > /dev/null
!pip install -q torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null
!pip install -q -U transformers > /dev/null
!pip install -q -U simpletransformers > /dev/null
import torch, transformers, tokenizers
torch.__version__, transformers.__version__, tokenizers.__version__
df = pd.read_csv('../input/imdb-synopsis-indonesian-movies/imdb_indonesian_movies_2.csv')
df.head()
df = df[["ringkasan_sinopsis", "judul_film"]]
df.columns = ['synopsis', 'title']
df
from simpletransformers import Seq2SeqModel

val = df.sample(frac=0.1)
train = df.drop(val.index)

args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 8,
    "num_train_epochs": 10,
}


model = Seq2SeqModel(encoder_decoder_type="bart",
                    encoder_decoder_name="facebook/bart-base",
                    args=args)
model.train_model(train)


result = model.eval_model(val)
print(result)
test_df = eval_df.sample(n=100)

for idx, row in test_df.iterrows():

    synopsis = row['synopsis']
    true_title = row['title']

    # Predict with trained BART model
    predicted_title = model.predict([plot])[0]

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Synopsis: {synopsis}\n\n\n')