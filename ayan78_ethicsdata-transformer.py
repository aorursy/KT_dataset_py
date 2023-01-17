!pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html

!pip install -U sentence-transformers
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from torch.utils.data import DataLoader

import math

from sentence_transformers import SentenceTransformer,  datasets, LoggingHandler, losses

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers.readers import STSDataReader

import logging

from datetime import datetime





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



logging.basicConfig(format='%(asctime)s - %(message)s',

                    datefmt='%Y-%m-%d %H:%M:%S',

                    level=logging.INFO,

                    handlers=[LoggingHandler()])
train_batch_size = 16

num_epochs = 4

model_save_path = '/kaggle/working/bert-base-ethics_training-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sts_reader = STSDataReader('/kaggle/input/stsbenchmark/')
model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
train_data = datasets.SentencesDataset(sts_reader.get_examples('COVID_Ethics_train.csv'), model)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)

train_loss = losses.CosineSimilarityLoss(model=model)
logging.info("Read dev dataset")

dev_data = datasets.SentencesDataset(examples=sts_reader.get_examples('COVID_Ethics_dev.csv'), model=model)

dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)

evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up

logging.info("Warmup-steps: {}".format(warmup_steps))
model.fit(train_objectives=[(train_dataloader, train_loss)],

          evaluator=evaluator,

          epochs=num_epochs,

          evaluation_steps=500,

          warmup_steps=warmup_steps,

          output_path=model_save_path)