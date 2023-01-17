import subprocess



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Install bert-as-service

!pip install bert-serving-server

!pip install bert-serving-client
# Download and unzip the pre-trained model

!wget http://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!unzip uncased_L-12_H-768_A-12.zip
# Start the BERT server

bert_command = 'bert-serving-start -model_dir /kaggle/working/uncased_L-12_H-768_A-12'

process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
# Start the BERT client

from bert_serving.client import BertClient

bc = BertClient()
# Compute embeddings for some test sentences

embeddings = bc.encode(['Embed a single sentence', 

                        'Can it handle periods? and then more text?', 

                        'how about periods.  and <p> html stuffs? <p>'])
embeddings.shape
embeddings