%%capture
# Install the latest Tensorflow version.
!pip3 install --upgrade tensorflow-gpu
# Install TF-Hub.
!pip3 install tensorflow-hub
!pip3 install seaborn
!pip3 install scipy
#@title Load the Universal Sentence Encoder's TF Hub module
from absl import logging
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import csv
import seaborn as sns
from scipy import spatial

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)


def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  cosSim = 1 - spatial.distance.cosine(message_embeddings_[0],message_embeddings_[1])
  return cosSim
  #print(cosSim)
  #plot_similarity(messages_, message_embeddings_, 90)


cossList = []
with open('../input/ComparisonDataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]}=> {row[1]} =>{row[2]}.')
            messages = [row[0],row[1]]
            coss = run_and_plot(messages)
            cossList.append(coss)
            print(str(line_count)+" : "+str(coss))
            line_count += 1
    print(f'Processed {line_count} lines.')
with open('/kaggle/working/use.txt', 'w', newline='') as fi:
   
    for i in cossList:
        if i>0.5:
            m = str(i)+','+'Yes'
            fi.write(m)
            fi.write("\n")
        else:
            m = str(i)+','+'No'
            fi.write(m)
            fi.write("\n")
    

#run_and_plot(messages)
               