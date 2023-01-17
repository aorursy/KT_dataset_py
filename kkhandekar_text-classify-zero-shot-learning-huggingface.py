!pip install -q git+https://github.com/huggingface/transformers.git
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



#import transformers

from transformers import pipeline
url = '../input/imdb-dataset-csv-file-with-reviews/IMDB Dataset.csv'

df = pd.read_csv(url,header='infer')

print("Total Records: ",df.shape[0])
classifier = pipeline("zero-shot-classification")
# Example



seq = "The GDP of many countries have been affected by this pandemic."

candidate_labels = ["politics", "public health", "economics"]



classifier(seq, candidate_labels)
seq = "The GDP of many countries have been affected by this pandemic."

candidate_labels = ["politics", "public health", "economics"]



classifier(seq, candidate_labels, multi_class=True)
sequences = [

    "Tenet is simply an incredible film with deep complex concepts to unravel well after the credits roll.",

    "The Social Dilemma is densely packed yet lively and entertaining documentary"

]

candidate_labels = ["positive", "negative"]



classifier(sequences, candidate_labels)
sequences = [

    "Tenet is simply an incredible film with deep complex concepts to unravel well after the credits roll.",

    "The Social Dilemma is densely packed yet lively and entertaining documentary"

]

candidate_labels = ["positive", "negative"]

hypothesis_template = "The sentiment of this review is {}."



classifier(sequences, candidate_labels,hypothesis_template=hypothesis_template)

classifier = pipeline("zero-shot-classification", model='xlm-roberta-large')
sequence = "El dilema social es un documental densamente lleno pero animado y entretenido" 

candidate_labels = ["positiva", "negativa"]



hypothesis_template = 'El sentimiento de esta revisión es {}.'



classifier(sequence, candidate_labels,hypothesis_template=hypothesis_template)