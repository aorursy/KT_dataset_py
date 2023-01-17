# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw = pd.read_csv('../input/news-summary/news_summary_more.csv', encoding='iso-8859-1')
text = raw.iloc[0,1]

text
import torch

import json 

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
model = T5ForConditionalGeneration.from_pretrained('t5-small')

tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device('cpu')
preprocess_text = text.strip().replace("\n","")

t5_prepared_Text = "summarize: "+preprocess_text

print ("original text preprocessed: \n", preprocess_text)
tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
summary_ids = model.generate(tokenized_text,

                                    num_beams=14,

                                    no_repeat_ngram_size=2,

                                    min_length=30,

                                    max_length=100,

                                    early_stopping=True)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print ("Summarized text: \n",output)
summary = pd.read_csv('../input/news-summary/news_summary.csv', encoding='iso-8859-1')

summary.head()
text = summary.iloc[0,5]

text
preprocess_text = text

t5_prepared_Text = "summarize: "+preprocess_text

print ("original text preprocessed: \n", preprocess_text)
tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
summary_ids = model.generate(tokenized_text,

                                    num_beams=10,

                                    no_repeat_ngram_size=3,

                                    min_length=100,

                                    max_length=250,

                                    early_stopping=True)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
output