# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
news = pd.read_csv("/kaggle/input/news-articles/Articles.csv", encoding="latin")

news['article_len'] = news.Article.str.len()

news["heading_len"] = news.Heading.str.len()



news = news.loc[news['article_len'] > 5000,["Article","Heading","article_len","heading_len"]].reset_index(drop=True)

print(news.shape[0])

news.head(2)
!pip install --upgrade transformers
#import transformers

from transformers import BartTokenizer, BartForConditionalGeneration

import torch

import time



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch_device)
model = BartForConditionalGeneration.from_pretrained("bart-large-cnn", output_past=True,).to(torch_device)

tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")

# data = list(news.Article[news['article_len'] > 5000])

# len(data)



# for i in news.Article:

#     print(i)
outputs = []

time_taken = []



for article in news.Article:



    st = time.time()

    article_input_ids = tokenizer.batch_encode_plus([article], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)

    summary_ids = model.generate(article_input_ids,

                                 num_beams=3,

                                 length_penalty=4.0,

                                 max_length=250,

                                 min_length=150,

                                 no_repeat_ngram_size=3)

    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    et = time.time() - st

    outputs.append( summary_txt )

    time_taken.append( et )

news["gen_summary"] = outputs

news["time_taken"] = time_taken

news
news.time_taken.mean()