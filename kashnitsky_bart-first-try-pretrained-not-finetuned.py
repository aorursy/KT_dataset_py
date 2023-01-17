!pip install -U rouge transformers > /dev/null
from tqdm.notebook import tqdm

tqdm.pandas()

from IPython.display import display, Markdown

from pathlib import Path



import numpy as np

import pandas as pd 



import torch

import transformers

from transformers import BartTokenizer, BartForConditionalGeneration



from nltk.tokenize import sent_tokenize
PATH_TO_CRYPTO_NEWS = Path('../input/news-about-major-cryptocurrencies-20132018-40k/')
train_df = pd.read_csv(PATH_TO_CRYPTO_NEWS / 'crypto_news_parsed_2013-2017_train.csv')

valid_df = pd.read_csv(PATH_TO_CRYPTO_NEWS / 'crypto_news_parsed_2018_validation.csv')



# readling empty strings is a bit different locally and here, but not a big deal 

train_df['text'].fillna(' ', inplace=True)
train_df.shape, valid_df.shape
def minimal_processing(s):

    return s.strip().replace('\r', '').replace('\n', ' ')
def extract_and_process_first_k_sent(text, k=3):

    

    sent_tok = sent_tokenize(text)

    

    if not sent_tok:

        return ' '

    

    result = " ".join([minimal_processing(sent.strip(' .').lower()) 

                                 for sent in sent_tok[:k]])

    

    return result
# train_texts = train_df['text'].progress_apply(lambda text: 

#                                               extract_and_process_first_k_sent(text))



valid_texts = valid_df['text'].progress_apply(lambda text: 

                                              extract_and_process_first_k_sent(text, k=10))
torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

model = BartForConditionalGeneration.from_pretrained('bart-large-cnn').to(torch_device)
example_text = train_df.loc[0, 'text']

example_title = train_df.loc[0, 'title']
display(Markdown('> **Title:** ' + example_title))

display(Markdown('> **Text:** ' + example_text))
article_input_ids = tokenizer.batch_encode_plus([example_text], 

                                                return_tensors='pt', 

                                                max_length=128)['input_ids'].to(torch_device)

summary_ids = model.generate(article_input_ids,

                             num_beams=4,

                             length_penalty=2.0,

                             max_length=20,

                             min_length=5,

                             no_repeat_ngram_size=3)



summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

display(Markdown('> **Summary:** ' + summary_txt))
bs = 32



val_summaries = []



for i in tqdm(range(0, len(valid_texts), bs)):



    article_input_ids = tokenizer.batch_encode_plus(valid_texts.iloc[i:i+bs].tolist(), 

                                                    return_tensors='pt', pad_to_max_length=True,

                                                    max_length=512)['input_ids'].to(torch_device)

    

    summary_ids = model.generate(article_input_ids,

                             num_beams=4,

                             length_penalty=2.0,

                             max_length=40,

                             min_length=5,

                             no_repeat_ngram_size=3)

    

    val_summaries.extend([tokenizer.decode(summary_ids[i].squeeze(), skip_special_tokens=True).lower()

            for i in range(len(summary_ids))])
val_summaries[:10]
valid_titles = valid_df['title'].str.lower().tolist()
from rouge import Rouge 



rouge = Rouge()

scores = rouge.get_scores(hyps=[el.split('.')[0] for el in val_summaries], refs=valid_titles, 

                          avg=True, ignore_empty=True)

scores
final_metric = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3

final_metric
val_res_df = pd.DataFrame({'title': valid_titles, 

                           'generated': val_summaries,

                          'text': valid_texts.values}).reset_index(drop=True)
val_rouge_scores = rouge.get_scores(hyps=val_summaries, refs=valid_titles, avg=False, ignore_empty=True)
val_res_df['rouge-1'] = [el['rouge-1']['f'] for el in val_rouge_scores]

val_res_df['rouge-2'] = [el['rouge-2']['f'] for el in val_rouge_scores]

val_res_df['rouge-L'] = [el['rouge-l']['f'] for el in val_rouge_scores]

val_res_df['avg_rouge'] = (val_res_df['rouge-1'] + val_res_df['rouge-2'] + val_res_df['rouge-L']) / 3
val_res_df.head()
def print_result(row):

    print('_' * 68)

    display(Markdown('> **Rouge:** ' + str(round(row['avg_rouge'], 3))))

    display(Markdown('> **Title:** ' + row['title']))

    display(Markdown('> **Text:** ' + row['text']))

    display(Markdown('> **Generated:** ' + row['generated']))

    print('_' * 68)
for _, row in val_res_df.sort_values(by='avg_rouge', ascending=False).head(30).iterrows():

    print_result(row)
for _, row in val_res_df.sort_values(by='avg_rouge', ascending=True).head(30).iterrows():

    print_result(row)
val_res_df.to_csv('val_set_with_bart_generated_titles.csv', index=None)