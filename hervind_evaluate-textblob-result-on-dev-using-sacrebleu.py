import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
from urllib.error import HTTPError
import textblob
import time
import re
dev_tcn = pd.read_csv('/kaggle/input/shopee-product-title-translation-open/dev_tcn.csv')
dev_en  = pd.read_csv('/kaggle/input/shopee-product-title-translation-open/dev_en.csv' )
dev = pd.concat([dev_en, dev_tcn], axis=1)
dev.head()
translation = []
for k in tqdm(range(len(dev))):

    try:
        one_translation = TextBlob(dev['text'][k]).translate(to="en")
        translation.append(one_translation)
        time.sleep(0.4)
        
    except (textblob.exceptions.NotTranslated, HTTPError) as e:
        print(k, e)
        if isinstance(e, textblob.exceptions.NotTranslated):
            translation.append("")
        else:
            break
            
textblob_translation = [x.string if isinstance(x, textblob.blob.TextBlob) else x for x in translation]
dev['textblob_translation'] = textblob_translation
dev.head()
!pip install sacrebleu
import sacrebleu
import matplotlib.pyplot as plt
refs = [list(dev['translation_output'])]
preds = list(dev['textblob_translation'])
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
refs = [list(dev['translation_output'].str.lower())]
preds = list(dev['textblob_translation'].str.lower())
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
def cleaning_string(my_string):
    my_string = re.sub(r"[^a-z0-9 ]+", ' ', my_string.lower()) # lowercase then change special char to '' 
    my_string = " ".join(my_string.split()) # remove white space

    return my_string
refs = [list(dev['translation_output'].map(cleaning_string))]
preds = list(dev['textblob_translation'].map(cleaning_string))
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
refs = [list(dev['translation_output'])]
preds = list(dev['translation_output'])
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
list_score_one_sample = []
for i in range(len(dev)):
    refs = [[list(dev['translation_output'].str.lower())[i]]]
    preds = list(dev['textblob_translation'].str.lower())[i]
    bleu = sacrebleu.corpus_bleu(preds, refs)
    list_score_one_sample.append(bleu.score)
plt.plot(list_score_one_sample);
