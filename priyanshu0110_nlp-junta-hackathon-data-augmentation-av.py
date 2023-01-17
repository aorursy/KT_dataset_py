!pip install translators
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tqdm.notebook import tqdm

# current version have logs, which is not very comfortable

import translators as ts

from multiprocessing import Pool

from tqdm import *
API = 'google'



def translator_constructor(api):

    if api == 'google':

        return ts.google

    elif api == 'bing':

        return ts.bing

    elif api == 'baidu':

        return ts.baidu

    elif api == 'sogou':

        return ts.sogou

    elif api == 'youdao':

        return ts.youdao

    elif api == 'tencent':

        return ts.tencent

    elif api == 'alibaba':

        return ts.alibaba

    else:

        raise NotImplementedError(f'{api} translator is not realised!')





def translate(x):

    try:

        return [x[0], translator_constructor(API)(x[1], from_lang, to_lang), x[2]]

    except:

        return [x[0], None, x[2]]





def imap_unordered_bar(func, args, n_processes: int = 48):

    p = Pool(n_processes, maxtasksperchild=100)

    res_list = []

    with tqdm(total=len(args)) as pbar:

        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):

            pbar.update()

            res_list.append(res)

    pbar.close()

    p.close()

    p.join()

    return res_list



CSV_PATH = '../input/analytics-vidhya-nlp-junta-hackathon/train.csv'

from_lang = 'en'

to_lang = 'es'

API = 'google'



df = pd.read_csv(CSV_PATH)

tqdm.pandas('Translation progress')

df[['review_id', 'user_review', 'user_suggestion']] = imap_unordered_bar(translate, df[['review_id', 'user_review', 'user_suggestion']].values)

df[['review_id', 'user_review', 'user_suggestion']].to_csv(f'train-{API}-{to_lang}.csv',index=False)
CSV_PATH = 'train-google-es.csv'

from_lang = 'es'

to_lang = 'en'

API = 'google'



df = pd.read_csv(CSV_PATH)

tqdm.pandas('Translation progress')

df[['review_id', 'user_review', 'user_suggestion']] = imap_unordered_bar(translate, df[['review_id', 'user_review', 'user_suggestion']].values)

df[['review_id', 'user_review', 'user_suggestion']].to_csv(f'train-{API}-{from_lang}-decoded.csv',index=False)
CSV_PATH = '../input/analytics-vidhya-nlp-junta-hackathon/test.csv'

from_lang = 'en'

to_lang = 'es'

API = 'google'



df = pd.read_csv(CSV_PATH)

df['user_suggestion'] =1

tqdm.pandas('Translation progress')

df[['review_id', 'user_review', 'user_suggestion']] = imap_unordered_bar(translate, df[['review_id', 'user_review', 'user_suggestion']].values)

df[['review_id', 'user_review']].to_csv(f'test-{API}-{to_lang}.csv',index=False)
CSV_PATH = 'test-google-en.csv'

from_lang = 'es'

to_lang = 'en'

API = 'google'



df = pd.read_csv(CSV_PATH)

df['user_suggestion'] =1

tqdm.pandas('Translation progress')

df[['review_id', 'user_review', 'user_suggestion']] = imap_unordered_bar(translate2, df[['review_id', 'user_review', 'user_suggestion']].values)

df[['review_id', 'user_review']].to_csv(f'test-{API}-{LANG}-decoded.csv',index=False)