!pip install git+https://github.com/ssut/py-googletrans.git
import numpy as np
import pandas as pd
from googletrans import Translator
from dask import bag, diagnostics
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv', index_col=['id'])
display(train, train.lang_abv.value_counts())
def translate(words, dest):
    dest_choices = ['zh-cn',
                    'ar',
                    'fr',
                    'sw',
                    'ur',
                    'vi',
                    'ru',
                    'hi',
                    'el',
                    'th',
                    'es',
                    'de',
                    'tr',
                    'bg'
                    ]
    if not dest:
        dest = np.random.choice(dest_choices)
        
    translator = Translator()
    decoded = translator.translate(words, dest=dest).text
    return decoded


#TODO: use a dask dataframe instead of all this
def trans_parallel(df, dest):
    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate, dest)
    hypo_bag =  bag.from_sequence(df.hypothesis.tolist()).map(translate, dest)
    with diagnostics.ProgressBar():
        premises = premise_bag.compute()
        hypos = hypo_bag.compute()
    df[['premise', 'hypothesis']] = list(zip(premises, hypos))
    return df

    
eng = train.loc[train.lang_abv == "en"].copy() \
           .pipe(trans_parallel, dest=None)

non_eng =  train.loc[train.lang_abv != "en"].copy() \
                .pipe(trans_parallel, dest='en')

train = train.append([eng, non_eng])

train.shape
train.to_csv('train_translate_all.csv', index=False)
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv', index_col=['id'])
display(test, test.lang_abv.value_counts())
eng = test.loc[test.lang_abv == "en"].copy() \
           .pipe(trans_parallel, dest=None)

non_eng =  test.loc[test.lang_abv != "en"].copy() \
                .pipe(trans_parallel, dest='en')

test = test.append([eng, non_eng])
test.to_csv('test_translate_all.csv', index=False)
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv', index_col=['id'])
eng = train.pipe(trans_parallel, dest='en')
eng.shape
eng.to_csv('train_translate_en.csv', index=False)
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv', index_col=['id'])
eng = test.pipe(trans_parallel, dest='en')
eng.to_csv('test_translate_en.csv', index=False)