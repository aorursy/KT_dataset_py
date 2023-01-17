!pip -q install googletrans
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import re
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_gui
tqdm.pandas(ncols=75) 

from googletrans import Translator
from dask import bag, diagnostics
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
def remove_space(text):
    return " ".join(text.split())

def remove_punctuation(text):
    return re.sub("[!@#$+%*:()'-]", ' ', text)

def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    return soup.get_text()

def remove_url(text):
    return re.sub(r"http\S+", "", text)

def translate(text):
    translator = Translator()
    return translator.translate(text, dest='en').text

def clean_text(text):
    text = remove_space(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    return text
train['premise'] = train.premise.progress_apply(lambda text : clean_text(text))
train['hypothesis'] = train.hypothesis.apply(lambda text : clean_text(text))
eng = train.loc[train.lang_abv == "en"].copy()
def translate_parallel(df):
    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate)
    hypothesis_bag = bag.from_sequence(df.hypothesis.tolist()).map(translate)
    with diagnostics.ProgressBar():
        premises = premise_bag.compute()
        hypothesis = hypothesis_bag.compute()
    df[['premise', 'hypothesis']] = list(zip(premises, hypothesis))
    return df
non_eng =  train.loc[train.lang_abv != "en"].copy().pipe(translate_parallel)
train_translated = eng.append(non_eng)
train_translated.head()
train.shape
train_translated.shape
train_translated.head()
train_translated.to_csv("train_english_translated.csv", index=False)
test_eng = test.loc[test.lang_abv == "en"].copy()
test_non_eng = test.loc[test.lang_abv != "en"].copy().pipe(translate_parallel)
test_translated = test_eng.append(test_non_eng)
print(test_translated.shape)
print(test.shape)
test_translated.to_csv("test_english_translated.csv", index=False)
test_translated.tail()
