from fastai.text import *

import html

import json

from sklearn.model_selection import train_test_split
BOS = 'xbos'  # beginning-of-sentence tag

FLD = 'xfld'  # data field tag



PATH=Path('/kaggle/input/bnwiki/lolol/lolol')
LM_PATH=Path('/temp')

LM_PATH.mkdir(exist_ok=True)
LANG_FILENAMES = [str(f) for f in PATH.rglob("*/*")]

print(len(LANG_FILENAMES))

LANG_FILENAMES[0:5]
LANG_TEXT = []

for i in LANG_FILENAMES:

    for line in open(i):

        LANG_TEXT.append(json.loads(line))

        

LANG_TEXT = pd.DataFrame(LANG_TEXT)
LANG_TEXT.to_csv(f"{LM_PATH}/wiki_bangla_corpus.csv", index=False)
LANG_TEXT = pd.read_csv(f"{LM_PATH}/wiki_bangla_corpus.csv")
data_lm = TextLMDataBunch.from_csv(LM_PATH,'wiki_bangla_corpus.csv')
learner=language_model_learner(data_lm,TransformerXL,pretrained=False,metrics=accuracy)
learner.load('/kaggle/input/lmtest/gen2')
learner.save_encoder('/kaggle/working/fine_tuned_enc')
import os

print(os.listdir("../input"))

with open('../input/40k-bangla-newspaper-article/40k_bangla_newspaper_article.p', 'rb') as f:

    data = pickle.load(f)

pik=pd.read_pickle('/kaggle/input/40k-bangla-newspaper-article/40k_bangla_newspaper_article.p')

pik=pd.DataFrame(pik)
pik.columns=['category','text','title']

pik=pik.dropna()

pik.to_csv('/kaggle/working/art.csv',index=False)
pik.isna().any()
pak=pd.read_csv('/kaggle/working/art.csv')

pak=pak.dropna()
from sklearn.model_selection import train_test_split

trainpak, validpak = train_test_split(pak, test_size=0.2)
data_lm = TextLMDataBunch.from_df('/tmp/',trainpak,validpak)
learner=language_model_learner(data_lm,TransformerXL,pretrained=False,metrics=accuracy)
learner.load_encoder('/kaggle/working/fine_tuned_enc')
learner.fit_one_cycle(15,0.1)
learner.recorder.plot_metrics()