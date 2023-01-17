%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.text import *

from fastai.text import *

!curl -0 http://u.cs.biu.ac.il/~yogo/hebwiki/wikipedia.raw.gz --output wikipedia.raw.gz

! zcat wikipedia.raw.gz >> wikipedia.txt

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

p = open("wikipedia.txt").read().split("\n")

wiki_data = (TextList.from_df(pd.DataFrame(p)) #change txt file to df and then read it via FastAI
            .split_by_rand_pct(0.1, seed=42) #split 
            .label_for_lm(ignore_empty=True)         #only lm - there isnt labeling   
            .databunch(bs=64, num_workers=1)) #make it as data banch like fastAI likes.

wiki_data.save('wiki_data_old.pkl')

from google.colab import drive
drive.mount('/content/drive')
data_path = Config.data_path()
name = 'hewiki'
path = data_path/name

wiki_data = load_data(path, '/content/drive/My Drive/Hebrew emotion recognition/Heb_ulmfit/wiki_heb_data.pkl')
# create the model from the data
learn = language_model_learner(wiki_data, AWD_LSTM, drop_mult=1.0)

lr = 1e-2
bs = 64
lr *= bs/48  # Scale learning rate by batch size


# we unfreeze all the model, beacuse he have totally random weight.
learn.unfreeze()
learn.fit_one_cycle(10, lr, moms=(0.8,0.7))

learn.save('heb_model_save_1')

learn.recorder.plot_losses()

learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))

learn.save('heb_model_save_2')
learn.recorder.plot_losses()

TEXT = "במהלך השנה 1948 קמה מדינת ישראל"
N_WORDS = 40
N_SENTENCES = 1
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.9) for _ in range(N_SENTENCES)))

# wiki_data = load_data(path, 'wiki_data.pkl')
# learn = language_model_learner(wiki_data, AWD_LSTM)
# learn.load("heb_model_save_4")
learn.export("wiki-heb.pkl")
