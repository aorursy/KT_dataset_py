!pip uninstall -y fastai
!pip install git+https://github.com/fastai/fastai.git
# show install info

#import fastai.utils.collect_env

#fastai.utils.collect_env.show_install()
!wget https://raw.githubusercontent.com/DaveSmith227/deep-elon-tweet-generator/master/musk-tweets-2010-to-2018.csv
from fastai.text import *
dfm = pd.read_csv('musk-tweets-2010-to-2018.csv')

dfm.head()
dfm['text'].apply(lambda x: len(x.split(' '))).mean() # average words per tweet
path = Path('.')

path.ls()
#len(dfm) # 6094 

bs = 64 

# create langugage model using our twitter data

data_lm = (TextList.from_df(dfm,path, cols=['text']) 

          #We randomly split and keep 10% (~609 tweets) for validation

           .random_split_by_pct(0.1)

          #We want to do a language model so we label accordingly

           .label_for_lm()

          .databunch(bs=bs))

data_lm.save('tmp_lm')
data_lm = TextLMDataBunch.load(path, 'tmp_lm', bs=bs)
# drop_mult is a parameter that controls the % of drop-out used

learn = language_model_learner(data_lm, AWD_LSTM, pretrained=True, drop_mult=0.30)
learn.lr_find()

learn.recorder.plot(skip_end=12)
learn.fit_one_cycle(8, 5e-2, moms=(0.8,0.7))
learn.save('tweet_head')

learn.load('tweet_head');
learn.unfreeze()
learn.fit_one_cycle(3, 1e-3, moms=(0.8,0.7))
learn.save('tweet_fine_tuned')

learn.load('tweet_fine_tuned');
TEXTS = ["xxbos","robots","tesla","falcon","i", "why","if", "that", "please"]

N_WORDS = 20 # average tweet length = 15 'words'
print("\n".join(str(i+1) + ". " + learn.predict(TEXTS[i], N_WORDS,no_unk=True, temperature=0.85) for i in range(len(TEXTS))))
print("\n".join(str(i+1) + ". " + (learn.beam_search(TEXTS[i], N_WORDS, temperature=0.85, top_k=6,beam_sz=20)).replace('Xxunk','').replace('xxunk','') for i in range(len(TEXTS))))