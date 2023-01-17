!pip uninstall -y fastai
!pip install git+https://github.com/fastai/fastai.git
# show install info

#import fastai.utils.collect_env

#fastai.utils.collect_env.show_install()
from fastai.text import *
path = untar_data(URLs.IMDB)

path.ls()
bs=64

data_lm = (TextList.from_folder(path)

           #Inputs: all the text files in path

            .filter_by_folder(include=['train', 'test', 'unsup']) 

           #We may have other temp folders that contain text files so we only keep what's in train and test

            .random_split_by_pct(0.2)

           #We randomly split and keep 20% (20,000 reviews) for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=bs))

data_lm.save('tmp_lm')
data_lm = TextLMDataBunch.load(path, 'tmp_lm', bs=bs)

data_lm.show_batch()
# drop_mult is a parameter that controls the % of drop-out used

learn = language_model_learner(data_lm, AWD_LSTM, pretrained=True, drop_mult=0.50) # was 0.3
learn.lr_find()

learn.recorder.plot(skip_end=12)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('imdb_head')

learn.load('imdb_head');
learn.unfreeze()
learn.fit_one_cycle(5, 1e-3, moms=(0.8,0.7))
learn.save('imdb_fine_tuned')

learn.load('imdb_fine_tuned');
TEXTS = ["xxbos the", "xxbos you", "xxbos well", "the","this","when","i really", "you can","if", "i was", "what"]

N_WORDS = 100 
print("\n".join(str(i+1) + ". " + learn.predict(TEXTS[i], N_WORDS,no_unk=True, temperature=0.85) for i in range(len(TEXTS))))
print("\n".join(str(i+1) + ". " + (learn.beam_search(TEXTS[i], N_WORDS, temperature=0.85, top_k=10,beam_sz=100)).replace('Xxunk','').replace('xxunk','') for i in range(len(TEXTS))))