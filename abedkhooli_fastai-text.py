# adapted from https://docs.fast.ai/text.html
# data at http://files.fast.ai/data/examples/imdb_sample.tgz
# slow
#!pip3 install git+https://github.com/fastai/fastai.git
#!pip3 install git+https://github.com/pytorch/pytorch
# install for cpu (set gpu off, make sure internet is connected in right pane)
!pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
!pip install -U fastai
# verify installation 
#import fastai
#fastai.show_install(1) # nice format
# download dataset 
!wget http://files.fast.ai/data/examples/imdb_sample.tgz
# unzip
!tar -xvzf imdb_sample.tgz 
# texts.csv is the file we will work with
!ls -la imdb_sample
# import needed libraries
from fastai import *
from fastai.text import * 
# define path
path = Path('imdb_sample')
path
# read and inspect data, label is target, isvalid flags validation set 
# both train and validation sets in same file
df = pd.read_csv(path/'texts.csv')
df.head()
# Language model data
data_lm = TextLMDataBunch.from_csv(path, 'texts.csv', bs=32)
# Classifier model data
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
# inspect data model
data_lm.show_batch(2)
# save work
data_lm.save()
data_clas.save()
# load saved
data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=32)
# create language model, downloads WT103 pretrained model
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2) # takes 6 - 7 minutes on cpu
# improve a bit more
learn.unfreeze()
learn.fit_one_cycle(2, 1e-3) # improve accuracy a abit more, takes around 20 minutes on cpu
# test language model prediction (sentence completion)
learn.predict("The movie was rather", n_words=10)
learn.save_encoder('ft_encoder') # save fine-tuned encoder
# create text classifier learner and train it
learnc = text_classifier_learner(data_clas, drop_mult=0.5)
learnc.load_encoder('ft_encoder')
learnc.fit_one_cycle(1, 1e-2) # around 6 minutes
learnc.freeze_to(-2) # tune last 2 layers of pretrained model
learnc.fit_one_cycle(3, slice(5e-3/2., 5e-3)) 
learnc.unfreeze() # improve a little more
learnc.fit_one_cycle(1, slice(2e-3/100, 2e-3))
# test classifier 
learnc.predict("it was the worst movie I ever saw. very bad.")
#another test 
learnc.predict("After all, it wasn't a bad movie to watch.")
