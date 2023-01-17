# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.text import *



def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

#Remember to use num_workers=0 when creating the DataBunch.
path = Path('../input/')

path.ls()
df = pd.read_csv(path/'train-balanced-sarcasm.csv')

df.head()
df['comment'][1]
#data = TextDataBunch.from_csv(path, 'train-balanced-sarcasm.csv', num_workers=0)
badinputs = df.loc[lambda x: x['comment'].isna()]

badinputs.head()
df = df.dropna()

df['comment'][56267:56270]
random_seed(123,True)

rand_df = df.assign(is_valid = np.random.choice(a=[True,False],size=len(df),p=[0.2,0.8]))

rand_df.head()
random_seed(1006,True)

bs=48

data_lm = (TextList.from_df(rand_df, path, cols="comment")

                .split_from_df(col='is_valid')

                .label_for_lm()

                .databunch(bs=bs))
data_lm.save('../working/data_lm.pkl')
data_lm = load_data('../working','data_lm.pkl',bs=bs)

data_lm.show_batch()
random_seed(100,True)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()

learn.recorder.plot(skip_end=15)
random_seed(111, True)

learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
random_seed(444,True)

learn.fit_one_cycle(10,2e-2,moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
learn.save_encoder('fine_tuned_enc')
data_clas = (TextList.from_df(rand_df, vocab=data_lm.vocab, cols="comment")

                .split_from_df(col='is_valid')

                .label_from_df(cols='label')

                .databunch(bs=bs))

data_clas.save('../working/data_clas.pkl')
data_clas = load_data('../working','data_clas.pkl',bs=bs)

data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot()
random_seed(678,True)

learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first')
learn.load('first');
random_seed(777,True)

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second')
learn.load('second');
random_seed(999,True)

learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(2e-3/(2.6**4),2e-3), moms=(0.8,0.7))
learn.save('third')
learn.load('third');
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn,preds,y,losses)

interp.plot_confusion_matrix()
losses,idxs = interp.top_losses()

len(data_clas.valid_ds)==len(losses)==len(idxs)

idxs[:10]
for i in range(10):

  print(df['comment'][idxs[i]],df['label'][idxs[i]],losses[i])
# Sarcastic

learn.predict("What could possibly go wrong?")
# Sincere

learn.predict("I think that is a really good idea.")
# Sarcastic

learn.predict("Obviously this is all your fault.")
# Sincere

learn.predict("Honestly, this is your fault.")
# ???

learn.predict("Good job, learner!")