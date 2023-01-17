import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.text import *

import seaborn as sns

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# training data

train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

train['comment_text'] = train['comment_text'].str.replace("\n"," ").replace('([“”¨«»®´·º½¾¿¡§£₤‘’=])', '')

train.head()
train.shape
# test data 

test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")

test['comment_text'] = test['comment_text'].str.replace("\n"," ").replace('([“”¨«»®´·º½¾¿¡§£₤‘’=])', '')

test.head()
test.shape
reviews_len = [len(x) for x in train["comment_text"]]

pd.Series(reviews_len).hist()

plt.show()

pd.Series(reviews_len).describe()
print(train['toxic'].value_counts())

print(train['severe_toxic'].value_counts())

print(train['obscene'].value_counts())

print(train['threat'].value_counts())

print(train['insult'].value_counts())

print(train['identity_hate'].value_counts())
# split data into training and validation set 

data = (TextList.from_df(train, cols='comment_text').split_by_rand_pct(0.3).label_for_lm().databunch(bs=48))
data.show_batch()
##saving 

data.save('/kaggle/working/data_lm_export.pkl')
path = '/kaggle/input/kernel4a04d4729c'
data_lm = load_data(path,'data_lm_export.pkl')
model_path = path + "/models/"
os.makedirs(model_path, exist_ok=True)
url = 'http://files.fast.ai/models/wt103_v1/' 

download_url(f'{url}lstm_wt103.pth', model_path + '/lstm_wt103.pth') 

download_url(f'{url}itos_wt103.pkl', model_path + '/itos_wt103.pkl')
learn =  language_model_learner(data_lm,AWD_LSTM, drop_mult=0.3)
learn.predict('This was such a great ', 50, temperature=1.1, min_p=0.001)
learn.export()
learner = load_learner(path)

learn.predict('This was such a great ', 50, temperature=1.1, min_p=0.001)
learn.predict('This was such a great ', 50, temperature=1.1, min_p=0.001)
learn.model_dir = ("/kaggle/working/")

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr
learn.fit_one_cycle(2, min_grad_lr)
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(2, min_grad_lr)
learn.save_encoder('ft_enc')
learn.recorder.plot_losses()
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.freeze_to(-2)

learn.fit_one_cycle(4, slice(5e-4, 2e-4), moms=(0.8,0.7))
learn.save_encoder('ft_enc_1')
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(2, 1e-5)
learn.save_encoder('ft_enc_2')
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.recorder.plot_losses()
learn.freeze_to(-2)

learn.fit_one_cycle(2, slice(5e-4, 2e-4), moms=(0.8,0.7))
learn.save_encoder('ft_enc_3')
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.unfreeze()

learn.fit_one_cycle(2, 1e-5, moms=(0.8,0.7))
learn.save_encoder('ft_enc_4')
label_cols = ['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']



test_datalist = TextList.from_df(test, cols='comment_text', vocab=data_lm.vocab)
data_clas = (TextList.from_df(train, cols='comment_text', vocab=data_lm.vocab).split_by_rand_pct(0.2).label_from_df(cols= label_cols , classes=label_cols).add_test(test_datalist).databunch(bs=32))



data_clas.show_batch()
data_clas.save('/kaggle/working/data_clas_export.pkl')
learn_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn_classifier.load_encoder('/kaggle/working/ft_enc_4')
learn_classifier.freeze()
learn_classifier.lr_find()

learn_classifier.recorder.plot(suggestion=True)
min_grad_lr = learn_classifier.recorder.min_grad_lr
learn_classifier.fit_one_cycle(2, min_grad_lr)
learn_classifier.lr_find()

learn_classifier.recorder.plot(suggestion=True)
min_grad_lr = learn_classifier.recorder.min_grad_lr
learn_classifier.recorder.plot_losses()
learn_classifier.freeze_to(-2)

learn_classifier.fit_one_cycle(4, slice(1e-3/100, 1e-3), moms=(0.8,0.7))
learn_classifier.save_encoder("ft_enc_5")
learn_classifier.lr_find()

learn_classifier.recorder.plot(suggestion=True)

min_grad_lr = learn_classifier.recorder.min_grad_lr
learn_classifier.recorder.plot_losses()
learn_classifier.unfreeze()

learn_classifier.fit_one_cycle(2, min_grad_lr, moms=(0.8,0.7))

learn_classifier.show_results()
test_id = test['id']

preds, target = learn_classifier.get_preds(DatasetType.Test, ordered=True)

labels = preds.numpy()



submission = pd.DataFrame({'id': test_id})

submission = pd.concat([submission, pd.DataFrame(preds.numpy(), columns = label_cols)], axis=1)
submission.head()


submission.to_csv('/kaggle/working/submission.csv', index=False)
from google.colab import files



df.to_csv('df.csv')

files.download('df.csv')
