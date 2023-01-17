import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import nltk

import re

import string

import json



from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator

from nltk import FreqDist

from nltk import sent_tokenize, word_tokenize

from nltk.stem.porter import PorterStemmer

from fastai.text import *





sns.set(style='whitegrid')

%matplotlib inline

warnings.filterwarnings('ignore')



with open('/kaggle/input/stopwords/engstopwords.json', 'r') as e:

    stopwords_eng = json.load(e)



train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

train_df.head()
print('Train size: ',train_df.shape)

print('Test size: ', test_df.shape)
train_df.columns
train_df.info()
train_df.isna().sum()
train_df.location.value_counts()[:20]
train_df.keyword.value_counts()[:20]

target = train_df.target.value_counts()

plt.figure(figsize=(10,5))

sns.countplot(x='target', data=train_df);

plt.title('Class Distribution');
def clean_txt(docs):

    # split into words

    speech_words = nltk.word_tokenize(docs)

    # convert to lower case

    lower_text = [w.lower() for w in speech_words]

    # prepare regex for char filtering

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    # remove punctuation from each word

    stripped = [re_punc.sub('', w) for w in lower_text]

    # remove remaining tokens that are not alphabetic

    words = [word for word in stripped if word.isalpha()]

    # filter out stop words

    words = [w for w in words if not w in  stopwords_eng]

    combined_text = ' '.join(words)

    return combined_text



train_df['text']= train_df['text'].apply(lambda x: clean_txt(x))

test_df['text']= test_df['text'].apply(lambda x: clean_txt(x))
real_mask = np.array(Image.open('/kaggle/input/twitter-pic/twitter_1.png'))

not_mask = np.array(Image.open('/kaggle/input/twitter-pic/twitter_2.png'))



real = train_df[train_df.target == 1]

Not = train_df[train_df.target == 0]



real_image_colors = ImageColorGenerator(real_mask)

not_image_colors = ImageColorGenerator(not_mask)



real_wordcloud = WordCloud(background_color="white", max_words=1000,mask=real_mask).generate(str(real['text']))

not_wordcloud = WordCloud(background_color="white", max_words=1000,mask=not_mask).generate(str(Not['text']))



f,ax=plt.subplots(1,2,figsize=(15,10))

ax[0].imshow(real_wordcloud.recolor(color_func=real_image_colors), interpolation="bilinear")

ax[0].set_title('Disaster Tweets',fontdict={'size':16,'weight':'bold'})

ax[0].axis("off");

ax[1].imshow(not_wordcloud.recolor(color_func=not_image_colors), interpolation="bilinear")

ax[1].set_title('Non Disaster Tweets',fontdict={'size':16,'weight':'bold'})

ax[1].axis("off");

plt.show()

data = (TextList.from_df(train_df, cols='text')

                .split_by_rand_pct(0.2)

                .label_for_lm()  

                .databunch(bs=128))

            

data.save()

data

data.show_batch()


learn = language_model_learner(data,AWD_LSTM, drop_mult=0.2, pretrained=True)

learn.fit_one_cycle(10, 1e-2)

learn.save('mini_train_lm')

learn.load('mini_train_lm')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

Learning_rate = learn.recorder.min_grad_lr

print(Learning_rate)

plt.show()
learn.fit_one_cycle(10, Learning_rate)

learn.save_encoder('finetune_train_encoder')
test_datalist = TextList.from_df(test_df, cols='text', vocab=data.vocab)



data_clas = (TextList.from_df(train_df, cols='text', vocab=data.vocab)

             .split_by_rand_pct(0.2)

             .label_from_df(cols= 'target')

             .add_test(test_df)

             .databunch(bs=128))



data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM,drop_mult=0.2,pretrained=True)

learn.load_encoder('finetune_train_encoder')
learn.freeze_to(-1)

learn.lr_find()

learn.recorder.plot(suggestion=True)

Learning_rate = learn.recorder.min_grad_lr

print(Learning_rate)

plt.show()

learn.fit_one_cycle(10, Learning_rate)

learn.recorder.plot_losses()

plt.show()



learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

Learning_rate = learn.recorder.min_grad_lr

print(Learning_rate)

plt.show()
learn.fit_one_cycle(10, Learning_rate)

learn.save('fl_train_class')
interp = TextClassificationInterpretation.from_learner(learn) 

interp.show_top_losses(10)
print(test_df.loc[2,'text'])

learn.predict(test_df.loc[2,'text'])
def get_preds_as_nparray(ds_type) -> np.ndarray:

    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in learn.data.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]
test_preds = get_preds_as_nparray(DatasetType.Test)

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sub['target'] = np.argmax(test_preds, axis=1)

sub.to_csv("submission.csv", index=False, header=True)

sub.head()