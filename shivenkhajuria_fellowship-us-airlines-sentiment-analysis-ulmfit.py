# Import necessary libraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# import fast.ai libraries for nlp

from fastai import *
from fastai.text import *

import fastai.utils.collect_env

fastai.utils.collect_env.show_install()
# bs=48
# bs=24
bs=192
path = Path('../input/twitter-airline-sentiment/')
file_name = 'Tweets.csv'
path.ls()
file_path = path / file_name
df_airline = pd.read_csv(file_path)
df_airline.head()
df_final = df_airline[['airline_sentiment', 'text']]
pd.set_option('display.max_colwidth',0)
df_final.head()
#check for missing values in data
df_final.isna().sum()
#Data is skewed more towards the negative sentiment
df_final['airline_sentiment'].value_counts()
#Visualize the sentiment for each airline
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x="airline", hue='airline_sentiment', data=df_airline)
plt.title("Airline Sentiment For Each Airline");

import re
regex = r"@(VirginAmerica|united|SouthwestAir|Delta|USAirways|AmericanAir)"
def text_replace(text):
    return re.sub(regex, '@airline', text, flags=re.IGNORECASE)

df_final['text'] = df_final['text'].apply(text_replace)
df_final.head(10)
#Split the dataframe randomly into train set and valid set. 
#TextLMDataBunch only accepts two separate dataframes for train and valid
train, valid = train_test_split(df_final, test_size=0.1)
moms = (0.8,0.7)
wd = 0.1
data_lm = TextLMDataBunch.from_df(path, train_df = train, valid_df = valid)
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, model_dir='/tmp/models')
learn.freeze()



#learn.model_dir='/kaggle/working/'
#To Find the proper learning rate, use the x value where the slope is steepest in the -y direction.
learn.lr_find()
learn.recorder.plot()
moms = (0.8,0.7)
wd = 0.1
lr = 1.0E-02
learn.fit_one_cycle(1, lr, moms=moms, wd=wd)
learn.unfreeze()
learn.fit_one_cycle(3, lr, moms=moms, wd=wd)

learn.predict('This flight sucks!', n_words=20)

learn.save_encoder('ft_enc')
train_valid, test = train_test_split(df_final, test_size=0.1)
train, valid = train_test_split(train_valid, test_size=0.1)

data_clas = TextClasDataBunch.from_df(path,train_df=train, valid_df = valid,test_df = test, vocab=data_lm.train_ds.vocab, 
                                      text_cols='text', label_cols='airline_sentiment', bs=48)


data_clas.show_batch()

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, model_dir='/tmp/models')
learn.load_encoder('ft_enc')
learn.freeze()
learn.lr_find()
learn.recorder.plot()
#The fast.ai ULMFIT method performs better if we do one epoch at a time for the classifier training
lr = 1.0E-03
learn.fit_one_cycle(1, lr, moms=moms, wd=wd)
learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.unfreeze()
lr /= 5
learn.fit_one_cycle(2, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.predict('this airline sucks!')

interp = TextClassificationInterpretation.from_learner(learn)
acc = accuracy(interp.preds, interp.y_true)
print('Accuracy: {0:.3f}'.format(acc))
interp.plot_confusion_matrix()
plt.title('Classifation Confusion Matrix')
#test_df = df_final
#test_df['pred_sentiment'] = test_df['text'].apply(lambda row: str(learn.predict(row)[0]))
#pred_sent_df = test_df.loc[(test_df['airline_sentiment'] == 'positive') & (test_df['pred_sentiment'] == 'negative')]
#pred_sent_df.head(20)


interp.show_top_losses(20)
