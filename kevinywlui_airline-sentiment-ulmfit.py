# Import fastai to use their ULMFiT implementation

from fastai.text import * 



# fastai needs us to specify a path sometimes

from pathlib import Path



# Import usual data science libraries

import pandas as pd

import seaborn as sns

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score
path = Path('../input/twitter-airline-sentiment/')

file_name = 'Tweets.csv'
file_path = path / file_name

df_full = pd.read_csv(file_path)

df_full.size
df_full.sample(10, random_state=0)
pd.set_option('display.max_colwidth', 0) # tweets aren't too long so let's just print it all
df = df_full[['airline_sentiment', 'text']]

df.sample(10)
df[['airline_sentiment', 'text']].isna().sum()
df['airline_sentiment'].value_counts()
df['airline_sentiment'].value_counts(normalize=True)
sns.countplot(y='airline', hue='airline_sentiment', data=df_full)
import re

regex = r"@(VirginAmerica|united|SouthwestAir|Delta|USAirways|AmericanAir)"

def text_replace(s):

    return re.sub(regex, '@airline', s, flags=re.IGNORECASE)
df['text'] = df['text'].apply(text_replace)
df['text'].sample(5)
train, valid = train_test_split(df, test_size=0.2)
moms = (0.8,0.7)

wd = 0.1
working_path = Path('./').resolve() # fastai needs a working path
data_lm = TextLMDataBunch.from_df(working_path, train, valid) # form the data bunch
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3) # this fetches the wiki103 model

learn.freeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 5.0E-02, moms=moms, wd=wd) # 5.0E-02 is LR with the steepest slope above
learn.unfreeze()
learn.fit_one_cycle(3, 5.0E-03, moms=moms, wd=wd)
learn.predict('My flight is great!', n_words=20)
learn.save_encoder('ft_enc')
train_valid, test = train_test_split(df, test_size=0.2)

train, valid = train_test_split(train_valid, test_size=0.2)
data_clas = TextClasDataBunch.from_df(working_path, train, valid, test_df=test, vocab=data_lm.train_ds.vocab, text_cols='text', label_cols='airline_sentiment', bs=32)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')

learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr = 3.0E-02

learn.fit_one_cycle(1, lr, moms=moms, wd=wd)
learn.freeze_to(-2)

lr /= 2

learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.freeze_to(-3)

lr /= 2

learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.unfreeze()

lr /= 5

learn.fit_one_cycle(3, slice(lr/(2.6**4), lr), moms=moms, wd=wd)
learn.predict('I love flying')
learn.predict('My flight was delayed')
learn.predict("Safe flight!")
interp = TextClassificationInterpretation.from_learner(learn)

acc = accuracy(interp.preds, interp.y_true)

print('Accuracy: {0:.3f}'.format(acc))
interp.plot_confusion_matrix()
scores = pd.DataFrame(interp.preds)

plt.figure(figsize=(12, 12))

fpr = dict()

tpr = dict()

thresh = dict()

for i, cls in zip(range(3), ['negative', 'neutral', 'positive']):

    score = scores[i].apply(lambda x: x.item())

    y_true = [x.item() == i for x in interp.y_true]

    fpr[i], tpr[i], thresh[i] = roc_curve(y_true, score, pos_label=True)

    auc = roc_auc_score(y_true, score)

    leg = "AUC: {0:.3f} -- {1}".format(float(auc), cls)

    plt.plot(fpr[i], tpr[i], label=leg)

    

plt.legend(loc="lower right", prop={'size': 28})

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
vc = df['airline_sentiment'].value_counts()

T = sum(vc) # number of total

P = vc[0] # number of Positive class

N = T - P # number of Negative class



# The following were computed from the ROC section. zip(Nfpr, Ntpr) gives a list of coordinates to the negative class ROC curve and Nthresh gives the corresponding thresholds.

Nfpr = fpr[0]

Ntpr = tpr[0]

Nthresh = thresh[0]



num_pts = len(Nfpr)
# This computes the cost as defined above where the FPR and TPR is given by Nfpr[i] and Ntpr[i]

def cost(C, i):

    return N*Nfpr[i] + C*P*(1-Ntpr[i]), Nthresh[i]
min(cost(1, i) for i in range(num_pts))[1]
min(cost(2, i) for i in range(num_pts))[1]