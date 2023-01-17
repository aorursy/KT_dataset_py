import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sn



from fastai.text import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



%matplotlib inline 
df = pd.read_csv('../input/Tweets.csv')

df.head()
df[['airline_sentiment', 'text']].isnull().sum()
df['airline_sentiment'].value_counts().plot(kind='bar')
df['tweet_length'] = df['text'].apply(len)

df.groupby(['tweet_length', 'airline_sentiment']).size().unstack().plot(kind='line')
rnd_state = 111

df_train, df_test = train_test_split(df, test_size=0.15, random_state = rnd_state)

df_train[['airline_sentiment', 'text']].to_csv('Tweets_train.csv', index=False, encoding='utf-8')

df_test[['airline_sentiment', 'text']].to_csv('Tweets_test.csv', index=False, encoding='utf-8')



data_lm = TextLMDataBunch.from_csv('.', 'Tweets_train.csv', valid_pct=0.15)

data_clas = TextClasDataBunch.from_csv('.', 'Tweets_train.csv', valid_pct=0.15, vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save('data_lm_export.pkl')

data_clas.save('data_clas_export.pkl')
data_lm = load_data('.', 'data_lm_export.pkl')

data_clas = load_data('.', 'data_clas_export.pkl', bs=32)
print("Preprocessed text:", data_lm.x[0])

print("\n")

print("Corresponding numerical sequence:", data_lm.x[0].data)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-02, moms=(0.8, 0.7))
learn.unfreeze()

learn.fit_one_cycle(10,1e-03, moms=(0.8, 0.7))
learn.predict("My experience was", n_words=10)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, 1e-2, moms=(0.8, 0.7))
learn.freeze_to(-2)

learn.fit_one_cycle(5, slice(1e-2/(2.6**4),1e-2), moms=(0.8, 0.7))
learn.unfreeze()

learn.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8, 0.7))
test_df = pd.read_csv("Tweets_test.csv", encoding="utf-8")

test_df['pred_sentiment'] = test_df['text'].apply(lambda row: str(learn.predict(row)[0]))

test_df['airline_sentiment'].value_counts().plot(kind='bar')
print("Test Accuracy: ", accuracy_score(test_df['airline_sentiment'], test_df['pred_sentiment']))
conf_matrix = confusion_matrix(y_true=test_df['airline_sentiment'].values, y_pred=test_df['pred_sentiment'].values, labels=['negative', 'neutral', 'positive'])

labels = ['negative', 'neutral', 'positive']

sn.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
pd.set_option('display.max_colwidth', -1)

test_df.loc[(test_df['airline_sentiment'] == 'positive') & (test_df['pred_sentiment'] == 'negative')].head()