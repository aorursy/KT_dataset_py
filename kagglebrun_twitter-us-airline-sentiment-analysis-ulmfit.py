from fastai.text import *



import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
path = '../input/twitter-airline-sentiment/Tweets.csv'

df = pd.read_csv(path, index_col = 'tweet_id')

#df[['airline_sentiment','airline','text']]

df.head()
pd.Series(df['airline_sentiment']).value_counts().plot(kind = "bar" , title = "Class Distribution", )
pd.crosstab(index = df["airline"] ,columns = df["airline_sentiment"] ).plot(kind = "bar", stacked=True, title="Sentiment Distribution by Airline", )
pd.Series(df["negativereason"]).value_counts().plot(kind = "bar" , title = "Reasons for Negative Reviews", )
pd.crosstab(index = df["airline"] ,columns = df["negativereason"] ).plot(kind = "bar", title = "Negative Reasons by Airline", stacked = True, figsize=(10,10))
df_final = df[['airline_sentiment', 'text']]

df_final.head()
train, val = train_test_split(df_final, test_size=0.1)

data_lm = TextLMDataBunch.from_df(path="/output/kaggle/working", train_df = train, valid_df = val)
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, model_dir='/output/kaggle/working')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2, moms=(0.8, 0.7))

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
#Testing the LM so far



learn.predict('Terrible customer service @UnitedAirlines!', n_words=10)
train_valid, test = train_test_split(df_final, test_size=0.1)

train, val = train_test_split(train_valid, test_size=0.1)
#creating classifer data bunch with vocab from the LM



data_clas = TextClasDataBunch.from_df(path,train,val,test, vocab=data_lm.train_ds.vocab, 

                                      text_cols='text', label_cols='airline_sentiment', bs=32)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, model_dir='/output/kaggle/working')
learn.load_encoder('fine_tuned_enc')

learn.freeze()
#finding lr for training process



learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
learn.recorder.plot_losses()
learn.freeze_to(-2)

learn.fit_one_cycle(3, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)

learn.fit_one_cycle(2, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
pred_class = learn.predict("Terrible Flight!")

print(f"Predicted Sentiment: {list(pred_class)[0]}")
test['pred_sentiment'] = test['text'].apply(lambda row: str(learn.predict(row)[0]))
learn.save('classifier')
test.head()
text_ci = TextClassificationInterpretation.from_learner(learn)

test_text = "@UnitedAirlines I am extremely disappointed."

text_ci.show_intrinsic_attention(test_text,cmap=cm.Blues)
text_ci.plot_confusion_matrix()
text_ci.show_top_losses(5)
accuracy_score(test['pred_sentiment'], test['airline_sentiment'])
f1_score(test['airline_sentiment'],test['pred_sentiment'],average = 'macro')
print(classification_report( test['airline_sentiment'], test['pred_sentiment']))