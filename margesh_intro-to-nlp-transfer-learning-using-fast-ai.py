import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/ireland-historical-news/irishtimes-date-text.csv')

df.head()
try:

    df = df.drop('publish_date', axis=1)

except:

    #already dropped

    pass



df.head()
df['headline_category'].value_counts()
print(len(df['headline_category'].value_counts()))
filtered_df = df[df.groupby('headline_category').headline_category.transform(len)>10000]

print(len(filtered_df['headline_category'].value_counts()))
filtered_df['headline_category'].value_counts()
plt.figure(figsize=(20,20))

f = sns.countplot(filtered_df['headline_category'])

f.set_xticklabels(f.get_xticklabels(), rotation=20, ha="right");
def sampling_k_elements(group, k=10000):

    return group.sample(k)



#Apply the function to all groups

balanced_df = filtered_df.groupby('headline_category').apply(sampling_k_elements).reset_index(drop=True)

balanced_df['headline_category'].value_counts()
balanced_df['category'] = balanced_df['headline_category'].astype("category").cat.codes

balanced_df.head()
np.random.seed(123)

balanced_df = balanced_df.iloc[np.random.permutation(len(balanced_df))]

cut1 = int(0.8 * len(balanced_df)) + 1

try:

    dropped_balanced_df = balanced_df.drop('headline_category', axis=1)

except:

    pass



df_train, df_valid = dropped_balanced_df[:cut1], dropped_balanced_df[cut1:]
print(df_train.shape)

df_train.head()
print(df_valid.shape)

df_valid.head()
df_valid['category'].value_counts()
category_numbers = dict(enumerate(balanced_df['headline_category'].astype("category").cat.categories))

print (category_numbers)
from fastai.text import *
data_lm = TextLMDataBunch.from_df(path="", train_df=df_train, valid_df = df_valid, text_cols="headline_text", label_cols="category")
data_lm.save('irish.pkl')
data_lm.show_batch(5)
data_lm.vocab.itos[:40]
learner = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.1)
learner.save_encoder('irish_encoder')
data_clas = TextClasDataBunch.from_df(path="", train_df=df_train, valid_df = df_valid, vocab=data_lm.train_ds.vocab, text_cols="headline_text",label_cols="category")
clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.1)
clas.load_encoder('irish_encoder')
clas.lr_find()

clas.recorder.plot()
clas.fit_one_cycle(5, 1e-02)
clas.freeze_to(-2)

clas.lr_find()

clas.recorder.plot(suggestion=True)
clas.fit_one_cycle(5, 5e-04, moms=(0.9,0.8))
clas.save_encoder('freeze_2_encoder')
clas.freeze_to(-3)

clas.lr_find()

clas.recorder.plot(suggestion=True)
clas.fit_one_cycle(5, 3.2e-05, moms=(0.95,0.85))
clas.save_encoder('freeze_3_encoder')
clas.unfreeze()

clas.lr_find()

clas.recorder.plot(suggestion=True)
clas.fit_one_cycle(3, 5e-05, moms=(0.95, 0.85))
clas.save_encoder('final_encoder')
clas.predict("Artist A's latest album is soaring through the charts")
print(category_numbers.get(7))
clas.predict("Beatles' latest album is soaring through the charts")[2].sum()
clas.predict("An underdog wins the worldcup 2-0")
category_numbers.get(18) #To know what category 18 belongs to, let us see...
def pred_classes(text):

    print(category_numbers.get(int(clas.predict(text)[1])))    
pred_classes("NIFTY falls down by 100 rupees")
pred_classes("Eggs and cholestrol - is eating many eggs really unhealthy for your heart?")
pred_classes("We need to do something now, or else our planet is doomed")
pred_classes("10 ways to improve your kitchen")
pred_classes("A couple gets 10 years for killing a teddy bear")