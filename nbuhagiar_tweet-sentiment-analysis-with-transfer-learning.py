# General Data Science

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Machine Learning

from fastai.text import TextLMDataBunch, language_model_learner, Transformer, TextClasDataBunch, text_classifier_learner



# Miscellaneous

import warnings

warnings.filterwarnings(action="once")

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv", encoding="latin", header=None)

df.head()
df = df[[0, 5]]

df.rename(columns={0: "target", 5: "text"}, inplace=True)

df.head()
df["target"] = df["target"].astype(str).str.replace("0", "negative").str.replace("4", "positive")

df = df.sample(frac=0.1, random_state=0)

limit = int(0.9 * len(df))

train = df[:limit]

val = df[limit:]
fig, ax = plt.subplots()

df["target"].value_counts().plot.bar(ax=ax)

ax.set_xlabel("Label")

ax.set_ylabel("Frequency")

fig.tight_layout()
data_path = "/kaggle/input/sentiment140/"

data_lm = TextLMDataBunch.from_df(data_path, train, val)

data_lm
data_lm.show_batch()
learn_lm = language_model_learner(data_lm, Transformer, drop_mult=0.5, model_dir="/kaggle/working")
learn_lm.lr_find()

learn_lm.recorder.plot()
learn_lm.fit_one_cycle(3, 3e-2)
learn_lm.save("lm_initial_fit")

learn_lm.unfreeze()
learn_lm.lr_find()

learn_lm.recorder.plot()
learn_lm.fit_one_cycle(3, slice(1e-6, 1e-4))
learn_lm.save("lm_second_fit")

learn_lm.save_encoder("encoder")
print(learn_lm.predict("WOW! I can't believe", n_words=10))

print(learn_lm.predict("WOW! I can't believe", n_words=10))



print(learn_lm.predict("And that's why", n_words=10))

print(learn_lm.predict("And that's why", n_words=10))
data_c = TextClasDataBunch.from_df(data_path, train, val)

data_c
data_c.show_batch()
learn_c = text_classifier_learner(data_c, Transformer, drop_mult=0.5, model_dir="/kaggle/working")

learn_c = learn_c.load_encoder("encoder")
learn_c.lr_find()

learn_c.recorder.plot()
learn_c.fit_one_cycle(3, 5e-3)
learn_c.save("c_initial_fit")

learn_c.unfreeze()
learn_c.lr_find()

learn_c.recorder.plot()
learn_c.fit_one_cycle(3, slice(3e-7, 3e-5))
print(learn_c.predict("I thought the movie was good but not great"))

print(learn_c.predict("That movie was really interesting. I was surprised how evil the villain was."))

print(learn_c.predict("Uh okay..."))

print(learn_c.predict("This news is upsetting."))

print(learn_c.predict("Yes, we finally did it!"))