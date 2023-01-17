import os

import re

import html as ihtml

import warnings

import random

warnings.filterwarnings('ignore')



os.environ["TFHUB_CACHE_DIR"] = "/tmp/"



from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import scipy

import umap



import tensorflow as tf

import tensorflow_hub as hub



import plotly_express as px

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_colwidth', -1)



SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_random_seed(SEED)



%matplotlib inline
umap.__version__
input_dir = '../input'



questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))

tags = pd.read_csv(os.path.join(input_dir, 'tags.csv'))

tag_questions = pd.read_csv(os.path.join(input_dir, 'tag_questions.csv'))
def clean_text(text, remove_hashtags=True):

    text = BeautifulSoup(ihtml.unescape(text), "lxml").text

    text = re.sub(r"http[s]?://\S+", "", text)

    if remove_hashtags:

        text = re.sub(r"#[a-zA-Z\-]+", "", text)

    text = re.sub(r"\s+", " ", text)        

    return text
questions['questions_full_text'] = questions['questions_title'] + ' '+ questions['questions_body']
sample_text = questions[questions['questions_full_text'].str.contains("&a")]["questions_full_text"].iloc[0]

sample_text
sample_text = clean_text(sample_text)

sample_text
%%time

questions['questions_full_text'] = questions['questions_full_text'].apply(clean_text)
questions['questions_full_text'].sample(2)
tag_questions.groupby(

    "tag_questions_tag_id"

).size().sort_values(ascending=False).to_frame("count").merge(

    tags, left_on="tag_questions_tag_id", right_on="tags_tag_id"

).head(10)
questions_id_medicine = set(tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id)

questions_id_engineering = set(tag_questions[tag_questions.tag_questions_tag_id == 54].tag_questions_question_id)

len(questions_id_medicine), len(questions_id_engineering), len(questions_id_medicine.intersection(questions_id_engineering))
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
import logging

from tqdm import tqdm_notebook

tf.logging.set_verbosity(logging.WARNING)

BATCH_SIZE = 128



sentence_input = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than

# approximately normalized.

sentence_emb = tf.nn.l2_normalize(embed(sentence_input), axis=1)



sentence_embeddings = []       

with tf.Session() as session:

    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    for i in tqdm_notebook(range(0, len(questions), BATCH_SIZE)):

        sentence_embeddings.append(

            session.run(

                sentence_emb, 

                feed_dict={

                    sentence_input: questions["questions_full_text"].iloc[i:(i+BATCH_SIZE)].values

                }

            )

        )
sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)

sentence_embeddings.shape
%%time

embedding = umap.UMAP(metric="cosine", n_components=2, random_state=42).fit_transform(sentence_embeddings)
df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])
df_emb_sample = df_se_emb.sample(10000)

fig, ax = plt.subplots(figsize=(12, 10))

plt.scatter(

    df_emb_sample["x"].values, df_emb_sample["y"].values, s=1

)

plt.setp(ax, xticks=[], yticks=[])

plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)

plt.show()
print(questions[df_se_emb.x > 10].shape[0])

questions[df_se_emb.x > 10].questions_full_text.sample(5)
print(questions[df_se_emb.y > 8].shape[0])

questions[df_se_emb.y > 8].questions_full_text.sample(5)
questions_id_medicine = tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id

questions_id_engineering = tag_questions[tag_questions.tag_questions_tag_id == 54].tag_questions_question_id

df_se_emb["tag"] = "none"

df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"

df_se_emb.loc[questions.questions_id.isin(questions_id_engineering), "tag"] = "engineering"

df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(

    set(questions_id_engineering)))), "tag"] = "both"
df_se_emb.tag.value_counts()
px.colors.qualitative.D3
df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()

df_emb_sample["tag"] = df_emb_sample.tag.astype("category")

df_emb_sample["size"] = 20

px.scatter(

    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",

    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,

    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid

    

)
questions_id_medicine = tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id

questions_id_biz = tag_questions[tag_questions.tag_questions_tag_id == 27292].tag_questions_question_id

df_se_emb["tag"] = "none"

df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"

df_se_emb.loc[questions.questions_id.isin(questions_id_biz), "tag"] = "business"

df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(

    set(questions_id_biz)))), "tag"] = "both"
df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()

df_emb_sample["tag"] = df_emb_sample.tag.astype("category")

df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()

df_emb_sample["tag"] = df_emb_sample.tag.astype("category")

df_emb_sample["size"] = 20

px.scatter(

    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",

    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,

    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid

    

)
df_se_emb["tag"] = "none"

df_se_emb.loc[questions.questions_id.isin(questions_id_engineering), "tag"] = "engineering"

df_se_emb.loc[questions.questions_id.isin(questions_id_biz), "tag"] = "business"

df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"

df_se_emb.loc[questions.questions_id.isin((set(questions_id_engineering).intersection(

    set(questions_id_biz)))), "tag"] = "none"

df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(

    set(questions_id_biz)))), "tag"] = "none"

df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(

    set(questions_id_engineering)))), "tag"] = "none"
df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()

df_emb_sample["tag"] = df_emb_sample.tag.astype("category")

df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()

df_emb_sample["tag"] = df_emb_sample.tag.astype("category")

df_emb_sample["size"] = 20

px.scatter(

    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",

    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,

    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid

    

)