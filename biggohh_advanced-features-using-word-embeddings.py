!pip install -q --upgrade nltk

!pip install -q umap-learn

import nltk

nltk.download('averaged_perceptron_tagger_ru')
from itertools import product



import pandas as pd

from IPython.display import display

from textblob.utils import strip_punc

from tqdm.auto import tqdm
datadir = "../input/competitive-data-science-predict-future-sales"
# Get the name and category name of each item, along with it's average price.

df = (

    pd.read_csv(f"{datadir}/sales_train.csv")

    .merge(pd.read_csv(f"{datadir}/items.csv"))

    .merge(pd.read_csv(f"{datadir}/item_categories.csv"))

    .groupby(["item_id", "item_category_name", "item_name"])["item_price"]

    .mean()

    .reset_index()

)

df
def remove_punctuation(text):

    return strip_punc(text, all=True)
df["item_name"] = df["item_name"].apply(remove_punctuation)

df["item_category_name"] = df["item_category_name"].apply(remove_punctuation)

with pd.option_context("display.max_colwidth", None):

    display(df.sort_values(["item_name", "item_category_name"]))
import gensim.downloader
en_model = gensim.downloader.load("glove-wiki-gigaword-300")
en_model.similar_by_word("xbox")
df["description"] = df["item_category_name"] + "\n" + df["item_name"]
import numpy as np

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# Remove any words that don't add meaning, such as "the", "in", "of", etc.

stopwords = stopwords.words("english")





def vectorize(row):

    text = row["description"]

    tokens = [

        word

        for word in word_tokenize(strip_punc(text.lower(), all=True))

        if word not in stopwords

    ]

    vecs = [en_model[tok] for tok in tokens if tok in en_model]

    if vecs:

        vector = np.mean(vecs, axis=0)

    else:

        vector = np.zeros_like(en_model.vectors[0])

    return vector
vecs = pd.DataFrame(

    data=np.array(df.apply(vectorize, axis=1).tolist()), index=df["description"]

)
import random



import seaborn as sns

from matplotlib import pyplot as plt

from umap import UMAP  # UMAP is faster than t-SNE



sns.set()





def plot_vectors(vectors, data=None, n_labels=0):

    fig, ax = plt.subplots(figsize=(16, 9))

    plt.close()

    ax.axis("off")

    seed = 42



    tsne = UMAP(n_components=2, random_state=seed)

    reduced = tsne.fit_transform(vectors)

    colours = np.log1p(data["item_price"])

    ax.scatter(reduced[:, 0], reduced[:, 1], c=colours, cmap="RdBu_r", alpha=0.2)



    random.seed(seed)

    for idx in random.sample(range(len(reduced)), n_labels):

        x, y = reduced[idx]

        name = data.iloc[idx]["description"]

        if len(name) > 37:

            name = name[:37] + "..."



        ax.annotate(

            name,

            (x, y),

            xycoords="data",

            xytext=(random.randint(-100, 100), random.randint(-100, 100)),

            horizontalalignment="right" if x < 0 else "left",

            textcoords="offset points",

            color="black",

            bbox=dict(boxstyle="round", fc=(0.03, 0.85, 0.37, 0.45), ec="none"),

            arrowprops=dict(arrowstyle="simple", linewidth=5, ec="none",),

        )



    fig.tight_layout()

    return fig
from sklearn.neighbors import NearestNeighbors



display(plot_vectors(vecs, data=df, n_labels=15))



nn = NearestNeighbors(n_neighbors=3, metric="cosine", n_jobs=-1).fit(vecs)



# Preview the nearest neighbours for the first few items.

with pd.option_context("display.max_colwidth", None):

    display(

        pd.DataFrame(

            df["description"].values[

                nn.kneighbors(vecs[:10], n_neighbors=3, return_distance=False)

            ]

        )

    )
cyrillic = set("АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя")



en, ru = 0, 0





def count_langs(text):

    global en, ru

    tokens = [

        set(word)

        for word in word_tokenize(strip_punc(text.lower(), all=True))

        if word not in stopwords

    ]



    for tok in tokens:

        if tok & cyrillic:

            ru += 1

        else:

            en += 1





df["description"].apply(count_langs)



# (Rough) percentage of words that are English

en / (en + ru)
# Note that this model was trained on POS (part-of-speech) tagged words.

# This means that we have to append a POS tag to the end of any Russian words before we look them up in this model.

ru_model = gensim.downloader.load("word2vec-ruscorpora-300")
print(en_model.similar_by_word("king", 1))



# The result for this should be 'царица', which means 'queen'

print(ru_model.similar_by_word("царь_NOUN", 1))
tsar_vector = ru_model.get_vector("царь_NOUN")

ru_model.similar_by_vector(tsar_vector, 2)
king_vector = en_model.get_vector("king")

en_model.similar_by_vector(king_vector, 2)
print(en_model.similar_by_vector(tsar_vector, 2))

print(ru_model.similar_by_vector(king_vector, 2))
!pip install -q transvec
from transvec.transformers import TranslationWordVectorizer



# transvec also includes a tokenizer that deals with the Russian POS tags that the pre-trained Russian model uses.

from transvec.tokenizers import EnRuTokenizer





word_pairs = pd.read_csv(f"{datadir}/../enru-word-pairs/ru_word_translations.csv")[["en", "ru"]]



# The transvec model takes the target language model first, followed by any source languages (you can provide more if you have a mix of more than two languages).

enru_model = TranslationWordVectorizer(

    en_model, ru_model, alpha=1, missing="ignore"

).fit(word_pairs)
# Our model can now automatically translate Russian words into a vector in English space.

# It doesn't get it right every time, but we can see that 6 out of the top ten words for "царь" ("tsar"), are correctly related by meaning.

# Note that if we provided an English word, the model would just default to the normal English-language vectors.

print(enru_model.similar_by_word("царь_NOUN"))
tokenizer = EnRuTokenizer()

tokens = df["description"].apply(tokenizer.tokenize)



item_vectors = pd.DataFrame(

    enru_model.transform(tokens), index=df["description"]

).fillna(0)
display(plot_vectors(item_vectors, data=df, n_labels=7))



nn = NearestNeighbors(n_neighbors=3, metric="cosine", n_jobs=-1).fit(item_vectors)



# Preview the nearest neighbours for the first few items.

with pd.option_context("display.max_colwidth", None):

    display(

        pd.DataFrame(

            df["description"].values[

                nn.kneighbors(item_vectors[:10], n_neighbors=3, return_distance=False)

            ]

        )

    )
# First, prepare the training data and test data into a single dataframe.



def denormalize(df):

    return df.merge(pd.read_csv(f"{datadir}/items.csv")).merge(

        pd.read_csv(f"{datadir}/item_categories.csv")

    )





train = (

    # Take the mean item price and item count for each month/shop/item combo

    pd.read_csv(f"{datadir}/sales_train.csv")

    .groupby(["date_block_num", "shop_id", "item_id"])

    .agg({"item_price": "mean", "item_cnt_day": ["mean", "sum"]})

    .reset_index()

)

train.columns = ["_".join([c for c in col if c]) for col in train.columns]

train.rename(columns={"item_cnt_day_sum": "item_cnt_month"}, inplace=True)

train = denormalize(train)



test = pd.read_csv(f"{datadir}/test.csv").drop("ID", axis=1)

test = denormalize(test)

test["date_block_num"] = 34



data = pd.concat([train, test])

data
# Next, prepare a nearest neighbours lookup table to allow us to look up similar items quickly.



items = data.groupby("item_id")[["item_category_name", "item_name"]].first()

items["description"] = items["item_category_name"] + "\n" + items["item_name"]

items = items.drop(columns=["item_category_name", "item_name"])



tokenizer = EnRuTokenizer()

tokens = items["description"].apply(tokenizer.tokenize)



# Index by item ID this time - it's easier to work with later on.

item_vectors = pd.DataFrame(enru_model.transform(tokens), index=items.index).fillna(0)

nn = NearestNeighbors(n_neighbors=3, metric="cosine", n_jobs=-1).fit(item_vectors)



# Number of neighbours we want to calculate for each item.

k = 3



all_neighbours = pd.DataFrame(

    nn.kneighbors(n_neighbors=k, return_distance=False), index=item_vectors.index,

)

all_neighbours
def add_nearest_neighbours(df, nns=all_neighbours):

    "Create a copy of df with extra columns containing the item IDs of the k most similar items (by description)"



    mergecol = "item_id"

    nn_ids = (

        nns.loc[df[mergecol]]

        .astype(np.int16)

        .rename(mapper=lambda x: f"{mergecol}_nn_{x + 1}", axis="columns")

    )



    return pd.concat([df, nn_ids.set_index(df.index)], axis=1)
# Add in the most similar item IDs to each row.

data = add_nearest_neighbours(data)

data
def lagjoin(df, groupon, features, lags, nns=[0], agg="mean", dtype=None):

    lagcols = pd.DataFrame(index=df.index)



    if isinstance(groupon, str):

        features = [groupon]

    if isinstance(features, str):

        features = [features]



    for lag, nn in tqdm(list(product(lags, nns))):

        # A lag of 0 means the current month. A nn of 0 means the current item, not a similar one.

        

        if not lag and not nn:

            # Duplicate of original data.

            continue



        shifted = df[groupon + features].groupby(groupon).agg(agg).reset_index()

        shifted["date_block_num"] += lag



        lgrpcols = [

            col if col != "item_id" or not nn else f"{col}_nn_{nn}" for col in groupon

        ]

        rgrpcols = groupon



        newfeatures = df.merge(

            shifted, left_on=lgrpcols, right_on=rgrpcols, how="left"

        )[[f + "_y" if f in df.columns else f for f in features]]

        newfeatures.columns = features



        colnames = [fcol + f"_lag_{lag}" if lag else fcol for fcol in features]

        if nn:

            colnames = [fcol + f"_nn_{nn}" if nn else fcol for fcol in colnames]



        if dtype is None:

            newdata = newfeatures.values

        else:

            newdata = newfeatures.values.astype(dtype)



        lagcols = pd.concat(

            [lagcols, pd.DataFrame(newdata, columns=colnames, index=lagcols.index)],

            axis=1,

        )



    return lagcols
newfeatures = lagjoin(

    data,

    

    # Try different groupings, e.g. month, item and shop ID.

    groupon=["date_block_num", "item_id"],

    

    # Can also be a list of columns.

    features="item_cnt_month",

    

    # A lag of 0 lets us get data for similar items in the same month.

    lags=[0, 1],

    

    # A nearest neighbour of 0 lets us get data for the same item in previous months.

    nns=[0, 1, 2, 3],

)



data = pd.concat([data, newfeatures], axis=1)

data