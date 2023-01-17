import re

import string

from functools import reduce



import numpy as np

import pandas as pd

import transformers

import torch

import plotly.graph_objects as go

from tqdm.notebook import tqdm





pd.set_option("display.max_colwidth", 300)
INPUT_DIR = "../input/tweet-sentiment-extraction"

train = pd.read_csv(f"{INPUT_DIR}/train.csv")

train.head()
train.isnull().sum()
train = train.dropna()



assert train.isnull().sum().eq(0).all()
# Ref: https://www.kaggle.com/parulpandey/basic-preprocessing-and-eda



def clean_text(text):

    """

    Does the following:

    - Make text lowercase

    - Remove text in square brackets

    - Remove links

    - Remove punctuation

    - Remove words containing numbers

    """

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train["clean"] = train["text"].map(clean_text)

train[["text", "clean"]].head()
train["num_words"] = train["clean"].str.split(" ").map(len)

train[["clean", "num_words"]].head()
not_too_short = train["num_words"] >= 10

not_too_short.sum()
raw_texts = train[not_too_short]["text"].sample(n=5000, random_state=42)

clean_texts = train.loc[raw_texts.index]["clean"]

selected_text = train.loc[raw_texts.index]["selected_text"]

sentiment = train.loc[raw_texts.index]["sentiment"]





train.loc[raw_texts.index][["text", "clean", "sentiment"]]
# Ref.: https://www.kaggle.com/abhishek/distilbert-use-features-oof/notebook



def chunks(l, n):

    """

    Yield successive n-sized chunks from l.

    

    Example

    -------

    >>> l = list(range(10))

    >>> for c in chunks(l, 3):

    ...     print(c)

    [0, 1, 2]

    [3, 4, 5]

    [6, 7, 8]

    [9]



    """

    for i in range(0, len(l), n):

        yield l[i:i + n]





def fetch_vectors(string_list, batch_size=64):

    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")

    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")

    model.to(DEVICE)



    fin_features = []

    total = len(string_list) // batch_size + 1

    for data in tqdm(chunks(string_list, batch_size), total=total):

        tokenized = []

        for x in data:

            x = " ".join(x.strip().split()[:300])

            tok = tokenizer.encode(x, add_special_tokens=True)

            tokenized.append(tok[:512])



        max_len = 512

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded).to(DEVICE)

        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        

        with torch.no_grad():

            last_hidden_states = model(input_ids, attention_mask=attention_mask)



        features = last_hidden_states[0][:, 0, :].cpu().numpy()

        fin_features.append(features)



    fin_features = np.vstack(fin_features)

    return fin_features
vectors = fetch_vectors(clean_texts)

vectors.shape
from sklearn.manifold import TSNE



reduced = TSNE(n_components=2).fit_transform(vectors)

reduced.shape
def wrap_text(text):

    """

    Insert <br> to wrap long text on a Plotly chart.



    Example

    -------

    >>> import string

    >>> text = "a b c d e f g h i"

    >>> wrap_text(text, 3)

    "a b c<br>d e f<br>g h i"



    """

    rows = [" ".join(c) for c in chunks(text.split(), 10)]

    return "<br>".join(rows)
hovertext = reduce(lambda a, b: a + "<br>" + b, [

    "# Raw text",

    raw_texts.map(wrap_text),

    "",

    "# Clean text",

    clean_texts.map(wrap_text),

    "",

    "# Selected text",

    selected_text.map(wrap_text),

    "",

    "# Sentiment",

    sentiment,

])



color = sentiment.map({

    "positive": "green",

    "neutral": "#bbbbbb",

    "negative": "red",

})



data = go.Scatter(

    x=reduced[:, 0],

    y=reduced[:, 1],

    mode="markers",

    hoverinfo="text",

    hovertext=hovertext,

    marker=dict(color=color),

)



go.Figure(data=data)