text_path = "../input/an-inquiry-into-the-nature-and-causes-of-wealth/An Inquiry.txt"

raw_text = open(text_path, "r").read()
type(raw_text), len(raw_text)
print(raw_text[0:1000])
text = raw_text[719:-18868]
from nltk.tokenize import word_tokenize
all_tokens = word_tokenize(text)
all_tokens[1:10]
from collections import Counter
token_counts = Counter(all_tokens)

token_counts.most_common(10)
# An example

"work".isalpha(), ",".isalpha()
alpha_tokens = [word for word in all_tokens if word.isalpha()]

len(all_tokens), len(alpha_tokens)
token_counts = Counter(alpha_tokens)

token_counts.most_common(10)
# An example of stopwords in English

from nltk.corpus import stopwords

stopwords.words("english")[10:20]
# May take a little time to run

alpha_tokens = [word.lower() for word in alpha_tokens]

cleaned_tokens = [word for word in alpha_tokens if not word in stopwords.words("english")]



len(cleaned_tokens), len(alpha_tokens)
token_counts = Counter(cleaned_tokens)

token_counts.most_common(10)
%matplotlib inline 

import matplotlib.pyplot as plt



# Configure the way the plots will look

plt.style.use([{

    "figure.dpi": 300,

    "figure.figsize":(12,9),

    "xtick.labelsize": "large",

    "ytick.labelsize": "large",

    "legend.fontsize": "x-large",

    "axes.labelsize": "x-large",

    "axes.titlesize": "xx-large",

    "axes.spines.top": False,

    "axes.spines.right": False,

},'seaborn-poster', "fivethirtyeight"])
from wordcloud import WordCloud

str_cleaned_tokens = " ".join(cleaned_tokens) # the word cloud needs raw text as argument not list

wc = WordCloud(background_color="white", width= 800, height= 400).generate(str_cleaned_tokens)

plt.imshow(wc)

plt.axis("off");
from nltk.tokenize import sent_tokenize, RegexpTokenizer
sentences = sent_tokenize(text)
print(sentences[100])
tokenizer = RegexpTokenizer(r'\w+')



def remove_stopwords(text, stopw = stopwords.words("english")):

    list_of_sentences = []

    

    for sentences in text:

        list_of_words = []

        for word in sentences:

            if not word in stopw:

                list_of_words.append(word)

        list_of_sentences.append(list_of_words)

    return list_of_sentences



def clean_sent(sentences):

    """Sentence must be a list containing string"""

    stopw = stopwords.words("english")

    # Lower each word in each sentence        

    sentences = [tokenizer.tokenize(sent.lower()) for sent in sentences]

    sentences = remove_stopwords(sentences)

    return sentences
cleaned_sentences = clean_sent(sentences)
print(cleaned_sentences[100])
from gensim.models import Word2Vec
model = Word2Vec(

    min_count= 10,# minimum word occurence 

    size = 300, # number of dimensions

    alpha = 0.01, #The initial learning rate

)
model.build_vocab(cleaned_sentences)

model.train(cleaned_sentences, total_examples = model.corpus_count, epochs = 60)
model.wv.most_similar("wealth")
model.wv.most_similar("france")
model.wv.most_similar("africa")
import pandas as pd

similar = pd.DataFrame(model.wv.most_similar("africa", topn= 10), columns = ["name", "height"])
similar.plot.barh(x = "name", y = "height");
model.wv.most_similar("king")
all_words = model.wv.vectors
def wv_to_df(model):

    all_wv = model.wv.vectors

    

    df = pd.DataFrame(

        all_wv,

        index = model.wv.vocab.keys(),

        columns = ["dim" + str(i+1) for i in range(all_wv.shape[1])]

    )

    return df
df = wv_to_df(model)
df.head()
df["idx"] = df.index

df.head()
ax = df.plot.scatter("dim1", "dim2")

for i, point in df.iterrows():

    ax.text(point.dim1 + 0.005, point.dim2 + 0.008, point.idx)
ax = df.plot.scatter("dim10", "dim20")

for i, point in df.iterrows():

    ax.text(point.dim10 + 0.005, point.dim20 + 0.008, point.idx)
def plot_region(df, x, y,label, x_bounds, y_bounds, s=35, ftsize = None):

    slices = df[

        (x_bounds[0] <= df[x]) &

        (df[x] <= x_bounds[1]) & 

        (y_bounds[0] <= df[y]) &

        (df[y] <= y_bounds[1])

    ]

    print(slices.shape)

    ax = slices.plot.scatter(x, y, s=s)

    for i, point in slices.iterrows():

        ax.text(point[x] + 0.005, point[y] + 0.005, point[label], fontsize = ftsize)
plot_region(df, "dim1", "dim2", "idx", (-0.5, 0), (-1, -0.3))
from sklearn.decomposition import PCA
pca_dimension_reduction = PCA(n_components= 2)

res =  pca_dimension_reduction.fit_transform(df.drop(columns = "idx"))
pca_coords = pd.DataFrame(res, columns = ["x", "y"])

pca_coords["words"] = df.index
pca_coords.head()
plot_region(pca_coords, "x", "y", "words", (0, 1), (1, 2))
from sklearn.manifold import TSNE
tsne_dimension_reduction = TSNE(n_components=2)

res = tsne_dimension_reduction.fit_transform(df.drop(columns = "idx"))
res.shape
tsne_coords = pd.DataFrame(res, columns = ["x", "y"])

tsne_coords["words"] = df.index
tsne_coords.plot.scatter(x = "x", y = "y")

plt.title("Book corpus in 2D");
plot_region(tsne_coords, "x", "y", "words", (5, 20), (20, 40))