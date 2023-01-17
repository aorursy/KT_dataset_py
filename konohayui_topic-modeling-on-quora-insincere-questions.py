import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import warnings, time, gc

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE

from wordcloud import WordCloud

np.random.seed(32)
color = sns.color_palette("Set2")
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation
output_notebook()

%matplotlib inline

train = pd.read_csv("../input/train.csv")
train.head()
train.isna().sum()
target_count = train["target"].value_counts()

plt.figure(figsize = (8, 5))
ax = sns.barplot(target_count.index, target_count.values)
rects = ax.patches
labels = target_count.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
train["quest_len"] = train["question_text"].apply(lambda x: len(x.split()))
sincere = train[train["target"] == 0]
insincere = train[train["target"] == 1]

plt.figure(figsize = (15, 8))
sns.distplot(sincere["quest_len"], hist = True, label = "sincere")
sns.distplot(insincere["quest_len"], hist = True, label = "insincere")
plt.legend(fontsize = 10)
plt.title("Questions Length Distribution by Class", fontsize = 12)
plt.show()
#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}
# Credit: https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

def clean_text(question):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    question = question.lower()
    #remove \n
    question = re.sub("\\n", "", question)
    #remove disteacting single quotes
    question = re.sub("\'", "", question)
    # remove new line characters
#     question = re.sub('s+', " ", question)
    
    #Split the sentences into words
    words = tokenizer.tokenize(question)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in stop_words and w not in punctuations]

    clean_sent = " ".join(words)
    # remove any non alphanum, digit character
#     clean_sent = re.sub("\W+", " ", clean_sent)
#     clean_sent = re.sub("  ", " ", clean_sent)
    
    return clean_sent
sincere["clean_question_text"] = sincere["question_text"].apply(lambda question: clean_text(question))
insincere["clean_question_text"] = insincere["question_text"].apply(lambda question: clean_text(question))
insincere.head()
cv = CountVectorizer(min_df = 10,
                     max_features = 100000,
                     analyzer = "word",
                     ngram_range = (1, 2),
                     stop_words = "english",
                     token_pattern = '[a-zA-Z]')

count_vectors = cv.fit_transform(insincere["clean_question_text"])
# params = {"n_components": [5, 10, 20, 30, 40, 50]}

# lda_model = LatentDirichletAllocation(n_components = n_topics, 
#                                       # we choose a small n_components for time convenient
#                                       # will find a appropriate n_components later 
#                                       learning_method = "online",
#                                       batch_size = 128,
#                                       evaluate_every = -1,
#                                       max_iter = 20,
#                                       random_state = 32,
#                                       n_jobs = -1)

# model = GridSearchCV(lda_model, param_grid = params)
# model.fit(count_vectors)

# best_lda_model = model.best_estimator_
# best_lda_model
n_topics = 8
lda_model = LatentDirichletAllocation(n_components = n_topics, 
                                      learning_method = "online",
                                      batch_size = 128,
                                      evaluate_every = -1,
                                      max_iter = 20,
                                      random_state = 32,
                                      n_jobs = -1)

question_topics = lda_model.fit_transform(count_vectors)
temp = question_topics
print("Log Likelihood: {} \nPerplexity: {}".format(lda_model.score(count_vectors), 
                                                   lda_model.perplexity(count_vectors)))
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_lda = tsne_model.fit_transform(question_topics)
question_topics = np.matrix(question_topics)
doc_topics = question_topics/question_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(insincere["question_text"]):
    lda_keys += [doc_topics[i].argmax()]
    
tsne_lda_df = pd.DataFrame(tsne_lda, columns = ["x", "y"])
tsne_lda_df["qid"] = insincere["qid"].values
tsne_lda_df["question"] = insincere["question_text"].values
tsne_lda_df["topics"] = lda_keys
tsne_lda_df["topics"] = tsne_lda_df["topics"].map(int)
import random

def generate_color():
    color = "#{:02x}{:02x}{:02x}".format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
colormap = np.array([generate_color() for t in range(n_topics)])
plot_lda = bp.figure(plot_width = 700, plot_height = 600, 
                    title = "LDA topics of Quora Questions",
                    tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                    x_axis_type = None, y_axis_type = None, min_border = 1)

source = ColumnDataSource(data = dict(x = tsne_lda_df["x"], y = tsne_lda_df["y"],
                         color = colormap[lda_keys],
                         qid = tsne_lda_df["qid"],
                         question = tsne_lda_df["question"],
                         topics = tsne_lda_df["topics"]))

plot_lda.scatter(x = "x", y = "y", color = "color", source = source)
hover = plot_lda.select(dict(type = HoverTool))
hover.tooltips = {"qid": "@qid","question": "@question", "topics": "@topics"}
show(plot_lda)
threshold = 0.5
idx = np.amax(temp, axis = 1) >= threshold
question_topics = temp[idx]
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_lda2 = tsne_model.fit_transform(question_topics)
new_insincere = insincere[["qid", "question_text"]].copy()
new_insincere = new_insincere[idx]
question_topics = np.matrix(question_topics)
doc_topics = question_topics/question_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(new_insincere["question_text"]):
    lda_keys += [doc_topics[i].argmax()]
    
tsne_lda_df2 = pd.DataFrame(tsne_lda2, columns = ["x", "y"])
tsne_lda_df2["qid"] = new_insincere["qid"].values
tsne_lda_df2["question"] = new_insincere["question_text"].values
tsne_lda_df2["topics"] = lda_keys
tsne_lda_df2["topics"] = tsne_lda_df2["topics"].map(int)
plot_lda = bp.figure(plot_width = 700, plot_height = 600, 
                    title = "LDA topics of Quora Questions",
                    tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                    x_axis_type = None, y_axis_type = None, min_border = 1)

source = ColumnDataSource(data = dict(x = tsne_lda_df2["x"], y = tsne_lda_df2["y"],
                         color = colormap[lda_keys],
                         qid = tsne_lda_df2["qid"],
                         question = tsne_lda_df2["question"],
                         topics = tsne_lda_df2["topics"]))

plot_lda.scatter(x = "x", y = "y", color = "color", source = source)
hover = plot_lda.select(dict(type = HoverTool))
hover.tooltips = {"qid": "@qid", "question": "@question", "topics": "@topics"}
show(plot_lda)
idx = np.amax(temp, axis = 1) < threshold
question_topics = temp[idx]
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_lda3 = tsne_model.fit_transform(question_topics)
new_insincere2 = insincere[["qid", "question_text"]].copy()
new_insincere2 = new_insincere2[idx]
question_topics = np.matrix(question_topics)
doc_topics = question_topics/question_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(new_insincere2["question_text"]):
    lda_keys += [doc_topics[i].argmax()]
    
tsne_lda_df3 = pd.DataFrame(tsne_lda3, columns = ["x", "y"])
tsne_lda_df3["qid"] = new_insincere2["qid"].values
tsne_lda_df3["question"] = new_insincere2["question_text"].values
tsne_lda_df3["topics"] = lda_keys
tsne_lda_df3["topics"] = tsne_lda_df2["topics"].map(int)
plot_lda = bp.figure(plot_width = 700, plot_height = 600, 
                    title = "LDA topics of Quora Questions",
                    tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                    x_axis_type = None, y_axis_type = None, min_border = 1)

source = ColumnDataSource(data = dict(x = tsne_lda_df3["x"], y = tsne_lda_df3["y"],
                         color = colormap[lda_keys],
                         qid = tsne_lda_df3["qid"],
                         question = tsne_lda_df3["question"],
                         topics = tsne_lda_df3["topics"]))

plot_lda.scatter(x = "x", y = "y", color = "color", source = source)
hover = plot_lda.select(dict(type = HoverTool))
hover.tooltips = {"qid": "@qid", "question": "@question", "topics": "@topics"}
show(plot_lda)
def create_wordcloud(i, data):
#     plt.subplot(int("52{}".format(ax+1)))
    wc =  WordCloud(max_words = 1000, stopwords = stop_words)
    wc.generate(" ".join(data))
    ax[int(i/2)][i%2].axis("off")
    ax[int(i/2)][i%2].set_title("Words Frequented in Topic {}".format(i), fontsize = 15)
    ax[int(i/2)][i%2].imshow(wc)
    
fig, ax = plt.subplots(4, 2, figsize = (25, 25))
for i in range(n_topics):
    text = tsne_lda_df[tsne_lda_df["topics"] == int(i)]["question"]
    create_wordcloud(int(i), text)
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

cor = squareform(pdist(tsne_lda2, metric = "euclidean"))
cor = squareform(pdist(tsne_lda2[:100], metric = "euclidean"))
labels = {}

for l, i in enumerate(tsne_lda_df2["qid"]):
    labels[l] = i
G = nx.Graph()

for i in range(cor.shape[0]):
    for j in range(cor.shape[1]):
        if i == j:
            G.add_edge(i, j, weight = 0)
        else:
            G.add_edge(i, j, weight = 1.0/cor[i, j])
            
G = nx.relabel_nodes(G, labels)
            
edges = [(i, j) for i, j, w in G.edges(data = True) if w["weight"] > 0.8]
edge_weight = dict([((u, v, ), int(d["weight"])) for u, v, d in G.edges(data = True)])

pos = nx.spring_layout(G)

plt.figure(figsize = (10, 8))
nx.draw_networkx_nodes(G, pos, node_size = 100, alpha = 0.5)
nx.draw_networkx_edges(G, pos, edgelist = edges, width = 1)
nx.draw_networkx_labels(G, pos, font_size = 8, font_family = "sans-serif")
plt.show()

