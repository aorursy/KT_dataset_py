# Import the standard toolkit...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ...and a few NLP specific things
import spacy
from spacy import displacy
from wordcloud import WordCloud

# ...and switch on "in notebook" charts, and make them a bit bigger!
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)

# ...then print a silly message to make it clear we're done
print("Reticulating splines... DONE")
def show_hipster_biz_name(a, b):
    from codecs import decode
    from zlib import adler32
    from IPython.display import HTML
    ns = 'mrcule/zvak/pbj/pbea/cvtrba/funpxyr/obngzna/pbyyne'
    qs = 'zbhfgnpur/nagvdhr/pebpurgrq/negvfnany'
    bs = 'pbzof/glcrjevgref/fyvccref/ancxvaf/jubyrsbbqf'
    def tr(n, c):
        c = decode(c, "rot13").split('/')
        return c[adler32(bytes(n, 'utf8')) % len(c)].title()
    n = "{} & {} {} {}".format(tr(a, ns), tr(b, ns), tr(a+b, qs), tr(b+a, bs))
    s = "font-family:serif;font-size:28pt;text-align:center;border:4px double black;padding:10px;"
    print("Your Hipster business name is:")
    display(HTML("<h1 style='{}'>{}</h1>".format(s, n)))

########################################################################################
# Edit the line below to include your name and your neighbour's name, then run the cell
########################################################################################

show_hipster_biz_name("Your Name", "Your Neighbour's Name")
# Experiment here!
# Load up the english language models... this takes a while!

nlp = spacy.load("en_core_web_lg")
print("{name}: {description}".format(**nlp.meta))
# Okay let's use SpaCy to process a simple sentence
# The fundamental operation is to create a stuctured "Doc" representation of a text. Let's take a look!

text = u"Pack my bag with twelve dozen liquor jugs."
doc = nlp(text)
doc.print_tree()
# But since we're in Jupyter we can do a lot better than that!
# The "Parts of Speech" e.g. VERB are drawn from the "Universal POS Tag" vocabulary
# Find out more at http://universaldependencies.org/u/pos/

options={'jupyter': True, 'options':{'distance': 120}}
displacy.render(doc, style='dep', **options)
# Spacy also ships with an "entity recogniser" -- it's pretty good!

ghostbusters = nlp(u"In the eponymous 1984 film, New York City celebrated the Ghostbusters with a ticker tape parade.")
displacy.render(ghostbusters, style="ent", **options)
# Use this cell to explore!

displacy.render(nlp(u"Explore Spacy here!"), style='dep', **options)
# The larger SpaCy models contain a list of words and their corresponding vectors

print("Document vectors have {} dimensions".format(len(doc.vector)))
print("And are not normalized e.g. this has length {}".format(np.linalg.norm(doc.vector)))
# Document vectors capture an intuitive notion of similarity
# Words that appear similar contexts are considered similar

def print_comparison(a, b):
    # Create the doc objects
    a = nlp(a)
    b = nlp(b)
    # Euclidean "L2" distance
    distance = np.linalg.norm(a.vector - b.vector)
    # Cosine similarity
    similarity = a.similarity(b)
    print("-" * 80)
    print("A: {}\nB: {}\nDistance: {}\nSimilarity: {}".format(a, b, distance, similarity))

text = "The cat sat on the mat"
print_comparison(text, "The feline lay on the carpet")
print_comparison(text, "Three hundred Theban warriors died that day")
print_comparison(text, "Ceci n'est pas une pipe")

# Use this cell to explore!

# Document vectors often also have a very interesting property sometimes called "linear substructure"
# Basically you can do arithmetic with words/concepts!

def vectorize(text):
    """Get the SpaCy vector corresponding to a text"""
    return nlp(text, disable=['parser', 'tagger', 'ner']).vector

from heapq import heappush, nsmallest, nlargest

def get_top_n(target_v, n=5):
    """Figure out the top-N words most similar to the target vector"""
    heap = []
    # SpaCy has a long list of words in `vocab` which we can pick from!
    for word in nlp.vocab:
        # Filter out mixed case and uncommon terms
        if not word.is_lower or word.prob < -15:
            continue
        distance = np.linalg.norm(target_v - word.vector)
        heappush(heap, (distance, word.text))
    return nsmallest(n, heap)


PUPPY, DOG, KITTEN = [vectorize(w) for w in ("puppy", "dog", "kitten")]

get_top_n(DOG - PUPPY + KITTEN)

# We can generalize that into a cute analogy finder

def print_analogy(a, b, c):
    """A is to B as C is to ???"""
    top_n = get_top_n(vectorize(b) - vectorize(a) + vectorize(c))
    best = [w for (s,w) in top_n if w not in (a,b,c)][0]
    print("{} is to {} as {} is to {}".format(a, b, c, best))
    
print_analogy("queen", "king", "woman")

# Use this cell to explore!
from sklearn.datasets import fetch_20newsgroups
raw_posts = fetch_20newsgroups()

print("Number of posts: {}".format(len(raw_posts.data)))
 # Source groups are listed in `target_names`
print("Newsgroups: {}".format(raw_posts.target_names))
 # Post text is in `data`
print("Sample post text:\n{0}\n{1}\n{0}".format('-' * 80, raw_posts.data[19]))
 # Post group is encoded in `target` as an index into `target_names`
print("Sample post group: {}".format(raw_posts.target_names[raw_posts.target[19]]))
# There's quite a lot of junk in there, headers etc. Fortunately sklearn can help a bit...
# We can pass through a special argument to strip out headers, footers and inline quotes

raw_posts = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
print("Sample post text:\n{0}\n{1}\n{0}".format('-' * 80, raw_posts.data[19]))
# Another key tool in the Python ecosystem is Pandas which is a library for working with tables of data
# We're going to convert the dataset in to a Panda DataFrame for ease of manipulation
# Don't worry too much about this -- it's not really our focus today -- but if you're interested you can
# find out more at http://pandas.pydata.org/

posts = pd.DataFrame({'text': raw_posts.data, 'group': [raw_posts.target_names[t] for t in raw_posts.target]})

# Many tools in the Python ecosystem are quite tightly integrated, so once we have DataFrame we can
# do things like plot it via the standard charting tool `matplotlib` which we importes as `plt` earlier
posts['group'].value_counts().plot(kind='bar', title="Per group document counts")
plt.show()
# One way to get a handle on a collection of documents (or "corpus") is to look at a wordcloud
# Thankfully someone has written a little library to help us do that

wc = WordCloud(background_color='white', width=1000, height=400, stopwords=[])
wc.generate(" ".join(t for t in posts[posts.group == 'rec.autos'].text)).to_image()
# Oh dear that wasn't much use... of course common words completely dominate!
# These are called "stopwords". It's common (if a little controversial these days...) to filter them out
# The wordcloud library we're using supports that

from wordcloud import STOPWORDS
better_stopwords = STOPWORDS.union({'may', 'one', 'will', 'also'})
wc = WordCloud(background_color='white', width=1000, height=400, stopwords=better_stopwords)
wc.generate(" ".join(t for t in posts[posts.group == 'rec.autos'].text)).to_image()
# Okay, that's more like it! Let's eyeball all the groups

for group in raw_posts.target_names:
    print("Wordcloud for {}".format(group))
    display(wc.generate(" ".join(t for t in posts[posts.group == group].text)).to_image())
# Looks okay, but comp.os.ms-windows.misc appears to be full of garbage
# Let's cull it (rather crudely...) and take another look

posts = posts[~posts.text.str.contains("AX")]
for group in raw_posts.target_names:
    print("Wordcloud for {}".format(group))
    display(wc.generate(" ".join(t for t in posts[posts.group == group].text)).to_image())
# Use this cell to explore!
# First let's get the documents into a suitable form
# Build a matrix by "stacking" the row vectors from SpaCy
# Takes about 20 seconds...

from sklearn.preprocessing import normalize

def vectorize(text):
    # Get the SpaCy vector -- turning off other processing to speed things up
    return nlp(text, disable=['parser', 'tagger', 'ner']).vector

# Now we stack the vectors and normalize them
# Inputs are typically called "X"
X = normalize(np.stack(vectorize(t) for t in posts.text))
print("X (the document matrix) has shape: {}".format(X.shape))
print("That means it has {} rows and {} columns".format(X.shape[0], X.shape[1]))
# Scikit Learn ships with a neat PCA implementation

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
print("X2 shape is {}".format(X2.shape))
# Okay let's take a look at it via matplotlib

def plot_groups(X, y, groups):
    for group in groups:
        plt.scatter(X[y == group, 0], X[y == group, 1], label=group, alpha=0.4)
    plt.legend()
    plt.show()
    
plot_groups(X2, posts.group, ('comp.os.ms-windows.misc', 'alt.atheism'))
CLUSTERS = 20

# First we fit the model...
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=CLUSTERS, random_state=1)
k_means.fit(X)

# Then we use it to predict clusters for each document...
# Again it's common to use yhat for a predicted value -- although we wouldn't expect these to
# correspond directly to the original groups
yhat = k_means.predict(X)

# Let's take a look at the distribution across classes
plt.hist(yhat, bins=range(CLUSTERS))
plt.show()
# To be honest that's not looking very healthy -- ideally we'd see a more even distribution
# Let's take a look at a couple of the big ones

plot_groups(X2, yhat, (1,14))
# Okay there are some definite (if rather blurry...) clusters there!
# Let's have a look at how our clusters relate to the original groups
def plot_cluster(c):
    posts[yhat == c]['group'].value_counts().plot(kind='bar', title="Cluster #{}".format(c))
    plt.show()

# Some are great matches...
plot_cluster(0)
# Some are not so great a match, but sensible (why...?)
plot_cluster(14)
# Some are just a bit random!
plot_cluster(9)
# Let's have a look at the wordclouds...
for c in range(CLUSTERS):
    print("Wordcloud for category #{}".format(c))
    display(wc.generate(" ".join(posts.text[yhat == c])).to_image())
# Use this cell to explore!