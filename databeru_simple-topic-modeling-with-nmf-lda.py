import pandas as pd
df = pd.read_csv("/kaggle/input/trump-tweets/realdonaldtrump.csv")
df.head(3)
print(f"Number of tweets: {df.shape[0]}")
# Add a few words to the stop words to avoid websites
from nltk.corpus import stopwords
stop_words = stopwords.words("english")+["http","https","www", "com"]

# Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words, ngram_range=(1,2))
tm = tfidf.fit_transform(df["content"])
# Non-Negative Matrix Factorization (NMF)
# Find two non-negative matrices (W, H) whose product 
# approximates the non- negative matrix X.
from sklearn.decomposition import NMF
nmf = NMF(n_components=10, random_state=0)

# fit the transfomed content with NMF
nmf.fit(tm)

# display the result
for index,topic in enumerate(nmf.components_):
    print(f"The top 20 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print("\n")
# Add a few words to the stop words to avoid websites
from nltk.corpus import stopwords
stop_words = stopwords.words("english")+["http","https","www", "com"]

# Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words, ngram_range=(1,2))
tm = tfidf.fit_transform(df["content"])


from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components = 10, n_jobs = -1, random_state = 0)

# fit the transfomed content with LDA
LDA.fit(tm)

# display the result
for index,topic in enumerate(LDA.components_):
    print(f"The top 20 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print("\n")