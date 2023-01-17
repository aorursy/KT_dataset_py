import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
papers = []
for year in [2016, 2017, 2018]:
    df = pd.read_csv("../input/emnlp-{}.csv".format(year))
    papers.append(df)

paper_df = pd.concat(papers).reset_index()
paper_df = paper_df.assign(title=paper_df["title"].apply(str.strip))
paper_df.head(5)
from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(stop_words="english", max_df=0.8, min_df=7, ngram_range=(1, 2))
vectorizer.fit(paper_df["title"])

year_counts = {}
words = vectorizer.get_feature_names()
for y in [2016, 2017, 2018]:
    vectors = vectorizer.transform(paper_df["title"][paper_df["year"] == y])
    counts = vectors.toarray().sum(axis=0)
    word_counts = {}
    for i in range(len(words)):
        word_counts[words[i]] = counts[i]
    year_counts[y] = word_counts

year_count_df = pd.DataFrame(year_counts)
year_count_df.head(5)
def show_rate(year_freq_df, rate_bound=0.003, max_lower_limit=0.5, min_lower_limit=0, figsize=(8, 12)):
    num_columns = year_freq_df.shape[1]
    year_total = year_freq_df.sum(axis=0)
    normalized_count = year_freq_df / year_total
    normalized_count = normalized_count[normalized_count.max(axis=1) > rate_bound]
    word_total = normalized_count.sum(axis=1)
    word_rates = (normalized_count.T / word_total).T
    word_rates = word_rates[word_rates.max(axis=1) > rate_bound]

    first_index = year_freq_df.columns[0]
    word_rates.sort_values(by=first_index, inplace=True)

    limited = word_rates[(word_rates.max(axis=1) > max_lower_limit) & \
                         (word_rates.min(axis=1) > min_lower_limit)]    
    limited.plot.barh(stacked=True, figsize=figsize)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=1, fontsize=12)
    plt.tight_layout()
    return limited
# Show New/Lost Topics
year_count_rate_df = show_rate(year_count_df)
# Show Continuing Topics
year_count_rate_df = show_rate(year_count_df, max_lower_limit=0.0, min_lower_limit=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer_tf = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=7, ngram_range=(1, 2))
vectorizer_tf.fit(paper_df["title"])

words = vectorizer_tf.get_feature_names()
year_tf = {}
for y in [2016, 2017, 2018]:
    vectors = vectorizer_tf.transform(paper_df["title"][paper_df["year"] == y])
    idfs = vectors.toarray().mean(axis=0)
    word_idfs = {}
    for i in range(len(words)):
        word_idfs[words[i]] = idfs[i]
    year_tf[y] = word_idfs

year_tf_df = pd.DataFrame(year_tf)
year_tf_df.head(5)
# Show New/Lost Topics
year_tf_rate_df = show_rate(year_tf_df)
# Show Continuing Topics
year_tf_rate_df = show_rate(year_tf_df, max_lower_limit=0.0, min_lower_limit=0.25)