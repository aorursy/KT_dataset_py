%matplotlib inline
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectKBest





def corr_matrix_of_important_words(term_doc_mat, word_list, scores, n_features_to_keep):

    selector = SelectKBest(k=n_features_to_keep).fit(term_doc_mat, scores)

    informative_words_index = selector.get_support(indices=True)

    labels = [word_list[i] for i in informative_words_index]

    data = pd.DataFrame(term_doc_mat[:,informative_words_index].todense(), columns=labels)

    data['Score'] = reviews.Score

    return(data.corr())



def heat_map(corrs_mat):

    sns.set(style="white")

    f, ax = plt.subplots(figsize=(11, 9))

    mask = np.zeros_like(corrs_mat, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True 

    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corrs_mat, mask=mask, cmap=cmap, ax=ax)

    



reviews = pd.read_csv('../input/Reviews.csv')

vectorizer = CountVectorizer(max_features = 500, stop_words='english')

term_doc_mat = vectorizer.fit_transform(reviews.Text)

word_list = vectorizer.get_feature_names()



corrs_large = corr_matrix_of_important_words(term_doc_mat, word_list, reviews.Score, 60)

print(corrs_large.Score.sort_values(inplace=False)[:-1])

corrs_small = corr_matrix_of_important_words(term_doc_mat, word_list, reviews.Score, 15)

heat_map(corrs_small)