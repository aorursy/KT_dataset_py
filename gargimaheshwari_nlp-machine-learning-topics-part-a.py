import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re



papers = pd.read_csv('../input/papers.csv')

papers.head()
papers = papers.drop(["id", "event_type", "pdf_name"], axis = 1)

papers.head()
groups = papers.groupby('year')

counts = groups.size()



plt.figure(figsize = (13, 8))

ax = sns.barplot(counts.index, counts.values, palette=("magma"))

ax.set_xlabel("Year",fontsize = 15, labelpad = 15)

plt.xticks(rotation = 90)

plt.show()
display(papers['title'].head())



papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))

papers['title_processed'] = papers['title_processed'].map(str.lower)



display(papers['title_processed'].head())
from sklearn.feature_extraction.text import CountVectorizer



def plot_10_most_common_words(count_data, count_vectorizer):

    words = count_vectorizer.get_feature_names()

    total_counts = np.zeros(len(words))

    for t in count_data:

        total_counts += t.toarray()[0]

    

    count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)

    words = [w[0] for w in count_dict[0:10]]

    counts = [w[1] for w in count_dict[0:10]]

    x_pos = np.arange(len(words))



    sns.barplot(x_pos, counts, palette=("plasma"))

    plt.xticks(x_pos, words, rotation = 90) 

    plt.xlabel('words', fontsize = 13)

    plt.ylabel('counts', fontsize = 13)

    plt.title('10 most common words', fontsize = 15)

    plt.show()

    

    return dict(count_dict)



count_vectorizer = CountVectorizer(stop_words = 'english')

count_data = count_vectorizer.fit_transform(papers['title_processed'])



plt.figure(figsize = (13, 8))

count_dict = plot_10_most_common_words(count_data, count_vectorizer)
data_dense = count_data.todense()

print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import LatentDirichletAllocation



import warnings

warnings.simplefilter("ignore", FutureWarning)



search_params = {'n_components': [20, 25, 30, 35], 'max_iter': [10, 20, 50]}

lda = LatentDirichletAllocation()

model = GridSearchCV(lda, cv = None, param_grid = search_params)

model.fit(count_data) #Takes a long time to run
best_lda_model = model.best_estimator_



print("Best Model's Params: ", model.best_params_)

print("Best Log Likelihood Score: ", model.best_score_)

print("Model Perplexity: ", best_lda_model.perplexity(count_data))
n_topics = [20, 25, 30, 35]

results = pd.DataFrame(model.cv_results_)

logl_10 = (results['mean_test_score'][i] for i in range(len(results)) if results['params'][i]['max_iter'] == 10)

logl_20 = (results['mean_test_score'][i] for i in range(len(results)) if results['params'][i]['max_iter'] == 20)

logl_50 = (results['mean_test_score'][i] for i in range(len(results)) if results['params'][i]['max_iter'] == 50)



plt.figure(figsize=(13, 8))

sns.lineplot(n_topics, list(logl_10), label='10')

sns.lineplot(n_topics, list(logl_20), label='20')

sns.lineplot(n_topics, list(logl_50), label='50')

plt.title("Choosing Optimal LDA Model", fontsize = 15)

plt.xlabel("Num Topics", fontsize = 13)

plt.ylabel("Log Likelyhood Scores", fontsize = 13)

plt.legend(title='Maximum Iterations', loc = 'best')

plt.show()