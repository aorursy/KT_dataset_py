import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
review_data = pd.read_csv('../input/food-review-data/food_reviews.csv')

review_data.head(5)
# review_data = review_data[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Text']]

review_data = review_data.iloc[:, [4,5,6,9]]

review_data.head(5)
helpful_reviews = review_data[review_data['HelpfulnessDenominator'] > 15]

len(helpful_reviews)
print('Number of null values:', np.count_nonzero(helpful_reviews.isnull()))

helpful_reviews = helpful_reviews[helpful_reviews['HelpfulnessNumerator'] >= 0]

helpful_reviews = helpful_reviews[helpful_reviews['HelpfulnessDenominator'] >= 0]

helpful_reviews.shape
helpful_reviews['Text'] = helpful_reviews['Text'].str.lower()

helpful_reviews.head(5)
helpful_reviews['Text'] = helpful_reviews['Text'].str.replace(r"[^(\s\w)/']", " ")

helpful_reviews.head(5)
help_values = helpful_reviews['HelpfulnessNumerator'] / helpful_reviews['HelpfulnessDenominator']



helpfulness = []

for i in np.arange(len(help_values.values)):

    if help_values.values[i] > .50:

        helpfulness.append(1)

    else:

        helpfulness.append(0)

      

helpful_reviews['Helpfulness'] = helpfulness

helpful_reviews.head(5)
num_helpful_reviews = sum(helpful_reviews['Helpfulness'])

print('Number of helpful reviews:', num_helpful_reviews)

print('Number of unhelpful reviews:', len(helpful_reviews['Helpfulness']) - num_helpful_reviews)
from sklearn.feature_extraction.text import TfidfVectorizer



helpful = helpful_reviews.copy()



# tf–idf, short for term frequency–inverse document frequency, is a numerical statistic 

# that is intended to reflect how important a word is to a document in a collection

text_vectorizer = TfidfVectorizer(min_df = 0.1, max_df=0.9, ngram_range=(1, 4), stop_words='english')

text_vectorizer.fit(helpful['Text'])
X_train = text_vectorizer.transform(helpful['Text'])

words = text_vectorizer.get_feature_names()

words
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression



feature_set = X_train

skf = model_selection.StratifiedKFold(n_splits=10)



# Grid Search automatically finds the best parameters for a model

gs = model_selection.GridSearchCV(estimator=LogisticRegression(),

                                  param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced']},

                                  cv = skf.get_n_splits(feature_set, helpful_reviews.Helpfulness),

                                  scoring='roc_auc')



gs.fit(X_train, helpful.Helpfulness)

gs.cv_results_
from sklearn.metrics import roc_auc_score, roc_curve



actuals = gs.predict(feature_set)        # gives prediction                   # accuracy, recall, precision calculations

Y_pred = gs.predict_proba(feature_set)   # gives probability of prediction    # ROC, AUC, RMSE calculations

plt.plot(roc_curve(helpful[['Helpfulness']], Y_pred[:,1])[0], roc_curve(helpful[['Helpfulness']], Y_pred[:,1])[1])

plt.title('ROC Curve');
Y_true = np.array(helpful['Helpfulness'].values)

roc_auc_score(Y_true, Y_pred[:,1].T)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import cluster, metrics

from scipy.spatial.distance import cdist 



helpful_reviews.head(5)
X = helpful_reviews.iloc[:,[0,1,2,4]].values

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = cluster.KMeans(n_clusters=k).fit(X)

    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

  

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
# Vectorizing texts

text_vectorizer = TfidfVectorizer(min_df = 0.05,  # ignore words w/ frequency lower than .05

                                 max_df=0.95,    # ignore words w/ frequency higher than .95

                                 ngram_range=(1, 2),  # include unigrams and bigrams

                                 stop_words='english')

text_vectorizer.fit(helpful_reviews['Text'])
kmeans_model = cluster.KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1,random_state=5).fit(X_train)



X_train = text_vectorizer.transform(helpful_reviews['Text'])    # turn words into vector

words = text_vectorizer.get_feature_names()                     # most useful words

words = np.array(words)  # to index later



# .argsort() returns the indices that would sort an array

sorted_vals = [kmeans_model.cluster_centers_[i].argsort() for i in range(0,np.shape(kmeans_model.cluster_centers_)[0])]



useful = set()

for i in range(len(kmeans_model.cluster_centers_)):

  # uses the indices of the sorted array to grab the last (best) 10 words of each cluster

  useful = useful.union(set(words[sorted_vals[i][-10:]]))    # union so no duplicate items

useful = list(useful)



useful
train = X_train[:,[np.argwhere(words==i)[0][0] for i in useful]]
helpful_reviews['cluster'] = kmeans_model.labels_

counts = helpful_reviews.groupby('cluster').count()

print('Cluster 0:', counts['Text'][0])

print('Cluster 1:', counts['Text'][1])

print('Cluster 2:', counts['Text'][2])

print('Cluster 3:', counts['Text'][3])
helpful_reviews.groupby('cluster').mean()
# adding Score column to top words

import scipy



# scores = helpful_reviews['Score'].values

scores = np.array(list(helpful_reviews['Score']))

scores = scores.reshape(11471, 1)

features = scipy.sparse.hstack((train,scipy.sparse.csr_matrix(scores))) # Stack sparse matrices horizontally (column wise)

features = scipy.sparse.csr_matrix(features) # Compressed Sparse Row matrix
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression



feature_set = features

skf = model_selection.StratifiedKFold(n_splits=10)





# Grid Search automatically finds the best parameters for a model

gs = model_selection.GridSearchCV(

    estimator=LogisticRegression(),

    param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced']},

    cv = skf.get_n_splits(feature_set, helpful_reviews['Helpfulness']),

    scoring='roc_auc')



gs.fit(features, helpful['Helpfulness'])

gs.cv_results_
gs.best_estimator_
# Coefficients represent the log-odds

# The fitted paramters of the best model

display(gs.best_estimator_.coef_,

        gs.best_estimator_.intercept_)
from sklearn.metrics import roc_auc_score, roc_curve



actuals = gs.predict(feature_set)        # gives prediction                   # accuracy, recall, precision calculations

Y_pred = gs.predict_proba(feature_set)   # gives probability of prediction    # ROC, AUC, RMSE calculations

plt.plot(roc_curve(helpful_reviews[['Helpfulness']], Y_pred[:,1])[0], roc_curve(helpful_reviews[['Helpfulness']], Y_pred[:,1])[1])

plt.title('ROC Curve');
Y_true = np.array(helpful['Helpfulness'].values)

roc_auc_score(Y_true, Y_pred[:,1].T)
# View top words and their corresponding coefficients by ascending order of coef

coefs = gs.best_estimator_.coef_[0]

words_coefs = pd.DataFrame({'words': useful[:19], 'coefficients': coefs[:19]})

words_coefs.sort_values(by='coefficients').reset_index().iloc[:,[1,2]]