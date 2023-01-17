%pylab inline

import numpy as np

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import wordcloud

from sqlalchemy import create_engine



from sklearn.feature_extraction.text import CountVectorizer



questions = pd.read_csv('../input/questions.csv')
# Cleaning

import re



def clean_statement(x):

    x = re.sub('-', ' ', x)

    x = re.sub('[,|.|?|\n]|\t', '', x)

    x = re.sub('n\'t', ' not', x)

    x = re.sub('All submissions for this problem are available', '', x)

    x = re.sub('Read problems statements in Mandarin Chinese|Russian|as well', '', x)

    x = re.sub('and|Vietnamese', '', x)

    return x



# clean_statement(questions.statement[0])



questions['statement'] = [clean_statement(x) for x in questions.statement.values]

questions.head()
from collections import Counter

Counter(questions.level).most_common()
questions.level.value_counts()
cloud = wordcloud.WordCloud(background_color='black', max_font_size=60, relative_scaling=.5).generate(' '.join(questions.statement))

plt.figure(figsize=(20,10))

plt.axis('off')

plt.imshow(cloud);
sns.countplot(questions.level)
solutions = pd.read_csv('../input/solutions.csv')

solutions.tail()
# make new categories

solutions['StatusGroup'] = solutions.Status.str.split('(').str[0]

g = sns.factorplot("StatusGroup", data=solutions, kind="count")

g.set_xticklabels(rotation=90)
from collections import Counter



# Let's only see those which were either right or wrong.

sol = solutions

sol = sol.loc[(sol.StatusGroup == 'wrong answer')|(sol.StatusGroup == 'accepted')]

Counter(sol.StatusGroup)
g = sns.factorplot("Language", data=sol, kind="count", aspect=1.7)

g.set_xticklabels(rotation=90)
sol['Success'] = (sol.StatusGroup == 'accepted').astype(int)



success_ratios = sol.groupby('UserID')['Success'].mean()

success_ratios.sort()



# plt.plot(success_ratios.values, label='SuccessRatio')

# plt.plot([0, 70000], [0.5, 0.5], '-.', label='Random')

# plt.legend()



# histogram of success ratios

sol['successRatio'] = sol.UserID.map(success_ratios)

sns.distplot(success_ratios)

# most people fall into high failure, high success, or equal failure:success ratio
# create question success ratio



success_ratios = sol.groupby('QCode')['Success'].mean()

success_ratios.sort()



# plt.plot(success_ratios.values, label='SuccessRatio')

# plt.plot([0, 70000], [0.5, 0.5], '-.', label='Random')

# plt.legend()



# histogram of success ratios

sol['question_success_ratio'] = sol.QCode.map(success_ratios)

sns.distplot(success_ratios)

# most people fall into high failure, high success, or equal failure:success ratio
sol.tail()
# Let's create a new table, that joins the question_success_ratio that we computed from solutions table to the questions table.

# Then we can ask questions about level vs question_success_ratio
success_to_level = pd.merge(questions, sol, how='inner', on='QCode', left_on=None, right_on=None,

         left_index=False, right_index=False, sort=True,

         suffixes=('_x', '_y'), copy=True, indicator=False)



success_to_level['statement'] = [clean_statement(x) for x in success_to_level.statement.values]



success_to_level[['level', 'statement','QCode', 'Title', 'question_success_ratio']].tail()
# drop duplicates

success_to_level = success_to_level[['level', 'question_success_ratio', 'statement','QCode', 'Title']].drop_duplicates()

success_to_level.head()
success_to_level.tail()
Counter(success_to_level.level).most_common()
# catenc = pd.factorize(success_to_level['level'])

# df['labels_enc'] = catenc[0]



# map all the level categories to integers ordered by difficulty

success_to_level['level'] = success_to_level['level'].map({'easy': 1, 'medium': 2, 'hard':3, 'beginner': 0, 'challenge':4})



# plot to see if we see any clusters

# plt.scatter(success_to_level.level, success_to_level.question_success_ratio)
success_to_level.tail()
from sklearn.cluster import KMeans

# plt.scatter(success_to_level['level'], success_to_level['question_success_ratio'])
import random



df = success_to_level[['level','question_success_ratio']]

# A = df.loc[start:end, idx[:, 'MSHP Bedrooms', :, :, 'C']]



# divide each value of level by 10, so that the scale is similar and narrower in range

# to that of question success ratio.



# add random noise to spread the visualization on the graph

df['level'] = df.level.apply(lambda x: x/6 + random.randint(1, 14)/100)

df.tail()
X = df.values

km = KMeans(n_clusters=3).fit(X)



Counter(km.labels_)
# these are the coordinates of the 3 centroids

km.cluster_centers_
predictions = km.predict(X)



plt.figure(figsize=(10,7))

plt.scatter(X[:, 0], X[:, 1], c=predictions)

plt.plot(km.cluster_centers_[0][0], km.cluster_centers_[0][1], 'o', label='Centroid 0')

plt.plot(km.cluster_centers_[1][0], km.cluster_centers_[1][1], 'o', label='Centroid 1')

plt.plot(km.cluster_centers_[2][0], km.cluster_centers_[2][1], 'o', label='Centroid 2')



plt.title('K Means Clustering on Level vs Success Ratio', size=20)

plt.xlabel('Question Difficulty Level')

plt.ylabel('Success Ratio from Submitted Solutions')

plt.legend();
# [('easy', 250),

#  ('medium', 230),

#  ('hard', 143),

#  ('beginner', 51),

#  ('challenge', 45)]



# Let's choose k=3 to cluster into easy, medium, hard.  With k-means, we can perhaps see if the success ratio

# for certain questions don't reflect their golden labels.  In other words, we'll be able to see easy questions that

# have 



# df = success_to_level[['question_success_ratio']]

X = success_to_level[['question_success_ratio']].values

km = KMeans(n_clusters=3).fit(X)



Counter(km.labels_)
# these are the coordinates of the 3 centroids

km.cluster_centers_
Counter(km.predict(X))
# we input only the question_success_ratio values into k-means.



df['predictions'] = km.predict(X)

df.tail()
from sklearn.metrics import silhouette_score



def plot_silhouette(data, clusters):

    '''

    Input:

        data - (DataFrame) Data to cluster on

        clusters - (list) List containing the number of clusters to check for

    Output:

        Plot showing the silhouette score for different numbers of centroids

    '''  

    df = data

    X = data.values

    

    vertical_axis = []

    

    for k in clusters:

        within_cluster_sum_of_squares = []

        km = KMeans(n_clusters=k).fit(X)

        centroids = km.cluster_centers_

        

#         for i in range(len(centroids)):

#             within_cluster_sum_of_squares.append(silhouette_score(df, km.labels_))

        

        vertical_axis.append(silhouette_score(df, km.labels_))



    plt.plot(clusters, vertical_axis)
num_clusters = list(range(3, 20, 3))



plot_silhouette(df, num_clusters)
scatter(df['level'], df['question_success_ratio'], alpha=.1)
X = df.values

km = KMeans(n_clusters=6).fit(X)



Counter(km.labels_)
# these are the coordinates of the 6 centroids

km.cluster_centers_
predictions = km.predict(X)



plt.figure(figsize=(10,7))

plt.scatter(X[:, 0], X[:, 1], c=predictions)



for i in range(6):

    plt.plot(km.cluster_centers_[i][0], km.cluster_centers_[i][1], 'o', label='Centroid '+str(i))



    

plt.title('K Means Clustering on Level vs Success Ratio', size=20)

plt.xlabel('Question Difficulty Level')

plt.ylabel('Success Ratio from Submitted Solutions')

plt.legend();
from sklearn.decomposition import PCA, kernel_pca

from sklearn.manifold import TSNE



base = '../input/program_codes/code/'

# # READ Stuff



sols = pd.read_csv('../input/solutions.csv', usecols=['QCode',

                                                      'SolutionID',

                                                      'Status', 'Language']).dropna()

first, second, third = [pd.read_csv(base+i+'.csv') for i in 'first,second,third'.split(',')]

# Merge stuff

code = pd.concat([first, second, third])

df = sols.merge(code, how='left', on='SolutionID')

del(sols);del(code);del(first);del(second);del(third)



df = df.dropna()



# Info

df.info()
Counter(df.Language).most_common()[:10]
# Let's consider solutions in either PYTH or C++11.



# We only use what we need. So we drop a few columns

df = df[['QCode', 'SolutionID','Solutions', 'Status', 'Language']]

df.Language = df.Language.str.split(' ').str[0]

df = df[(df.Language == 'C++11') | (df.Language == 'PYTH')]

df.info()
Counter(df.Language).most_common()
df.tail()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.Solutions, df.Language, test_size=0.4, random_state=0)

print("type of text_train: {}".format(type(X_train)))

print("length of X_train: {}".format(len(X_train)))

list(X_train)[15]  # C++
list(X_train)[17]  # Python
print("Number of documents in test data: {}".format(len(X_test)))
vect = CountVectorizer().fit(X_train)

X_train = vect.transform(X_train)
# apply CountVectorizer for X_test

vect = CountVectorizer().fit(X_test)

X_test = vect.transform(X_test)
print("X_train:\n{}".format(repr(X_train)))
# We can access the vocabulary to return a list 

# where each entry corresponds to one feature.





feature_names = vect.get_feature_names()

print("Number of features: {}".format(len(feature_names)))

print("First 20 features:\n{}".format(feature_names[:20]))

print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))

print("Every 2000th feature:\n{}".format(feature_names[::2000]))
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=10)

print("Mean cross-validation accuracy: {}".format(np.mean(scores)))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid.fit(X_train, y_train)

print("Best cross-validation score: {}".format(grid.best_score_))

print("Best parameters: ", grid.best_params_)
scores = cross_val_score(LogisticRegression(), X_test, y_test, cv=10)

print("Test Data:")

print("Mean cross-validation accuracy: {}".format(np.mean(scores)))
import sklearn

# assert sklearn.__version__ == '0.18' # Make sure we are in the modern age

from sklearn.feature_extraction.text import CountVectorizer



# Here are some good defaults

max_features=1000

max_df=0.95,  

min_df=2,

max_features=1000,

stop_words='english'



from nltk.corpus import stopwords

stop = stopwords.words('english')



# document-term matrix A

vectorized = CountVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')



a = vectorized.fit_transform(questions.statement)

a.shape



# so n is 1474, and m is 1000
from sklearn.decomposition import NMF

model = NMF(init="nndsvd",

            n_components=4,

            max_iter=200)
W = model.fit_transform(a)

H = model.components_



print("W:", W.shape)

print("H:", H.shape)
vectorizer = vectorized



terms = [""] * len(vectorizer.vocabulary_)

for term in vectorizer.vocabulary_.keys():

    terms[vectorizer.vocabulary_[term]] = term

    

# Have a look that some of the terms

terms[-5:]

# terms
import numpy as np



for topic_index in range(H.shape[0]):  # H.shape[0] is k

    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]

    term_ranking = [terms[i] for i in top_indices]

    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))
model = NMF(init="nndsvd",

            n_components=15,

            max_iter=200)



W = model.fit_transform(a)

H = model.components_



vectorizer = vectorized



terms = [""] * len(vectorizer.vocabulary_)

for term in vectorizer.vocabulary_.keys():

    terms[vectorizer.vocabulary_[term]] = term



for topic_index in range(H.shape[0]):  # H.shape[0] is k

    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]

    term_ranking = [terms[i] for i in top_indices]

    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))
from sklearn.feature_extraction.text import TfidfVectorizer



# document-term matrix A

tfidf_vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')



vectorized = tfidf_vectorizer



a = vectorized.fit_transform(questions.statement)



model = NMF(init="nndsvd",

            n_components=15,

            max_iter=200)



W = model.fit_transform(a)

H = model.components_





vectorizer = vectorized



terms = [""] * len(vectorizer.vocabulary_)

for term in vectorizer.vocabulary_.keys():

    terms[vectorizer.vocabulary_[term]] = term



for topic_index in range(H.shape[0]):  # H.shape[0] is k

    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]

    term_ranking = [terms[i] for i in top_indices]

    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))
# this counter counts the number of question staments primarily beloning to each of the above topics.



Counter([np.argmax(i) for i in W]).most_common()
hist([np.argmax(i) for i in W])

plt.title("Counts of Questions per Topic")
from sklearn.decomposition.online_lda import LatentDirichletAllocation



# document-term matrix A

tfidf_vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')

vectorized = tfidf_vectorizer

vectorized = vectorized.fit_transform(questions.statement)



lda = LatentDirichletAllocation(n_topics=4,

                                max_iter=5,

                                learning_method='online',

                                learning_offset=50.,

                                random_state=42)



lda.fit(vectorized), lda.components_.shape
def print_top_words(model, feature_names, n_top_words=15):

    for topic_idx, topic in enumerate(model.components_):

        print("Topic #%d:" % topic_idx, end = "")

        print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()



print("Topics in LDA model:")



# print(type(tfidf_vectorizer))

# print(type(vectorized))



vectorizer = tfidf_vectorizer



tf_feature_names = vectorizer.get_feature_names()  # TfidfVectorizer

print_top_words(lda, tf_feature_names)
lda = LatentDirichletAllocation(n_topics=15,

                                max_iter=5,

                                learning_method='online',

                                learning_offset=50.,

                                random_state=42)



lda.fit(vectorized)





print("Topics in LDA model:")



# print(type(tfidf_vectorizer))

# print(type(vectorized))



vectorizer = tfidf_vectorizer



tf_feature_names = vectorizer.get_feature_names()  # TfidfVectorizer

print_top_words(lda, tf_feature_names)
# exceeds kaggle kernel memory limit

#vectorizer = TfidfVectorizer(max_df=0.95,  

#                            min_df=2,

#                            max_features=1000,

#                            stop_words='english')

#

#vectorized = vectorizer.fit_transform(df['Solutions'].values.astype('U'))  ## Even astype(str) would work
#model = NMF(init="nndsvd",

#            n_components=15,

#            max_iter=200)

#W = model.fit_transform(vectorized)

#H = model.components_

#

#for topic_index in range(H.shape[0]):

#    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]

#    term_ranking = [terms[i] for i in top_indices]

#    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))
# Here are the results:



#Topic 0: google_conversion_id, got, present, recognition, size, light, y1, similarly, 01, 05

#Topic 1: got, called, gives, colored, spaces, determine, having, units, moving, level

#Topic 2: rectangles, got, present, size, light, recognition, www, y1, st, _gaq

#Topic 3: submit, special, removed, girl, light, got, replace, present, www, similarly

#Topic 4: codechef, 23, recognition, inside, base, associated, st, constraints, ak, 02

#Topic 5: aim, list, got, aj, match, scores, element, cell, read, ajax

#Topic 6: able, ri, constraints, www, inside, fibonacci, max, st, middle, base

#Topic 7: ingredients, password, paths, list, wo, _gaq, coordinates, winning, longest, localtime

#Topic 8: occurs, 01, containing, task, dist, operations, launched, ri, constraints, time

#Topic 9: described, present, inside, ri, constraints, y1, element, 50, 45, contests

#Topic 10: straight, 01, y1, light, johnny, size, 19, max, given, 02

#Topic 11: lacs, constraints1, gives, scores, just, previous, got, ri, ki, middle

#Topic 12: lineconstraints1, small, girl, distinct, constraints, got, ri, position, known, write

#Topic 13: acm, recognition, similar, johnny, inside, constraints, st, goes, id, list

#Topic 14: table, element, scores, previous, 01, got, spend, taken, results, consecutive



from sklearn.model_selection import train_test_split



# Create train/test split with labels

train_data, test_data, train_target, test_target = train_test_split(questions.statement,

                                                                    questions.level,

                                                                    test_size=.2,

                                                                    random_state=10)





# Transform train data from a list of strings into a matrix of frequency counts

vectorizer_count = CountVectorizer()



# fit the instance to the training data

vectorized_count_train_data = vectorizer_count.fit_transform(train_data)



print("There are {:,} words in the vocabulary.".format(len(vectorizer_count.vocabulary_)))

print("'{}' appears {:,} times.".format('tree', vectorizer_count.vocabulary_['tree']))



from sklearn.naive_bayes import MultinomialNB



# Create an instance of the Naive Bayes class 

clf = MultinomialNB()

# Call fit method

clf.fit(vectorized_count_train_data, train_target)



# Score the test data with your NB model

accuracy = clf.score(vectorizer_count.transform(test_data), test_target)



print("The accuracy on the test data is {:.2%}".format(accuracy))
from sklearn.feature_extraction.text import TfidfVectorizer



# Transform train data from a list of strings into a matrix of frequency counts

vectorizer_tf_idf = TfidfVectorizer()
vectorized_train_data = vectorizer_tf_idf.fit_transform(train_data)



# Create an instance of the Naive Bayes class 

clf_tf_idf = MultinomialNB()

# Call fit method

clf_tf_idf.fit(vectorized_train_data, train_target)
print("The accuracy on the test data is {:.2%}".format(clf.score(vectorizer_tf_idf.transform(test_data), test_target)))
from sklearn.metrics import precision_score

from sklearn.metrics import confusion_matrix



# [('medium', 507),

#  ('easy', 478),

#  ('hard', 290),

#  ('challenge', 100),

#  ('beginner', 99)]



cm_train = confusion_matrix(train_target,

                            clf_tf_idf.predict(vectorizer_tf_idf.transform(train_data)), labels = ['easy', 'medium'])

cm_train
cm_test = confusion_matrix(test_target,

                           clf_tf_idf.predict(vectorizer_tf_idf.transform(test_data)),

                           labels = ['easy', 'medium'])

cm_test
# Thus in binary classification, the count of true negatives is

# :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is

# :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.



# true negatives is quadrant II, true positibes is quadrant I

print('true negatives: ', cm_test[0][0])

print('false negatives: ', cm_test[1][0])

print('true positives: ', cm_test[1][1])

print('false positives: ', cm_test[0][1])
import matplotlib.pyplot as plt

import numpy as np



def show_confusion_matrix(C,class_labels=['0','1']):

    """

    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function

    class_labels: list of strings, default simply labels 0 and 1.



    Draws confusion matrix with associated metrics.

    Source: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

    """

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    

    # true negative, false positive, etc...

    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];



    NP = fn+tp # Num positive examples

    NN = tn+fp # Num negative examples

    N  = NP+NN



    fig = plt.figure(figsize=(8,8))

    ax  = fig.add_subplot(111)

    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)



    # Draw the grid boxes

    ax.set_xlim(-0.5,2.5)

    ax.set_ylim(2.5,-0.5)

    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)

    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)

    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)

    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)



    # Set xlabels

    ax.set_xlabel('Predicted Label', fontsize=16)

    ax.set_xticks([0,1,2])

    ax.set_xticklabels(class_labels + [''])

    ax.xaxis.set_label_position('top')

    ax.xaxis.tick_top()

    # These coordinate might require some tinkering. Ditto for y, below.

    ax.xaxis.set_label_coords(0.34,1.06)



    # Set ylabels

    ax.set_ylabel('True Label', fontsize=16, rotation=90)

    ax.set_yticklabels(class_labels + [''],rotation=90)

    ax.set_yticks([0,1,2])

    ax.yaxis.set_label_coords(-0.09,0.65)

    

    # Fill in initial metrics: tp, tn, etc...

    ax.text(0,0,

            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,1,

            'False Neg: %d'%fn,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,0,

            'False Pos: %d'%fp,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))

    

    ax.text(1,1,

            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    # Fill in secondary metrics: accuracy, true pos rate, etc...

    ax.text(2,0,

            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,1,

            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,2,

            'Accuracy: %.2f'%((tp+tn+0.)/N),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,2,

            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,2,

            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))

    

    plt.tight_layout()

    plt.show()

    

def show_confusion_matrix(C,class_labels=['0','1']):

    """

    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function

    class_labels: list of strings, default simply labels 0 and 1.



    Draws confusion matrix with associated metrics.

    Source: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

    """

    import matplotlib.pyplot as plt

    import numpy as np

    

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    

    # true negative, false positive, etc...

    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];



    NP = fn+tp # Num positive examples

    NN = tn+fp # Num negative examples

    N  = NP+NN



    fig = plt.figure(figsize=(8,8))

    ax  = fig.add_subplot(111)

    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)



    # Draw the grid boxes

    ax.set_xlim(-0.5,2.5)

    ax.set_ylim(2.5,-0.5)

    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)

    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)

    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)

    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)



    # Set xlabels

    ax.set_xlabel('Predicted Label', fontsize=16)

    ax.set_xticks([0,1,2])

    ax.set_xticklabels(class_labels + [''])

    ax.xaxis.set_label_position('top')

    ax.xaxis.tick_top()

    # These coordinate might require some tinkering. Ditto for y, below.

    ax.xaxis.set_label_coords(0.34,1.06)



    # Set ylabels

    ax.set_ylabel('True Label', fontsize=16, rotation=90)

    ax.set_yticklabels(class_labels + [''],rotation=90)

    ax.set_yticks([0,1,2])

    ax.yaxis.set_label_coords(-0.09,0.65)



    # Fill in initial metrics: tp, tn, etc...

    ax.text(0,0,

            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,1,

            'False Neg: %d'%fn,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,0,

            'False Pos: %d'%fp,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))





    ax.text(1,1,

            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    # Fill in secondary metrics: accuracy, true pos rate, etc...

    ax.text(2,0,

            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,1,

            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,2,

            'Accuracy: %.2f'%((tp+tn+0.)/N),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,2,

            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,2,

            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))

    

    plt.tight_layout()

    plt.show()
show_confusion_matrix(cm_test,

                     class_labels=["easy", "medium"])
# this considers all the classifiers



from sklearn.model_selection import KFold



# Conduct k-fold cross validation

k_fold = KFold(n_splits=10,

               random_state=42)



accuracy_by_fold = []



data = questions.statement

target = questions.level



for train_index, test_index in k_fold.split(data):

    data_train, data_test = np.array(data)[train_index], np.array(data)[test_index]

    target_train, target_test = np.array(target)[train_index], np.array(target)[test_index]

    vectorized_train_data = vectorizer_count.fit_transform(data_train)

    clf.fit(vectorized_train_data, target_train)

    accuracy = clf.score(vectorizer_count.transform(data_test), target_test)

    accuracy_by_fold.append(accuracy)



accuracy_by_fold
from sklearn.cross_validation import cross_val_score



cm_train
from statistics import mean

print("The average accuracy on the test data is {:.2%}".format(mean(accuracy_by_fold)))
X_test = vectorizer_count.transform(test_data)

y_predict = clf.predict(X_test)
tn, fp, fn, tp = cm_train.ravel()

precision = tp/(tp+fp)

recall = tp/(tp+fn)



f1_score = (2*precision*recall) / (precision+recall)

print("train data f1_score: ", f1_score)  # on train data
accuracy_train = (tp+tn)/(tp+tn+tp+fn)



# train data
tn, fp, fn, tp = cm_test.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



f1_score = (2*precision*recall) / (precision+recall)

print("test data f1_score", f1_score)  # on test data
accuracy = (tp+tn)/(tp+tn+tp+fn)

print("train data accuracy: ", accuracy_train)

print("test data accuracy: ", accuracy)  # test data
from textblob import TextBlob

# Fit TextBlob to the question statements.

testimonial = TextBlob(" ".join(questions.statement))



testimonial.sentiment, testimonial.sentiment.polarity
# as a point of reference, let's see what the sentiment and subjectivity is on a toy string



testimonial = TextBlob("have a great day")



testimonial.sentiment, testimonial.sentiment.polarity
# for less time spent training, let's consider only solutions written in Python.

sol = sol[(sol.Language == 'PYTH')]



# prepare the MemTaken column

sol["memory_space"] = sol.MemTaken.apply(lambda x: x.replace("M", ""))

# sol.tail()

# sol["MemTaken"] = [float(sol["MemTaken"][i].replace("M", "")) for i in range(len(sol.MemTaken))]



sol.tail()
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

# kernel SVM on the dataset by using the MinMaxScaler for preprocessing



# there aren't enough features to try SVM like this.

# Let's make predictions on the question difficulty based off success_ratios from the submitted solutions.

# X = success_to_level.question_success_ratio

# y = success_to_level.level



# use these features instead

# We could try to predict Success based on TimeTaken, MemTaken, and question_success_ratio

X = sol[['TimeTaken', 'memory_space', 'question_success_ratio']]

y = sol['Success']



# load and split the data

X_train, X_test, y_train, y_test = train_test_split(X, y)



# compute minimum and maximum on the training data

scaler = MinMaxScaler().fit(X_train)



# rescale the training data

X_train_scaled = scaler.transform(X_train)



svm = SVC()

# # learn an SVM on the scaled training data

svm.fit(X_train_scaled, y_train)

# scale the test data and score the scaled data

X_test_scaled = scaler.transform(X_test)



print("SVM")

print("Train score: {:.2f}".format(svm.score(X_train_scaled, y_train)))

print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# let's check the counts of success labels to determine baseline and reason about class imbalance:



Counter(sol.Success)
print(X_train.shape)

print(X_test.shape)
# rescale the training data

X_train_scaled = scaler.transform(X_train)



# print dataset properties before and after scaling

print("transformed shape: {}".format(X_train_scaled.shape))

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))

print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))

print("per-feature minimum after scaling:\n {}".format(

X_train_scaled.min(axis=0)))

print("per-feature maximum after scaling:\n {}".format(

X_train_scaled.max(axis=0)))
# transform test data

X_test_scaled = scaler.transform(X_test)

# print test data properties after scaling

print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))

print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
svm = SVC()



# # learning an SVM on the scaled training data



svm.fit(X_train_scaled, y_train)



X_scaled_d = scaler.fit_transform(X)
# scale the test data and score the scaled data

X_test_scaled = scaler.transform(X_test)

print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# preprocessing using zero mean and unit variance scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



# learning an SVM on the scaled training data

svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set

print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# We could try to predict Success based on TimeTaken, MemTaken, and question_success_ratio



X = sol[['TimeTaken', 'memory_space', 'question_success_ratio']]

y = sol['Success']



#from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import train_test_split

# load and split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=5, random_state=2)

forest.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))