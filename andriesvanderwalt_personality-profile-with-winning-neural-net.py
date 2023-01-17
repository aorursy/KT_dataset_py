# Basic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import pickle
from sklearn.externals import joblib

# NLP imports
import nltk

# SK-Learn
# NLP and preprocessing
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             TfidfTransformer,
                                             CountVectorizer)

# Classifiers and ML
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import (FunctionTransformer,
                                   StandardScaler,
                                   label_binarize)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_val_score)
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier,
                              BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.preprocessing import DenseTransformer

# Show plots within notebook
%matplotlib inline

# Set plotting style
sns.set_style('white')

train = pd.read_csv('../input/mbti_1.csv')

all_personalities = train.copy()
all_personalities.info()

all_viz = all_personalities.copy()

all_viz['mind'] = [word[0] for word in all_viz['type']]
all_viz['energy'] = [word[1] for word in all_viz['type']]
all_viz['nature'] = [word[2] for word in all_viz['type']]
all_viz['tactics'] = [word[3] for word in all_viz['type']]

count_person = all_viz.groupby('type').count().sort_values('posts',
                                                           ascending=False)

f, ax = plt.subplots(figsize=(12, 8))
count_person['posts'].plot(kind='bar', edgecolor='black',
                           linewidth=1.2, width=0.8)
ax.set_xlabel('Personalities')
ax.set_ylabel('Number of people')
ax.set_title('Number of posts per personality type');

def count_person(df, type, one, two):
    """Count the number of individuals with each of the type personality
    type at each of the four axes

    Parameters:
    -----------
    df -- Input datafram containing posts and personality types
    type -- The personality axis to be assessed
    one -- The first personality in this axis
    two -- The second personality in this axis

    """
    one_count = 0
    two_count = 0
    for i in df[type]:
        if i == one:
            one_count += 1
        else:
            two_count += 1

    return one_count, two_count

i_count, e_count = count_person(all_viz, 'mind', 'I', 'E')
n_count, s_count = count_person(all_viz, 'energy', 'N', 'S')
f_count, t_count = count_person(all_viz, 'nature', 'F', 'T')
j_count, p_count = count_person(all_viz, 'tactics', 'J', 'P')

personality_axes = ['mind', 'energy', 'nature', 'tactics',
                    'mind_l', 'energy_l', 'nature_l', 'tactics_l']

count_axes = pd.DataFrame([[i_count, e_count], [n_count, s_count],
                           [f_count, t_count], [j_count, p_count],
                           ['I', 'E'], ['N', 'S'],
                           ['F', 'T'], ['J', 'P']],
                          index=personality_axes)
count_axes = count_axes.T
count_axes

plt.rcParams.update({'font.size': 14})

f, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
plt.tight_layout(pad=3)
cols = ['skyblue', 'sandybrown']

ax[0, 0].bar(count_axes['mind_l'], count_axes['mind'],
            edgecolor='black', linewidth=1.2, width=0.8, color=cols)
ax[0, 0].set_xlabel('Mind')
ax[0, 0].set_ylabel('Number of people')
ax[0, 0].set_title('Number of posts for all four personality axes')

ax[0, 1].bar(count_axes['energy_l'], count_axes['energy'],
            edgecolor='black', linewidth=1.2, width=0.8, color=cols)
ax[0, 1].set_xlabel('Energy')
ax[0, 1].set_ylabel('Number of people')

ax[1, 0].bar(count_axes['nature_l'], count_axes['nature'],
            edgecolor='black', linewidth=1.2, width=0.8, color=cols)
ax[1, 0].set_xlabel('Nature')
ax[1, 0].set_ylabel('Number of people')

ax[1, 1].bar(count_axes['tactics_l'], count_axes['tactics'],
            edgecolor='black', linewidth=1.2, width=0.8, color=cols)
ax[1, 1].set_xlabel('Tactics')
ax[1, 1].set_ylabel('Number of people');
twt_tkn = TweetTokenizer()
all_viz['tkn_s'] = all_viz.apply(lambda row: twt_tkn.tokenize(row['posts']),
                                 axis=1)
all_viz['lenght_words'] = all_viz['tkn_s'].apply(len)

f, ax = plt.subplots(figsize=(10, 6))
all_viz['lenght_words'].hist(bins=20, edgecolor='black',
                             linewidth=1.2, grid=False,
                             color='skyblue')
ax.set_xlabel('Number of words')
ax.set_ylabel('Number of persons')
ax.set_title('The number of words posted by each person');

all_viz[['lenght_words', 'type']].hist(bins=20, by='type',
                                       edgecolor='black',
                                       linewidth=1.2,
                                       grid=False,
                                       color='skyblue',
                                       figsize=(10, 10));

all_personalities.columns = ['personality', 'post']

all_personalities.head()

def add_indi_labels(X):
    """Add individual labels for all four personality axes.

    Parameters:
    -----------
    X -- Dataframe containing a column 'type' which contains the Meyers
    Briggs's Type Indicator parsonality in the format 'INFJ'

    Returns:
    -----------
    full_df -- A dataframe which now includes four additional columns for each
    personality axis. These columns are encoded in 0's and 1's, corresponding
    to the labels of the new columns, with 1 indicating that the sample has
    said label and 0 indicating the complement personalit at that asix.
    """

    full_df = X
    full_df['I'] = full_df['type'].apply(lambda x: x[0] == 'I').astype('int')
    full_df['N'] = full_df['type'].apply(lambda x: x[1] == 'N').astype('int')
    full_df['F'] = full_df['type'].apply(lambda x: x[2] == 'F').astype('int')
    full_df['J'] = full_df['type'].apply(lambda x: x[3] == 'J').astype('int')

    return full_df


def subsample(X, pers_axis):
    """Create individual dataframes for each personlaity axis, and subsample
    to the personality which is the lowest in that axis.

    Parameters:
    -----------
    X -- Dataframe containing a column 'personality' which contains the Meyers
    Briggs's Type Indicator parsonality in the format 'INFJ'
    pers_axis -- String with the personality axis of which the the new
    dataframe should be made, options are 'mind', 'energy', 'nature', 'tactics'

    Returns:
    -----------
    subsampled_df -- A Dataframe specific to a personality axis, which contains
    only the posts per individual and a column corresponding to the personality
    axis specified in pers_axis. The labels in the personality axis have been
    binerized, where 1's indicate identity with label and 0's indicate the
    other personality.
    """

    f_df = X.copy()
    f_df['mind'] = [word[0] for word in f_df.loc[:, 'personality']]
    f_df['energy'] = [word[1] for word in f_df.loc[:, 'personality']]
    f_df['nature'] = [word[2] for word in f_df.loc[:, 'personality']]
    f_df['tactics'] = [word[3] for word in f_df.loc[:, 'personality']]

    min_c = f_df.groupby(pers_axis).count().min()
    max_t = f_df.groupby(pers_axis).count().idxmax()
    min_t = f_df.groupby(pers_axis).count().idxmin()

    max_df = f_df[f_df.loc[:, pers_axis] == max_t[0]].sample(int(min_c[0]))
    min_df = f_df[f_df.loc[:, pers_axis] == min_t[0]]
    new_df = pd.concat([max_df, min_df]).sample(frac=1)

    X = new_df.loc[:, 'post']
    y = new_df.loc[:, pers_axis]

    labels = label_binarize(y, classes=[str(min_t[0]), str(max_t[0])])
    code_lbs = np.ravel(labels)
    X_df = pd.DataFrame(X).reset_index(drop=True)
    subsampled_df = X_df.join(pd.DataFrame(code_lbs, columns=[max_t[0]]))
    return subsampled_df

def save_obj(obj, name):
    """Using pickle, save objects from python.

    Save obj as a pk1 file in the ./obj/ directory within the current
    directory. Create the obj directory before using this function.

    Parameters:
    -----------
    obj -- Object to be saved.
    name -- Name which object is to be saved as.
    """

    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Using pickle, load objects to python.

    Load obj from a pk1 file in the ./obj/ directory within the current
    directory.This function specifically loads object created using the
    save_obj function.

    Parameters:
    -----------
    name -- Name of object to be loaded, without its extention.

    Returns:
    -----------
    Python object previously 'pickled'.
    """

    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

df_mind = subsample(all_personalities, 'mind')
df_nature = subsample(all_personalities, 'nature')
df_energy = subsample(all_personalities, 'energy')
df_tactics = subsample(all_personalities, 'tactics')
train_labeled = []
train_labeled = add_indi_labels(train)
df_mind.head()
def preprocess(stringss):
    """Preprocess text for Natural Language Processing.

    This function preprocesses text data for Natural Language processing
    by removing punctuation, except emotive punctuation ('!', '?').

    Parameters:
    -----------
    stringss -- Input string which is to be processed.

    Returns:
    -----------
    String of which the punctuation has been removed.
    """

    import string
    keep_punc = ['?', '!']
    punctuation = [str(i) for i in string.punctuation]
    punctuation = [punc for punc in punctuation if punc not in keep_punc]
    s = ''.join([punc for punc in stringss if punc not in punctuation])
    return s


def make_stopwords(other_stopwords, remove_from):
    """Make a list of stopwords to be removed during Natural Language Processing.

    Function which creates a list of words commonly used in the English
    language, which can be supplemented with more specified (other_stopwords)
    words, or from which specified words (remove_from) are removed.

    Parameters:
    -----------
    other_stopwords -- Additional stopwords to be included in the list of
    stopwords.
    remove_from -- Words to be excluded from the list of stopwords

    Returns:
    -----------
    stopwords_punc_personality -- A list of stopwords.
    """

    from nltk.corpus import stopwords
    stopw_all = stopwords.words('english')
    stopwords_punc = [word for word in stopw_all if word not in remove_from]
    stopwords_punc_personality = other_stopwords + stopwords_punc
    return stopwords_punc_personality

personalities = ['infj', 'intj', 'isfj', 'istj',
                 'infp', 'intp', 'isfp', 'istp',
                 'enfj', 'entj', 'esfj', 'estj',
                 'enfp', 'entp', 'esfp', 'estp', '...']
empty_list = []
stopwords_all = make_stopwords(personalities, empty_list)

stops = ['...']
words_to_keep = stopwords_all[0:52]
stopwords_no_pro = make_stopwords(stops, words_to_keep)

stopwords_pers_no_pro = make_stopwords(personalities, words_to_keep)

def tokenize(string, token=TweetTokenizer(), lemma=WordNetLemmatizer()):
    """Tokenize sentences into words.

    Create word tokens from strings. This function uses the TweetTokwnizer by
    default to tokenize words from sentences, as well as lemmatizing the words
    and changing full url's to their domains (e.g., change
    https://www.youtube.com/watch?v=TjJUqWXK68Y will return www.youtube.com).

    Parameters:
    -----------
    string -- Input string which is to be processed.
    tokenzier -- Tokenizer to be used in creating tokens,
                 default=TweetTokenizer()
    lemmatizer -- Lemmatizer to be used in lemmatization
                  default=WordNetLemmatizer()

    Returns:
    -----------
    Returns a string of words which have been tokenized, lemmatized and urls
    which has been removed.
    """

    token = TweetTokenizer()
    lemma = WordNetLemmatizer()

    tokens = token.tokenize(string)
    toks = []
    for i in tokens:
        lemmas = lemma.lemmatize(i)
        toks.append(lemmas)

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    websites = []
    for i in toks:
        websites.append(re.findall(pattern_url, i))
    websites2 = [x for x in websites if x]
    websites3 = []
    for i in websites2:
        if str(i).split('/')[2] not in websites3:
            websites3.append(str(i).split('/')[2])

    for idx, i in enumerate(toks):
        for a in websites3:
            if str(a) in i:
                toks[idx] = a

    df = ' '.join(toks)

    return df

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(df_mind['post'],
                                                  df_mind['I'])
X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(df_energy['post'],
                                                  df_energy['N'])
X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(df_nature['post'],
                                                  df_nature['F'])
X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(df_tactics['post'],
                                                  df_tactics['P'])

train_tests = {'mind': [X_tr_m, X_te_m, y_tr_m, y_te_m],
                'energy': [X_tr_e, X_te_e, y_tr_e, y_te_e],
                'nature': [X_tr_n, X_te_n, y_tr_n, y_te_n],
                'tactics': [X_tr_t, X_te_t, y_tr_t, y_te_t]}

vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=stopwords_no_pro,
                             tokenizer=tokenize)

pipeline = Pipeline([
    ('features', vectorizer),
    ('classify', LogisticRegression())
])

scores = {}
for k, i in train_tests.items():
    pipeline.fit(i[0], i[2])
    scores[str(k) + '_score'] = pipeline.score(i[0], i[2])
    predict = pipeline.predict(i[1])
    predict_train = pipeline.predict(i[0])
    scores[str(k) + '_confusion'] = metrics.confusion_matrix(i[3], predict)
    scores[str(k) + '_accuracy_train'] = metrics.accuracy_score(i[2],
                                                                predict_train)
    scores[str(k) + '_accuracy_test'] = metrics.accuracy_score(i[3],
                                                               predict)
    scores[str(k) + '_prec_test'] = metrics.precision_score(i[3],
                                                            predict,
                                                            average='weighted')
    scores[str(k) + '_recall_test'] = metrics.recall_score(i[3],
                                                           predict,
                                                           average='weighted')
    scores[str(k) + '_f1_train'] = metrics.f1_score(i[2],
                                                    predict_train,
                                                    average='weighted')
    scores[str(k) + '_f1_test'] = metrics.f1_score(i[3],
                                                   predict,
                                                   average='weighted')

accuracy = pd.DataFrame([scores['mind_accuracy_test'],
                         scores['energy_accuracy_test'],
                         scores['nature_accuracy_test'],
                         scores['tactics_accuracy_test']],
                        columns=['accuracy'])
axes = pd.DataFrame(['Mind', 'Energy', 'Nature', 'Tactics'],
                    columns=['type'])
accuracy_b = accuracy.join(axes)

f1 = pd.DataFrame([scores['mind_f1_test'],
                   scores['energy_f1_test'],
                   scores['nature_f1_test'],
                   scores['tactics_f1_test']],
                  columns=['accuracy'])
axes = pd.DataFrame(['Mind', 'Energy', 'Nature', 'Tactics'],
                    columns=['type'])
f1_b = f1.join(axes)

f, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)

ax[0].bar(accuracy_b['type'], accuracy_b['accuracy'],
          color=['steelblue', 'khaki', 'mediumseagreen', 'darksalmon'])
ax[0].set_title('Accuracy of base classification model')
ax[1].bar(f1_b['type'], f1_b['accuracy'],
          color=['dodgerblue', 'gold', 'forestgreen', 'tomato'])
ax[1].set_title('F1 score of base classification model');

def grid_search_p_model(grid, trains_tests):
    """Perform a gridsearch on multiple binary axes.

    This function performs multiple gridsearches on each axis specified in
    the train_tests sets. For each of the axes a gridsearch will be performed
    on the grid instance specified. Following the gridsearch, the models
    are assessed for the parameters, best score, accuracy, recall, precision
    and the f1 score. This all is output into a dictionary, with a key
    specifying the metric and for which model.

    Parameters:
    -----------
    grid -- GridSearchCV instance created with specific parameters to be
    searched
    trains_tests -- dictionary of which the keys are an axes for binary
    classification, and the values are test train splits along that axis.

    Returns:
    -----------
    Dictionary object where keys indicate the axis along with various metrics.
    The metrics assessed are:
        -> Model with best parameters
        -> Accuracy score of the best model
        -> The cross validation results
        -> The train accuracy score
        -> The confusion matrix
        -> Test accuracy score
        -> Precision
        -> Recall
        -> F1 Test score
        -> F1 Train score

    """
    s = {}
    for k, i in trains_tests.items():
        grid.fit(i[0], i[2])
        s[str(k) + '_best_params'] = grid.best_params_
        s[str(k) + '_best_score'] = grid.best_score_
        s[str(k) + '_grid_results'] = grid.cv_results_
        s[str(k) + '_score'] = grid.score(i[0], i[2])
        predict = grid.predict(i[1])
        predict_train = grid.predict(i[0])
        s[str(k) + '_confusion'] = metrics.confusion_matrix(i[3],
                                                            predict)
        s[str(k) + '_accuracy_train'] = metrics.accuracy_score(i[2],
                                                               predict_train)
        s[str(k) + '_accuracy_test'] = metrics.accuracy_score(i[3],
                                                              predict)
        s[str(k) + '_prc_test'] = metrics.precision_score(i[3],
                                                          predict,
                                                          average='weighted')
        s[str(k) + '_recall_test'] = metrics.recall_score(i[3],
                                                          predict,
                                                          average='weighted')
        s[str(k) + '_f1_train'] = metrics.f1_score(i[2],
                                                   predict_train,
                                                   average='weighted')
        s[str(k) + '_f1_test'] = metrics.f1_score(i[3],
                                                  predict,
                                                  average='weighted')
    return s

param_grid = {'features__stop_words': [stopwords_all,
                                       stopwords_no_pro,
                                       stopwords_pers_no_pro,
                                       None],
              'features__tokenizer': [tokenize, None],
              'features__preprocessor': [preprocess, None]}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline, param_grid, refit='accuracy', verbose=4,
                    n_jobs=-1, scoring=scoring, return_train_score=True)

scr_tkn = grid_search_p_model(grid, train_tests)
scr_tkn['mind_best_params']
pipeline = Pipeline([
    ('features', TfidfVectorizer(preprocessor=preprocess,
                                 stop_words=stopwords_all)),
    ('classify', LogisticRegression())
])

param_grid = {'features__min_df': [0.0, 0.1, 0.2, 0.3],
              'features__max_features': [1000, 5000,
                                         50000, 100000],
              'features__ngram_range': [(1, 1), (1, 2), (1, 3)]}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline, param_grid, refit='accuracy',
                    verbose=4, n_jobs=-1,
                    scoring=scoring, return_train_score=True)

scr_vec = grid_search_p_model(grid, train_tests)

print('The mind axis:', scr_vec['mind_best_params'])
print('The energy axis:', scr_vec['energy_best_params'])
print('The nature axis:', scr_vec['nature_best_params'])
print('The tactics axis:', scr_vec['tactics_best_params'])

vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words=stopwords_all)

pipeline_classif = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', LogisticRegression())
])

param_grid = {'classify':[
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    MLPClassifier(alpha=1, max_iter=2000)
]}
scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_classif, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)

scr_bsc = grid_search_p_model(grid, train_tests)

accu = scr_bsc['mind_grid_results']['mean_test_accuracy']
names = []
for i in scr_bsc['mind_grid_results']['param_classify']:
    names.append(str(i)[:7])

accu_b = pd.DataFrame([accu, names], index=['score', 'clf']).T

time = scr_bsc['mind_grid_results']['mean_fit_time']

time_b = pd.DataFrame([time, names], index=['score', 'clf']).T

f, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=1)

ax.bar(accu_b['clf'], accu_b['score'],
       color=['gold', 'limegreen', 'tan', 'darkorange',
              'seagreen', 'darkcyan', 'royalblue', 'mediumpurple',
              'palevioletred', 'orchid', 'crimson'])
ax.set_title('Accuracy of base classification model')
ax.tick_params(rotation=90)

f, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=1)

ax.bar(time_b['clf'], time_b['score'],
       color=['gold', 'limegreen', 'tan', 'darkorange',
              'seagreen', 'darkcyan', 'royalblue', 'mediumpurple',
              'palevioletred', 'orchid', 'crimson'])
ax.set_title('Time taken to fit base model')
ax.tick_params(rotation=90)

vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=stopwords_no_pro)

pipeline_logis = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', LogisticRegression())
])

penalty = ['l1', 'l2']
C = np.logspace(0, 3, 4)
solver = ['liblinear', 'saga']

param_grid = {'classify__C': C,
              'classify__penalty': penalty,
              'classify__solver': solver,
              'features__max_features': [1001, 10000, 50000]}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_logis, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)
scr_log = grid_search_p_model(grid, train_tests)
print('Mind test F1 score:', scr_log['mind_f1_test'])
print('Energy test F1 score:', scr_log['energy_f1_test'])
print('Nature test F1 score:', scr_log['nature_f1_test'])
print('Tactics test F1 score:', scr_log['tactics_f1_test'])

vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words=stopwords_all)

pipeline_knn = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', KNeighborsClassifier())
])

neighbours = [2, 5, 10, 50, 100]

param_grid = {'classify__n_neighbors': neighbours}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_knn, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)

scores_knn = grid_search_p_model(grid, train_tests)
print('Mind test F1 score:', scores_knn['mind_f1_test'])
print('Energy test F1 score:', scores_knn['energy_f1_test'])
print('Nature test F1 score:', scores_knn['nature_f1_test'])
print('Tactics test F1 score:', scores_knn['tactics_f1_test'])
accu = scores_knn['mind_grid_results']['mean_test_accuracy']
neigh = list(scores_knn['mind_grid_results']['param_classify__n_neighbors'])

accu_b = pd.DataFrame([accu, neigh], index=['score', 'neigh']).T

f, ax = plt.subplots(figsize=(8, 6))

plt.plot(accu_b['neigh'], accu_b['score'])
ax.set_title('Accuracy at various number of neighbors')
ax.set_xlabel('Neighbors')
ax.set_ylabel('Test Accuracy');

vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words=stopwords_all)

pipeline_svm = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', SVC())
])

kernel = ['linear',  'rbf']
C = np.logspace(-1, 2, 4)
gamma = np.logspace(-4, 0, 4)

param_grid = {'classify__kernel': kernel,
              'classify__C': C,
              'classify__gamma': gamma}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_svm, param_grid, refit='accuracy', verbose=4,
                   n_jobs=4, scoring=scoring, return_train_score=True)

scores_svm = grid_search_p_model(grid, train_tests)


print('Mind test F1 score:', scores_svm['mind_f1_test'])
print('Energy test F1 score:', scores_svm['energy_f1_test'])
print('Nature test F1 score:', scores_svm['nature_f1_test'])
print('Tactics test F1 score:', scores_svm['tactics_f1_test'])

vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words=stopwords_all)

pipeline_lda = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', LinearDiscriminantAnalysis())
])

solver = ['svd', 'lsqr']
n_components = [2, 50, 100, 500, None]

param_grid = {'classify__solver': solver,
              'classify__n_components': n_components}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_lda, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)

scores_lda = grid_search_p_model(grid, train_tests)


print('Mind test F1 score:', scores_lda['mind_f1_test'])
print('Energy test F1 score:', scores_lda['energy_f1_test'])
print('Nature test F1 score:', scores_lda['nature_f1_test'])
print('Tactics test F1 score:', scores_lda['tactics_f1_test'])

vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words=stopwords_all)

pipeline_nn = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', MLPClassifier(max_iter=2000))
])

layers = [(1000,), (500,), (100, )]
alpha = np.logspace(-4, 0, 5)
activation = ['logistic', 'relu']

param_grid = {'classify__hidden_layer_sizes': layers,
              'classify__alpha': alpha,
              'classify__activation': activation}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_nn, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)

scores_nn = grid_search_p_model(grid, train_tests)

print('Mind test F1 score:', scores_nn['mind_f1_test'])
print('Energy test F1 score:', scores_nn['energy_f1_test'])
print('Nature test F1 score:', scores_nn['nature_f1_test'])
print('Tactics test F1 score:', scores_nn['tactics_f1_test'])

vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=stopwords_no_pro)

pipeline_rf = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD(n_components=1000)),
    ('classify', RandomForestClassifier())
])

n_estimators = [100, 250, 500, 1000]
max_features = ['auto', 'log2', 'sqrt']
criterion = ["gini", "entropy"]
bootstrap = [True, False]
max_depth = [3, 5, None]

param_grid = {'classify__n_estimators': n_estimators,
              'classify__max_features': max_features,
              'classify__criterion': criterion,
              'classify__bootstrap': bootstrap,
              'classify__max_depth': max_depth}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_rf, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)
scores_rnf = grid_search_p_model(grid, train_tests)

print('Mind test F1 score:', scores_rnf['mind_f1_test'])
print('Energy test F1 score:', scores_rnf['energy_f1_test'])
print('Nature test F1 score:', scores_rnf['nature_f1_test'])
print('Tactics test F1 score:', scores_rnf['tactics_f1_test'])

accu_m = [scr_log['mind_f1_test'], scores_knn['mind_f1_test'],
          scores_lda['mind_f1_test'], scores_svm['mind_f1_test'],
          scores_nn['mind_f1_test'], scores_rnf['mind_f1_test']]

accu_e = [scr_log['energy_f1_test'], scores_knn['energy_f1_test'],
          scores_lda['energy_f1_test'], scores_svm['energy_f1_test'],
          scores_nn['energy_f1_test'], scores_rnf['energy_f1_test']]

accu_n = [scr_log['nature_f1_test'], scores_knn['nature_f1_test'],
          scores_lda['nature_f1_test'], scores_svm['nature_f1_test'],
          scores_nn['nature_f1_test'], scores_rnf['nature_f1_test']]

accu_t = [scr_log['tactics_f1_test'], scores_knn['tactics_f1_test'],
          scores_lda['tactics_f1_test'], scores_svm['tactics_f1_test'],
          scores_nn['tactics_f1_test'], scores_rnf['tactics_f1_test']]

names = ['Logistic', 'KNN', 'LDA', 'SVM', 'NeuralNetw', 'RandomF']

accu_mind = pd.DataFrame([accu_m, names], index=['score', 'clf']).T
accu_energy = pd.DataFrame([accu_e, names], index=['score', 'clf']).T
accu_nature = pd.DataFrame([accu_n, names], index=['score', 'clf']).T
accu_tactics = pd.DataFrame([accu_t, names], index=['score', 'clf']).T

f, ax = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
plt.tight_layout(pad=3, h_pad=8)
cols = ['gold', 'limegreen', 'darkorange',
        'darkcyan', 'royalblue', 'mediumpurple']
ax[0, 0].bar(accu_mind['clf'], accu_mind['score'], color=cols)
ax[0, 0].set_title('F1 test score of best mind classification models')
ax[0, 0].tick_params(rotation=90)

ax[0, 1].bar(accu_energy['clf'], accu_energy['score'], color=cols)
ax[0, 1].set_title('F1 test score of best energy classification models')
ax[0, 1].tick_params(rotation=90)

ax[1, 0].bar(accu_nature['clf'], accu_nature['score'], color=cols)
ax[1, 0].set_title('F1 test score of best nature classification models')
ax[1, 0].tick_params(rotation=90)

ax[1, 1].bar(accu_tactics['clf'], accu_tactics['score'], color=cols)
ax[1, 1].set_title('F1 test score of best tactics classification models')
ax[1, 1].tick_params(rotation=90);

vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=stopwords_no_pro)

pipeline_svd = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD()),
    ('classify', LogisticRegression(penalty='l1', C=1000))
])

num_components = [50, 100, 500, 1000, 5000]

param_grid = {
    'svd__n_components': num_components
}
scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_svd, param_grid, refit='accuracy', verbose=4,
                    n_jobs=-1, scoring=scoring, return_train_score=True)

scores_svd = grid_search_p_model(grid, train_tests)

print('Mind test F1 score:', scores_svd['mind_f1_test'])
print('Energy test F1 score:', scores_svd['energy_f1_test'])
print('Nature test F1 score:', scores_svd['nature_f1_test'])
print('Tactics test F1 score:', scores_svd['tactics_f1_test'])

accu = scores_svd['mind_grid_results']['mean_test_accuracy']
neigh = list(scores_svd['mind_grid_results']['param_svd__n_components'])

accu_b = pd.DataFrame([accu, neigh], index=['score', 'neigh']).T

f, ax = plt.subplots(figsize=(8, 6))

plt.plot(accu_b['neigh'], accu_b['score'])
ax.set_title('Accuracy at various number of components')
ax.set_xlabel('Components')
ax.set_ylabel('Test Accuracy');

vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=stopwords_no_pro)
classifier = LogisticRegression(penalty='l1', C=1000)

pipeline_svd = Pipeline([
    ('features', vectorizer),
    ('densify', DenseTransformer()),
    ('svd', TruncatedSVD()),
    ('classify', AdaBoostClassifier())
])

booster = [GradientBoostingClassifier(),
           AdaBoostClassifier(classifier),
           BaggingClassifier(classifier)]
n_estimators = [50, 100, 500, 1000]
param_grid = {
    'classify': booster,
    'classify__n_estimators': n_estimators
}
scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
grid = GridSearchCV(pipeline_svd, param_grid, refit='accuracy', verbose=4,
                    n_jobs=4, scoring=scoring, return_train_score=True)

scores_ensm = grid_search_p_model(grid, train_tests)

print('Mind test F1 score:', scores_ensm['mind_f1_test'])
print('Energy test F1 score:', scores_ensm['energy_f1_test'])
print('Nature test F1 score:', scores_ensm['nature_f1_test'])
print('Tactics test F1 score:', scores_ensm['tactics_f1_test'])

train_labeled.head()
X_nn = train_labeled['posts']
y_nn = train_labeled[['I', 'N', 'F', 'J']]

X_train, X_test, y_train, y_test = train_test_split(X_nn, y_nn)
pipeline = Pipeline([ 
    ('vectorize', TfidfVectorizer(stop_words=stopwords_no_pro)), 
    ('svd', TruncatedSVD(n_components=500)),
    ('scaler', StandardScaler()),
    ('classify', MLPClassifier(max_iter=2000))
])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

predict = pipeline.predict(X_test)
print('The classification scores: ','\n',
      metrics.classification_report(y_test, predict))

print('This is the mind accuracy score of my model',
      metrics.accuracy_score(y_test['I'], pd.DataFrame(predict)[0]))
print('This is the energy accuracy score of my model',
      metrics.accuracy_score(y_test['N'], pd.DataFrame(predict)[1]))
print('This is the nature accuracy score of my model',
      metrics.accuracy_score(y_test['F'], pd.DataFrame(predict)[2]))
print('This is the tactics accuracy score of my model',
      metrics.accuracy_score(y_test['J'], pd.DataFrame(predict)[3]))

param_grid = {
              'classify__hidden_layer_sizes': [(5000,), (1000,),
                                               (500,), (250, )],
              'classify__alpha': [1e-1, 1e-3, 1e-5, 1e-7],
              'classify__activation': ['logistic', 'relu'],
              'svd__n_components': [250, 500, 1000]}

grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=4, n_jobs=4)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_score_

grid.score(X_test, y_test)

pipeline = Pipeline([
    ('vectorize', TfidfVectorizer(stop_words=stopwords_no_pro)),
    ('svd', TruncatedSVD(n_components=250)),
    ('scaler', StandardScaler()),
    ('classify', MLPClassifier(max_iter=2000, activation='logistic',
                               alpha=1e-05, hidden_layer_sizes=(1000,)))
])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

predict = pd.DataFrame(pipeline.predict(X_test))

scr = pd.DataFrame([[metrics.accuracy_score(y_test['I'], predict[0]),
                     metrics.accuracy_score(y_test['N'], predict[1]),
                     metrics.accuracy_score(y_test['F'], predict[2]),
                     metrics.accuracy_score(y_test['J'], predict[3])],
                    [metrics.matthews_corrcoef(y_test['I'], predict[0]),
                     metrics.matthews_corrcoef(y_test['N'], predict[1]),
                     metrics.matthews_corrcoef(y_test['F'], predict[2]),
                     metrics.matthews_corrcoef(y_test['J'], predict[3])],
                    [metrics.f1_score(y_test['I'], predict[0]),
                     metrics.f1_score(y_test['N'], predict[1]),
                     metrics.f1_score(y_test['F'], predict[2]),
                     metrics.f1_score(y_test['J'], predict[3])],
                    [metrics.recall_score(y_test['I'], predict[0]),
                     metrics.recall_score(y_test['N'], predict[1]),
                     metrics.recall_score(y_test['F'], predict[2]),
                     metrics.recall_score(y_test['J'], predict[3])],
                    [metrics.precision_score(y_test['I'], predict[0]),
                     metrics.precision_score(y_test['N'], predict[1]),
                     metrics.precision_score(y_test['F'], predict[2]),
                     metrics.precision_score(y_test['J'], predict[3])]],
                   index=['Accuracy', 'Matthews', 'F1',
                          'Recall', 'Precision'],
                   columns=['Mind', 'Energy', 'Nature', 'Tactics']).T

scr
accu_m = [scr_log['mind_f1_test'], scores_knn['mind_f1_test'],
          scores_lda['mind_f1_test'], scores_svm['mind_f1_test'],
          scores_nn['mind_f1_test'], scores_rnf['mind_f1_test'],
          metrics.f1_score(y_test['I'], predict[0])]

accu_e = [scr_log['energy_f1_test'], scores_knn['energy_f1_test'],
          scores_lda['energy_f1_test'], scores_svm['energy_f1_test'],
          scores_nn['energy_f1_test'], scores_rnf['energy_f1_test'],
          metrics.f1_score(y_test['N'], predict[1])]

accu_n = [scr_log['nature_f1_test'], scores_knn['nature_f1_test'],
          scores_lda['nature_f1_test'], scores_svm['nature_f1_test'],
          scores_nn['nature_f1_test'], scores_rnf['nature_f1_test'],
          metrics.f1_score(y_test['F'], predict[2])]

accu_t = [scr_log['tactics_f1_test'], scores_knn['tactics_f1_test'],
          scores_lda['tactics_f1_test'], scores_svm['tactics_f1_test'],
          scores_nn['tactics_f1_test'], scores_rnf['tactics_f1_test'],
          metrics.f1_score(y_test['J'], predict[3])]

names = ['Logistic', 'KNN', 'LDA', 'SVM',
         'NeuralNetw', 'RandomF', 'MultilabelNN']

accu_mind = pd.DataFrame([accu_m, names], index=['score', 'clf']).T
accu_energy = pd.DataFrame([accu_e, names], index=['score', 'clf']).T
accu_nature = pd.DataFrame([accu_n, names], index=['score', 'clf']).T
accu_tactics = pd.DataFrame([accu_t, names], index=['score', 'clf']).T


f, ax = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
plt.tight_layout(pad=3, h_pad=8)
cols = ['gold', 'limegreen', 'darkorange',
        'darkcyan', 'royalblue', 'mediumpurple', 'crimson']
ax[0, 0].bar(accu_mind['clf'], accu_mind['score'], color=cols)
ax[0, 0].set_title('F1 test score of best mind classification models')
ax[0, 0].tick_params(rotation=90)

ax[0, 1].bar(accu_energy['clf'], accu_energy['score'], color=cols)
ax[0, 1].set_title('F1 test score of best energy classification models')
ax[0, 1].tick_params(rotation=90)

ax[1, 0].bar(accu_nature['clf'], accu_nature['score'], color=cols)
ax[1, 0].set_title('F1 test score of best nature classification models')
ax[1, 0].tick_params(rotation=90)

ax[1, 1].bar(accu_tactics['clf'], accu_tactics['score'], color=cols)
ax[1, 1].set_title('F1 test score of best tactics classification models')
ax[1, 1].tick_params(rotation=90);

