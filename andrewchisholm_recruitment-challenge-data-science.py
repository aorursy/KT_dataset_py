import os
print(os.listdir("../input"))
!head "../input/amazon_cells_labelled.txt"
import os
import random

DATA_PATH = '../input'
LABEL_MAP= {
    1: 'positive',
    0: 'negative'
}

TRAIN_FILES = {'yelp_labelled.txt', 'imdb_labelled.txt'}
TEST_FILES = {'amazon_cells_labelled.txt'}

def iter_instances_at_path(path):
    with open(path, 'rt') as f:
        for i, line in enumerate(f):
            sentence, label = line.split('\t')
            
            # some very basic data norm and validation, break-out if it gets more complicated
            sentence = sentence.strip()
            label = int(label)
            assert sentence
            assert label in LABEL_MAP

            yield {
                'sentence': sentence,
                'label': label,
                'source': {
                    'path': os.path.basename(path),
                    'idx': i
                },
            }

files = TRAIN_FILES.union(TEST_FILES)
items = [i for p in files for i in iter_instances_at_path(os.path.join(DATA_PATH, p))]
print("Read {:d} items from {:d} files...".format(len(items), len(files)))
items[:1]
import spacy

# use spacy for the low-level cookie-cutter text processing (i.e. tokenization)
# we will also make use of word vectors and dependency parse features
nlp = spacy.load('en_core_web_lg')

def preprocess_item(item):
    item['doc'] = nlp(item['sentence'])
    return item
from tqdm import tqdm_notebook as tqdm
items = [preprocess_item(i) for i in tqdm(items)]
items[:1]
from collections import Counter
# it really depends what we count as a "word"
counts = Counter()
for i in items:
    if i['source']['path'] in TRAIN_FILES:
        counts.update(t.text.lower() for t in i['doc'])
print("Most common tokens:")
counts.most_common(10)
counts = Counter()
for i in items:
    if i['source']['path'] in TRAIN_FILES:
        for t in i['doc']:
            if not t.is_stop and not t.is_punct and not t.lemma_ in nlp.Defaults.stop_words:
                counts[t.lemma_] += 1
print("Most common non-stop, non-punctuation, lemmatized tokens:")
counts.most_common(10)
import numpy

# compute the cosine similarity between a pair of vectors
# we will use this for word-vector features
def sim(a, b):
    res = numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
    if numpy.isnan(res):
        return 0.
    return res
DELEX_POS_TAGS = {'PROPN', 'NOUN'}
def delexicalize(doc):
    tks = []
    for t in doc:
        if t.pos_ in DELEX_POS_TAGS:
            if not tks or tks[-1] != t.pos_:
                tks.append(t.pos_)
        else:
            tks.append(t.lemma_)
    return tks
def extract_bag_of_ngrams(item):
    tokens = [t.lemma_.lower() for t in item['doc']]    
    bigrams = ['|'.join(ngram) for ngram in zip(tokens, tokens[1:])]
    for t in set(tokens + bigrams):
        yield t, True

def extract_root_tokens(item):
    for t in item['doc']:
        if t.dep_ == 'ROOT':
            yield t.lemma_, True

def extract_delex_ngrams(item):
    tokens = [t for t in delexicalize(item['doc'])]
    bigrams = ['|'.join(ngram) for ngram in zip(tokens, tokens[1:])]
    for t in set(bigrams):
        yield t, True

from scipy.spatial.distance import cosine as cosine_distance
good = nlp('good').vector
bad = nlp('bad').vector
gv = good - bad

def extract_doc_vect_sim(item):
    yield 'good', sim(item['doc'].vector, good)
    yield 'bad', sim(item['doc'].vector, bad)
    yield 'proj', sim(item['doc'].vector, gv)
    
    token_sims = [sim(t.vector, gv) for t in item['doc']]
    yield 'max(proj)', max(token_sims)
    yield 'min(proj)', min(token_sims)
    yield 'direction', numpy.argmax(token_sims) > numpy.argmin(token_sims)

FEATURES = [
    ('bow', extract_bag_of_ngrams),
    ('root', extract_root_tokens),
    ('vect', extract_doc_vect_sim),
    ('delex', extract_delex_ngrams)
]

def get_features_for_item(item):
    features = {}
    for tag, extractor in FEATURES:
        for key, value in extractor(item):
            features[tag+':'+key] = value
    return features

for i in tqdm(items):
    i['features'] = get_features_for_item(i)
train = [i for i in items if i['source']['path'] in TRAIN_FILES]
test = [i for i in items if i['source']['path'] in TEST_FILES]

random.shuffle(train)
random.shuffle(test)

# in real task we might split off a separate dev-set here for model validation + feature engineering
# e.g:
# dev_split_idx = len(train)//4
# dev, train = train[:dev_split_idx], train[dev_split_idx:]
# print('Dev:', len(dev))

# for the sake of this task, we'll just use cross-val for hyperparams selection and just note
# that we're implicitly p-hacking the held-out eval when feature engineering
print('Train:', len(train))
print('Test', len(test))
def items_to_dataset(items):
    X, Y = [], []
    for i in items:
        X.append(i['features'])
        Y.append(i['label'])
    return X, Y
train_X, train_Y = items_to_dataset(train)
test_X, test_Y = items_to_dataset(test)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

grid = {
    "n_estimators": [25, 100, 200],
    "max_depth": [3, None],
}

clf = Pipeline((
    ('vectorizer', DictVectorizer()),
    ('classifier', GridSearchCV(RandomForestClassifier(), param_grid=grid, cv=5))
))

model = clf.fit(train_X, train_Y)
model.steps[-1][1].best_estimator_
from sklearn.metrics import accuracy_score

y_pred = model.predict(test_X)
print('Accuracy:', accuracy_score(test_Y, y_pred))
# to calculate a bootstrapped ci we just iteratively resample instances
# with replacement then look at the distribution of scores for some metric (i.e. accuracy)
def compute_bootstrapped_ci(y_pred, y, ci, n_samples=10000):
    y, y_pred = numpy.array(y), numpy.array(y_pred)

    scores = []
    for _ in range(n_samples):
        idxs = numpy.random.randint(len(y_pred), size=len(y_pred))
        scores.append(accuracy_score(y[idxs], y_pred[idxs]))

    bounds = (100-ci)/2
    return numpy.percentile(scores, [bounds, 100-bounds])
print("Confidence intervals for ACC:")
print("95%:", compute_bootstrapped_ci(y_pred, test_Y, 95))
print("99%:", compute_bootstrapped_ci(y_pred, test_Y, 99))
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

measures = precision_recall_fscore_support(test_Y, y_pred, average=None)

# let's turn these into a nice table for printing
measure_map = ['precision', 'recall', 'fscore', 'support']
class_measures = defaultdict(dict)
for m, measure in enumerate(measures):
    for c, result in enumerate(measure):
        class_measures[LABEL_MAP[c]][measure_map[m]] = result

# for some feature sets, pos/neg p/r will be unbalanced
print(''.rjust(30), ''.join(m.rjust(10) for m in measure_map))
for c, measures in class_measures.items():
    print(c.rjust(30), ''.join('{:.3f}'.format(measures[m]).rjust(10) for m in measure_map))
%matplotlib inline
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision, recall, _ = precision_recall_curve(test_Y, [p[1] for p in model.predict_proba(test_X)])
plt.figure(figsize=(6, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
# what are the most important features?
selected_clf = model.steps[-1][1].best_estimator_
feature_importances = sorted(zip(selected_clf.feature_importances_, model.steps[0][1].get_feature_names()), reverse=True)
feature_importances[:10]
feature_importance_ranks = {k:i for i, (_, k) in enumerate(feature_importances)}
# for each mistake, we have lots of features but generally only need to see the most important ones to interpret decisions
def get_top_features(features, limit=10):
    return sorted(features.keys(), key=lambda k: feature_importance_ranks.get(k, len(feature_importance_ranks)))[:limit]

mistakes = []
for item, label, probs in zip(test, test_Y, model.predict_proba(test_X)):
    if label != numpy.argmax(probs):
        mistakes.append((numpy.max(probs), LABEL_MAP[item['label']], item['sentence'], get_top_features(item['features'])))
sorted(mistakes, reverse=True)[:10]
