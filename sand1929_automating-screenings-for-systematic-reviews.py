import numpy as np
import pickle
import json
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import comb
import xgboost as xgb
def parse_medline(infile):
    data = []
    types = {'PMID': str, 'TI': str}
    doc = {}
    prev = None
    for line in infile:
        if line == '\n':
            if 'TI' in doc:
                data.append(doc)
            doc = {}
        elif line[4] == '-':
            key = line[:4].strip()
            value = line[5:].strip()
            if key in types:
                doc[key] = types[key](value)
                prev = key
            else:
                prev = None
        else:
            if prev == 'TI':
                doc[prev] += ' ' + line.strip()
    return data
with open('/kaggle/input/cohen-reviews/2004_TREC_ASCII_MEDLINE_1', encoding="ISO-8859-1") as f:
    data = parse_medline(f)
with open('/kaggle/input/cohen-reviews/2004_TREC_ASCII_MEDLINE_2', encoding="ISO-8859-1") as f:
    data.extend(parse_medline(f))
with open('/kaggle/input/cohen-labels/epc-ir.clean.tsv') as f:
    relevance = f.readlines()
judgements = {}
id_to_topic = {}
for line in relevance:
    tokens = line.split()
    pmid = tokens[2]
    judgements[pmid] = tokens[4] == 'I'
    id_to_topic[pmid] = tokens[0]
def get_features(data):
    vectorizer = CountVectorizer(max_df=1.0, ngram_range=(1,1), stop_words='english')
    X_bow = vectorizer.fit_transform([x['TI'] for x in data]).todense()
    cos_sims = cosine_similarity(X_bow, X_bow)

    X_cos = []
    for i in range(X_bow.shape[0]):
        dists = [0] * 24
        for sim in cos_sims[i]:
            for i in range(24):
                if sim > .04 * (i + 1):
                    dists[i] += 1 / X_bow.shape[0]
        X_cos.append(dists)
    X_cos = np.array(X_cos)
    return X_cos

def get_features_per_topic(topics, data):
    Xs = []
    y = []
    for topic in topics: 
        print(topic)
        topic_data = []
        relevant = []
        irrelevant = []

        for x in data:
            if id_to_topic.get(x['PMID']) == topic:
                topic_data.append(x)

        y.extend([judgements[x['PMID']] for x in topic_data])
        Xs.append(get_features(topic_data))
    X = np.concatenate(Xs, axis=0)
    return X, y
# Topics of the reviews from the Cohen dataset.
# We randomly ordered these for cross-validation, and this is the random ordering we generated.
# For the sake of reproducibility, we've recorded the random ordering that yielded the results in Section 5 (Results) here.
all_topics = [
    'OralHypoglycemics',
    'Triptans',
    'NSAIDS',
    'ACEInhibitors',
    'ProtonPumpInhibitors',
    'BetaBlockers',
    'Opiods',
    'ADHD',
    'CalciumChannelBlockers',
    'UrinaryIncontinence',
    'Antihistamines',
    'SkeletalMuscleRelaxants',
    'AtypicalAntipsychotics',
    'Statins',
    'Estrogens'
]
X_folds = []
y_folds = []
for i in range(0, 15, 3):
    X, y = get_features_per_topic(all_topics[i:i+3], data)
    X_folds.append(X)
    y_folds.append(y)
# Returns i_{R95} and WSS@95 as described in Section 4 (Experiments)
def get_wss(ys, preds):
    rankings = [x for _,x in sorted(zip(preds,ys), reverse=True)]
    for i in range(len(rankings)):
        if sum(rankings[:i]) >= sum(rankings) * .95:
            return i, .95 - i / len(rankings)
# The significance test described in Section 4 (Experiments). Returns the p-value
def test_sig(ys, i):
    N = len(ys)
    R = sum(ys)
    r = R * .95
    return comb(N - R, i - r, exact=True) * comb(R, r - 1, exact=True) / comb(N, i - 1, exact=True) * (R - r + 1) / (N - i + 1)
# Experiments on the Cohen dataset
param = {'max_depth': 2, 'eta': 0.3, 'objective': 'binary:logistic', 'scale_pos_weight': 10}
num_round = 20

for fold in range(5):
    X_train = np.concatenate(X_folds[0:fold] + X_folds[fold+1:], axis=0)
    y_train = sum(y_folds[0:fold] + y_folds[fold+1:], [])
    X_test = X_folds[fold]
    y_test = y_folds[fold]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    i, wss = get_wss(y_test, preds)
    print("WSS@95 for fold {}: {}".format(fold, wss))
    print("p-value for fold {}: {}".format(fold, test_sig(y_test, i)))
# Experiments on COVID-19-related reviews
dataset_paths = [
    ['/kaggle/input/litreviewdata/litReviewData/cortegiani-2020-03-10/pubmed.json'],
    ['/kaggle/input/litreviewdata/litReviewData/kapoor-2020-03-30/pubmed.json', '/kaggle/input/litreviewdata/litReviewData/kapoor-2020-03-30/google_scholar.json'],
    ['/kaggle/input/litreviewdata/litReviewData/purssell-2020-01-30/pubmed.json']
]

X_train = np.concatenate(X_folds, axis=0)
y_train = sum(y_folds, [])
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(param, dtrain, num_round)

for idx, paths in enumerate(dataset_paths):
    data = []
    for path in paths:
        with open(path, encoding="ISO-8859-1") as f:
            data.extend(json.load(f))
    y_test = [row['relevant'] for row in data]
    X_test = get_features(data)
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = bst.predict(dtest)
    i, wss = get_wss(y_test, preds)
    print("WSS@95 for review {}: {}".format(idx, wss))
    print("p-value for review {}: {}".format(idx, test_sig(y_test, i)))
with open('/kaggle/input/search-results/animal_models.txt', encoding="ISO-8859-1") as f:
    data = parse_medline(f)
X = xgb.DMatrix(get_features(data))
preds = bst.predict(X)
rankings = [x for _,x in sorted(zip(preds, data), key=lambda x:x[0], reverse=True)]
for i, article in enumerate(rankings):
    print('{}.'.format(i))
    print('PMID: {}'.format(article['PMID']))
    print('Title: {}'.format(article['TI']))
    print()
with open('/kaggle/input/search-results/healthcare.txt', encoding="ISO-8859-1") as f:
    data = parse_medline(f)
X = xgb.DMatrix(get_features(data))
preds = bst.predict(X)
rankings = [x for _,x in sorted(zip(preds, data), key=lambda x:x[0], reverse=True)]
for i, article in enumerate(rankings):
    print('{}.'.format(i))
    print('PMID: {}'.format(article['PMID']))
    print('Title: {}'.format(article['TI']))
    print()
