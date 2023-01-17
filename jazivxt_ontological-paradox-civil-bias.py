import string

import numpy as np

import pandas as pd

from sklearn import *

from nltk.corpus import stopwords

from multiprocessing import Pool, cpu_count

from warnings import filterwarnings as fw; fw("ignore")



train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

train.shape, test.shape, sub.shape
mcol = [train.columns[i] for i in [11, 13, 14, 17, 19, 21, 22, 29, 31]]

train.drop(columns=[c for c in train.columns if c not in ['id', 'comment_text', 'target']+mcol], inplace=True)



for c in ['target']+mcol:

    train[c] = np.where(train[c] >= 0.5, 1, 0)

train['weights'] = train[mcol[0]].astype(str).str.cat(train[mcol[1:]].astype(str), sep='')

train['weights'] = train['weights'].map(lambda x: int(x, 2))
def transform_df(df):

    sw = set(stopwords.words("english"))

    df = pd.DataFrame(df)

    df['np'] = df['comment_text'].map(lambda x: len([c for c in str(x) if c in string.punctuation]))

    df['nu'] = df['comment_text'].map(lambda x: len([w for w in str(x).split(' ') if w.isupper()]))

    df['nt'] = df['comment_text'].map(lambda x: len([w for w in str(x).split(' ') if w.istitle()]))

    df['len'] = df['comment_text'].map(lambda x: len(str(x)))

    df['wc'] = df['comment_text'].map(lambda x: len(str(x).split(' ')))

    df['wcu'] = df['comment_text'].map(lambda x: len(set(str(x).split(' '))))

    df['mwl'] = df['comment_text'].map(lambda x: np.mean([len(w) for w in str(x).split(' ')]))

    df['wcu%'] = df['wcu'] / df['wc']

    df['comment_text'] = df['comment_text'].str.lower()

    df['comment_text'] = df['comment_text'].str.replace('[^a-z ]',' ', regex=True)

    df['comment_text'] = df['comment_text'].str.replace('    ',' ', regex=False)

    df['comment_text'] = df['comment_text'].str.replace('   ',' ', regex=False)

    df['comment_text'] = df['comment_text'].str.replace('  ',' ', regex=False)

    df['sw'] = df['comment_text'].map(lambda x: len([w for w in str(x).split(' ') if w in sw]))

    df['swu'] = df['comment_text'].map(lambda x: len(set([w for w in str(x).split(' ') if w in sw])))

    df['lenc'] = df['comment_text'].map(lambda x: len(str(x)))

    df['wcc'] = df['comment_text'].map(lambda x: len(str(x).split(' ')))

    df['wcuc'] = df['comment_text'].map(lambda x: len(set(str(x).split(' '))))

    df['mwlc'] = df['comment_text'].map(lambda x: np.mean([len(w) for w in str(x).split(' ')]))

    df['wcuc%'] = df['wcuc'] / df['wcc']

    return df



def multi_transform(df):

    p = Pool(cpu_count())

    df = p.map(transform_df, np.array_split(df, cpu_count()))

    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)

    p.close(); p.join()

    return df



train = multi_transform(train)

test = multi_transform(test)
from tqdm import tqdm

tqdm.pandas()



col = ['np', 'nu', 'nt', 'len', 'wc', 'wcu', 'mwl','wcu%', 'sw', 'swu', 'lenc', 'wcc', 'wcuc', 'mwlc', 'wcuc%', 'tscore']

word_toxicity = {}



def get_word_toxicity(s,toxicity):

    global word_toxicity

    for w in str(s).split(' '):

        if w in word_toxicity:

            word_toxicity[w]['Count'] += 1

            word_toxicity[w]['ToxicitySum'] += toxicity

            word_toxicity[w]['Toxicity'] = word_toxicity[w]['ToxicitySum'] / word_toxicity[w]['Count']

        else:

            word_toxicity[w] = {'Count': 1, 'ToxicitySum': toxicity, 'Toxicity':  toxicity}



_ = train.progress_apply(lambda r:  get_word_toxicity(r['comment_text'], r['target']), axis=1)



def score_text1(s):

    global word_toxicity

    score = 0.

    max_toxicity = 0.

    wc = len(str(s).split(' '))

    for w in str(s).split(' '):

        if w in word_toxicity:

            if word_toxicity[w]['Count'] > 1:

                if word_toxicity[w]['Toxicity'] > max_toxicity:

                    max_toxicity = word_toxicity[w]['Toxicity']

                score += word_toxicity[w]['Toxicity']

    return ((score / wc) * 0.9) + (max_toxicity * 0.1) 



train['tscore'] = train.progress_apply(lambda r:  score_text1(r['comment_text']), axis=1)

test['tscore'] = test.progress_apply(lambda r:  score_text1(r['comment_text']), axis=1)
vsize = 70



def load_vectors(path, size=30):

    f = open(path, 'r', encoding='utf-8')

    d = {}

    for l in f:

        w = l.rstrip().split(' ')

        d[w[0]] = list(map(float, w[1:size+1]))

    return d



w2v = load_vectors('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', vsize)



def get_comment_vectors(s, size=30):

    s = str(s).split(' ')

    v = np.zeros(size)

    denom = 0

    for w in s:

        if w in w2v:

            v = np.add(v, w2v[w])

            denom += 1

    if denom > 0:

        v /= denom

    return v



trainx = [get_comment_vectors(s, vsize) for s in train['comment_text'].values]

trainx = pd.DataFrame(trainx, columns=['crawl_300d_2M_vec+' + str(i) for i in range(vsize)])

train = pd.concat((train, trainx), axis=1)



testx = [get_comment_vectors(s, vsize) for s in test['comment_text'].values]

testx = pd.DataFrame(testx, columns=['crawl_300d_2M_vec+' + str(i) for i in range(vsize)])

test = pd.concat((test, testx), axis=1)



col += ['crawl_300d_2M_vec+' + str(i) for i in range(vsize)]
def auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def subgroup_auc(df, subgroup):

    subgroup_examples = df[df[subgroup]]

    return auc(subgroup_examples['target'], subgroup_examples['pred'])



def bpsn_auc(df, subgroup):

    subgroup_negative_examples = df[df[subgroup] & ~df['target']]

    non_subgroup_positive_examples = df[~df[subgroup] & df['target']]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return auc(examples['target'], examples['pred'])



def bnsp_auc(df, subgroup):

    subgroup_positive_examples = df[df[subgroup] & df['target']]

    non_subgroup_negative_examples = df[~df[subgroup] & ~df['target']]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return auc(examples['target'], examples['pred'])



def bm_for_model(df):

    global mcol

    records = []

    for subgroup in mcol:

        record = {'subgroup': subgroup, 'subgroup_size': len(df[subgroup])}

        record['subgroup_auc'] = subgroup_auc(df,subgroup)

        record['bpsn_auc'] = bpsn_auc(df, subgroup)

        record['bnsp_auc'] = bnsp_auc(df, subgroup)

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)



def power_mean(series, p=-5):

    total = sum(np.power(series, p))

    return np.power(total / len(series), 1 / p)



def get_final_metric(df):

    global mcol

    for c in ['target'] + mcol:

        df[c] = np.where(df[c] >= 0.5, True, False)

    overall_auc = metrics.roc_auc_score(df['target'], df['pred'])

    df = bm_for_model(df)

    bias_score = np.average([power_mean(df['subgroup_auc']), power_mean(df['bpsn_auc']), power_mean(df['bnsp_auc'])])

    return (overall_auc * 0.25) + ((1 - 0.25) * bias_score)



def lgb_toxic_metric(preds, dtrain):

    global mcol

    labels = dtrain.get_label()

    weights =  dtrain.get_weight()

    df = pd.DataFrame(weights, columns=['weights'])

    df['weights'] = df['weights'].map(lambda x: np.binary_repr(x, width=len(mcol)))

    for i in range(len(mcol)):

        df[mcol[i]] = df['weights'].map(lambda x: int(str(x)[i]))

    df['target'] = labels

    df['pred'] = preds

    score = get_final_metric(df)

    return 'toxic_metric', score, True
import lightgbm as lgb



params = {'learning_rate':0.2, 'max_depth':8, 'objective':'binary', 'metric':'auc', 'num_leaves':32, 'feature_fraction':0.9,'bagging_fraction':0.8, 'bagging_freq':5}



folds = 3

test['prediction'] = 0.0

for fold in range(folds):

    x1, x2, y1, y2, w1, w2 = model_selection.train_test_split(train[col], train['target'], train['weights'], test_size=0.3, random_state=fold+7)

    model = lgb.train(params, lgb.Dataset(x1, label=y1, weight=w1), 100, lgb.Dataset(x2, label=y2, weight=w2), early_stopping_rounds=10,  verbose_eval=2, feval=lgb_toxic_metric)

    test['prediction'] += model.predict(test[col], num_iteration=model.best_iteration)

test['prediction'] = (test['prediction'] / folds).clip(0,1)

test['prediction'] = test['prediction'].map(lambda x: x * 0.05 if x < 0.14 else x * 1.15).clip(0.000001,0.999999) #Blend kernel submission factors ** magic numbers **

test[['id', 'prediction']].to_csv('submission.csv', index=False)