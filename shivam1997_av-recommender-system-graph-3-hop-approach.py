import numpy as np
import pandas as pd
from tqdm import tqdm
def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    actual = list(actual)
    predicted = list(predicted)
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
df = pd.read_csv('../input/av-recommendation-systems/train_mddNHeX/train.csv')
df = df.sort_values(by=['user_id', 'challenge_sequence']).reset_index(drop=True)
df.head()
df_train = df[df['challenge_sequence'] <= 10].reset_index(drop=True)
df_test = df[df['challenge_sequence'] > 10].reset_index(drop=True)
df_train['value'] = 1
df_train_adj = df_train.pivot(index='user_id', columns='challenge', values='value').fillna(0)
%%time
## Hop 2
hop = np.dot(df_train_adj.T, df_train_adj)
np.shape(hop)
%%time
## Hop 3
hop = np.dot(hop, df_train_adj.T).T
np.shape(hop)
%%time
hop_arg_sort = np.fliplr(hop.argsort(axis=1))[:, :13]
hop_arg_sort.shape
tmp = pd.DataFrame(hop_arg_sort, index=df_train_adj.index).apply(lambda x: df_train_adj.columns[x])
tmp = tmp.T[df_test['user_id'].unique()].T
tmp
%%time
challenge_pred = []

for ix in tqdm(range(len(tmp))):
    c_t = df_train['challenge'][df_train['user_id'] == tmp.index[ix]].values
    challenge = [c for c in tmp.iloc[ix] if c not in c_t][:3]
    challenge_pred.append(challenge)
challenge_true = pd.DataFrame(df_test.groupby(by='user_id')['challenge'].agg(list)).T[df_test['user_id'].unique()].T['challenge'].to_list()
np.shape(challenge_true), np.shape(challenge_pred)
mapk(challenge_true, challenge_pred, k=3)
