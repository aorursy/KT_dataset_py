# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
dftrain = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/train.csv')

dftest = pd.read_csv('/kaggle/input/av-recommendation-systems/test_HLxMpl7/test.csv')

challenge = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')



sub = pd.read_csv('/kaggle/input/av-recommendation-systems/sample_submission_J0OjXLi_DDt3uQN.csv')
df = dftrain.append(dftest,ignore_index=True)

df.shape
dftrain[dftrain['challenge_sequence'].isin([11,12,13])]
train=df.copy()



train['challenge'] = train['challenge'].astype("category")

# train['challenge'].cat.codes

d_challenge = dict(enumerate(train['challenge'].cat.categories))



train['user_id'] = train['user_id'].astype("category")

# train['challenge'].cat.codes

d_users = dict(enumerate(train['user_id'].cat.categories))
# d
train['challenge'] = train['challenge'].cat.codes

train['user_id'] = train['user_id'].cat.codes
# !pip install faiss
import pandas as pd

import scipy.sparse as sparse

import numpy as np

import random

import implicit

from sklearn.preprocessing import MinMaxScaler



sparse_content_person = sparse.csr_matrix((train['challenge_sequence'].astype(float), (train['challenge'], train['user_id'])))

sparse_person_content = sparse.csr_matrix((train['challenge_sequence'].astype(float), (train['user_id'], train['challenge'])))





model = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.3, iterations=500,calculate_training_loss=True) #best

# model = implicit.bpr.BayesianPersonalizedRanking(factors=300, regularization=0.01, iterations=100)

# model = implicit.lmf.LogisticMatrixFactorization(factors=200, regularization=0.2, iterations=50)

# model = implicit.approximate_als.FaissAlternatingLeastSquares(approximate_similar_items=True, approximate_recommend=True, nlist=5502, nprobe=2000, use_gpu=False)





# factors=100, regularization=0.01, dtype=<type 'numpy.float32'>, use_native=True, use_cg=True, use_gpu=False, iterations=15, calculate_training_loss=False, num_threads=0, random_state=None



alpha = 40

# alpha=40 best for als

data = (sparse_content_person * alpha).astype('double')

model.fit(data)
from tqdm import tqdm_notebook as tqdm
# train[train['user_id']==4577]



test_user_ids = train[train['user_sequence'].isin(dftest['user_sequence'])]['user_id'].unique()
# dftrain[dftrain['user_id']==4580].merge(challenge,right_on='challenge_ID',left_on='challenge',how='left')
# dftest[dftest['user_id']==4577].merge(challenge,right_on='challenge_ID',left_on='challenge',how='left')
# challenge[challenge['challenge_ID'].isin(['CI25344','CI25338','CI25345'])]

test_user_ids
dftest[dftest['user_id']==d_users[1]]['challenge'].values
user_challenge=[]

from tqdm import tqdm_notebook as tqdm

for user in tqdm(test_user_ids):

    recs = model.recommend(user,sparse_person_content,N=3,recalculate_user=True)

    for k in range(len(recs)):

        user_challenge.append([str(d_users[user])+'_'+str(k+11),d_challenge[recs[k][0]]])



# user_challenge
# user_challenge
# user_challenge=[]

# from tqdm import tqdm_notebook as tqdm

# for user in tqdm(test_user_ids):

#     recs = model.recommend(user,sparse_person_content,N=10)

    

# #     print([d_challenge[recs[k][0]] for k in range(len(recs)) if d_challenge[recs[k][0]] in dftest[dftest['user_id']==d_users[user]]['challenge'].values])

    

#     user_challenge.append([[str(d_users[user])+'_'+str(k+11),d_challenge[recs[k][0]]] for k in range(len(recs)) if 

#                            d_challenge[recs[k][0]] not in dftest[dftest['user_id']==d_users[user]]['challenge'].values][:3])

# #     for k in range(len(recs)):

# #         if d_challenge[recs[k][0]] not in dftest[dftest['user_id']==d_users[user]]['challenge'].values:

# #             user_challenge.append([str(d_users[user])+'_'+str(k+11),d_challenge[recs[k][0]]])

# #         else:

# #             print(d_users[user])

# #     break

# # user_challenge
# user_challenge
# user_challenge
sub = pd.DataFrame(user_challenge,columns = ['user_sequence','challenge'])

sub
# sub['challenge'].value_counts()
sub.to_csv('av_implicit_kv12_als.csv',index=False)