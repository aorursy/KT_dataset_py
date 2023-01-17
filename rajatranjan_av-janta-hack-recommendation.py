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
train = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/train.csv')

test = pd.read_csv('/kaggle/input/av-recommendation-systems/test_HLxMpl7/test.csv')

challenge = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')



sub = pd.read_csv('/kaggle/input/av-recommendation-systems/sample_submission_J0OjXLi_DDt3uQN.csv')
wide_train = train.pivot_table(index = "user_id", columns="challenge_sequence", values="challenge", aggfunc= lambda x : x).reset_index()

wide_train
wide_train.drop(["user_id"], axis =1, inplace = True)
chal=[]

rows = []

for index, row in wide_train.iterrows():

    r = row.map(str).values

    chal.extend(r)

# rows
wide_test = test.pivot_table(index = "user_id", columns="challenge_sequence", values="challenge", aggfunc= lambda x : x).reset_index()
test_ids = wide_test['user_id']
wide_test.drop(["user_id"], axis =1, inplace = True)

for index, row in wide_test.iterrows():

    r = row.map(str).values

    chal.extend(r)
chal=list(set(chal))

from sklearn.preprocessing import LabelEncoder

l=LabelEncoder()

l.fit(chal)

for col in range(1,14):

    wide_train[col]=l.transform(wide_train[col])

    wide_train[col]=wide_train[col].astype(int)
# wide_train

for col in range(1,11):

    wide_test[col]=l.transform(wide_test[col])

    wide_test[col]=wide_test[col].astype(int)

wide_test.head()
X=wide_train.drop([11,12,13],axis=1)

y=wide_train[[11,12,13]]
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss,mean_squared_log_error



X_trn, X_val, y_trn, y_val = train_test_split(X, y.values, test_size=0.25, random_state=1994)

X_test = wide_test

import sklearn

# sklearn.neighbors.VALID_METRICS
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestNeighbors

from sklearn.multioutput import MultiOutputClassifier,ClassifierChain

from sklearn.linear_model import LogisticRegression

# for metric in ['wminkowski','yule','russellrao','jaccard','dice','manhattan','l2','l1','canberra','cosine','haversine','russellrao']:



# clf = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5,algorithm='auto',weights='distance',leaf_size=100,metric='hamming'))  #hamming best

preds=[]

for mertc in ['hamming','manhattan','l1','canberra']:

    clf = ClassifierChain(KNeighborsClassifier(n_neighbors=5,algorithm='auto',weights='distance',leaf_size=100,metric=mertc),random_state=1994)  #hamming best

    # clf = RandomForestClassifier()  #hamming best

    _ = clf.fit(X,y)

    preds.append(clf.predict(X_test))

#     print(mertc,mapk(y_val,predictions_val_lgb))

from scipy.stats import mode

mode(preds,0)[0][0].shape
pred_modes = mode(preds,0)[0][0].astype(int)
# yule 9.58791156110376e-05

# russellrao 7.670329248883007e-05

# jaccard 7.670329248883007e-05

# dice 7.670329248883007e-05

# manhattan 0.13074395801773125

# l2 failed

# l1 0.13074395801773125

# canberra 0.1301527034714632

# cosine 0.08696874980025185

# haversine failed

# russellrao 7.670329248883007e-05
# clf.fit(X,y)

# pred = clf.predict(X_test)

# pred = pred.astype(int)


for i in range(0,3):

    wide_test[11+i]=l.inverse_transform(pred_modes[:,i])
wide_test['user_id']=test_ids
wide_test[['user_id',11,12,13]]
submission  = pd.melt(wide_test[['user_id',11,12,13]],id_vars="user_id",var_name="sequence", value_name="challenge" )
submission['user_sequence'] = submission['user_id'].map(str)+"_"+submission['sequence'].map(str)

submission[['user_sequence','challenge']].to_csv('AV_recomm_kv10.csv',index=False)