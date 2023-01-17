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
train = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/train.csv')

challenge = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')

test = pd.read_csv('/kaggle/input/av-recommendation-systems/test_HLxMpl7/test.csv')

print(train.shape, test.shape, challenge.shape)
challenge.head()
for col in challenge.columns:

    print(f'{col}: {challenge[col].nunique()}')
test.head(20)
print(f'The number of unique users in train set is: {train.user_id.nunique()}')

print(f'The number of unique challenges in train set is: {train.challenge.nunique()}')



print(f'The number of unique users in test set is: {test.user_id.nunique()}')

print(f'The number of unique challenges in test set is: {test.challenge.nunique()}')
print(len(np.intersect1d(train.challenge, test.challenge)))

print(len(np.intersect1d(train.user_id, test.user_id)))
def apk(actual, predicted, k):

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



def mapk(actual, predicted, k):

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
col = challenge.columns

col = ['challenge' if x =='challenge_ID' else x for x in col]

challenge.columns = col
train_ch = pd.merge(train, challenge, on = 'challenge', how = 'left')

test_ch = pd.merge(test, challenge, on = 'challenge', how = 'left')
train_ch.head(26)
print(train_ch.shape, test_ch.shape)
submit = pd.read_csv('/kaggle/input/av-recommendation-systems/sample_submission_J0OjXLi_DDt3uQN.csv')

submit.head()
users = train.user_id

val_user = list(users.sample(15000, random_state = 0))
!pip install turicreate

import turicreate as tc
train_sf = tc.SFrame.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/train.csv')

test_sf = tc.SFrame.read_csv('/kaggle/input/av-recommendation-systems/test_HLxMpl7/test.csv')

train_sf = train_sf.append(test_sf)

#training_data, validation_data = tc.recommender.util.random_split_by_user(train_sf, 'user_id', 'challenge')
train_sf = train_sf.sort('user_id')

train_sf['rating'] = 10.0

training_data = train_sf[train_sf['challenge_sequence'] < 11]

validation_data = train_sf[train_sf['challenge_sequence'] >= 11]
challenge_sf = tc.SFrame.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')

challenge_sf = challenge_sf.rename({'challenge_ID': 'challenge'})

challenge_sf['category_id']= challenge_sf['category_id'].fillna(0).astype(str)

challenge_sf['programming_language']= challenge_sf['programming_language'].astype(str)

mean = np.mean(challenge['total_submissions'])

challenge_sf['total_submissions']= challenge_sf['total_submissions'].fillna(mean)
#tc.config.set_num_gpus(0)

#del model

model = tc.recommender.ranking_factorization_recommender.create(training_data, 'user_id', 'challenge'

                                                                , ranking_regularization = 0.09

                                                                , max_iterations = 400

                                                                , num_factors = 222

                                                                #, regularization= 1e-4

                                                                , target = 'challenge_sequence'

                                                                #, item_data = challenge_sf#[['challenge', 'total_submissions']]

                                                                , num_sampled_negative_examples = 7

                                                                ,verbose = True

                                                               )
validation_data.head()
users = validation_data['user_id'].unique()

recommend = model.recommend(users= users, k=3)

#recommend2 = model2.recommend(users= users, k=3)
recommend = recommend.sort(['user_id', 'rank'])

#recommend2 = recommend2.sort(['user_id', 'rank'])
validation_data = validation_data.sort(['user_id', 'challenge_sequence'])
recommend_df = pd.DataFrame(recommend)

#recommend2_df = pd.DataFrame(recommend2)



predicted = recommend_df.groupby('user_id').agg(lambda x : ','.join(list(x))).reset_index().reindex(columns=recommend_df.columns)

#predicted2 = recommend2_df.groupby('user_id').agg(lambda x : ','.join(list(x))).reset_index().reindex(columns=recommend2_df.columns)



actual_df = pd.DataFrame(validation_data)

actual = actual_df.groupby('user_id').agg(lambda x : ','.join(list(x))).reset_index().reindex(columns=actual_df.columns)
actual.head()
actual_list = []

for x in actual.challenge:

    actual_list.append([y for y in x.split(',')])

actual_list[:5]
#predicted2.head()
pred_list = []

for x in predicted.challenge:

    pred_list.append([y for y in x.split(',')])

pred_list[:5]
#pred2_list = []

#for x in predicted2.challenge:

#    pred2_list.append([y for y in x.split(',')])

#pred2_list[:5]
[print(a,p) for a,p in zip(actual_list[-5:], pred_list[-5:])]
#[print(a,p) for a,p in zip(actual_list[-5:], pred2_list[-5:])]
print(mapk(actual_list, pred_list, 3))
#print(mapk(actual_list, pred2_list, 3))
recommend.head()
recommend2.head()
training_data[training_data['user_id'] == 113838]
users = test_sf['user_id'].unique()

users = np.sort(users)
users
pred_rec = model.recommend(users= users, k=3)

submit = pd.read_csv('/kaggle/input/av-recommendation-systems/sample_submission_J0OjXLi_DDt3uQN.csv')
#pred_rec2 = model2.recommend(users= users, k=3)
#pred_rec2.tail()
submit['challenge'] = pred_rec['challenge']

submit.to_csv('turi_tuned_1977.csv', index= False)
submit['challenge'] = pred_rec2['challenge']

submit.to_csv('turi_model2.csv', index= False)
submit.head(10)
challenge = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')

challenge.head()
challenge.challenge_series_ID = challenge.challenge_series_ID.fillna('None')

challenge.challenge_series_ID = challenge.challenge_series_ID.astype('str')



challenge.category_id = challenge.category_id.fillna('None')

challenge.category_id = challenge.category_id.astype('str')
from sklearn.preprocessing import LabelEncoder

le_series = LabelEncoder()

challenge.challenge_series_ID = le_series.fit_transform(challenge.challenge_series_ID)



le_cat = LabelEncoder()

challenge.category_id = le_cat.fit_transform(challenge.category_id)
challenge = pd.get_dummies(challenge, columns = ['author_gender'], drop_first = True)
challenge.head()
ch = challenge[['programming_language', 'challenge_series_ID', 'total_submissions', 'category_id', 'author_gender_M']]

ch.total_submissions = ch.total_submissions.fillna(np.mean(ch.total_submissions))
ch.isnull().sum()
from sklearn.metrics import pairwise

matrix = pairwise.cosine_similarity(ch, ch)
sim_df = pd.DataFrame(matrix, columns = challenge.challenge_ID)

sim_df.index = challenge.challenge_ID
sim_df.head()
#sim_df = pd.merge(sim_df, challenge, on = 'challenge_ID')

#sim_df['publish_date'] = (sim_df['publish_date']).astype('str')
#date = sim_df[sim_df['challenge_ID'] == 'CI23714'].publish_date

sim_df.nlargest(13, 'CI24915')['CI24915']
train.head(13)
challenge = pd.read_csv('/kaggle/input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')

challenge.head()

col = ['programming_language', 'total_submissions', 'author_gender']

ch = challenge[col]

ch = pd.get_dummies(ch, columns = [ 'programming_language', 'author_gender'], drop_first = False)

ch.total_submissions = ch.total_submissions.fillna(np.mean(ch.total_submissions))

print(ch.shape)

matrix = pairwise.cosine_similarity(ch, ch)

sim2_df = pd.DataFrame(matrix, columns = challenge.challenge_ID)

sim2_df.index = challenge.challenge_ID
sim2_df.head()
sim2_df.nlargest(500, 'CI24915')['CI24915']
train.head(13)