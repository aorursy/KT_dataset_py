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

sgd = pd.read_csv('/kaggle/input/submiss/SGD.csv')
turi = pd.read_csv('/kaggle/input/submiss/TURI.csv')
sgd.head()
int((sgd['user_sequence'][0]).split('_')[0])
user_id = []
seq = []
for i in range(0,len(sgd)):
    user_id.append((int((sgd['user_sequence'][i]).split('_')[0])))
for i in range(0,len(sgd)):
    seq.append((int((sgd['user_sequence'][i]).split('_')[1])))
sgd['challenge_sequence'] = seq
turi['challenge_sequence'] = seq
sgd['user'] = user_id
turi['user'] = user_id
sgd.head()
turi.head()
actual_df = pd.DataFrame(turi)
actual = actual_df.groupby('user').agg(lambda x : ','.join(list(x))).reset_index().reindex(columns=actual_df.columns)
actual_list = []
for x in actual.challenge:
    actual_list.append([y for y in x.split(',')])
new_turi = pd.DataFrame()
new_turi['user'] = actual['user']
new_turi['ch'] = actual_list
new_turi
actual_df = pd.DataFrame(sgd)
actual = actual_df.groupby('user').agg(lambda x : ','.join(list(x))).reset_index().reindex(columns=actual_df.columns)
actual_list = []
for x in actual.challenge:
    actual_list.append([y for y in x.split(',')])
new_sgd = pd.DataFrame()
new_sgd['user'] = actual['user']
new_sgd['ch'] = actual_list


train = pd.read_csv('/kaggle/input/train.csv')
train.head()
train.dtypes
#challenge = pd.read_csv('/kaggle/input/challenge_data.csv')
test = pd.read_csv('/kaggle/input/input-data/test.csv')
all_data = train.append(test, ignore_index = True)
all_data.head()
all_data['ratings'] = all_data['challenge_sequence']
all_data.head()
challenge = pd.read_csv('/kaggle/input/challenge_data.csv')
challenge.shape
actual = [3,2,1]
predicted = [3,4,5]
actual = list(actual)
predicted = list(predicted)

if len(predicted)>3:
    predicted = predicted[:3]

score = 0.0
num_hits = 0.0

for i,p in enumerate(predicted):
    if p in actual and p not in predicted[:i]:
        num_hits += 1.0
        score += num_hits / (i+1.0)
        
if not actual:
    print(0.0)

print(score / min(len(actual), 3))


#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = all_data.pivot(index='user_id', 
                                                          columns='challenge', 
                                                          values='ratings').fillna(0)

users_items_pivot_matrix_df.head(10)
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
users_items_pivot_matrix[:10]
users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
users_items_pivot_sparse_matrix
#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 50
#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
U.shape
Vt.shape
sigma = np.diag(sigma)
sigma.shape
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings
#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
user_id = list(users_items_pivot_matrix_df.index)
user_dict = {}
counter = 0 
for i in user_id:
    user_dict[i] = counter
    counter += 1
#item_dict ={}
#df = books_metadata[['book_id', 'title']].sort_values('book_id').reset_index()

#for i in range(df.shape[0]):
#    item_dict[(df.loc[i,'book_id'])] = df.loc[i,'title']

item_dict = {}
df = challenge[['challenge_ID']].reset_index()

for i in range(df.shape[0]):
    item_dict[(df.loc[i,'challenge_ID'])] = df.loc[i,'challenge_ID']

#counter = 0
#for i in item_id:
#    item_id[counter] = i
#    counter += 1
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.05,
                no_components=150)
#users_items_pivot_sparse_matrix
model = model.fit(users_items_pivot_sparse_matrix,
                  epochs=100,
                  num_threads=16, verbose=True)
challenge.head()
#books_metadata_selected = challenge[['challenge_ID', 'programming_language', 'challenge_series_ID', 'total_submissions', 
#                                          'author_ID', 'author_gender', 'author_org_ID']]
books_metadata_selected = challenge[['challenge_ID', 'programming_language','author_gender']]
#books_metadata_selected['total_submissions'].fillna(np.mean(books_metadata_selected.total_submissions), inplace = True)
books_metadata_selected['programming_language'].fillna(books_metadata_selected['programming_language'].mode()[0], inplace=True)
#books_metadata_selected['challenge_series_ID'].fillna(books_metadata_selected['challenge_series_ID'].mode()[0], inplace=True)
#books_metadata_selected['author_ID'].fillna(books_metadata_selected['author_ID'].mode()[0], inplace=True)
books_metadata_selected['author_gender'].fillna(books_metadata_selected['author_gender'].mode()[0], inplace=True)
#books_metadata_selected['author_org_ID'].fillna(books_metadata_selected['author_org_ID'].mode()[0], inplace=True)
books_metadata_selected_transformed = pd.get_dummies(books_metadata_selected, columns = ['programming_language', 'author_gender'])
books_metadata_selected_transformed = books_metadata_selected_transformed.reset_index().drop('index', axis=1)
books_metadata_selected_transformed.head()
books_metadata_csr = csr_matrix(books_metadata_selected_transformed.drop('challenge_ID', axis=1).values)
books_metadata_csr
def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 3, show = True):
    rec = []
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items), item_features=books_metadata_csr))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
#        print ("User: " + str(user_id))
#        print("Known Likes:")
        counter = 1
        for i in known_items:
 #           print(str(counter) + '- ' + i)
            counter+=1

  #      print("\n Recommended Items:")
        counter = 1
        for i in scores:
   #         print(str(counter) + '- ' + i)
            counter+=1
            rec.append(i)
    return rec
ans = sample_recommendation_user(model, users_items_pivot_matrix_df, 4577, user_dict, item_dict)
print(ans)
list_user = test['user_id'].unique()
len(list_user)
rec = []
for i in list_user:
    ans = []
    ans = sample_recommendation_user(model, users_items_pivot_matrix_df, i, user_dict, item_dict)
    for j in ans:
        rec.append(j)
len(rec)
potential = list(cf_preds_df[4576].sort_values(ascending=False)[:13].index)
print(potential)
ans = []
present = list(test[test['user_id'] == 4577]['challenge'])
print(present)
count = 0
while(count != 3):
    if potential[0] in present:
        potential.remove(potential[0])
    else:
        count = count + 1
        ans.append(potential[0])
        potential.remove(potential[0])
print(ans)
        
list_user = test['user_id'].unique()
len(list_user)
rec = []
for i in list_user:
    potential = list(cf_preds_df[i].sort_values(ascending=False)[:13].index)
    present = list(test[test['user_id'] == i]['challenge'])
    count = 0
    while(count != 3):
        if potential[0] in present:
            potential.remove(potential[0])
        else:
            count = count + 1
            rec.append(potential[0])
            potential.remove(potential[0])
    

su = pd.read_csv('/kaggle/input/sample_submission_J0OjXLi_DDt3uQN.csv')
d = {'user_sequence':list(su['user_sequence']),'challenge':rec}
submission = pd.DataFrame(d)
submission.to_csv('submit.csv',index = False)
