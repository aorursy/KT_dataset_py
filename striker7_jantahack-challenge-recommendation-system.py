import numpy as np
import pandas as pd
import gc 
pd.set_option('display.max_columns', 50)
import warnings
warnings.filterwarnings("ignore")
import scipy
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import random
import datetime
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import LabelEncoder
# from google.colab import drive
# drive.mount('/content/drive')
train = pd.read_csv(r"https://raw.githubusercontent.com/shrikantnarayankar15/Insaid-ML-advanced-project/master/train.csv")
test = pd.read_csv(r"https://raw.githubusercontent.com/shrikantnarayankar15/Insaid-ML-advanced-project/master/test.csv")
challenge = pd.read_csv(r"https://raw.githubusercontent.com/shrikantnarayankar15/Insaid-ML-advanced-project/master/challenge_data.csv")
train.info()
test.info()
challenge.info()
challenge.isnull().sum()
challenge.isnull().sum().plot(kind='barh')
challenge.columns
challenge['total_submissions'] = challenge['total_submissions'].fillna(challenge['total_submissions'].mean())
categorical_features = ['challenge_series_ID',
        'author_ID', 'author_gender',
       'author_org_ID', 'category_id']
challenge[categorical_features] = challenge[categorical_features].apply(lambda x:x.fillna(x.mode()[0]))
challenge.isnull().sum()
challenge.info()
challenge['programming_language'].value_counts().plot(kind='bar')
challenge['challenge_series_ID'].value_counts().nlargest(10).plot(kind='bar')
challenge.groupby('challenge_series_ID')['total_submissions'].sum().nlargest(10)
challenge.groupby('challenge_series_ID')['total_submissions'].sum().nlargest(10).plot(kind='bar')
challenge['author_ID'].value_counts().nlargest(10).plot(kind='bar')
challenge['author_ID'].value_counts().nlargest(10)
challenge['author_gender'].value_counts().plot(kind='bar')
challenge['author_org_ID'].value_counts().nlargest(5).plot(kind='bar')
train['challenge'].value_counts().nlargest(10).plot(kind='bar')
train[train['challenge_sequence']==1]['challenge'].value_counts().nlargest(10)
train[train['challenge_sequence']==2]['challenge'].value_counts().nlargest(10)
challenge = challenge.rename(columns={'challenge_ID':'challenge'})
challenge['publish_year'] = pd.DatetimeIndex(challenge['publish_date']).year
train_challenge = pd.merge(train,challenge,on=["challenge"],how="left")
train_challenge.groupby('programming_language')['user_id'].count().plot(kind='bar')
train_challenge.groupby('programming_language')['user_id'].count()
train_challenge.groupby('challenge_series_ID')['user_id'].count().nlargest(10)
train_challenge.groupby('publish_year')['user_id'].count().reset_index().sort_values(by=['publish_year'],ascending=False).rename(columns={'user_id':'user_count'})[['publish_year','user_count']].plot(x='publish_year')
train_challenge.groupby('publish_year')['user_id'].count().sort_values(ascending=False)
le =LabelEncoder()
challenge["challenge_series_ID"] = le.fit_transform(challenge["challenge_series_ID"].astype(str))
challenge["total_submissions"] = challenge["total_submissions"].fillna(challenge["total_submissions"].mean()).astype(int)
challenge["category_id"] = challenge["category_id"].fillna(challenge["category_id"].mean()).astype(int)
challenge['publish_date'] = pd.DatetimeIndex(challenge['publish_date']).year
combine_set=pd.concat([train,test], ignore_index=True)
mer_train = pd.merge(train,challenge,on=["challenge"],how="left")
mer_test = pd.merge(test,challenge,on=["challenge"],how="left")
mer_df = mer_train.append(mer_test).reset_index(drop=True)
total_data = mer_df.pivot_table(index='challenge',columns='user_id',values='challenge_sequence').fillna(0)
from scipy.sparse import csr_matrix
total_data_matrix = csr_matrix(total_data.values)
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm = 'brute')
model_knn.fit(total_data_matrix)
def distance_func(total_data, distances,indices):
  every = {}
  for i in range(0, len(distances.flatten())):
      if i == 0:
        pass
      else:
        every[total_data.index[indices.flatten()[i]]] = distances.flatten()[i]
  return every
all_s = {}
for i in total_data.index:
    distances, indices = model_knn.kneighbors(total_data.loc[i].values.reshape(1, -1), n_neighbors = 10)
    data = distance_func(total_data, distances,indices)
    all_s[i] = data
final_df = pd.DataFrame(columns=['user_sequence','challenge'])
counter = 0
for user_id in test.user_id.unique():
  challenge_ids_of_user = test[test.user_id==user_id]['challenge']
  all_sss = {}
  for i in challenge_ids_of_user:
    # distances, indices = model_knn.kneighbors(total_data_train.loc[i].values.reshape(1, -1), n_neighbors = 4)
    # data = distance_func(total_data_train, distances,indices)
    if i in all_s:
      data = all_s[i]
      for key, value in data.items():
        if key in all_sss:
          if value < all_sss[key]:
            all_sss.update({key:value})
        else:
          all_sss.update({key:value})
      # all_sss.update(data)
  for i in challenge_ids_of_user:
    if i in all_sss:
      del all_sss[i]
  challenges = [*dict(sorted(all_sss.items(), key=lambda x:x[1])[:3])]
  if len(challenges) == 0:
    final_df.loc[counter,:] = str(user_id)+'_11', '0'
    final_df.loc[counter+1,:] = str(user_id)+'_12', '0'
    final_df.loc[counter+2,:] = str(user_id)+'_13', '0'
    counter += 3
    continue
  if len(challenges) != 3:
    for i in range(3):
      challenges.append(challenges[0])
  
  final_df.loc[counter,:] = str(user_id)+'_11', challenges[0]
  final_df.loc[counter+1,:] = str(user_id)+'_12', challenges[1]
  final_df.loc[counter+2,:] = str(user_id)+'_13', challenges[2]
  counter += 3
final_df.to_csv('submit.csv', index=False)
!pip install turicreate
import turicreate as tc
tc.config.set_num_gpus(1)
# user_df and challenge_df is the data which we will pass to the turicreate which will extract features and use it further to increase recommendation
user_data = mer_df.groupby("user_id")["challenge_series_ID"].agg(lambda x: pd.Series.mode(x)[0]).to_frame()
user_data = user_data.reset_index()
user_data = tc.SFrame(user_data)
challenge_data = tc.SFrame(challenge)
combine_set_tc=tc.SFrame(combine_set)
m=tc.item_similarity_recommender.create(combine_set_tc, user_id='user_id',
                                                            item_id='challenge',
                                                            target='challenge_sequence',
                                                            user_data=user_data,
                                                            item_data = challenge_data,
                                                            similarity_type = "cosine",
                                                            )
results = m.recommend(test["user_id"].unique().tolist(),k=3)["challenge"]
submission_sample =pd.read_csv(r'https://raw.githubusercontent.com/shrikantnarayankar15/Insaid-ML-advanced-project/master/sample_submission.csv')

submission_sample["challenge"] = np.array(results).reshape(-1,1)

submission_sample.to_csv("submit_target.csv",index=False)