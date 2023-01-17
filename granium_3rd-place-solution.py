import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/ucu-data/traindata.csv", index_col = 0)
test = pd.read_csv("../input/ucu-machine-learning-inclass/testdata.csv", index_col = 0)
data = pd.concat([train, test], axis = 0, sort=False)
data['duration'] = np.log(data.duration)
campaigns = data.campaign.unique()
campaign_stats = data.groupby('campaign').mean().loc[:,['age', 'balance', 'duration']]
data['campaign_avg_balance'] = data['campaign'].apply(lambda x: campaign_stats.loc[x][1])
data['campaign_avg_duration'] = data['campaign'].apply(lambda x: campaign_stats.loc[x][2])
jobs = data.job.unique()
jobs_stats = data.groupby('job').mean().loc[:,['age', 'balance', 'duration']]
data['job_avg_age'] = data['job'].apply(lambda x: jobs_stats.loc[x][0])
data['job_avg_balance'] = data['job'].apply(lambda x: jobs_stats.loc[x][1])
edu = data.education.unique()
edu_stats = data.groupby('education').mean().loc[:,['age', 'balance', 'duration']]
data['edu_avg_age'] = data['education'].apply(lambda x: edu_stats.loc[x][0])
data['edu_avg_balance'] = data['education'].apply(lambda x: edu_stats.loc[x][1])
marital = data.marital.unique()
m_stats = data.groupby('marital').mean().loc[:,['age', 'balance', 'duration']]
data['marital_avg_age'] = data['marital'].apply(lambda x: m_stats.loc[x][0])
data['marital_avg_balance'] = data['marital'].apply(lambda x: m_stats.loc[x][1])
def add_custom_clustering(data_in, cols, feature,k = 20):
    data = data_in.copy()
    cluster_data = data[cols]
    c = KMeans(n_clusters = k)
    c.fit(cluster_data)
    Y = c.predict(cluster_data)
    data[feature] = Y.astype(object)
    return data
data = add_custom_clustering(data, ['age'], 'age_cluster')
data = add_custom_clustering(data, ['balance'], 'balance_cluster')
data = add_custom_clustering(data, ['duration'], 'duration_cluster')
data = add_custom_clustering(data, ['age', 'balance', 'duration'], 'abd_cluster')
data = add_custom_clustering(data, ['day'], 'day_cluster')
data = add_custom_clustering(data, ['balance', 'duration'], 'bd_cluster')
#Prepare categorical features for catboost
cols = list(data.columns)
cols.remove('y')
categorical_features = [cols.index(col_name)for col_name in list(data.dtypes[data.dtypes == object].index)]
train = data[data.y.notnull()]
test = data[data.y.isna()]

train_pool = catboost.Pool(data=train.drop(['y'], axis = 1),
                           label=train['y'],
                           cat_features=categorical_features)
#This parameters are not tuned 
params = {'loss_function': 'Logloss',
          'eval_metric': 'AUC', 
          'depth': 5,
          'l2_leaf_reg': 25,
          'od_type': 'Iter',
          'od_wait': 250,
          'iterations': 200,
          'learning_rate': 0.03,
          'one_hot_max_size': 22}
cv_res = catboost.cv(pool=train_pool, params=params, fold_count=5) # Cross validation 5 folds
clf = catboost.CatBoostClassifier(**params)
clf.fit(train_pool)
list(zip(clf.feature_names_, clf.feature_importances_)) # Feature importances from catboost 
def submit(predictions, filename):
    submission = pd.read_csv('../input/ucu-data/sample_submission.csv')
    submission.y = predictions
    submission.to_csv(filename, index=False)
predictions = clf.predict_proba(test.drop(['y'], axis = 1))[:, 1] # Get probabilities for `1` prediction
submit(predictions, 'submission-kernel.csv')
