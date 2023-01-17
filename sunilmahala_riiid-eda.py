
#important links
## https://github.com/riiid/ednet
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.model_selection import train_test_split
import operator
import random

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('fivethirtyeight')

import riiideducation
import dask.dataframe as dd
import  pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# train_df= pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
#                 usecols=[1, 2, 3,4,7,8,9],nrows=10**6, dtype={'timestamp': 'int64', 'user_id': 'int32' ,
#                                                   'content_id': 'int16','content_type_id': 'int8',
#                                                   'answered_correctly':'int8',
#                                                   'prior_question_elapsed_time': 'float32',
#                                                   'prior_question_had_explanation': 'boolean'}
#               )
train_df= pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                nrows=10**6, dtype={'timestamp': 'int64', 'user_id': 'int32' ,
                                                  'content_id': 'int16','content_type_id': 'int8',
                                    'task_container_id':'int16','user_answer':'int8',
                                                  'answered_correctly':'int8',
                                                  'prior_question_elapsed_time': 'float32',
                                                  'prior_question_had_explanation': 'boolean'}
              )
train_df.head(10)
data = train_df
temp = data[data.user_id == np.random.choice(data.user_id.unique())].sort_values("timestamp")
print (temp.shape)
temp.head(10)
data['timestamp'].describe()
# checking null
data['timestamp'].isnull().sum()
f = plt.figure(figsize=(16, 8))
gs = f.add_gridspec(1, 2)

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0, 0])
    data['timestamp'].hist(bins = 50,color='orange')
    plt.title("Timestamp Distribution")

data = data.sort_values(['user_id','timestamp'])
data.head(20)
data = data.sort_values(['user_id','task_container_id'])
data.head(20)
def countplot(column):
    plt.figure(dpi=100)
    sns.countplot(train_data[column])
    plt.show()
# sns.distplot(data['timestamp'],color='yellow')
# plt.show()
data['timestamp'].hist()
plt.show()
# task contaioner id not in increasing order 
data['task_container_id'] = (
    data
    .groupby('user_id')['task_container_id']
    .transform(lambda x: pd.factorize(x)[0])
    .astype('int16')
)
data.head()
data['time_elapsed'] = data.groupby('user_id')['timestamp'].apply(lambda x: x- x.shift(1))
data['time_elapsed'] = data['time_elapsed'].fillna(data.groupby('user_id')['time_elapsed'].transform('mean'))
data.head(5)
time_correct = data[data['answered_correctly']==1]['task_container_id']
time_wrong = data[data['answered_correctly']!=1]['task_container_id']
# https://glowingpython.blogspot.com/2012/09/boxplot-with-matplotlib.html
plt.boxplot([time_correct, time_wrong])
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Words in project title')
plt.grid()
plt.show()

plt.figure(figsize=(10,3))
sns.kdeplot(time_correct ,label="correct ans", bw=0.6)
sns.kdeplot(time_wrong,label="wrong ans", bw=0.6)
plt.title('time stamp')
plt.xlabel('')
plt.legend()
plt.show()

time_correct = data[data['answered_correctly']==1]['time_elapsed']
time_wrong = data[data['answered_correctly']!=1]['time_elapsed']
plt.figure(figsize=(10,3))
sns.kdeplot(time_correct ,label="correct ans", bw=0.6)
sns.kdeplot(time_wrong,label="wrong ans", bw=0.6)
plt.title('time stamp')
plt.xlabel('')
plt.legend()
plt.show()
def plot_by_dv(data,col):
    time_correct = data[data['answered_correctly']==1][col]
    time_wrong = data[data['answered_correctly']!=1][col]
    plt.figure(figsize=(10,3))
    sns.kdeplot(time_correct ,label="correct ans", bw=0.6)
    sns.kdeplot(time_wrong,label="wrong ans", bw=0.6)
    plt.title('time stamp')
    plt.xlabel('')
    plt.legend()
    plt.show()
# unique user id 
print("total data",len(data),len(data['user_id'].unique()))
data['total_question_attemp'] = data.groupby('user_id')['user_id'].transform('count')
data['total_question_attemp_correct'] = data.groupby('user_id')['answered_correctly'].transform('sum')
data['total_question_attemp_correct'] = data['total_question_attemp_correct']/data['total_question_attemp']
data.head()
plot_by_dv(data,'total_question_attemp')
plot_by_dv(data,'total_question_attemp_correct')
# unique question  
print("total data",len(data),len(data['content_id'].unique()))
data['question_attemp'] = data.groupby('content_id')['content_id'].transform('count')
data['question_ans_correct'] = data.groupby('content_id')['answered_correctly'].transform('sum')
data['question_ans_correct'] = data['question_ans_correct']/data['question_attemp']
data.head(10)
plot_by_dv(data,'question_attemp')
plot_by_dv(data,'question_ans_correct')
def stack_plot(data, xtick, col2, col3='total'):
     ind = np.arange(data.shape[0])

     plt.figure(figsize=(20,5))
     p1 = plt.bar(ind, data[col3].values)
     p2 = plt.bar(ind, data[col2].values)
     plt.ylabel('Projects')
     plt.title('Number of projects aproved vs rejected')
     plt.xticks(ind, list(data[xtick].values))
     plt.legend((p1[0], p2[0]), ('total', 'accepted'))
     plt.show()


def univariate_barplots(data, col1, col2, top=False):
	temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()
	# Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
	temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg(total='count')).reset_index()['total']
	temp['Avg'] = pd.DataFrame(data.groupby(col1)[col2].agg(Avg='mean')).reset_index()['Avg']

	temp.sort_values(by=['total'],inplace=True, ascending=False)

	if top:
		temp = temp[0:top]

	stack_plot(temp, xtick=col1, col2=col2, col3='total')
	print(temp.head(5))
	print("="*50)
	#print(temp.tail(5))
data.groupby(['content_type_id','answered_correctly']).agg('count').iloc[:,:1]
data[data['content_type_id']==1].head()
## removing latures data
data = data[data['content_type_id']!=1]
data.head(10)
data['task_container_id'].value_counts()
data['user_answer'].value_counts()
data['answered_correctly'].value_counts()
data['prior_question_elapsed_time'].isna().sum()
data['prior_question_elapsed_time'] = data['prior_question_elapsed_time'].fillna(data.groupby('user_id')['prior_question_elapsed_time'].transform('mean'))
data.head()
plot_by_dv(data,'prior_question_elapsed_time')
data['prior_question_had_explanation'].value_counts()
data['prior_question_had_explanation'].isnaprior_question_had_explanation().sum()
data['prior_question_had_explanation'] = \
data['prior_question_had_explanation'].fillna(data['prior_question_had_explanation'].mode()[0])
data.head()

train = train[train.content_type_id == False]
#arrange by timestamp
train = train.sort_values(['timestamp'], ascending=True)

train.drop(['timestamp','content_type_id'], axis=1,   inplace=True)

results_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])
results_c.columns = ["answered_correctly_content"]

results_u = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum'])
results_u.columns = ["answered_correctly_user", 'sum']
X = train.iloc[:,:]
X = pd.merge(X, results_u, on=['user_id'], how="left")
X = pd.merge(X, results_c, on=['content_id'], how="left")
X=X[X.answered_correctly!= -1 ]
X=X.sort_values(['user_id'])
Y = X[["answered_correctly"]]
X = X.drop(["answered_correctly"], axis=1)
X.head()
# fill_mode = lambda col: col.fillna(col.mode())
# X = X.apply(fill_mode, axis=1)
X.head()
X["prior_question_had_explanation"].value_counts()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
X['prior_question_had_explanation'].fillna(X['prior_question_had_explanation'].mode()[0], inplace=True)
X['prior_question_elapsed_time'].fillna(X['prior_question_elapsed_time'].mean(), inplace=True)

X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])
X.head()

X = X[['answered_correctly_user', 'answered_correctly_content', 'sum','prior_question_elapsed_time','prior_question_had_explanation_enc']] 
#X.fillna(0.5,  inplace=True)

Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size = 0.2, shuffle=False, random_state=42)
print("XGBoost version:", xgb.__version__)
%%time
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

x_cfl=XGBClassifier(objective='binary:logistic',eval_metric= 'auc',tree_method = 'gpu_hist',
                    n_jobs=-1)

prams={
    'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
     'n_estimators':[100,200,500,1000,2000,3000,4000],
     'max_depth':[3,5,10],
    'colsample_bytree':[0.1,0.3,0.5,1],
    'subsample':[0.1,0.3,0.5,1]
}

random_cfl=RandomizedSearchCV(x_cfl,param_distributions=prams,verbose=10,n_jobs=-1,cv=3)
random_cfl.fit(Xt, Yt)
val_pred = x_cfl.predict(Xv)
    
# CV score
score = roc_auc_score(Yv, val_pred)
print(f"AUC = {score}")
