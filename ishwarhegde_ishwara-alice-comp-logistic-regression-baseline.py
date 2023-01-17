# Import libraries and set desired options

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns



import pickle

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

from scipy.sparse import hstack

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import TimeSeriesSplit
# Read the training and test data sets

time_list=["time%s"%i for i in range(1,11)]

site_list=["site%s"%i for i in range(1,11)]

#parse all time columns instead of just time1

train_df = pd.read_csv('../input/train_sessions.csv',

                       index_col='session_id', parse_dates=time_list) 

test_df = pd.read_csv('../input/test_sessions.csv',

                      index_col='session_id', parse_dates=time_list)



# Sort the data by time

train_df = train_df.sort_values(by='time1')



# Look at the first rows of the training set

train_df.head()



# Our target variable

y_train = train_df['target'].values



# United dataframe of the initial data 

full_df = pd.concat([train_df, test_df])



# Index to split the training and test data sets



idx_split = train_df.shape[0]
#Add a column that counts the number of non empty entries in each row

train_df['n_sites'] = train_df[site_list].apply(lambda x: 10- (x.isnull().sum()), axis='columns')

test_df['n_sites'] = test_df[site_list].apply(lambda x: 10- (x.isnull().sum()), axis='columns')



#Extract only the time variables 

train_time  =  train_df[time_list] 

test_time=test_df[time_list]


def get_total_time(row): 

    time_length = row.shape[0] - 1 

    i = time_length 

    while pd.isnull( row [ i ]): 

        i -= 1 

    return (row[i] - row[0]) / np.timedelta64(1,'s')
%%time

total_time_train = []

for row in train_time.values:

    total_time_train.append(get_total_time(row))

total_time_train = np.array(total_time_train).reshape(-1,1).astype(int)



total_time_test = []

for row in test_time.values:

    total_time_test.append(get_total_time(row))

total_time_test = np.array(total_time_test).reshape(-1,1).astype(int)



train_df["total_time"]=total_time_train

test_df["total_time"]=total_time_test
#Months in the data

full_df["month"]=full_df["time1"].dt.month

full_df.groupby(['target',"month"]).count()



#Notice that Alice does not use in certain months. Therefore months feature might be useful



train_df["month"]=train_df["time1"].dt.month

test_df["month"]=test_df["time1"].dt.month

#I checkthe day of the week i.e. weekday/weekend 

#Note: 0: Monday... 1: Sunday



full_df["day"]=full_df["time1"].dt.dayofweek

full_df.groupby(['target',"day"]).count()



#Alice does not seem to use it much on Wed/Sunday



train_df["day"]=train_df["time1"].dt.dayofweek

test_df["day"]=test_df["time1"].dt.dayofweek

# I also add a date feature which is to account for the day of the month



train_df["date"]=train_df["time1"].dt.day

test_df["date"]=test_df["time1"].dt.day
#Checking if there exists a pattern for hours of use 

full_df["hour"]=full_df["time1"].dt.hour

g = sns.FacetGrid(full_df, col="target")

g.map(sns.distplot, "hour")





train_df["hour"]=train_df["time1"].dt.hour

test_df["hour"]=test_df["time1"].dt.hour



#Alice does not seem to use the computer much at night i.e. beyond 1800 hours. Therefore I will add a dummy for night

train_df["night"]= np.where(train_df['hour']>=18, 1, 0)

test_df["night"]= np.where(test_df['hour']>=18, 1, 0)

# Change site1, ..., site10 columns type to integer and fill NA-values with zeros

sites = ['site%s' % i for i in range(1, 11)]

train_df[sites] = train_df[sites].fillna(0).astype('int')

test_df[sites] = test_df[sites].fillna(0).astype('int')



# Load websites dictionary

with open(r"../input/site_dic.pkl", "rb") as input_file:

    site_dict = pickle.load(input_file)



# Create dataframe for the dictionary

sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])

print(u'Websites total:', sites_dict.shape[0])

# Our target variable

y_train = train_df['target'].values.reshape(-1,1)



# United dataframe of the initial data 

full_df = pd.concat([train_df.drop("target",axis=1), test_df])



# Index to split the training and test data sets

idx_split = train_df.shape[0]
train_df.shape

y_train.shape
# small

time_features_train=train_df[["month","hour","day","night","date","total_time", "n_sites"]]



time_features_test=test_df[["month","hour","day","night","date", "total_time","n_sites"]]



train_df[sites].fillna(0).to_csv('train_sessions_text.txt', 

                                 sep=' ', index=None, header=None)

test_df[sites].fillna(0).to_csv('test_sessions_text.txt', 

                                sep=' ', index=None, header=None)

train=train_df[sites]

train.shape
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=50000)

with open('train_sessions_text.txt') as inp_train_file:

    X_train = tfidf.fit_transform(inp_train_file)

with open('test_sessions_text.txt') as inp_test_file:

    X_test = tfidf.transform(inp_test_file)

print(X_train.shape, X_test.shape)



# Stacking the time_features

from scipy.sparse import coo_matrix, hstack

X_train=hstack([X_train,time_features_train])

X_test=hstack([X_test,time_features_test])

X_train=X_train.tolil() 

X_test=X_test.tolil()
#Cross validation for hyperparameter tuning 

# Create regularization penalty space



# Create regularization hyperparameter space

C = np.logspace(-2, 2, 10)



# Create hyperparameter options

hyperparameters = dict(C=C)

#define the base model 

logistic=LogisticRegression(random_state=17)



#define cv model

time_split = TimeSeriesSplit(n_splits=10)

clf = GridSearchCV(logistic, hyperparameters, cv=time_split, verbose=0)



#fit the same training data so cv doesn't "peek" at the rest of the training data

best_local_model = clf.fit(X_train,y_train)
# View best hyperparameters

print('Best C:', best_local_model.best_estimator_.get_params()['C'])

C_star=best_local_model.best_estimator_.get_params()['C']

# Perform cv with time -awarness 





logistic = LogisticRegression(C=C_star, random_state=17, solver='lbfgs', max_iter=500)

cv_score_time=cross_val_score(logistic,X_train,y_train,cv=time_split,n_jobs=1)

print(cv_score_time)

print(cv_score_time.mean())
#Tuning the Tfidf Model

max_features=[10000,25_000,50_000,75_0000,100_000]

n_gram_range=[(1,2),(1,3),(1,5)]

hyperparameters=dict(max_features=max_features,n_gram_range=n_gram_range)

print(hyperparameters)
# Function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
# Train the model on the whole training data set

# Use random_state=17 for reproducibility



# Make a prediction for test data set

y_test = logistic.predict_proba(X_test)[:, 1]



# Write it to the file which could be submitted

write_to_submission_file(y_test, 'baseline_time_tfidf_time_sites.csv')