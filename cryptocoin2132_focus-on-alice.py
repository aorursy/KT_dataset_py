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
# Alice competition
# -> interpret weigths with eli5
! pwd
PATH_TO_DATA = '../input/'
SEED = 17
train_df = pd.read_csv(PATH_TO_DATA+'catch-me-if-you-can/train_sessions.csv',index_col='session_id')
test_df = pd.read_csv(PATH_TO_DATA+'catch-me-if-you-can/test_sessions.csv',index_col='session_id')

train_df
test_df
train_df.info()
# change time columns into datetime format

time_col = ['time%s'%i for i in range(1,11)]
time_col
train_df[time_col] = train_df[time_col].apply(pd.to_datetime)
test_df[time_col] = test_df[time_col].apply(pd.to_datetime)
test_df.info()
site_col = ['site%s' %i for i in range(1,11)]
import pickle
# open file of sites and index
with open(PATH_TO_DATA+'mlcourse/site_dic.pkl','rb') as file :
    site2id = pickle.load(file)
    
    
site2id
# list of websites containing youtube
pd.Series(site2id)[pd.Series(site2id).index.str.contains('youtube')]
# the dict file is site(key): id(value) [site2id]
# we want instead id2site because our dataframe contains id and we want to change it with sites

id2site = {v:k for (k,v) in site2id.items()}
id2site
id2site[12836]
## need to do a bag of words with sites name instead of id
# BoW needs a list with each element being 1 observation to create several columns
# simplest textformatting is countvectorizer 
# td idf is more complex text formatting

# change id of site into name
# -> fillna(0) for NAN values
# in the dict create index 0 : unknown site 

id2site[0] = 'unknown'

# before going to sparse data we need to sort by date to be able to do CV with timeseries
train_df = train_df.sort_values(by = 'time1')


# list train sites : 1 row = 1 element in the list separated by space
#.tolist() : from series to list


train_sessions = train_df[site_col].fillna(0).apply(lambda row : ' '.join([id2site[i] for i in row]),axis = 1).tolist()

test_sessions = test_df[site_col].fillna(0).apply(lambda row : ' '.join([id2site[i] for i in row]),axis = 1).tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
# diff btw countvectorizer and tfidvectorizer?
# countvectorizer counts word frequency in the document (row)
# tfidf vectorizer counts word frequency in the document and adjust inversely proportionate 
# to the freauency in the corpus (whole data)
# Why? words like 'the' and 'we' appear in every document -> they dont tell us much about what makes a document unique
# but if a word like "crepuscular" appear a lot in a document but not freaquently in the corpus
# then it will give us much more info on the document 

# in countvectorizer 'the' has much more weights than 'crepuscular' which makes no sense
# in conclusion : it is a way to penalize frequent words


vectorizer = TfidfVectorizer(ngram_range=(1,5), max_features = 50000, tokenizer = lambda s : s.split())

# we want to split words only by space not by dots or commas
# -> tokenizer allows us to define how to split words
# -> string.split() does it on spaces
train_sessions[0].split()
# only columns from sites into sparse format
X_train_sites = vectorizer.fit_transform(train_sessions)
X_test_sites = vectorizer.transform(test_sessions)
X_train_sites
y_train = train_df['target'].values #numpy
y_train
train_times, test_times = train_df[time_col], test_df[time_col]

train_times
X_train_sites.shape, X_test_sites.shape
# sample of new features
vectorizer.get_feature_names()[10000:10010]
## CV schedule with timeseries
from sklearn.model_selection import TimeSeriesSplit
time_split = TimeSeriesSplit(n_splits = 10)
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=SEED,solver ='liblinear',C=1)
# liblinear is the typical gradient descent 
from sklearn.model_selection import cross_val_score
%%time
# we will do a cross validation with time split -> need to order train with time date

# train only with sites data

cv_score1 = cross_val_score(estimator=logit,X=X_train_sites,y=y_train, cv = time_split, n_jobs = -1,scoring = 'roc_auc')
cv_score1, cv_score1.mean()
# train with all sites data

logit.fit(X_train_sites,y_train)
! pip install eli5
import eli5
# new columns name
vectorizer.get_feature_names()
# Model feature weigths with eli5

eli5.show_weights(estimator = logit, feature_names = vectorizer.get_feature_names(), top = 30)
from IPython.display import display_html
#Display the HTML representation of an object
display_html(eli5.show_weights(estimator = logit, feature_names = vectorizer.get_feature_names(), top = 30))
# these are the websites that are the most descriptive of Alice.
# interesting enough she is not using gmail ...
# now let's predict and make a submission file 

# predict 
logit_test_pred = logit.predict_proba(X_test_sites)[:,1]

logit_test_pred
# submission
predicted_df = pd.DataFrame(logit_test_pred,columns = ['target'], index = np.arange(1,logit_test_pred.shape[0]+1))

predicted_df
predicted_df.to_csv('subm1.csv',index_label = 'session_id')
# a helper function for writing predictions to a file 

def write_to_submission_file(predicted_labels,output_file,target_name = 'target',index_label='session_id'):
    predicted_df = pd.DataFrame(predicted_labels,columns = [target_name], index = np.arange(1,predicted_labels.shape[0]+1))
    predicted_df.to_csv(output_file,index_label = index_label)
write_to_submission_file(logit_test_pred,'subm1.csv')
# a function that performs CV, model training, displaying feature importance,
# making predictions and forming submission file

def train_and_predict(model,X_train,y_train,X_test,site_feature_names = vectorizer.get_feature_names(),new_feature_names = None,cv=time_split,scoring='roc_auc'
                     ,top_n_features_to_show = 30,submission_file_name='submission.csv'):
    
    cv_scores = cross_val_score(estimator = model,X=X_train,y=y_train,cv=cv,scoring=scoring,n_jobs=-1)
    print('CV scores', cv_scores)
    print('CV mean: {}, CV std: {}'.format(cv_scores.mean(),cv_scores.std()))
    
    model.fit(X_train,y_train)
    
    if new_feature_names:
        all_feature_names = site_feature_names + new_feature_names
    else:
        
        all_feature_names = site_feature_names
    
    display_html(eli5.show_weights(model,feature_names = all_feature_names, top = top_n_features_to_show))
    
    if new_feature_names:
        print('New feature weights:')
        print(pd.DataFrame({'feature': new_feature_names,
                            'coef': model.coef_.flatten()[-len(new_feature_names):]}))
        
    test_pred = model.predict_proba(X_test)[:,1]
    
    
    write_to_submission_file(test_pred,submission_file_name)
    
    return cv_scores
cv_scores1 = train_and_predict(model = logit, X_train=X_train_sites,y_train=y_train,X_test=X_test_sites,
                              site_feature_names= vectorizer.get_feature_names(),cv=time_split,
                               submission_file_name= 'subm1.csv')
# let's focus now on time columns
# intuition : people visit websites at specific moments 
# -> what is the distribution of visiting hours?
session_start_hour = train_times['time1'].apply(lambda ts : ts.hour).values
session_start_hour
%matplotlib inline
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

plt.figure(figsize=(12,8))
sns.countplot(session_start_hour);
# compare now distribution of target 1 and target 0
fig,ax =plt.subplots(1,2,figsize = (12,8))

sns.countplot(session_start_hour[y_train == 1],ax=ax[0])
ax[0].set_title('Alice')
ax[0].set(xlabel = 'Session start hour')
sns.countplot(session_start_hour[y_train == 0],ax=ax[1])
ax[1].set_title('Others')
ax[1].set(xlabel = 'Session start hour');

# Alice prefers 4-5 pm for browsing
# -> create time features : morning, day, evening, night

morning = ((session_start_hour >= 7) & (session_start_hour <= 11)).astype('int')

day = ((session_start_hour >= 12) & (session_start_hour <= 18)).astype('int')

evening = ((session_start_hour >= 19) & (session_start_hour <= 23)).astype('int')

night = ((session_start_hour >= 0) & (session_start_hour <= 6)).astype('int')


pd.crosstab([morning,day,evening,night],y_train, rownames=['morning','day','evening','night'],colnames=['target'])
# we will create and add new features (morning,...)
# we will also keep a flag of whether we add an hour feature or not (overfitting???)
from scipy.sparse import hstack
objects_to_hstack = [X_train_sites, morning.reshape(-1,1),day.reshape(-1,1),evening.reshape(-1,1),night.reshape(-1,1)]
    
#add together -> the method to add new features with sparse matrix is to use hstack() from scipy.sparse 
# dont forget to reshape(-1,1) to be able to concatenate
X_train_with_times1 = hstack(objects_to_hstack)

X_train_with_times1
# now that we added features for train set let's do it for test set 
# but now with a function
def add_time_features(df_times,X_sparse,add_hour = True):
    """Add time features to sparse matrix"""
    
    hour = df_times['time1'].apply(lambda ts : ts.hour)
    
    morning = ((hour >= 7) & (hour <= 11)).astype(int).values.reshape(-1,1)
    day = ((hour >= 12) & (hour <= 18)).astype(int).values.reshape(-1,1)
    evening = ((hour >= 19) & (hour <= 23)).astype(int).values.reshape(-1,1)
    night  = ((hour >= 0) & (hour <= 6)).astype(int).values.reshape(-1,1)
    
    objects_to_hstack = [X_sparse,morning,day,evening,night]
    feature_names = ['morning','day','evening','night']
    
    if add_hour:
        # adding hour features if required
        
        objects_to_hstack.append(hour.values.reshape(-1,1)/24)
        # divided by 24 to normalize
        
        feature_names.append('hour')
        
    
    X = hstack(objects_to_hstack)
    
    return X,feature_names
# add time features for train and test sparse matrix

X_train_with_times1, new_feat_names = add_time_features(train_times, X_train_sites)
X_test_with_times1 , _= add_time_features(test_times, X_test_sites)
# same number of columns
X_train_with_times1.shape , X_test_with_times1.shape
new_feat_names
## TimeSeries CrossValidation with ROC_AUC
# we will use our function train_and_predict to make CV + graphs and feature importance + fit & predict_proba
cv_scores2 =train_and_predict(model=logit,X_train=X_train_with_times1,y_train=y_train,
                 X_test = X_test_with_times1, new_feature_names = new_feat_names,
                 submission_file_name= 'subm2.csv',cv=time_split)
# it looks like hour colums is very important
cv_scores2 > cv_scores1

# we see an increase in nearly every folds compared to previous crossvalidation
# better results 
#LB : 0.91803 -> 0.93132

# hour feature looks too important -> suspicious
# redo the work without hour

X_train_with_times2, new_feat_names = add_time_features(df_times=train_times,
                                                       X_sparse=X_train_sites,
                                                      add_hour= False)
X_test_with_times2, _ = add_time_features(df_times=test_times,X_sparse=X_test_sites,
                                         add_hour = False)
cv_scores3 = train_and_predict(model = logit, X_train=X_train_with_times2,y_train=y_train,
                              X_test=X_test_with_times2,new_feature_names=new_feat_names,
                              submission_file_name='subm3.csv')
# CV is more stable without hour 
# and the prediction is better 
cv_scores3 > cv_scores1
# better for every folds
cv_scores3 > cv_scores2
# only better in half time 
# but we choose the third one because LESS VARIATION in CV
# LB : 0.93132 -> 0.94522
# -> hour feature leads to overfitting -> better not to add it
## Conclusion
# Basically when you see a feature too important 
# first reflex is to drop it and see if the results drop as well
# if not then this feature is causing overfitting
# for this submission we will create a new feature : sesssion duration
# first time we will do it incorrectly (without scaling)
# axis = 1 gives max row wise and timedelta64[ms] expressed in millisecond (float) then int to express it in int
(train_times.max(axis = 1) - train_times.min(axis = 1)).astype('timedelta64[ms]').astype('int')
                                                               
# redo in a function form
# remember hstack accepts a list of 2D arrays -> .values.reshape(-1,1)

def add_session_duration_incrorrect(df_times,X_sparse):
    new_feat = (df_times.max(axis = 1) - df_times.min(axis = 1)).astype('timedelta64[ms]').astype('int')
    return hstack([X_sparse,new_feat.values.reshape(-1,1)])


# add feature for train and test sparse matrix

X_train_with_time_incorrect = add_session_duration_incrorrect(train_times,X_train_with_times2)

X_test_with_time_incorrect = add_session_duration_incrorrect(test_times,X_test_with_times2)
cv_scores4 = train_and_predict(model=logit,X_train=X_train_with_time_incorrect,
                              y_train=y_train,X_test= X_test_with_time_incorrect,
                               new_feature_names=new_feat_names+['sess_duration'],
                              cv = time_split, submission_file_name= 'subm4.csv')
# HUGE DETERIORATION OF RESULTS !!!!
# reason ? session duration is expressed in millisecond -> high values 
# Too much weight given -> more difficult to do optimal Gradient Descent
# NEED to perform feature scaling
train_durations = (train_times.max(axis = 1) - train_times.min(axis=1)).astype('timedelta64[ms]').astype('int')
test_durations = (test_times.max(axis = 1) - test_times.min(axis=1)).astype('timedelta64[ms]').astype('int')
from sklearn.preprocessing import StandardScaler
# scaling features (fit_tansform on train and transform on test)

scaler = StandardScaler()

train_dur_scaled = scaler.fit_transform(train_durations.values.reshape(-1,1))
test_dur_scaled = scaler.transform(test_durations.values.reshape(-1,1))

#.values.reshape is not necessary but it is for hstack -> we can do it now
train_durations
train_dur_scaled
X_train_with_time_correct = hstack([X_train_with_times2,train_dur_scaled])

X_test_with_time_correct = hstack([X_test_with_times2,test_dur_scaled])
X_train_with_time_correct.shape, X_test_with_time_correct.shape
cv_scores5 = train_and_predict(model = logit, X_train=X_train_with_time_correct,
                              y_train=y_train, X_test=X_test_with_time_correct,
                              new_feature_names=new_feat_names+['sess_duration'],
                              cv = time_split, submission_file_name='subm5.csv')
cv_scores5 > cv_scores3
# New model better on 9 folds over 10 
# LB : 0.94522 -> 0.94616
# A bit better 
# a really good practice (especially to come up with new features)
# is to look at new kernels (from competition) to get new ideas
# we will add here month (june,july,...) and day of week (Monday,Tuesday,...) + yearmonth (202001,...202012)
# weekday
train_times['time1'].apply(lambda ts: ts.weekday())
# month
train_times['time1'].apply(lambda ts: ts.month)
# year month
# divided by 100.000 for scalin reasons
train_times['time1'].apply(lambda ts : 100 * ts.year + ts.month)/1e5
# create a function to apply on train and test

def add_day_month(df_times,X_sparse):
    weekday = df_times['time1'].apply(lambda ts: ts.weekday())
    month = df_times['time1'].apply(lambda ts: ts.month)
    yearmonth = df_times['time1'].apply(lambda ts : 100 * ts.year + ts.month)/1e5
    
    objects_to_hstack = [X_sparse,weekday.values.reshape(-1,1),month.values.reshape(-1,1),
                        yearmonth.values.reshape(-1,1)]
    feature_names = ['weekday','month','yearmonth']
    
    X_new = hstack(objects_to_hstack)
    
    return X_new, feature_names
X_train_final, more_feat_names = add_day_month(train_times,X_train_with_time_correct)

X_test_final,_ = add_day_month(test_times,X_test_with_time_correct)
cv_scores6 = train_and_predict(model = logit, X_train=X_train_final,y_train=y_train,
                              X_test= X_test_final,new_feature_names= new_feat_names+ ['sess_duration'] + more_feat_names,
                              cv =time_split, submission_file_name='subm6.csv')
cv_scores6 > cv_scores5

# new model better on 6 folds especially the last ones
# but mean and std are lower

# however LB is better : 0.94616 -> 0.95059
# Once we have no more ideas of feature engineering
# we can start thinking about Hyperparametrization
# with pipelines we can avoid data leakage on CV
# we could hyperparam:
# ngram_range
# max_features
# CountVectorizer vs TfidfVectorizer
# C (logReg)
# in this code now we will limit ourselves to hyperparam only C
from sklearn.model_selection import GridSearchCV
c_values = np.logspace(-2,2,20) 
# we could have used wider range first

logit_grid_searcher = GridSearchCV(estimator = logit, param_grid= {'C':c_values},
                                  scoring = 'roc_auc', cv =time_split,n_jobs = -1,
                                  verbose = True)
%%time
logit_grid_searcher.fit(X_train_final,y_train)
logit_grid_searcher.best_score_, logit_grid_searcher.best_params_
# new model after hyperparametrization with best_params_

final_model = logit_grid_searcher.best_estimator_

final_model
cv_scores7 = train_and_predict(model= final_model,X_train=X_train_final,y_train=y_train,
                              X_test= X_test_final,new_feature_names= new_feat_names + ['sess_duration'] + more_feat_names,
                              cv = time_split, submission_file_name='subm7.csv')
cv_scores7 > cv_scores6
# Tuning hyperparam helped only for 6 folds
# LB :0.95059 -> 0.95051

# -> less than previous submission
#### HINT #####
# our CV schema is not perfect !
# hint : is all training set needed for a good prediction?


# my personal answer : 
# No we need to learn only on latest data ! 
# Why? because very old data can be misleading
# in other word people change their behavior through time
# and it is not relevant to learn previous behavior
# to predict new behavior

# E.g. learning DAILY price fluctuation of bitcoin in 2012
# will not help us make good trading decisions in 2020
# it could even be misleading
# summarize 7 submissions

cv_means = [cv_score.mean() for cv_score in [cv_scores1, cv_scores2,cv_scores3,cv_scores4,cv_scores5,cv_scores6,cv_scores7]]

cv_stds = [cv_score.std() for cv_score in [cv_scores1, cv_scores2,cv_scores3,cv_scores4,cv_scores5,cv_scores6,cv_scores7]]

public_lb_scores = [0.91803,0.93132,0.94522,0.67018,0.94616,0.95059,0.95051]
cv_means
cv_stds
df_scores = pd.DataFrame({'CV_mean':cv_means,'CV_stds':cv_stds,'LB':public_lb_scores}
             ,index= range(1,len(cv_means)+1))

df_scores
# correlation exists between CV and LB
# but not perfect
# which submission to choose ?

# ANSWER
# a popular solution is to treat mean cv and LB results with weights,
# proportionally to train and test size

# However as here there is a time component
# test set is then per se more important (latest data)
# we will give arbitrary weight (60%) - based on no theory only prectical experience

# we could also use std scores to help selecting# the best submission
weight = 0.6
df_scores['cv_lb_weighted'] = weight * df_scores['LB'] + (1-weight) * df_scores['CV_mean']
df_scores
# the idea to use weight is to avoid trusting only 
# LB because it could lead to overfitting the test set
# we should trust our cv especially if the schema is well designed
# best value is submission 7
! cp subm7.csv submission.csv

# questions:
# 1.how is it possible that with just using tfidf we have better results than countvectorizer
# knowing that we are using entropy the same way we do with log reg
# 2. how come by adding only one feature we are overfitting?
# 3. how come that feature scaling improves our performance?