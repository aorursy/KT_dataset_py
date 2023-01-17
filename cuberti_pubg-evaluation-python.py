!pip install kaggle
!pip install fastai==0.7.0
!pip install torchtext==0.2.3
!pip install pdpbox
%load_ext autoreload
%autoreload 2

%matplotlib inline
import pandas as pd
from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import *
from sklearn.model_selection import *
from IPython.display import display

from sklearn.metrics import *
from scipy.cluster import hierarchy as hc
import seaborn as sns

from pdpbox import pdp
from plotnine import *
import feather
%%time
train = pd.read_csv(f'../input/train_V2.csv')
test = pd.read_csv(f'../input/test_V2.csv')
samp_sub = pd.read_csv(f'../input/sample_submission_V2.csv')
train.head()
train.shape
#Take a smaller% sample of the data for training
perc = 0.1
train_small=train.sample(n = round(train.shape[0]*perc), random_state = 42)
train.columns
train_small.shape
train_cats(train_small)
#Show that the categories have been successfully assigned
train_small.matchType.cat.categories
df, y, nas = proc_df(train_small, 'winPlacePerc', max_n_cat=19)
df.shape
df.head()
#Create random training and testing samples
#Use only reduced columns of model to predict random forest 
X_train, X_test, y_train, y_test = train_test_split(
    df, y, 
    test_size=0.5, random_state=42)
print("X Train Shape: " + str(X_train.shape))
print("y Train Shape: " + str(y_train.shape))
print("X Test Shape: " + str(X_test.shape))
print("y Test Shape: " + str(y_test.shape))
def print_score(m, X_train, y_train, X_test, y_test, verbose = True):
  if verbose:
    print("R**2 Score of model: " + str(m.score(X_train, y_train)))
    print("Mean Abs Error: " + str(mean_absolute_error(y_test, m.predict(X_test))))
    print("OOB Score: " + str(m.oob_score_))
  else:
    return m.score(X_train, y_train), mean_absolute_error(y_test, m.predict(X_test)), m.oob_score_;
%%time 
#Sample run to test that random forrest is working and function above is working correctly

set_rf_samples(10000)
m = RandomForestRegressor(n_estimators = 40, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt',
                          oob_score = True
                          )
m.fit(X_train, y_train)
print_score(m, X_train, y_train, X_test, y_test)
#Check out variable importance of columns of training data
#Show the top 15 most impoartant variables from initial random forest
pd.DataFrame(m.feature_importances_, index = df.columns, columns=['importance']).sort_values(by = 'importance', ascending = False)[:15]
fi = rf_feat_importance(m, df) #Print out top ten important features
fi[:30].plot('cols', 'imp', 'barh', figsize=(12,7), color='blue')
to_keep = fi[fi.imp>0.001].cols;
to_keep
#This gives the top 20 features
#Use only reduced columns of model to predict random forest 
df_keep = train_small[to_keep].copy()
train_cats(df_keep)
df_keep, _, nas = proc_df(df_keep, max_n_cat=18)
print(df_keep.head())
X_train, X_test, y_train, y_test = train_test_split(
    df_keep, train_small.winPlacePerc, 
    test_size=0.5, random_state=42)
print("X Train Shape: " + str(X_train.shape))
print("y Train Shape: " + str(y_train.shape))
print("X Test Shape: " + str(X_test.shape))
print("y Test Shape: " + str(y_test.shape))
df_keep_all=get_sample(df_keep.join(train_small.winPlacePerc), 500)
sns.regplot(x="walkDistance", y = 'winPlacePerc', data = df_keep_all, ci = 99)
def plot_pdp(feat, clusters = None, feat_name = None):
  feat_name = feat_name or feat
  p = pdp.pdp_isolate(m, x, model_features = x.columns,feature=feat)
  return pdp.pdp_plot(p, feat_name, plot_lines = True,
                     cluster = clusters is not None, n_cluster_centers = clusters)
%%time 
set_rf_samples(10000)
m = RandomForestRegressor(n_estimators = 10, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt')
m.fit(X_train, y_train)
print(m.score(X_train,y_train))
print(mean_absolute_error(y_test, m.predict(X_test)))
#learn here that removing some of the useless columns helps with model performance
X_train.columns
x=get_sample(X_train, 500)
plot_pdp(feat='maxPlace')
feats = ['swimDistance', 'kills']
p = pdp.pdp_interact(m, x, model_features = x.columns, features = feats)
pdp.pdp_interact_plot(p, feats)
%%time
corr = np.round(scipy.stats.spearmanr(get_sample(df_keep, 1000)).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method = 'average')
fig = plt.figure(figsize=(16,10))
dendogram = hc.dendrogram(z, labels = df_keep.columns, orientation='left', leaf_font_size = 16)
plt.show()
def get_oob(X_train, y_train):
    m = RandomForestRegressor(n_estimators = 10, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt',
                             oob_score = True)
    m.fit(X_train, y_train)
    return m.oob_score_
df_keep.columns
#Recall that df_keep has a reduced number of columns that are relevant to selecting a winner
#Get a baseline
get_oob(X_train, y_train)
for c in ('kills', 'killStreaks', 'numGroups', 'maxPlace'):
  print(c, get_oob(X_train.drop(c, axis = 1), y_train))
to_drop = ['kills', 'numGroups']
get_oob(X_train.drop(to_drop, axis = 1), y_train)
%%time
m = RandomForestRegressor(n_estimators = 20, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt',
                         oob_score = True)
m.fit(X_train, y_train)
print_score(m, X_train, y_train, X_test, y_test)
feats = ['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired', 'heals',
       'longestKill', 'damageDealt', 'rideDistance', 'kills', 'assists',
       'swimDistance', 'headshotKills', 'matchDuration', 'numGroups', 'DBNOs',
       'maxPlace', 'killStreaks', 'revives', 'rankPoints', 'winPoints']
res = pd.DataFrame([],columns = ['Feature', 'R**2', 'MAE','OOB'])
r2, mae, oob = print_score(m, X_train, y_train, X_test, y_test, verbose = False)
res = res.append(pd.DataFrame({'Feature':['All Features'], 'R**2':[r2], "MAE":[mae], 'OOB':oob}))
%%time
for f in feats:
    df_subs = X_train.drop(f, axis = 1)
    
    df_test = X_test.drop(f, axis = 1)
    m = RandomForestRegressor(n_estimators = 20, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt', oob_score = True)
    m.fit(df_subs, y_train)
    r2, mae, oob = print_score(m, df_subs,y_train, df_test, y_test, verbose = False)
    res = res.append(pd.DataFrame({'Feature':[f], 'R**2':[r2], "MAE":[mae], 'OOB':oob}))

res.set_index('Feature')    
res.sort_values(by='MAE')
def custom_style(row):
    font = 'normal'
    if row.values[1] == 'All Features':
        font = 'bold'
    return ['font-weight: %s' % font]*len(row.values)

res.reset_index().sort_values(by = "MAE").style.apply(custom_style, axis=1)
%%time
#Inital MAE score 0.0896
#drop_feats = ['killStreaks', 'swimDistance', 'assists','heals','rideDistance','DBNOs']
drop_feats = ['swimDistance']
df_subs = X_train.drop(drop_feats, axis = 1)  
df_test = X_test.drop(drop_feats, axis = 1)

m = RandomForestRegressor(n_estimators = 20, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=100, 
                          max_features = 'sqrt',
                         oob_score = True)
m.fit(df_subs, y_train)
print_score(m, df_subs, y_train, df_test, y_test)
fi = rf_feat_importance(m, df_subs) #Print out top ten important features
fi[:20].plot('cols', 'imp', 'barh', figsize=(12,7), color='blue')
#Keep track of features that are going to be used for final model
kept_features = df_subs.columns
#kept_features = kept_features.append(pd.Index(['winPlacePerc']))
train_f = train[kept_features]
train_cats(train_f)
train_f.head()
train_f.describe(include = 'all')
df, _, nas = proc_df(train_f)
print("training df shape: " +str(df.shape))
print("training y shape: " +str(train.winPlacePerc.shape))
y = train.winPlacePerc.fillna(0).values
from scipy import stats
plt.hist(y)
%%time
set_rf_samples(5000)
m = RandomForestRegressor(n_estimators = 100, 
                          random_state = 42,
                          criterion = 'mae', n_jobs = 4, 
                          min_samples_leaf=1, 
                          max_features = 'sqrt', oob_score = True)

m.fit(df, y)
test_f = test[kept_features]
train_cats(test_f)
test_df, _, nas = proc_df(test_f)
test_df.head()
pred_score = m.predict(test_df)
pred_score
test.shape
pred_score.shape
samp_sub.head()
my_submission = pd.DataFrame({'Id': test.Id, 'winPlacePerc': pred_score})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
