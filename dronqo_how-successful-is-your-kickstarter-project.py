%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from collections import Counter
from wordcloud import WordCloud


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# read in dataset
kick = pd.read_csv('../input/ks-projects-201801.csv', encoding='latin1')
# basic statistics
print('Dimension of the dataset:', kick.shape)

# show first 5 observations of dataframe for exploration purposes
kick.head()
# name of the columns
kick.columns
# data type of each column
kick.get_dtype_counts()
# number of missing values per column
kick.isnull().sum().sort_values(ascending = False)
# remove rows with missing values
kick.dropna(inplace=True)
# number of unique values per column, sorted by descending order
kick.T.apply(lambda x: x.nunique(), axis=1).sort_values(ascending=False)
def categorical_with_per_count(kick, feature):
    '''
    Calculate frequency of the categorical feature with % and count base.
    Sorted on the descending order.
    '''
    
    # calculate frequency on % and value
    freq_merged = pd.concat([kick[feature].value_counts(normalize=True) * 100,
                             kick[feature].value_counts(normalize=False)], axis=1)
    # rename columns
    freq_merged.columns = [feature + '_%', feature + '_count']
    return freq_merged
categorical_with_per_count(kick, 'state')
# keep `failed` and `successful` states
kick.query("state in ['failed', 'successful']", inplace=True)
# class balance of the dataframe
categorical_with_per_count(kick, 'state')
# select features for further analysis
kick = kick.loc[:, ['name', 'category', 'main_category', 'deadline',
                    'usd_goal_real', 'launched', 'state', 'country']]
# rename `usd_goal_real` to `goal`
kick.rename(columns={'usd_goal_real':'goal'}, inplace=True)
# frequency of the main category
categorical_with_per_count(kick, 'main_category')
# change dimension of the plot
dims = (10, 8)
fig, ax = plt.subplots(figsize = dims)

# barplot of the main categories by descending order
sns.countplot(
    y=kick.main_category,
    order = kick['main_category'].value_counts().index
)
# top 10 the most frequent country
categorical_with_per_count(kick, 'country').head(n=10)
kick.goal.describe()
# calculate frequency of the goal: the most popular goal
categorical_with_per_count(kick, 'goal').head(n=10)
# combine different plots into one: goal and log(goal)
dims = (14, 8)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=dims)
sns.distplot(kick.goal, ax=ax1)
sns.distplot(np.log1p(kick.goal), ax=ax2)
# convert strings to `datetime`
kick['launced'] = pd.to_datetime(kick.launched)
kick['deadline'] = pd.to_datetime(kick.deadline)
kick = kick.sort_values('launced')
def show_wordcloud(data, title = None):
    '''Split names by space and generate word counts.'''
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
# successful Projects
show_wordcloud(kick[kick.state == 'successful']['name'])
# failed Projects
show_wordcloud(kick[kick.state == 'failed']['name'])
# initialize new data frame
kk = pd.DataFrame()
# length of the name
kk['name_len'] = kick.name.str.len()
# if name contains a question mark
kk['name_is_question'] = (kick.name.str[-1] == '?').astype(int)
# if name contains an exclamation mark
kk['name_is_exclamation'] = (kick.name.str[-1] == '!').astype(int)
# if name is uppercase
kk['name_is_upper'] = kick.name.str.isupper().astype(float)
def count_non_character(row):
    '''Number of non character in the sentence'''
    return sum((0 if c.isalpha() else 1 for c in str(row)))
# number of non character in the name
kk['name_non_character'] = kick.name.apply(count_non_character)
# number of words in the name
kk['name_number_of_word'] = kick.name.apply(lambda x: len(str(x).split(' ')))
# We generate new feature based on ratio between vowels and other alpha characters
def countVowelstoLettersRatio(s):
    '''Count ratio between vowels and letters'''
    s = str(s)
    count = 1  
    vowels = 0
    for i in s:
        if i.isalpha():
            count = count + 1
            if i in 'aeiou':
                vowels = vowels + 1
    return ((vowels * 1.0) / count)

# for each name calculate vowels ratio
kk['name_vowel_ratio'] = kick.name.apply(countVowelstoLettersRatio)
# create indicator variable for `country` variable
kk['country_is_us'] = (kick.country == 'US').astype(int)
kk['Goal_1000'] = kick.goal.apply(lambda x: x // 1000)
kk['Goal_500'] = kick.goal.apply(lambda x: x // 500)
kk['Goal_10'] = kick.goal.apply(lambda x: x // 10)
# log transformation of `goal` to reduce skewness 
kick['goal'] = np.log1p(kick.goal)
kk['goal'] = kick['goal']
from datetime import datetime
import time

def to(dt):
    '''Add timestamp as a value'''
    return time.mktime(dt.timetuple())

kick['timestamp'] = kick['launced'].apply(to)    
# We will create data frames containing only single main category
categories = set(kick.main_category)
frames = {}
for ct in categories:
    frames[ct] = kick[kick['main_category'] == ct]
# We will use Progressbar to track progress as it istime consuming operation
import pyprind
pbar = pyprind.ProgBar(331675)


def getElementsInRange(cat,end,week):
    '''Get number of launched projects in given range from (end - week) to end'''
    global pbar
    pob = frames[cat]
    start = end - pd.DateOffset(weeks = week)
    # as we sorted our projects by launch date earlier geting number of projects in given date range is easy
    value = pob['launced'].searchsorted(end)[0] - pob['launced'].searchsorted(start)[0]
    pbar.update()
    return value
# Number of projects in same category for last week    
kk['Last_Week'] = kick.apply(lambda x: getElementsInRange(x['main_category'],x['launced'],1),axis = 1) 
pbar = pyprind.ProgBar(331675)
# Number of projects in same category for last month    
kk['Last_Month'] = kick.apply(lambda x: getElementsInRange(x['main_category'],x['launced'],4),axis = 1) 
pbar = pyprind.ProgBar(331675)
# Number of projects in same category for last year    
kk['Last_Year'] = kick.apply(lambda x: getElementsInRange(x['main_category'],x['launced'],52),axis = 1) 
pbar = pyprind.ProgBar(331675)
# Number of projects in same category for last 3 months  
kk['Last_3_Month'] = kick.apply(lambda x: getElementsInRange(x['main_category'],x['launced'],13),axis = 1)
pbar = pyprind.ProgBar(331675)
# Number of projects in same category for last 6 months  
kk['Last_6_Month'] = kick.apply(lambda x: getElementsInRange(x['main_category'],x['launced'],26),axis = 1)
def getDelta(a,b):
    '''Get diffence in days between launch and deadline'''
    return (a - b).days

# Duration of the project   
kk['Duration'] = kick.apply(lambda x: getDelta(x['deadline'],x['launced']),axis = 1)
## Month of launch
kk['Month'] = kick['launced'].apply(lambda x : x.month)
# Hour at which project was launched
kk['Hour'] = kick['launced'].apply(lambda x : x.hour)
## Month of deadline
kk['deadline_month'] = kick['deadline'].apply(lambda x : x.month)
# indicator feature for weekend
kk['isLaunchWeekend'] = kick['launced'].apply(lambda x : int(x.weekday() > 5))
kk['Category'] = kick['category']
kk['main_category'] = kick['main_category']
def getRangeMean(cat,end,week):
    global pbar
    pob = frames[cat]
    start = end - pd.DateOffset(weeks = week)
    value = pob.iloc[pob['launced'].searchsorted(start)[0]:pob['launced'].searchsorted(end)[0]]['goal'].mean()
    pbar.update()
    return value
pbar = pyprind.ProgBar(331675)
# Mean goal for category last month
kk['mean_goal_in_category_last_month'] = kick.apply(lambda x: getRangeMean(x['main_category'],x['launced'],4),axis = 1) 
def getRangeMedian(cat,end,week):
    global pbar
    pob = frames[cat]
    start = end - pd.DateOffset(weeks = week)
    value = pob.iloc[pob['launced'].searchsorted(start)[0]:pob['launced'].searchsorted(end)[0]]['goal'].median()
    pbar.update()
    return value
pbar = pyprind.ProgBar(331675)
# Median goal for category last month
kk['median_goal_in_category_last_month'] = kick.apply(lambda x: getRangeMedian(x['main_category'],x['launced'],4),axis = 1) 
pbar = pyprind.ProgBar(331675)
# Mean goal for category last month
kk['mean_goal_in_category_last_year'] = kick.apply(lambda x: getRangeMean(x['main_category'],x['launced'],52),axis = 1) 
pbar = pyprind.ProgBar(331675)
# Median goal in category last month
kk['median_goal_in_category_last_year'] = kick.apply(lambda x: getRangeMedian(x['main_category'],x['launced'],52),axis = 1) 
kk['median_goal_Last_6_Month'] = kick.apply(lambda x: getRangeMedian(x['main_category'],x['launced'],26),axis = 1)
kk['mean_goal_Last_6_Month'] = kick.apply(lambda x: getRangeMean(x['main_category'],x['launced'],26),axis = 1)
kk['mean_goal_Last_Week'] = kick.apply(lambda x: getRangeMean(x['main_category'],x['launced'],1),axis = 1)
kk['median_goal_Last_Week'] = kick.apply(lambda x: getRangeMedian(x['main_category'],x['launced'],1),axis = 1)
kk = kk.fillna(0)   # fill created NAs with 0s
# include state of project
kk['state'] = kick.state
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.violinplot(x="main_category", y="Duration", hue= 'state', data=kk, split=True)
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.countplot(x='Category', palette="pastel",data= kk[kk['state'] == 'successful'].groupby("Category")
              .filter(lambda x: len(x) > 3000), ax=ax)
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.countplot(x='Category', palette="pastel",data= kk[kk['state'] == 'failed']
              .groupby("Category").filter(lambda x: len(x) > 4500), ax=ax)
frame = pd.DataFrame()
counts = kk.Category.value_counts()
succ = kk[kk.state == 'successful'].Category.value_counts()


rows_list = []
for i in counts.keys():
    dict1 = {}
    # get input row in dictionary format
    # key = col_name
    dict1['Category'] = i
    dict1['Percent'] = succ[i] / counts[i] * 100
    rows_list.append(dict1)

frame = pd.DataFrame(rows_list)   
so = frame.sort_values('Percent').tail(10)
high = frame.sort_values('Percent').head(10)
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.barplot(x='Category',y = 'Percent', palette="pastel",data = so, ax=ax)
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.barplot(x='Category',y = 'Percent', palette="pastel",data = high, ax=ax)
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.distplot(kk.name_vowel_ratio[kk.state == 'successful'], kde=True,color='g', rug=False, ax = ax);
sns.distplot(kk.name_vowel_ratio[kk.state == 'failed'], kde=True,color='r', rug=False, ax = ax);
corr = kk.corr()
dims = (16, 10)
fig, ax = plt.subplots(figsize = dims)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,ax = ax)
# seperate dependent and independent part into seperate variables
y = (kk.state == 'successful').astype(int)
x = kk.drop(['state'], axis = 1)
# transform target variable into categorical value
class_le = LabelEncoder()
y = class_le.fit_transform(y.values)
# create dummy variable for main category and category variables
x_pca = pd.get_dummies(x, columns = ['Category','main_category'], drop_first=True)
# explained variance ratio with cumulative sum
pca = PCA(n_components=10, random_state=1)
X_pca = pca.fit_transform(x_pca.values)
# plot cumulative variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.title('Explained variance ratio by Principal components-PCA', fontsize=16)
plt.step(range(1, 11), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.axvline(2, linestyle=':', label='n_components chosen', c='red')
plt.legend(prop=dict(size=12))

plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='0 - successful')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='1 - failed')

plt.title('Variation of the classes based on first two components', fontsize=16)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='best', prop=dict(size=12))

plt.show()
# perform One-hot-encoding
x = pd.get_dummies(x, columns = ['main_category','Category'], drop_first=True)
# initialize balanced class indeces
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

# initialize values into different variables
for train_index, test_index in sss.split(x, y):
    X_train, X_test = x.iloc[train_index, :].reset_index(drop=True), x.iloc[test_index, :].reset_index(drop=True)
    y_train, y_test = y[train_index], y[test_index]
print('Train dataset class distribution:')
print(np.bincount(y_train))

print('\nTest dataset class distribution:')
print(np.bincount(y_test))
# initialize list of the transformations
pipe_rf = Pipeline(steps=[
    ('std', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3))
])

# list of the parameters to test
param_grid = [
    {
        'rf__max_depth': [3, 4, 5],
        'rf__min_samples_leaf': [4, 5]
    }
]

# initialize grid search
estimator = GridSearchCV(
    pipe_rf,
    cv=StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=1), # preserve class balance
    param_grid=param_grid,
    scoring='roc_auc'
)

# fit train data
estimator.fit(X_train, y_train)
print('Best Grid Search result :', estimator.best_score_)
print('Best parameter :', estimator.best_params_)
# get best classifier
clf_rf = estimator.best_estimator_

# predict test data set
y_pred_rf = estimator.predict(X_test)
# test data set auc error
print('Train data ROC/AUC :', )
print('Test data ROC/AUC :', roc_auc_score(y_true=y_test, y_score=y_pred_rf))

# confusion matrix
print('\nConfusion matrix')
print(confusion_matrix(y_true=y_test, y_pred=y_pred_rf))

# classification matrix
print('\nClassification matrix')
print(classification_report(y_true=y_test, y_pred=y_pred_rf))
# initialize list of the transformations
pipe_lr = Pipeline(steps=[
    ('std', StandardScaler()),
    ('lr', LogisticRegression(penalty='l1', C=0.001, random_state=0))
])

param_grid = [
    {
        'lr__C': [0.001, 0.01, 0.1, 1],
        'lr__penalty': ['l1', 'l2']  # regularizatian parameter
    }
]

estimator = GridSearchCV(
    pipe_lr,
    cv=StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=1), # preserve class balance
    param_grid=param_grid,
    scoring='roc_auc',
)

# fit training data
estimator.fit(X_train, y_train)
print('Best Grid Search result :', estimator.best_score_)
print('Best parameter :', estimator.best_params_)
# get best estimator
clf_lr = estimator.best_estimator_

# predict test data set
y_pred_lr = estimator.predict(X_test)
# test data set auc error
print('Train data ROC/AUC :', )
print('Test data ROC/AUC :', roc_auc_score(y_true=y_test, y_score=y_pred_lr))

# confusion matrix
print('\nConfusion matrix')
print(confusion_matrix(y_true=y_test, y_pred=y_pred_lr))

# classification matrix
print('\nClassification matrix')
print(classification_report(y_true=y_test, y_pred=y_pred_lr))
# initialize list of the transformations

# We will use parameters that were found based on experiments made on local machine

clf_gbm = lgb.LGBMClassifier(
        n_estimators=1000,
        num_leaves=25,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=9,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
)


# fit training data
clf_gbm.fit(X_train,
              y_train,
              eval_metric='auc', 
              verbose=0)
# predict test data set
y_pred_gbm = clf_gbm.predict(X_test)
# test data set auc error
print('Test data ROC/AUC :', roc_auc_score(y_true=y_test, y_score=y_pred_gbm))

# confusion matrix
print('\nConfusion matrix')
print(confusion_matrix(y_true=y_test, y_pred=y_pred_gbm))

# classification matrix
print('\nClassification matrix')
print(classification_report(y_true=y_test, y_pred=y_pred_gbm))
# feature importance for Light GBM
predictor_columns = X_train.columns
feat_import = list(zip(predictor_columns, list(clf_gbm.feature_importances_)))
ns_df = pd.DataFrame(data = feat_import, columns=['Feat_names', 'Importance'])
ns_df_sorted = ns_df.sort_values(['Importance', 'Feat_names'], ascending = [False, True])

ns_df_sorted
