import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('/kaggle/input/mldub-comp1/train_data.csv')
test = pd.read_csv('/kaggle/input/mldub-comp1/test_data.csv')
sample_sub = pd.read_csv('/kaggle/input/mldub-comp1/sample_sub.csv')
train.head()
train.describe()
sns.pairplot(train, diag_kind="kde")
train.info()
train.isnull().sum()
fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=train['comments'], y=train['target_variable'], color=('yellowgreen'), alpha=0.5)
#plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Comments scatter plot', fontsize=15, weight='bold' )

fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=train['videos'], y=train['target_variable'], color=('yellowgreen'), alpha=0.5)
#plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Videos scatter plot', fontsize=15, weight='bold' )

fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=train['photos'], y=train['target_variable'], color=('yellowgreen'), alpha=0.5)
#plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Photos scatter plot', fontsize=15, weight='bold' )
feature = 'comments'
df = train.copy()
upper_limit = df[feature].quantile(0.95)
lower_limit = df[feature].quantile(0.05)
df.loc[(df[feature] > upper_limit),feature] = upper_limit
df.loc[(df[feature] < lower_limit),feature] = lower_limit
fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=df['comments'], y=df['target_variable'], color=('yellowgreen'), alpha=0.5)
#plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Comments scatter plot (no outliers)', fontsize=15, weight='bold' )
df['comments'].hist()
def numeric(df, feature_list):
    df_tmp = pd.DataFrame()
    for feature in feature_list:
        df_tmp[feature] = df[feature]
        upper_limit = df_tmp[feature].quantile(0.95)
        lower_limit = df_tmp[feature].quantile(0.05)
        df_tmp.loc[(df_tmp[feature] > upper_limit),feature] = upper_limit
        df_tmp.loc[(df_tmp[feature] < lower_limit),feature] = lower_limit
        df_tmp[feature] = np.log1p(np.abs(df_tmp[feature]))
    return df_tmp
df_tmp = pd.DataFrame()
df_tmp['added_year'] =  pd.to_datetime(train['date_added']).dt.year
df_tmp['current_year'] = pd.to_datetime('2020-01-01')
df_tmp['current_year'] = df_tmp['current_year'].dt.year
df_tmp['age'] = df_tmp['current_year'] - df_tmp['added_year']
df_tmp['age'].hist()
df_tmp['target_variable'] = train['target_variable']
fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=df_tmp['age'], y=df_tmp['target_variable'], color=('yellowgreen'), alpha=0.5)
#plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Age scatter plot (no outliers)', fontsize=15, weight='bold' )
def age(df):
    df_tmp = pd.DataFrame()
    df_tmp['added_year'] =  pd.to_datetime(df['date_added']).dt.year
    df_tmp['current_year'] = pd.to_datetime('2020-01-01')
    df_tmp['current_year'] = df_tmp['current_year'].dt.year
    df_tmp['age'] = df_tmp['current_year'] - df_tmp['added_year']
    df_tmp['age_log'] = np.log1p(np.abs(df_tmp['age']))
    return pd.DataFrame(df_tmp['age_log'])
age(train)
df[feature].fillna('').str.len()
def text_chars_len(df, feature_list):
    df_tmp = pd.DataFrame()
    for feature in feature_list:
        df_tmp[feature+'_chars_len'] = np.log1p(np.abs(df[feature].fillna('').str.len()))
    return df_tmp
text_length = ['about', 'name', 'origin', 'origin_place', 'other_text', 'tags']
text_chars_len(train, text_length)
text_chars_len(train, text_length).hist()
print(train['status'].unique())
print(test['status'].unique())
train['status'].value_counts()
test['status'].value_counts()
for df in [train, test]:
    df.loc[~df['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
def get_status_frequency():
    df_tmp = pd.DataFrame()
    df_tmp['status'] = train['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    freq_dict = df_tmp['status'].value_counts().to_dict()
    return freq_dict

def status_frequency(df):
    df_tmp = pd.DataFrame()
    freq_dict = get_status_frequency()
    df_tmp['status'] = df['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    df_tmp['status'] = np.log1p(np.abs(df_tmp['status'].apply(lambda x: freq_dict[x])))
    return df_tmp
status_frequency(train)
status_frequency(train).hist()
train['type'].unique()
train['type'].value_counts()
def dum_type(df):
    df_tmp = pd.DataFrame()
    df_concat = pd.concat([train['type'].str.lower(), test['type'].str.lower()])
    df_concat.fillna('unknown', inplace = True) # It can be either Other or Unknown 
    freq_dict = df_concat.value_counts().head(100).to_dict()
    item_list = [key for key in freq_dict]   
    df_tmp['type_dum'] = df['type'].str.lower().fillna('unknown')
    df_tmp.loc[~df_tmp['type_dum'].isin(item_list), 'type_dum'] = 'other'
    df_tmp = pd.get_dummies(df_tmp)
    return df_tmp
dum_type(train)
def dum_status(df):
    df_tmp = pd.DataFrame()
    df_tmp['status'] = df['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    df_tmp = pd.get_dummies(df_tmp)
    return df_tmp
dum_status(train)
def cat_status(df):
    df_tmp = pd.DataFrame()
    df_tmp['status_cat'] = df['status']
    df_tmp.loc[~df_tmp['status_cat'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status_cat'] = 'Unknown'
    mapping = {'Deadpool': 0,
               'Submission': 1,
               'Confirmed': 2,
               'Unknown': -1}
    df_tmp['status_cat'] = df_tmp['status_cat'].apply(lambda x: mapping[x]).astype(int)
    return df_tmp
cat_status(train)
def numeric_origin_year(df):
    df_tmp = pd.DataFrame()
    df_tmp['origin_year_numeric'] = df['origin_year'].apply(lambda x: 1900 if x == 'Unknown' else 2020 - int(x)).astype(int)
    df_tmp['origin_year_numeric'] = np.log1p(np.abs(df_tmp['origin_year_numeric']))
    return df_tmp
numeric_origin_year(train)
numeric_origin_year(train).hist()
train.isnull().sum()
def text_mining(df, feature):
    df_tmp = pd.DataFrame()
    lemmatizer = WordNetLemmatizer()
    train_text = train[feature].fillna('')
    test_text = test[feature].fillna('')
    text = pd.concat([train_text, test_text]).apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)))
    df[feature+'_norm'] = df[feature].fillna('').apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)))
    word_vectorizer = TfidfVectorizer(min_df=0.05, stop_words = 'english',analyzer='word', token_pattern=r"(?u)\b[^\d\W]+\w\b")
    word_vectorizer.fit(text)
    df_tmp = word_vectorizer.transform(df[feature+'_norm'].fillna(''))
    df_tmp = pd.DataFrame(df_tmp.toarray(), columns=word_vectorizer.get_feature_names())
    df_tmp = df_tmp.add_prefix(feature+'_')
    print('Text mining {} - done'.format(feature))
    return df_tmp
text_mining(train, 'tags')
# Numeric features from dataset
def numeric(df, feature_list):
    df_tmp = pd.DataFrame()
    for feature in feature_list:
        df_tmp[feature] = df[feature]
        upper_limit = df_tmp[feature].quantile(0.95)
        lower_limit = df_tmp[feature].quantile(0.05)
        df_tmp.loc[(df_tmp[feature] > upper_limit),feature] = upper_limit
        df_tmp.loc[(df_tmp[feature] < lower_limit),feature] = lower_limit
        df_tmp[feature] = np.log1p(np.abs(df_tmp[feature]))
    return df_tmp

# Add numeric feature "age"
def age(df):
    df_tmp = pd.DataFrame()
    df_tmp['added_year'] =  pd.to_datetime(df['date_added']).dt.year
    df_tmp['current_year'] = pd.to_datetime('2020-01-01')
    df_tmp['current_year'] = df_tmp['current_year'].dt.year
    df_tmp['age'] = df_tmp['current_year'] - df_tmp['added_year']
    df_tmp['age_log'] = np.log1p(np.abs(df_tmp['age']))
    return pd.DataFrame(df_tmp['age'])

# Add numeric features: count chars in the text and apply log function
def text_chars_len(df, feature_list):
    df_tmp = pd.DataFrame()
    for feature in feature_list:
        df_tmp[feature+'_chars_len'] = np.log1p(np.abs(df[feature].fillna('').str.len()))
    return df_tmp

# Get frequency of values in categorical field "status"
def get_status_frequency():
    df_tmp = pd.DataFrame()
    df_tmp['status'] = train['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    freq_dict = df_tmp['status'].value_counts().to_dict()
    return freq_dict

# Add numeric feature "status" frequency
def status_frequency(df):
    df_tmp = pd.DataFrame()
    freq_dict = get_status_frequency()
    df_tmp['status'] = df['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    df_tmp['status'] = np.log1p(np.abs(df_tmp['status'].apply(lambda x: freq_dict[x])))
    return df_tmp

# Add dummy features of categorical field "type"
def dum_type(df):
    df_tmp = pd.DataFrame()
    df_concat = pd.concat([train['type'].str.lower(), test['type'].str.lower()])
    df_concat.fillna('unknown', inplace = True) # It can be either Other or Unknown 
    freq_dict = df_concat.value_counts().head(100).to_dict()
    item_list = [key for key in freq_dict]   
    df_tmp['type_dum'] = df['type'].str.lower().fillna('unknown')
    df_tmp.loc[~df_tmp['type_dum'].isin(item_list), 'type_dum'] = 'other'
    df_tmp = pd.get_dummies(df_tmp)
    return df_tmp

# Add dummy features of categorical field "status"
def dum_status(df):
    df_tmp = pd.DataFrame()
    df_tmp['status'] = df['status']
    df_tmp.loc[~df_tmp['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
    df_tmp = pd.get_dummies(df_tmp)
    return df_tmp

# Add categorical feature on field "status"
def cat_status(df):
    df_tmp = pd.DataFrame()
    df_tmp['status_cat'] = df['status']
    df_tmp.loc[~df_tmp['status_cat'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status_cat'] = 'Unknown'
    mapping = {'Deadpool': 0,
               'Submission': 1,
               'Confirmed': 2,
               'Unknown': -1}
    df_tmp['status_cat'] = df_tmp['status_cat'].apply(lambda x: mapping[x]).astype(int)
    return df_tmp

# Add numeric feature on field "origin_year"
def numeric_origin_year(df):
    df_tmp = pd.DataFrame()
    df_tmp['origin_year_numeric'] = df['origin_year'].apply(lambda x: 1900 if x == 'Unknown' else 2020 - int(x)).astype(int)
    df_tmp['origin_year_numeric'] = np.log1p(np.abs(df_tmp['origin_year_numeric']))
    return df_tmp

# Text mining
def text_mining(df, feature):
    df_tmp = pd.DataFrame()
    lemmatizer = WordNetLemmatizer()
    train_text = train[feature].fillna('')
    test_text = test[feature].fillna('')
    text = pd.concat([train_text, test_text]).apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)))
    df[feature+'_norm'] = df[feature].fillna('').apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)))
    word_vectorizer = TfidfVectorizer(min_df=0.05, stop_words = 'english',analyzer='word', token_pattern=r"(?u)\b[^\d\W]+\w\b")
    word_vectorizer.fit(text)
    df_tmp = word_vectorizer.transform(df[feature+'_norm'].fillna(''))
    df_tmp = pd.DataFrame(df_tmp.toarray(), columns=word_vectorizer.get_feature_names())
    df_tmp = df_tmp.add_prefix(feature+'_')
    print('Text mining {} - done'.format(feature))
    return df_tmp

# Assemble all features in dataset
def dataset(df): 
    df_numerics = numeric(df, numerics)
    df_age = age(df)
    df_text_chars_len = text_chars_len(df, text_length)
    df_status_frequency = status_frequency(df)
    df_dum_type = dum_type(df)
    df_dum_status = dum_status(df)
    df_cat_status = cat_status(df)
    df_numeric_origin_year = numeric_origin_year(df)
    df_text_mining = text_mining(df, 'tags')

    df_tmp = df_numerics.copy()
    df_tmp = df_tmp.merge(df_age, left_index=True, right_index=True) 
    df_tmp = df_tmp.merge(df_text_chars_len, left_index=True, right_index=True)   
    df_tmp = df_tmp.merge(df_status_frequency, left_index=True, right_index=True)    
    df_tmp = df_tmp.merge(df_dum_type, left_index=True, right_index=True)    
    df_tmp = df_tmp.merge(df_dum_status, left_index=True, right_index=True) 
    df_tmp = df_tmp.merge(df_numeric_origin_year, left_index=True, right_index=True)
    df_tmp = df_tmp.merge(df_text_mining, left_index=True, right_index=True)    
    return df_tmp

# Main body
numerics = ['comments', 'photos', 'videos']
text_length = ['about', 'name', 'origin', 'origin_place', 'other_text', 'tags']
X_train = dataset(train)
X_test = dataset(test)
y_train = train['target_variable']
def normalize(df):
    return ((df-df.mean())/df.std())
X_train = normalize(X_train)
X_test = normalize(X_test)
X_train.head()
df = X_train.copy()
df['target_variable'] = train['target_variable']
numcorr = df.corr()
Num=numcorr['target_variable'].sort_values(ascending=False).head(100).to_frame()
cm = sns.light_palette("cyan", as_cmap=True)
plt = Num.style.background_gradient(cmap=cm)
plt
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [70,80,90],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [3,5,7],
    'n_estimators': [60, 70, 80]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
grid_search.best_params_
cv = KFold(3, shuffle=True, random_state=42)
# Update parametrs from grid search
model = RandomForestRegressor(bootstrap=False,
                              max_depth=90,
                              max_features='sqrt',
                              min_samples_leaf=1,
                              min_samples_split=7,
                              n_estimators=60,
                              random_state=42,
                              n_jobs=-1,
                              verbose=2,
                            )
cv_results = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=cv,
                             scoring='neg_mean_squared_error')
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('RMSE: {}'.format(np.sqrt([-x for x in cv_results])))
print('RMSE MEAN: {}'.format(np.sqrt([-x for x in cv_results]).mean()))
preds = model.predict(normalize(X_test))
submission = pd.DataFrame()
submission['id'] = test['id']
submission['target_variable'] = preds
submission.to_csv('submission.csv', index=False)