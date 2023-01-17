%load_ext autoreload

%autoreload 2

%matplotlib inline

seed = 93
# Basic

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import scipy

import os

import json

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from pandas_summary import DataFrameSummary

from IPython.display import display



# Sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.decomposition import TruncatedSVD
PATH = '../input/'
!ls {PATH}
train_raw = pd.read_csv(f'{PATH}/train/train.csv', low_memory=False)

test_raw = pd.read_csv(f'{PATH}/test/test.csv', low_memory=False)
train_raw.shape, test_raw.shape
train_raw.head(3)
test_raw.head(3)
train_raw.describe().T
test_raw.describe().T
train_raw.isnull().sum(axis=0)
train_raw.isnull().sum(axis=1).sort_values(ascending=False).head(10)
train_raw['from_dataset'] = 'train'

test_raw['from_dataset'] = 'test'

alldata = pd.concat([train_raw, test_raw], axis = 0)
feats_counts = alldata.nunique(dropna = False)
feats_counts.sort_values()[:10]
alldata.fillna('NaN', inplace=True)
alldata.head(5).T
train_raw['AdoptionSpeed'].value_counts().sort_index().plot('bar');
sns.set(style="darkgrid")

plt.figure(figsize=(14,6))

ax = sns.countplot(x="from_dataset", data=alldata, hue='Type')

plt.title('Number cats and dogs in train and test sets');
plt.figure(figsize=(14,6))

plt.ylabel('Age')

plt.plot(alldata['Age'], '.');
alldata['Age'].value_counts().head(20)
alldata['age_mod_12'] = alldata['Age'].apply(lambda x: True if (x%12)==0 else False)
# dogs

dogs = alldata.loc[alldata['Type'] == 1]

# cats

cats = alldata.loc[alldata['Type'] == 2]
dogs['Name'].value_counts().head(20)
cats['Name'].value_counts().head(20)
alldata['NaN_name'] = alldata['Name'].apply(lambda x: True if str(x) == 'NaN' else False)
alldata['NaN_name'].value_counts()
alldata[alldata['Name'].apply(lambda x: len(str(x))) < 3]['Name'].unique()
alldata['name_len_one_or_two'] = alldata['Name'].apply(lambda x: True if len(str(x)) < 3 else False)
alldata['name_len_one_or_two'].value_counts()
alldata[alldata['Name'].apply(lambda x: len(str(x))) == 3]['Name'].unique()
plt.figure(figsize=(14,6))

plt.ylabel('Fee')

plt.plot(alldata['Fee'], '.');
train_raw['RescuerID'].value_counts().head(15)
test_raw['RescuerID'].value_counts().head(15)
top_20_rescuers = list(train_raw['RescuerID'].value_counts()[:20].index)

top_20_data = train_raw.loc[train_raw['RescuerID'].isin(top_20_rescuers)]
plt.figure(figsize=(10,4))

top_20_data['AdoptionSpeed'].value_counts().sort_index().plot('bar');

plt.title('AdoptionSpeed of the top20 rescuers');
plt.figure(figsize=(10,4))

train_raw['AdoptionSpeed'].value_counts().sort_index().plot('bar');

plt.title('Adoptionspeed in the whole training sample');
def desc_len_feature(df):

    descs = np.stack([item for item in df.Description])

    desc_len = [len(item) for item in descs]

    # Add the features to the dataframe

    df['desc_length'] = desc_len

    

def rescue_count_feature(df):

    rescuers_df = pd.DataFrame(df.RescuerID)

    rescuer_counts = rescuers_df.apply(pd.value_counts)

    rescuer_counts.columns = ['rescue_count']

    rescuer_counts['RescuerID'] = rescuer_counts.index

    df = df.merge(rescuer_counts, how='left', on='RescuerID')

    return df



def name_len_feature(df):

    names = np.stack([item for item in df.Name])

    name_len = [len(item) for item in names]

    df['name_length'] = name_len

    return df

# We have the convert the IDs into categories in order to create the features.

alldata['RescuerID'] = alldata.RescuerID.astype('category')

alldata['RescuerID'] = alldata.RescuerID.cat.codes

desc_len_feature(alldata)

name_len_feature(alldata)

alldata = rescue_count_feature(alldata)
# Kudos to https://www.kaggle.com/artgor/exploration-of-data-step-by-step

def parse_sentiment_files(datatype):

    sentiments = {}

    for filename in os.listdir('../input/' + datatype + '_sentiment'):

        with open('../input/' + datatype + '_sentiment/' +  filename, 'r') as f:

            sentiment = json.load(f)

            pet_id = filename.split('.')[0]

            sentiments[pet_id] = {}

            sentiments[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']

            sentiments[pet_id]['score'] = sentiment['documentSentiment']['score']

            sentiments[pet_id]['language'] = sentiment['language']

            

    return sentiments



def sentiment_features(df, sentiments):

    df['lang'] = df['PetID'].apply(lambda x: sentiments[x]['language']

                                 if x in sentiments else 'no')

    df['magnitude'] = df['PetID'].apply(lambda x: sentiments[x]['magnitude']

                                       if x in sentiments else 0)

    df['score'] = df['PetID'].apply(lambda x: sentiments[x]['magnitude']

                                   if x in sentiments else 0)

    return df
train_sentiment = parse_sentiment_files('train')

test_sentiment = parse_sentiment_files('test')

sentiment_features(alldata, train_sentiment);

sentiment_features(alldata, test_sentiment);
alldata.head()
cols = ['Breed1']

for col in cols:

    frequencies = dict(alldata[col].value_counts()/alldata[col].shape[0])

    alldata[col + '_frequency'] = alldata[col].apply(lambda x: frequencies[x])
headlines = ["President trump won the election", "The world was shocked",

              "Barcelona won the champions league"]
vectorizer = CountVectorizer(analyzer='word')

X = vectorizer.fit_transform(headlines)

columns = [x for x in vectorizer.get_feature_names()]

pd.DataFrame(X.todense(), columns=columns)
count_matrix = pd.DataFrame(X.todense(), columns=columns)

tfidf = TfidfTransformer()

inverse_frequencies = tfidf.fit_transform(count_matrix)

pd.DataFrame(inverse_frequencies.todense(), columns=columns)
def tfidf_features(corpus):

    tfv = TfidfVectorizer(min_df=2,  max_features=None,

        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',

        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,)

    tfv.fit(list(corpus))

    X = tfv.transform(corpus)

    return X

    

def svd_features(df, freq_matrix, n_comps=1):

    svd = TruncatedSVD(n_components=n_comps) #Choose 20 most relevant ones.

    svd.fit(freq_matrix)

    freq_matrix = svd.transform(freq_matrix)

    freq_matrix = pd.DataFrame(freq_matrix, columns=['svd_{}'.format(i) for i in range(n_comps)])

    df = pd.concat((df, freq_matrix), axis=1)

    return df
X = tfidf_features(alldata['Description']); X
alldata=svd_features(alldata, X)
alldata.head()
# Store PetID for later

train_pet_ids = train_raw.PetID

test_pet_ids = test_raw.PetID
alldata = alldata.drop(['Description', 'PetID', 'Name', 'lang', 'RescuerID'], axis=1)
# Split the feature engineered dataframe back into test and train sets.

train = alldata.loc[alldata['from_dataset'] == 'train']

test = alldata.loc[alldata['from_dataset'] == 'test']
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(train['AdoptionSpeed'])

y = pd.DataFrame(y, columns=['AdoptionSpeed'])
train = train.drop(['from_dataset', 'AdoptionSpeed'], axis=1)

test = test.drop(['from_dataset', 'AdoptionSpeed'], axis=1);
m = RandomForestClassifier(n_estimators=500, random_state=seed,

                           max_features='sqrt',

                           min_samples_leaf=25, n_jobs=-1);

m.fit(train, y);
def rf_feature_importance(df, m):

    df = pd.DataFrame({'cols' : df.columns, 'imp' : m.feature_importances_}).sort_values('imp', ascending=False)

    return df
fi = rf_feature_importance(train, m)

fi[:25].plot('cols', 'imp', kind='barh', legend=False);
train.shape
to_keep = fi[fi.imp>0.005].cols; len(to_keep)

train = train[to_keep].copy()

test = test[to_keep].copy()

train.shape
train.head(3)
m = RandomForestClassifier(n_estimators=500, random_state=seed,

                           max_features='sqrt',

                           min_samples_leaf=1, n_jobs=-1)



test_preds = np.zeros(test.shape[0])

results=[]

n_folds = 4

cv = StratifiedKFold(n_splits=n_folds, random_state=seed)

for (train_idx, valid_idx) in cv.split(train,y):

    m.fit(train.iloc[train_idx], y.iloc[train_idx])

    score = metrics.cohen_kappa_score(y.loc[valid_idx], m.predict(train.iloc[valid_idx]), weights='quadratic')

    results.append(score)

    y_test = m.predict(test)

    test_preds += y_test.reshape(-1)/n_folds

    

mean = np.mean(results)

std = np.std(results)
print(f'Mean kappa score: {mean} with std: {std}')
fi = rf_feature_importance(train, m)

fi[:25].plot('cols', 'imp', kind='barh', legend=False);
df_sub = pd.DataFrame({'PetID' : test_pet_ids})

df_sub['AdoptionSpeed'] = test_preds

df_sub['AdoptionSpeed'] = df_sub['AdoptionSpeed'].astype(int)

df_sub.head()
df_sub.to_csv(f'submission.csv', index=False)