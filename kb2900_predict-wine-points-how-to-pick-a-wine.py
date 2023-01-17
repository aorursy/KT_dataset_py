import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

sns.set_context('notebook', font_scale=1.5)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

print("Before removing duplicates:", len(wine_reviews))

wine_reviews.tail()
# if we count for description, we can see there are multiple duplicate rows for each description

# wine_reviews.description.value_counts(dropna=False)



# dedup based on all columns

wine_reviews = wine_reviews.drop_duplicates()

print("Removing duplicates based on all columns:", len(wine_reviews))
# dedup based on description

wine_reviews_ddp = wine_reviews.drop_duplicates('description')

print("Removing duplicates based on description:", len(wine_reviews_ddp))
# full join two dedupped data and find the rows only in the first with '_merge' flag

wine_reviews_all = wine_reviews.merge(wine_reviews_ddp, how='outer', indicator=True)

dup_wine_desc = wine_reviews_all[wine_reviews_all['_merge']=='left_only'].description



wine_reviews_all[wine_reviews_all['description'].isin(dup_wine_desc)]
# just to illustrate the difference

wine_reviews.reset_index().tail()
wine_reviews = wine_reviews.reset_index(drop = True)
wine_reviews.describe()
print('Skewness=%.3f' %wine_reviews['points'].skew())

print('Kurtosis=%.3f' %wine_reviews['points'].kurtosis())

sns.distplot(wine_reviews['points'], bins=20, kde=True);
print('Skewness=%.3f' %wine_reviews['price'].skew())

print('Kurtosis=%.3f' %wine_reviews['price'].kurtosis())

sns.distplot(wine_reviews['price'].dropna());
print('Skewness=%.3f' %np.log(wine_reviews['price']).skew())

print('Kurtosis=%.3f' %np.log(wine_reviews['price']).kurtosis())

sns.distplot(np.log(wine_reviews['price']).dropna());
sns.set(style = 'whitegrid', rc = {'figure.figsize':(8,6), 'axes.labelsize':12})

sns.scatterplot(x = 'price', y = 'points', data = wine_reviews);
sns.boxplot(x = 'points', y = 'price', palette = 'Set2', data = wine_reviews, linewidth = 1.5);
wine_reviews['points'].corr(wine_reviews['price'])
wine_cat = wine_reviews.select_dtypes(include=['object']).columns

print('n rows: %s' %len(wine_reviews))

for i in range(len(wine_cat)):

    c = wine_cat[i]

    print(c, ': %s' %len(wine_reviews[c].unique()))
wine_reviews['country'].value_counts()
print(wine_reviews['region_2'].isna().sum())

wine_reviews['region_2'].value_counts()
fig, ax = plt.subplots(1, 1, figsize = (12, 7))

col_order = wine_reviews.groupby(['country'])['points'].aggregate(np.median).reset_index().sort_values('points')

p = sns.boxplot(x = 'country', y = 'points', palette = 'Set3', data = wine_reviews, order = col_order['country'], linewidth = 1.5)

plt.setp(p.get_xticklabels(), rotation = 90)

ax.set_xlabel('');
fig, ax = plt.subplots(1, 1, figsize = (10, 6))

col_order = wine_reviews.groupby(['region_2'])['points'].aggregate(np.median).reset_index().sort_values('points')

p = sns.boxplot(x = 'region_2', y = 'points', palette = 'Set3', data = wine_reviews, order = col_order['region_2'], linewidth = 1.5)

plt.setp(p.get_xticklabels(), rotation = 60)

ax.set_xlabel('');
wine_reviews['word_count'] = wine_reviews['description'].apply(lambda x: len(str(x).split(" ")))

sns.boxplot(x = 'points', y = 'word_count', palette = 'Set3', data = wine_reviews, linewidth = 1.5);
print(wine_reviews.isnull().sum())
# calculate percentage of missing values

wine_missing = pd.DataFrame(wine_reviews.isnull().sum()/len(wine_reviews.index) * 100)

wine_missing.columns = ['percent']

wine_missing
# first, we know that region_2 has nearly 60% of missing values so drop it

wine_reviews.drop(['region_2'], inplace = True, axis = 1, errors = 'ignore')



# second, it is not sensible to replace na with most frequent category for designation and region_1, so I create a new Unkown category

wine_reviews['designation'].fillna('Unknown', inplace = True)

wine_reviews['region_1'].fillna('Unknown', inplace = True)



# last, replace na with median for numeric variable price

wine_reviews['price'].fillna((wine_reviews['price'].median()), inplace = True)

wine_reviews.tail()
wine_reviews[wine_reviews['country'].isna()]
wine_reviews[wine_reviews.winery.isin(['Tsililis', 'Büyülübağ', 'Chilcas'])]
wine_reviews.loc[wine_reviews.designation == 'Askitikos', 'country'] = 'Greece'

wine_reviews.loc[wine_reviews.designation == 'Askitikos', 'province'] = 'Thessaly'



wine_reviews.loc[wine_reviews.designation == 'Shah', 'country'] = 'Turkey'

wine_reviews.loc[wine_reviews.designation == 'Shah', 'province'] = 'Marmara'



# As I have said, San Rafael is located in Maule Region; for simplicity, I assign 'Maule Valley' in line with other rows

wine_reviews.loc[wine_reviews.designation == 'Piedra Feliz', 'country'] = 'Chile'

wine_reviews.loc[wine_reviews.designation == 'Piedra Feliz', 'province'] = 'Maule Valley'

wine_reviews.loc[wine_reviews.designation == 'Piedra Feliz', 'region_1'] = 'San Rafael'
wine_reviews[wine_reviews.designation.isin(['Askitikos', 'Shah', 'Piedra Feliz'])]
enc_cols = wine_reviews.drop(['description', 'points', 'price', 'designation', 'winery'], axis = 1)

dummies = pd.get_dummies(enc_cols, prefix = ['country', 'province', 'region_1', 'variety']) 



# combined with log transformed 'price'

X_encoded = pd.concat([np.log(wine_reviews['price']), dummies], axis = 1)

X_encoded.shape
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



wine_desc = pd.DataFrame({'description': wine_reviews['description']})
wine_desc['clean_desc'] = wine_desc['description'].apply(lambda x: x.lower())

wine_desc['clean_desc'].head()
wine_desc['clean_desc'] = wine_desc['clean_desc'].str.replace('[^\w\s]', '')

wine_desc['clean_desc'].head()
wine_desc['clean_desc'] = wine_desc['clean_desc'].str.replace('[0-9]+', '')

wine_desc['clean_desc'].head()
stop_words = stopwords.words('english')

wine_desc['clean_desc'] = wine_desc['clean_desc'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))

wine_desc['clean_desc'].head()
# stem words

porter = PorterStemmer()

wine_desc['clean_desc'][:10].apply(lambda x: ' '.join([porter.stem(w) for w in x.split()]))
# lemmatization

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()



def get_wordnet_tag(nltk_tag):

    if nltk_tag.startswith('J'):

        return wordnet.ADJ

    elif nltk_tag.startswith('V'):

        return wordnet.VERB

    elif nltk_tag.startswith('N'):

        return wordnet.NOUN

    elif nltk_tag.startswith('R'):

        return wordnet.ADV

    else:          

        return None



def lemmatize(sentence):

    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

    lemmatized_sentence = []

    for word, tag in tagged:

        wntag = get_wordnet_tag(tag)

        if wntag is None:

            lemmatized_sentence.append(lemmatizer.lemmatize(word))

        else:

            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos = wntag))

    return ' '.join(lemmatized_sentence)



wine_desc['clean_desc'] = wine_desc['clean_desc'].apply(lambda x: lemmatize(x))

wine_desc['clean_desc'][:10]
pd.Series(' '.join(wine_desc['clean_desc']).split()).value_counts()[:10]
wine_desc.head()
from sklearn.feature_extraction.text import CountVectorizer

X_desc = wine_desc['clean_desc']



count_vectorizer = CountVectorizer(max_features = 1000)

count_vectorizer.fit(X_desc)

X_count = count_vectorizer.transform(X_desc)

print(X_count.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer(max_features = 1000)

tfidf_vectorizer.fit(X_desc)

X_tfidf = tfidf_vectorizer.transform(X_desc)

print(X_tfidf.shape)
from keras.preprocessing.text import Tokenizer

# create a tokenizer 

tokenizer = Tokenizer()

tokenizer.fit_on_texts(wine_desc['clean_desc'].values)

word_index = tokenizer.word_index



# load the pre-trained word-embedding vectors 

embeddings_index = {}

with open('../input/glove6b100dtxt/glove.6B.100d.txt') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs



# create token-embedding mapping

embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
vocab_size = len(word_index) + 1

nonzeros = np.count_nonzero(np.count_nonzero(embedding_matrix, axis = 1))

nonzeros / vocab_size
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

y = wine_reviews['points'].values

X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(X_count, y, test_size = 0.25, random_state = 12)



# train the model

rf = RandomForestRegressor()

rf.fit(X_train_count, y_train_count)



# test the model

y_pred_rf_count = rf.predict(X_test_count)



from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_count, y_pred_rf_count))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test_count, y_pred_rf_count))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_count, y_pred_rf_count)))
rf.score(X_test_count, y_test_count)
X_count_df = pd.DataFrame(X_count.toarray())

X = pd.concat([X_encoded, X_count_df], axis = 1)

X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(X, y, test_size = 0.25, random_state = 12)



# train the model

rf = RandomForestRegressor()

rf.fit(X_train_count, y_train_count)



# test the model

y_pred_rf_count = rf.predict(X_test_count)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_count, y_pred_rf_count))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test_count, y_pred_rf_count))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_count, y_pred_rf_count)))
rf.score(X_test_count, y_test_count)
from pprint import pprint

pprint(rf.get_params())
# tried to commit with following codes, but failed



# from sklearn.model_selection import RandomizedSearchCV

# params that will be sampled from

# max_depth = [int(x) for x in np.linspace(40, 100, num = 7)]

# max_depth.append(None)

# random_params = {'n_estimators': [200, 400, 600, 800, 1000, 1200],

#                 'max_depth': max_depth,

#                 'min_samples_split': [2, 3],

#                 'min_samples_leaf': [2, 3]}



# rf = RandomForestRegressor()



# to reduce the time, just sample 50 combinations

# rf_count_rd = RandomizedSearchCV(estimator = rf, param_distributions = random_params, n_iter = 50, cv = 3, verbose = 1)

# rf_count_rd.fit(X_train_count, y_train_count)

# rf_count_rd.best_params_



# random search with best performance parameters

# rf_count_rd_best = rf_count_rd.best_estimator_

# y_pred_rf_count_rd = rf_count_rd_best.predict(X_test_count)



# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_count, y_pred_rf_count_rd))  

# print('Mean Squared Error:', metrics.mean_squared_error(y_test_count, y_pred_rf_count_rd))  

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_count, y_pred_rf_count_rd)))
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size = 0.25, random_state = 12)



# train the model

rf = RandomForestRegressor()

rf.fit(X_train_tfidf, y_train_tfidf)



# test the model

y_pred_rf_tfidf = rf.predict(X_test_tfidf)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_tfidf, y_pred_rf_tfidf))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf)))
rf.score(X_test_tfidf, y_test_tfidf)
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

X = pd.concat([X_encoded, X_tfidf_df], axis = 1)

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X, y, test_size = 0.25, random_state = 12)



# train the model

rf = RandomForestRegressor()

rf.fit(X_train_tfidf, y_train_tfidf)



# test the model

y_pred_rf_tfidf = rf.predict(X_test_tfidf)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_tfidf, y_pred_rf_tfidf))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf)))
rf.score(X_test_tfidf, y_test_tfidf)
# still did not work as it took too much time to run;

# plan to run it locally using Jupyter notebook and push it to my Github



# params that will be sampled from

# max_depth = [int(x) for x in np.linspace(40, 100, num = 7)]

# max_depth.append(None)

# random_params = {'n_estimators': [200, 400, 600, 800, 1000, 1200],

#                 'max_depth': max_depth,

#                 'min_samples_split': [2, 3],

#                 'min_samples_leaf': [2, 3]}



# rf = RandomForestRegressor()

# rf_tfidf_rd = RandomizedSearchCV(estimator = rf, param_distributions = random_params, n_iter = 50, cv = 3, verbose = 1)

# rf_tfidf_rd.fit(X_train_tfidf, y_train_tfidf)

# rf_tfidf_rd.best_params_



# random search with best performance parameters

# rf_tfidf_rd_best = rf_tfidf_rd.best_estimator_

# y_pred_rf_tfidf_rd = rf_tfidf_rd_best.predict(X_test_count)



# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_tfidf, y_pred_rf_tfidf_rd))  

# print('Mean Squared Error:', metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf_rd))  

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_tfidf, y_pred_rf_tfidf_rd)))