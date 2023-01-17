# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string



plt.rcParams.update({'font.size': 14})



# Load data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



print (train.shape, test.shape, sub_sample.shape)
train.head()
train.duplicated().sum()
train = train.drop_duplicates().reset_index(drop=True)
# Class balance

# train.target.value_counts()

sns.countplot(y=train.target);
# NA data

train.isnull().sum()
test.isnull().sum()
# Check number of unique keywords, and whether they are the same for train and test sets

print (train.keyword.nunique(), test.keyword.nunique())

print (set(train.keyword.unique()) - set(test.keyword.unique()))
# Most common keywords

plt.figure(figsize=(9,6))

sns.countplot(y=train.keyword, order = train.keyword.value_counts().iloc[:15].index)

plt.title('Top 15 keywords')

plt.show()

# train.keyword.value_counts().head(10)
kw_d = train[train.target==1].keyword.value_counts().head(10)

kw_nd = train[train.target==0].keyword.value_counts().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(kw_d, kw_d.index, color='c')

plt.title('Top keywords for disaster tweets')

plt.subplot(122)

sns.barplot(kw_nd, kw_nd.index, color='y')

plt.title('Top keywords for non-disaster tweets')

plt.show()
top_d = train.groupby('keyword').mean()['target'].sort_values(ascending=False).head(10)

top_nd = train.groupby('keyword').mean()['target'].sort_values().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(top_d, top_d.index, color='pink')

plt.title('Keywords with highest % of disaster tweets')

plt.subplot(122)

sns.barplot(top_nd, top_nd.index, color='yellow')

plt.title('Keywords with lowest % of disaster tweets')

plt.show()
# Check number of unique keywords and locations

print (train.location.nunique(), test.location.nunique())
# Most common locations

plt.figure(figsize=(9,6))

sns.countplot(y=train.location, order = train.location.value_counts().iloc[:15].index)

plt.title('Top 15 locations')

plt.show()
raw_loc = train.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = train[train.location.isin(top_loc)]



top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(train.target))

plt.xticks(rotation=80)

plt.show()
# Fill NA values

for col in ['keyword','location']:

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')



def clean_loc(x):

    if x == 'None':

        return 'None'

    elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':

        return 'World'

    elif 'New York' in x or 'NYC' in x:

        return 'New York'    

    elif 'London' in x:

        return 'London'

    elif 'Mumbai' in x:

        return 'Mumbai'

    elif 'Washington' in x and 'D' in x and 'C' in x:

        return 'Washington DC'

    elif 'San Francisco' in x:

        return 'San Francisco'

    elif 'Los Angeles' in x:

        return 'Los Angeles'

    elif 'Seattle' in x:

        return 'Seattle'

    elif 'Chicago' in x:

        return 'Chicago'

    elif 'Toronto' in x:

        return 'Toronto'

    elif 'Sacramento' in x:

        return 'Sacramento'

    elif 'Atlanta' in x:

        return 'Atlanta'

    elif 'California' in x:

        return 'California'

    elif 'Florida' in x:

        return 'Florida'

    elif 'Texas' in x:

        return 'Texas'

    elif 'United States' in x or 'USA' in x:

        return 'USA'

    elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:

        return 'UK'

    elif 'Canada' in x:

        return 'Canada'

    elif 'India' in x:

        return 'India'

    elif 'Kenya' in x:

        return 'Kenya'

    elif 'Nigeria' in x:

        return 'Nigeria'

    elif 'Australia' in x:

        return 'Australia'

    elif 'Indonesia' in x:

        return 'Indonesia'

    elif x in top_loc:

        return x

    else: return 'Others'

    

train['location_clean'] = train['location'].apply(lambda x: clean_loc(str(x)))

test['location_clean'] = test['location'].apply(lambda x: clean_loc(str(x)))
top_l2 = train.groupby('location_clean').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l2.index, y=top_l2)

plt.axhline(np.mean(train.target))

plt.xticks(rotation=80)

plt.show()
leak = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv", encoding='latin_1')

leak['target'] = (leak['choose_one']=='Relevant').astype(int)

leak['id'] = leak.index

leak = leak[['id', 'target','text']]

merged_df = pd.merge(test, leak, on='id')

sub1 = merged_df[['id', 'target']]

sub1.to_csv('submit_1.csv', index=False)
import re



test_str = train.loc[417, 'text']



def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # Remove link

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text



print("Original text: " + test_str)

print("Cleaned text: " + clean_text(test_str))
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'



def process_text(df):

    

    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    # df['hashtags'].fillna(value='no', inplace=True)

    # df['mentions'].fillna(value='no', inplace=True)

    

    return df

    

train = process_text(train)

test = process_text(test)
from wordcloud import STOPWORDS



def create_stat(df):

    # Tweet length

    df['text_len'] = df['text_clean'].apply(len)

    # Word count

    df['word_count'] = df["text_clean"].apply(lambda x: len(str(x).split()))

    # Stopword count

    df['stop_word_count'] = df['text_clean'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    # Punctuation count

    df['punctuation_count'] = df['text_clean'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # Count of hashtags (#)

    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))

    # Count of mentions (@)

    df['mention_count'] = df['mentions'].apply(lambda x: len(str(x).split()))

    # Count of links

    df['link_count'] = df['links'].apply(lambda x: len(str(x).split()))

    # Count of uppercase letters

    df['caps_count'] = df['text_clean'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))

    # Ratio of uppercase letters

    df['caps_ratio'] = df['caps_count'] / df['text_len']

    return df



train = create_stat(train)

test = create_stat(test)



print(train.shape, test.shape)
train.corr()['target'].drop('target').sort_values()
from nltk import FreqDist, word_tokenize



# Make a set of stop words

stopwords = set(STOPWORDS)

# more_stopwords = {'https', 'amp'}

# stopwords = stopwords.union(more_stopwords)
# Unigrams

word_freq = FreqDist(w for w in word_tokenize(' '.join(train['text_clean']).lower()) if 

                     (w not in stopwords) & (w.isalpha()))

df_word_freq = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])

top20w = df_word_freq.sort_values('count',ascending=False).head(20)



plt.figure(figsize=(8,6))

sns.barplot(top20w['count'], top20w.index)

plt.title('Top 20 words')

plt.show()
plt.figure(figsize=(16,7))

plt.subplot(121)

freq_d = FreqDist(w for w in word_tokenize(' '.join(train.loc[train.target==1, 'text_clean']).lower()) if 

                     (w not in stopwords) & (w.isalpha()))

df_d = pd.DataFrame.from_dict(freq_d, orient='index', columns=['count'])

top20_d = df_d.sort_values('count',ascending=False).head(20)

sns.barplot(top20_d['count'], top20_d.index, color='c')

plt.title('Top words in disaster tweets')

plt.subplot(122)

freq_nd = FreqDist(w for w in word_tokenize(' '.join(train.loc[train.target==0, 'text_clean']).lower()) if 

                     (w not in stopwords) & (w.isalpha()))

df_nd = pd.DataFrame.from_dict(freq_nd, orient='index', columns=['count'])

top20_nd = df_nd.sort_values('count',ascending=False).head(20)

sns.barplot(top20_nd['count'], top20_nd.index, color='y')

plt.title('Top words in non-disaster tweets')

plt.show()
# Bigrams



from nltk import bigrams



plt.figure(figsize=(16,7))

plt.subplot(121)

bigram_d = list(bigrams([w for w in word_tokenize(' '.join(train.loc[train.target==1, 'text_clean']).lower()) if 

              (w not in stopwords) & (w.isalpha())]))

d_fq = FreqDist(bg for bg in bigram_d)

bgdf_d = pd.DataFrame.from_dict(d_fq, orient='index', columns=['count'])

bgdf_d.index = bgdf_d.index.map(lambda x: ' '.join(x))

bgdf_d = bgdf_d.sort_values('count',ascending=False)

sns.barplot(bgdf_d.head(20)['count'], bgdf_d.index[:20], color='pink')

plt.title('Top bigrams in disaster tweets')

plt.subplot(122)

bigram_nd = list(bigrams([w for w in word_tokenize(' '.join(train.loc[train.target==0, 'text_clean']).lower()) if 

              (w not in stopwords) & (w.isalpha())]))

nd_fq = FreqDist(bg for bg in bigram_nd)

bgdf_nd = pd.DataFrame.from_dict(nd_fq, orient='index', columns=['count'])

bgdf_nd.index = bgdf_nd.index.map(lambda x: ' '.join(x))

bgdf_nd = bgdf_nd.sort_values('count',ascending=False)

sns.barplot(bgdf_nd.head(20)['count'], bgdf_nd.index[:20], color='yellow')

plt.title('Top bigrams in non-disaster tweets')

plt.show()
import category_encoders as ce



# Target encoding

features = ['keyword', 'location_clean']

encoder = ce.TargetEncoder(cols=features)

encoder.fit(train[features],train['target'])



train = train.join(encoder.transform(train[features]).add_suffix('_target'))

test = test.join(encoder.transform(test[features]).add_suffix('_target'))
from sklearn.feature_extraction.text import CountVectorizer



# CountVectorizer



# Links

vec_links = CountVectorizer(min_df = 5, analyzer = 'word', token_pattern = r'https?://\S+') # Only include those >=5 occurrences

link_vec = vec_links.fit_transform(train['links'])

link_vec_test = vec_links.transform(test['links'])

X_train_link = pd.DataFrame(link_vec.toarray(), columns=vec_links.get_feature_names())

X_test_link = pd.DataFrame(link_vec_test.toarray(), columns=vec_links.get_feature_names())



# Mentions

vec_men = CountVectorizer(min_df = 5)

men_vec = vec_men.fit_transform(train['mentions'])

men_vec_test = vec_men.transform(test['mentions'])

X_train_men = pd.DataFrame(men_vec.toarray(), columns=vec_men.get_feature_names())

X_test_men = pd.DataFrame(men_vec_test.toarray(), columns=vec_men.get_feature_names())



# Hashtags

vec_hash = CountVectorizer(min_df = 5)

hash_vec = vec_hash.fit_transform(train['hashtags'])

hash_vec_test = vec_hash.transform(test['hashtags'])

X_train_hash = pd.DataFrame(hash_vec.toarray(), columns=vec_hash.get_feature_names())

X_test_hash = pd.DataFrame(hash_vec_test.toarray(), columns=vec_hash.get_feature_names())

print (X_train_link.shape, X_train_men.shape, X_train_hash.shape)
_ = (X_train_link.transpose().dot(train['target']) / X_train_link.sum(axis=0)).sort_values(ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(x=_, y=_.index)

plt.axvline(np.mean(train.target))

plt.title('% of disaster tweet given links')

plt.show()
_ = (X_train_men.transpose().dot(train['target']) / X_train_men.sum(axis=0)).sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=_.index, y=_)

plt.axhline(np.mean(train.target))

plt.title('% of disaster tweet given mentions')

plt.xticks(rotation = 50)

plt.show()
hash_rank = (X_train_hash.transpose().dot(train['target']) / X_train_hash.sum(axis=0)).sort_values(ascending=False)

print('Hashtags with which 100% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==1].index))

print('Total: ' + str(len(hash_rank[hash_rank==1])))

print('Hashtags with which 0% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==0].index))

print('Total: ' + str(len(hash_rank[hash_rank==0])))
# Tf-idf for text

from sklearn.feature_extraction.text import TfidfVectorizer



vec_text = TfidfVectorizer(min_df = 10, ngram_range = (1,2), stop_words='english') 

# Only include >=10 occurrences

# Have unigrams and bigrams

text_vec = vec_text.fit_transform(train['text_clean'])

text_vec_test = vec_text.transform(test['text_clean'])

X_train_text = pd.DataFrame(text_vec.toarray(), columns=vec_text.get_feature_names())

X_test_text = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names())

print (X_train_text.shape)
# Joining the dataframes together



train = train.join(X_train_link, rsuffix='_link')

train = train.join(X_train_men, rsuffix='_mention')

train = train.join(X_train_hash, rsuffix='_hashtag')

train = train.join(X_train_text, rsuffix='_text')

test = test.join(X_test_link, rsuffix='_link')

test = test.join(X_test_men, rsuffix='_mention')

test = test.join(X_test_hash, rsuffix='_hashtag')

test = test.join(X_test_text, rsuffix='_text')

print (train.shape, test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler



features_to_drop = ['id', 'keyword','location','text','location_clean','text_clean', 'hashtags', 'mentions','links']

scaler = MinMaxScaler()



X_train = train.drop(columns = features_to_drop + ['target'])

X_test = test.drop(columns = features_to_drop)

y_train = train.target



lr = LogisticRegression(solver='liblinear', random_state=777) # Other solvers have failure to converge problem



pipeline = Pipeline([('scale',scaler), ('lr', lr),])



pipeline.fit(X_train, y_train)

y_test = pipeline.predict(X_test)



submit = sub_sample.copy()

submit.target = y_test

submit.to_csv('submit_lr.csv',index=False)
print ('Training accuracy: %.4f' % pipeline.score(X_train, y_train))
# F-1 score

from sklearn.metrics import f1_score



print ('Training f-1 score: %.4f' % f1_score(y_train, pipeline.predict(X_train)))
# Confusion matrix

from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y_train, pipeline.predict(X_train)))
# Cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)

cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print('Cross validation F-1 score: %.3f' %np.mean(cv_score))
# Top features

plt.figure(figsize=(16,7))

s1 = pd.Series(np.transpose(lr.coef_[0]), index=X_train.columns).sort_values(ascending=False)[:20]

s2 = pd.Series(np.transpose(lr.coef_[0]), index=X_train.columns).sort_values()[:20]

plt.subplot(121)

sns.barplot(y=s1.index, x=s1)

plt.title('Top positive coefficients')

plt.subplot(122)

sns.barplot(y=s2.index, x=s2)

plt.title('Top negative coefficients')

plt.show()
# Feature selection

from sklearn.feature_selection import RFECV



steps = 20

n_features = len(X_train.columns)

X_range = np.arange(n_features - (int(n_features/steps)) * steps, n_features+1, steps)



rfecv = RFECV(estimator=lr, step=steps, cv=cv, scoring='f1')



pipeline2 = Pipeline([('scale',scaler), ('rfecv', rfecv)])

pipeline2.fit(X_train, y_train)

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(np.insert(X_range, 0, 1), rfecv.grid_scores_)

plt.show()
print ('Optimal no. of features: %d' % np.insert(X_range, 0, 1)[np.argmax(rfecv.grid_scores_)])
selected_features = X_train.columns[rfecv.ranking_ == 1]

X_train2 = X_train[selected_features]

X_test2 = X_test[selected_features]
# lr2 = LogisticRegression(solver='liblinear', random_state=37)

pipeline.fit(X_train2, y_train)

cv2 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=456)

cv_score2 = cross_val_score(pipeline, X_train2, y_train, cv=cv2, scoring='f1')

print('Cross validation F-1 score: %.3f' %np.mean(cv_score2))
from sklearn.model_selection import GridSearchCV



grid={"C":np.logspace(-2,2,5), "penalty":["l1","l2"]}

lr_cv = GridSearchCV(LogisticRegression(solver='liblinear', random_state=20), grid, cv=cv2, scoring = 'f1')



pipeline_grid = Pipeline([('scale',scaler), ('gridsearch', lr_cv),])



pipeline_grid.fit(X_train2, y_train)



print("Best parameter: ", lr_cv.best_params_)

print("F-1 score: %.3f" %lr_cv.best_score_)
# Submit fine-tuned model



y_test2 = pipeline_grid.predict(X_test2)

submit2 = sub_sample.copy()

submit2.target = y_test2

submit2.to_csv('submit_lr2.csv',index=False)
# Top features with fine-tuned model

plt.figure(figsize=(16,7))

s1 = pd.Series(np.transpose(lr.coef_[0]), index=X_train2.columns).sort_values(ascending=False)[:20]

s2 = pd.Series(np.transpose(lr.coef_[0]), index=X_train2.columns).sort_values()[:20]

plt.subplot(121)

sns.barplot(y=s1.index, x=s1)

plt.title('Top positive coefficients')

plt.subplot(122)

sns.barplot(y=s2.index, x=s2)

plt.title('Top negative coefficients')

plt.show()
# Error analysis

y_hat = pipeline_grid.predict_proba(X_train2)[:,1]

checker = train.loc[:,['text','keyword','location','target']]

checker['pred_prob'] = y_hat

checker['error'] = np.abs(checker['target'] - checker['pred_prob'])



# Top 50 mispredicted tweets

error50 = checker.sort_values('error', ascending=False).head(50)

error50 = error50.rename_axis('id').reset_index()

error50.target.value_counts()
pd.options.display.max_colwidth = 200



error50.loc[0:10,['text','target','pred_prob']]