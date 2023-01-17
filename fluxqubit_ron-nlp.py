import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train  = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print("Train : ",train.shape)

print("*"*10)

print("Test : ",test.shape)
train.head(10)

#train.tail(10)
train = train.drop_duplicates().reset_index(drop=True)
sns.countplot(x= "target",data=train)
train.target.value_counts()
train.isnull().sum()
test.isnull().sum()
print(train.keyword.nunique(), test.keyword.nunique())
plt.figure(figsize=(12,8))

sns.countplot(x="keyword",

              data=train,

              order=train.keyword.value_counts().iloc[:15].index

             )

plt.title('Top Keywords')

plt.show()
dist = train[train.target==1].keyword.value_counts().head()

#dist

plt.figure(figsize=(9,6))

sns.barplot(dist,dist.index)

plt.show()
nondist = train[train.target==0].keyword.value_counts().head()

#nondist

plt.figure(figsize=(9,6))

sns.barplot(nondist,nondist.index)

plt.show()
distribution_dist = train.groupby('keyword').mean()['target'].sort_values(ascending=False).head(10)

#distribution_dist



plt.figure(figsize=(9,6))

sns.barplot(distribution_dist,distribution_dist.index)

plt.title("Distribution of keywords for higher risk")

plt.show()
distribution_nondist = train.groupby('keyword').mean()['target'].sort_values().head(10)

#distribution_nondist

plt.figure(figsize=(9,6))

sns.barplot(distribution_nondist,distribution_nondist.index)

plt.title("Distribution of Non Disasters keywords for lower risk")

plt.show()
print (train.location.nunique(), test.location.nunique())
plt.figure(figsize=(9,6))

sns.countplot(y=train.location, order = train.location.value_counts().iloc[:15].index)

plt.title('Top locations')

plt.show()
for clmn in ['keyword','location']:

    train[clmn]= train[clmn].fillna('None')

    test[clmn]= test[clmn].fillna('None')
def setlocname(x):

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

    else: return 'Others'

    
import string
train['locations'] = train['location'].apply(lambda x: setlocname(str(x)))

test['locations'] = test['location'].apply(lambda x:setlocname(str(x)))
plt.figure(figsize=(9,6))

sns.countplot(y=train.locations, order = train.locations.value_counts().iloc[:15].index)

plt.title('Top Updated locations')

plt.show()
top_l2 = train.groupby('locations').mean()['target'].sort_values(ascending=False)

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
#Clean text data berfore converting as vector

import re

    

def preprocessing_text(text):

    text = re.sub(r'https?://\S+','',text)

    text = re.sub(r'\n',' ',text)

    text = re.sub('\s+',' ',text).strip()

    return text
dummystr = train.loc[417,'text']

print(dummystr)

print(preprocessing_text(dummystr))
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'
def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'
def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'
def text_process(df):

    df['text_clean'] = df['text'].apply(lambda x: preprocessing_text(x))

    df['hash'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mention'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    

    return df
train = text_process(train)

test = text_process(test)
train.head()

from wordcloud import STOPWORDS

import string
def make_wordcloud(df):

    df['text_len'] = df['text_clean'].apply(len)

    df['wordcount'] = df['text_clean'].apply(lambda x : len(str(x).split()))

    df['stop_word_count'] = df['text_clean'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    df['punctuation_count'] = df['text_clean'].apply(lambda x :len([c for c in str(x) if c in string.punctuation]))

    df['hashtag_count'] = df['hash'].apply(lambda x :len(str(x).split()))

    df['mention_count'] = df['mention'].apply(lambda x:len(str(x).split()))

    df['link_count'] = df['links'].apply(lambda x:len(str(x).split()))

    df['caps_count'] = df['text_clean'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))

    df['caps_ratio'] = df['caps_count'] / df['text_len']

    return df

    
train = make_wordcloud(train)

test = make_wordcloud(test)

train.head()
train.corr()
TCor = train.corr()

mask = np.triu(np.ones_like(TCor, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(TCor, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
train.corr()['target'].drop('target').sort_values()
#http://www.clker.com/cliparts/a/1/a/5/1242249442627091102Flammable-symbol.svg.med.png

#from PIL import Image

#import requests

#from wordcloud import WordCloud

#mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/a/1/a/5/1242249442627091102Flammable-symbol.svg.med.png', stream=True).raw))



# This function takes in your text and your mask and generates a wordcloud. 

#def generate_wordcloud(words, mask):

 #   word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(words)

 #   plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')

 #   plt.imshow(word_cloud)

 #   plt.axis('off')

 #   plt.tight_layout(pad=0)

#  plt.show()



#generate_wordcloud(train['text_clean'], mask)
import category_encoders as ce
features = ['keyword','locations']

encoder = ce.TargetEncoder(cols=features)

encoder.fit(train[features],train['target'])



train = train.join(encoder.transform(train[features]).add_suffix('_target'))

test = test.join(encoder.transform(test[features]).add_suffix('_target'))
from sklearn.feature_extraction.text import CountVectorizer
vec_links = CountVectorizer(min_df=5,analyzer='word',token_pattern = r'https?://\S+')

link_vec = vec_links.fit_transform(train['links'])

link_vec_test = vec_links.transform(test['links'])

X_train_link = pd.DataFrame(link_vec.toarray(), columns=vec_links.get_feature_names())

X_test_link = pd.DataFrame(link_vec_test.toarray(), columns=vec_links.get_feature_names())
vec_men = CountVectorizer(min_df = 5)

men_vec = vec_men.fit_transform(train['mention'])

men_vec_test = vec_men.transform(test['mention'])

X_train_men = pd.DataFrame(men_vec.toarray(), columns=vec_men.get_feature_names())

X_test_men = pd.DataFrame(men_vec_test.toarray(), columns=vec_men.get_feature_names())
vec_hash = CountVectorizer(min_df = 5)

hash_vec = vec_hash.fit_transform(train['hash'])

hash_vec_test = vec_hash.transform(test['hash'])

X_train_hash = pd.DataFrame(hash_vec.toarray(), columns=vec_hash.get_feature_names())

X_test_hash = pd.DataFrame(hash_vec_test.toarray(), columns=vec_hash.get_feature_names())
hash_rank = (X_train_hash.transpose().dot(train['target']) / X_train_hash.sum(axis=0)).sort_values(ascending=False)

print('Hashtags with which 100% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==1].index))

print('Total: ' + str(len(hash_rank[hash_rank==1])))

print('Hashtags with which 0% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==0].index))

print('Total: ' + str(len(hash_rank[hash_rank==0])))
from sklearn.feature_extraction.text import TfidfVectorizer
vec_text = TfidfVectorizer(min_df = 10 ,ngram_range=(1,2),stop_words= 'english')

text_vec = vec_text.fit_transform(train['text_clean'])

text_vec_test = vec_text.transform(test['text_clean'])

X_train_text = pd.DataFrame(text_vec.toarray(), columns=vec_text.get_feature_names())

X_test_text = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names())
print(X_train_text.shape,X_test_text.shape)
train = train.join(X_train_link, rsuffix='_link')

train = train.join(X_train_men, rsuffix='_mention')

train = train.join(X_train_hash ,rsuffix='_hashtag')

train = train.join(X_train_text ,rsuffix='_text')



test = test.join(X_test_link, rsuffix='_link')

test = test.join(X_test_men, rsuffix='_mention')

test = test.join(X_test_hash ,rsuffix='_hashtag')

test = test.join(X_test_text ,rsuffix='_text')



print (train.shape, test.shape)
#Logistic
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
features_to_drop = ['id', 'keyword','location','text','locations','text_clean', 'hash', 'mention','links']
scale = MinMaxScaler()
train.head()

#/X_train = train.drop(columns = features_to_drop + ['target'] )
X_train = train.drop(columns = features_to_drop + ['target'] )
X_train.head()
X_test = test.drop(columns = features_to_drop)
X_test.head()
y_train = train.target
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', random_state=777)

pipeline = Pipeline ([('scale', scale),('lr',lr)])

pipeline.fit(X_train,y_train)

y_test = pipeline.predict(X_test)



sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submit = sub_sample.copy()

submit.target = y_test

submit.to_csv('submit_lr.csv',index=False)
print ('Accuracy for Train: %.4f' % pipeline.score(X_train, y_train))
from sklearn.metrics import f1_score

print ('F1 score: %.4f' % f1_score(y_train, pipeline.predict(X_train)))
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y_train, pipeline.predict(X_train)))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=12, test_size=0.2,random_state=143)

cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')



print('Cross validation F-1 score: %.3f' %np.mean(cv_score))
from sklearn.feature_selection import RFECV



steps = 20

n_features = len(X_train.columns)

X_range = np.arange(n_features - (int(n_features/steps)) * steps, n_features+1, steps)



rfecv = RFECV(estimator=lr, step=steps, cv=cv, scoring='f1')



pipeline2 = Pipeline([('scale',scale), ('rfecv', rfecv)])

pipeline2.fit(X_train, y_train)

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(np.insert(X_range, 0, 1), rfecv.grid_scores_)

plt.show()
selected_features = X_train.columns[rfecv.ranking_ == 1]

X_train2 = X_train[selected_features]

X_test2 = X_test[selected_features]
pipeline.fit(X_train2, y_train)

cv2 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=456)

cv_score2 = cross_val_score(pipeline, X_train2, y_train, cv=cv2, scoring='f1')

print('Cross validation F-1 score: %.3f' %np.mean(cv_score2))
from sklearn.model_selection import GridSearchCV



grid={"C":np.logspace(-2,2,5), "penalty":["l1","l2"]}

lr_cv = GridSearchCV(LogisticRegression(solver='liblinear', random_state=20), grid, cv=cv2, scoring = 'f1')



pipeline_grid = Pipeline([('scale',scale), ('gridsearch', lr_cv),])



pipeline_grid.fit(X_train2, y_train)



print("Best parameter: ", lr_cv.best_params_)

print("F-1 score: %.3f" %lr_cv.best_score_)
y_test2 = pipeline_grid.predict(X_test2)

submit2 = sub_sample.copy()

submit2.target = y_test2

submit2.to_csv('submit_lr2.csv',index=False)
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