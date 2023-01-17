import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
print('Training data shape (rows, cols): ', df_train.shape)

df_train.head()
# keyword and location columns have some nulls

df_train.info()
print('Test data shape (rows, cols): ', df_test.shape)

df_test.head()
# keyword and location columns have some nulls

df_test.info()
# Null check

df_train['keyword'].isnull().value_counts() / df_train.shape[0]
df_train['location'].isnull().value_counts() / df_train.shape[0]
df_train['text'].isnull().value_counts() / df_train.shape[0]
df_test['keyword'].isnull().value_counts() / df_test.shape[0]
df_test['location'].isnull().value_counts() / df_test.shape[0]
df_test['text'].isnull().value_counts() / df_test.shape[0]
# Target Distribution (0 or 1)

dist_class = df_train['target'].value_counts()

labels = ['Non-disaster tweet', 'Disaster tweet']



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Target Count")



ax2.pie(dist_class,

        labels=labels,

        counterclock=False,

        startangle=90,

        autopct='%1.1f%%',

        pctdistance=0.7)

plt.title("Target Frequency Proportion")

plt.show
disaster_tweet_length = df_train[df_train['target']==1]['text'].str.len()

nondisaster_tweet_length = df_train[df_train['target']==0]['text'].str.len()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



ax1.hist(disaster_tweet_length, color='red')

ax1.set_title("Disaster Tweets")



ax2.hist(nondisaster_tweet_length, color='green')

ax2.set_title("Non-Disaster Tweets")



fig.suptitle("Characters in tweets")

plt.show()
disaster_tweet_words = df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))

nondisaster_tweet_words = df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



ax1.hist(disaster_tweet_words, color='red')

ax1.set_title("Disaster Tweets")



ax2.hist(nondisaster_tweet_words, color='green')

ax2.set_title("Non-Disaster Tweets")



fig.suptitle("Words in tweets")

plt.show()
df_train_keyword = pd.DataFrame({

    'keyword': df_train['keyword'].value_counts().index,

    'count': df_train['keyword'].value_counts().values

})



df_train_location = pd.DataFrame({

    'location': df_train['location'].value_counts().index,

    'count': df_train['location'].value_counts().values

})



print('Number fo unique keywords in training data: ', df_train_keyword.shape[0])



px.bar(

    df_train_keyword,

    x='keyword',

    y='count',

    title="Each unique keyword count in training data"

).show()



px.bar(

    df_train_location,

    x=df_train_location['location'][:20],

    y=df_train_location['count'][:20],

    title="Top 20 location countin training data"

).show()
df_train[df_train['target'] == 1]['keyword'].value_counts()
df_train[df_train['target'] == 0]['keyword'].value_counts()
df_train[df_train['target'] == 1]['location'].value_counts()
df_train[df_train['target'] == 0]['location'].value_counts()
df_test_keyword = pd.DataFrame({

    'keyword': df_test['keyword'].value_counts().index,

    'count': df_test['keyword'].value_counts().values

})



df_test_location = pd.DataFrame({

    'location': df_test['location'].value_counts().index,

    'count': df_test['location'].value_counts().values

})



print('Number fo unique keywords in test data: ', df_test_keyword.shape[0])



px.bar(

    df_test_keyword,

    x='keyword',

    y='count',

    title="Each unique keyword count in test data"

).show()



px.bar(

    df_test_location,

    x=df_test_location['location'][:20],

    y=df_test_location['count'][:20],

    title="Top 20 location count in test data"

).show()
disaster_tweet = dict(df_train[df_train['target']==1]['keyword'].value_counts())



stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(disaster_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
nondisaster_tweet = dict(df_train[df_train['target']==0]['keyword'].value_counts())



wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(nondisaster_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
test_tweet = dict(df_test['keyword'].value_counts())



wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(test_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



import string

def remove_punc(text):

    table = str.maketrans('','',string.punctuation)

    return text.translate(table)
# https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt

slang_abbrev_dict = {

    'AFAIK': 'As Far As I Know',

    'AFK': 'Away From Keyboard',

    'ASAP': 'As Soon As Possible',

    'ATK': 'At The Keyboard',

    'ATM': 'At The Moment',

    'A3': 'Anytime, Anywhere, Anyplace',

    'BAK': 'Back At Keyboard',

    'BBL': 'Be Back Later',

    'BBS': 'Be Back Soon',

    'BFN': 'Bye For Now',

    'B4N': 'Bye For Now',

    'BRB': 'Be Right Back',

    'BRT': 'Be Right There',

    'BTW': 'By The Way',

    'B4': 'Before',

    'B4N': 'Bye For Now',

    'CU': 'See You',

    'CUL8R': 'See You Later',

    'CYA': 'See You',

    'FAQ': 'Frequently Asked Questions',

    'FC': 'Fingers Crossed',

    'FWIW': 'For What It\'s Worth',

    'FYI': 'For Your Information',

    'GAL': 'Get A Life',

    'GG': 'Good Game',

    'GN': 'Good Night',

    'GMTA': 'Great Minds Think Alike',

    'GR8': 'Great!',

    'G9': 'Genius',

    'IC': 'I See',

    'ICQ': 'I Seek you',

    'ILU': 'I Love You',

    'IMHO': 'In My Humble Opinion',

    'IMO': 'In My Opinion',

    'IOW': 'In Other Words',

    'IRL': 'In Real Life',

    'KISS': 'Keep It Simple, Stupid',

    'LDR': 'Long Distance Relationship',

    'LMAO': 'Laugh My Ass Off',

    'LOL': 'Laughing Out Loud',

    'LTNS': 'Long Time No See',

    'L8R': 'Later',

    'MTE': 'My Thoughts Exactly',

    'M8': 'Mate',

    'NRN': 'No Reply Necessary',

    'OIC': 'Oh I See',

    'OMG': 'Oh My God',

    'PITA': 'Pain In The Ass',

    'PRT': 'Party',

    'PRW': 'Parents Are Watching',

    'QPSA?': 'Que Pasa?',

    'ROFL': 'Rolling On The Floor Laughing',

    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',

    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',

    'SK8': 'Skate',

    'STATS': 'Your sex and age',

    'ASL': 'Age, Sex, Location',

    'THX': 'Thank You',

    'TTFN': 'Ta-Ta For Now!',

    'TTYL': 'Talk To You Later',

    'U': 'You',

    'U2': 'You Too',

    'U4E': 'Yours For Ever',

    'WB': 'Welcome Back',

    'WTF': 'What The Fuck',

    'WTG': 'Way To Go!',

    'WUF': 'Where Are You From?',

    'W8': 'Wait',

    '7K': 'Sick:-D Laugher'

}



def unslang(text):

    if text.upper() in slang_abbrev_dict.keys():

        return slang_abbrev_dict[text.upper()]

    else:

        return text
def tokenization(text):

    text = re.split('\W+', text)

    return text



stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):

    text = [word for word in text if word not in stopword]

    return text



def stemming(text):

    text = [stemmer.stem(word) for word in text]

    return text
for datas in [df_train, df_test]:

    datas['cleaned_text'] = datas['text'].apply(lambda x : remove_url(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : remove_html(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : remove_emoji(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : unslang(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : remove_punc(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : tokenization(x.lower()))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : remove_stopwords(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : stemming(x))

    datas['cleaned_text'] = datas['cleaned_text'].apply(lambda x : ' '.join(x))
df_train.head(10)
df_train['text'][100]
df_train['cleaned_text'][100]
df_test.head(10)
df_test['text'][100]
df_test['cleaned_text'][100]
vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(df_train['cleaned_text'])
# df_train_feature = df_train[['keyword', 'location', 'cleaned_text']]

# X = vectorizer.transform(df_train_feature).todense()

X = vectorizer.transform(df_train['cleaned_text']).todense()

y = df_train['target'].values
print(X.shape)

print(y.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
# from sklearn.model_selection import GridSearchCV



# params = {

#     'max_depth': list(range(5, 11)),

#     'learning_rate': list(np.arange(0.05, 0.30, 0.05)),

#     'gamma': list(np.arange(0.01, 0.06, 0.01)),

#     'min_child_weight': list(range(1, 6)),

    

#     # fixed params

#     'n_estimators' : [1500],

#     'n_jobs': [4],

#     'objective': ['binary:logistic'],

#     'eval_metric' : ['logloss'],

#     'random_state': [42]

# }



# model = XGBClassifier(tree_method='gpu_hist')

# cv = GridSearchCV(model, params, cv=5, n_jobs=4, scoring='roc_auc')



# cv.fit(X_train, y_train)
xgb_classifir = XGBClassifier(tree_method='gpu_hist',

                              learning_rate=0.1,

                              num_round=1000,

                              max_depth=10,

                              min_child_weight=2,

                              colsample_bytree=0.8,

                              subsample=0.9,

                              gamma=0.4,

                              reg_alpha=1e-5,

                              reg_lambda=1,

                              n_estimators=2000,

                              objective='binary:logistic',

                              eval_metric=["auc", "logloss", "error"],

                              early_stopping_rounds=50)



# https://www.coursera.org/learn/competitive-data-science/lecture/wzi5a/hyperparameter-tuning-ii
%%time

xgb_classifir.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_valid, y_valid)])
y_pred_xgb = xgb_classifir.predict(X_valid)
type(y_pred_xgb)
print(y_pred_xgb)
confusion_matrix(y_valid, y_pred_xgb)
accuracy_score(y_valid, y_pred_xgb)
f1_score(y_valid, y_pred_xgb)
fpr, tpr, _ = roc_curve(y_valid, y_pred_xgb)

auc_score = auc(fpr, tpr)
# clear current figure

plt.clf()



plt.title('ROC Curve')

plt.plot(fpr, tpr, marker='.', label='AUC = {:.2f}'.format(auc_score))



# it's helpful to add a diagonal to indicate where chance 

# scores lie (i.e. just flipping a coin)

plt.plot([0,1],[0,1],'r--')



plt.xlim([-0.1,1.1])

plt.ylim([-0.1,1.1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.legend(loc='lower right')

plt.show()
predictions = [round(value) for value in y_pred_xgb]
# retrieve performance metrics

results = xgb_classifir.evals_result()

epochs = len(results['validation_0']['auc'])

x_axis = range(0, epochs)



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))



# plot auc

ax1.plot(x_axis, results['validation_0']['auc'], label='Train')

ax1.plot(x_axis, results['validation_1']['auc'], label='Validation')

ax1.set_title("XGBoost AUC")

ax1.legend()



# plot log loss

ax2.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax2.plot(x_axis, results['validation_1']['logloss'], label='Validation')

ax2.set_title("XGBoost Log Loss")

ax2.legend()



# plot classification error

ax3.plot(x_axis, results['validation_0']['error'], label='Train')

ax3.plot(x_axis, results['validation_1']['error'], label='Validation')

ax3.set_title("XGBoost Classification Error")

ax3.legend()
# Suppport Vecter Machine

from sklearn.svm import SVC



svm_classifier = SVC(kernel='rbf')

svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_valid)

print(y_pred_svm)
# K-Nearest neighbour

from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier(n_neighbors = 7,weights = 'distance',algorithm = 'brute')

knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_valid)

print(y_pred_knn)
X_testset = vectorizer.transform(df_test['cleaned_text']).todense()
print(X_testset.shape)
y_test_pred_ave = (xgb_classifir.predict(X_testset) + svm_classifier.predict(X_testset) + knn_classifier.predict(X_testset)) / 3

print(y_test_pred_ave)
y_test_pred = np.where(y_test_pred_ave >= 0.5, 1, 0)

print(y_test_pred)
submission_file = pd.DataFrame({'id': df_test['id'], 'target': y_test_pred})

submission_file
submission_file.to_csv('submission_xgb_20200130.csv', index = False)