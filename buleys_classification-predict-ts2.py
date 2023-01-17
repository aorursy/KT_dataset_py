# Install Comet and import Experiment class
!pip install comet_ml
from comet_ml import Experiment
# Start experiment
experiment = Experiment(api_key="XXXXXXXX",
                        project_name="XXXXXXXX",
                        workspace="XXXXXX")
# Set experiment name, new name for each run
experiment.set_name('XXXXXX')
# Data wrangling
import numpy as np 
import pandas as pd 

# Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Text processing
import string
import re
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Data processing
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split

# Model imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier 

# Metrics
from sklearn.metrics import accuracy_score, log_loss, precision_score 
from sklearn.metrics import recall_score, precision_recall_curve, f1_score, classification_report, confusion_matrix
from collections import defaultdict


# Kaggle input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Silence warnings for clean flow of notebook
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/climate-change-belief-analysis/train.csv')
test = pd.read_csv('/kaggle/input/climate-change-belief-analysis/test.csv')
display(df.head())
display(test.head())
# Detect and remove NaN values as well as duplicate rows
df = df.drop_duplicates(subset=['message'])
display(df.isnull().sum())
display(test.isnull().sum())
def remove_blanks(df):
    """
    Takes in a dataframe, detects empty strings and removes them.

    Parameters:
    ---------
    DataFrame 

    Returns:
    ---------
    DataFrame:Dataframe with no empty strings

    """
    blanks = []
    for index, tweet in enumerate(df['message']):
        if type(tweet) == str:
            if tweet in ['', ' ']:
                blanks.append(index)
    print(blanks)
    return df.drop(blanks)
# Remove blanks in train and test datasets
df = remove_blanks(df)
test = remove_blanks(test)
def clean_text(text):
    """
    Takes in text, cleans it by making it lowercase,
    removes links/urls, punctuations etc. and returns it a clean text.

    Parameters:
    ---------
    text (str):String text

    Returns:
    ---------
    str:Clean text

    """
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', 'URL', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text
# Cleaning text in train and test datasets
df['clean_tweet'] = df['message'].apply(lambda x: clean_text(x))
test['clean_tweet'] = test['message'].apply(lambda x: clean_text(x))
stop_words = stopwords.words('english')  # Assign stop_words list


def remove_stopword(text):
    """
    Takes in text and removes stop words.

    Parameters:
    ---------
    text (str):String text

    Returns:
    ---------
    str:Text without stop words

    """
    return [word for word in text.split() if word not in stop_words]
# Removing stop words in train and test datasets
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: remove_stopword(x))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: remove_stopword(x))
def join_tweet(text):
    return ' '.join(text)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: join_tweet(x))
df.head()
# Original tweets
for index, text in enumerate(df['message'][0:5]):
    print('Tweet %d:\n' % (index+1), text)
# Clean tweets
for index, text in enumerate(df['clean_tweet'][0:6]):
    print('Tweet %d:\n' % (index+1), text)
df.head()
# Checking number of values under each sentiment
df['sentiment'].value_counts()
# Plot sentiment distribution
fig, ax = plt.subplots(figsize=(10, 8))
graph = sns.countplot(x='sentiment', data=df, ax=ax)
plt.title('Distribution of sentiment group count')
# Simple word split to get an idea of the raw tweet length
# Will add new column for count that can be removed after analysis
df['word count'] = df['message'].apply(lambda t: len(t.split()))
df.head()
# Get a number summary of the word count variable
df.groupby('sentiment')['word count'].describe()
# Plot tweet word count distribution
fig, ax = plt.subplots(figsize=(10, 5))

# create graphs
sns.kdeplot(df['word count'][df['sentiment'] == -1], shade=True, label='Anti')
sns.kdeplot(df['word count'][df['sentiment'] == 0], shade=True,
            label='Neutral')
sns.kdeplot(df['word count'][df['sentiment'] == 1], shade=True, label='Pro')
sns.kdeplot(df['word count'][df['sentiment'] == 2], shade=True, label='News')

# set title and labels
plt.xlabel('Count of words in tweet')
plt.ylabel('Density')
plt.title('Distribution of Tweet Word Count')
plt.show()
# Create new column to check tweet character length
df['count_characters'] = df['message'].apply(lambda c: len(c))
df.head()
# Get a number summary of the word count character variable
df.groupby('sentiment')['count_characters'].describe()
# Plot tweet character count distribution
fig, ax = plt.subplots(figsize=(10, 5))

# Create graphs
sns.kdeplot(df['count_characters'][df['sentiment'] == -1], shade=True,
            label='Anti')
sns.kdeplot(df['count_characters'][df['sentiment'] == 0], shade=True,
            label='Neutral')
sns.kdeplot(df['count_characters'][df['sentiment'] == 1], shade=True,
            label='Pro')
sns.kdeplot(df['count_characters'][df['sentiment'] == 2], shade=True,
            label='News')

# Set title and label
plt.xlabel('Total Character Count in Tweet')
plt.ylabel('Density')
plt.title('Distribution of Tweet Character Count')
plt.show()
# Repeat for punctuation
df['punctuation_count'] = df['message'].apply(lambda x: len([i for i in str(x)
                                              if i in string.punctuation]))

# Get a number summary of the panctuation count variable
df.groupby('sentiment')['punctuation_count'].describe()
# Plot tweet punctuation count distribution
fig, ax = plt.subplots(figsize=(10, 5))

# Create graphs
sns.kdeplot(df['punctuation_count'][df['sentiment'] == -1], shade=True,
            label='Anti')
sns.kdeplot(df['punctuation_count'][df['sentiment'] == 0], shade=True,
            label='Neutral')
sns.kdeplot(df['punctuation_count'][df['sentiment'] == 1], shade=True,
            label='Pro')
sns.kdeplot(df['punctuation_count'][df['sentiment'] == 2], shade=True,
            label='News')

# Set title and label
plt.xlabel('Count of punctuation')
plt.ylabel('Density')
plt.title('Distribution of Tweet punctuation Count')
plt.show()
def hashtags_extract(text):
    """
    Takes in text, exctracts the hashtags in the text and stores them
    in a list.

    Parameters:
    ---------
    text (str):String text

    Returns:
    ---------
    list of str:Hahtags

    """
    text = str(text).lower()
    hashtags = []
    for token in text.split():
        if token.startswith('#'):
            hashtags.append(token[1:])
    return hashtags
# Extract hashtags from train data and group them by sentiment
df['hashtags'] = df['message'].apply(lambda x: hashtags_extract(x))

df_ht = df.groupby('sentiment')['hashtags'].sum()
df_ht.head()
# Creating a hashtag list for every sentiment
news_ht = df_ht.loc[1, ]
pro_ht = df_ht.loc[2, ]
neutral_ht = df_ht.loc[0, ]
anti_ht = df_ht.loc[-1]
# Plot the top n hashtags for each sentiment
my_feqlist = []
my_dframe = []
for index, ht in enumerate([news_ht, pro_ht, neutral_ht, anti_ht]):
    my_feqlist.append(nltk.FreqDist(ht))
    my_dframe.append(pd.DataFrame({'Hashtag': list(my_feqlist[index].keys()),
                     'Count': list(my_feqlist[index].values())}))
    # selecting top 10 most frequent hashtags
    my_dframe[index] = my_dframe[index].nlargest(columns="Count", n=10)

fig, ax = plt.subplots(4, 1, figsize=(15, 30))
sns.barplot(data=my_dframe[0], x="Hashtag", y="Count", ax=ax[0])
sns.barplot(data=my_dframe[1], x="Hashtag", y="Count", ax=ax[1])
sns.barplot(data=my_dframe[2], x="Hashtag", y="Count", ax=ax[2])
sns.barplot(data=my_dframe[3], x="Hashtag", y="Count", ax=ax[3])

sentiments = ['News', 'Pro', 'Neutral', 'Anti']
for index, sent in enumerate(sentiments):
    ax[index].set(ylabel='Count')
    ax[index].set(xlabel='Hashtags')
    ax[index].set(title=f'Hashtags Countplot for {sent} Sentiment')
plt.show()
df_1 = df[df['sentiment'] == 1]
df_2 = df[df['sentiment'] == 2]
df_0 = df[df['sentiment'] == 0]
df_minus_1 = df[df['sentiment'] == -1]

All_messages = " ".join(sent for sent in df['message'])
messages_1 = " ".join(sent for sent in df_1['message'])
messages_2 = " ".join(sent for sent in df_2['message'])
messages_0 = " ".join(sent for sent in df_0['message'])
messages_minus_1 = " ".join(sent for sent in df_minus_1['message'])
#fig, ax = plt.subplots(5, 1, figsize  = (35,40))
# Create and generate a word cloud image:
wordcloud_all = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(All_messages)
wordcloud_news = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(messages_1)
wordcloud_pro = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(messages_2)
wordcloud_neutral = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(messages_0)
wordcloud_anti = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(messages_minus_1)

wordcloud_dict = {wordcloud_all:'All Sentiment',
                  wordcloud_news:'News',
                  wordcloud_pro:'Pro',
                  wordcloud_neutral:'Neutral',
                  wordcloud_anti:'Anti'}

# Display the generated image:
for k, v in wordcloud_dict.items():
    
    fig, ax = plt.subplots(figsize = (15,7))
    plt.imshow(k, interpolation='bilinear')
    plt.title(f'Tweets under {v} Class', fontsize=30)
    plt.axis('off')
    
    plt.show()
def http_extractor(df):

    df["web_pages"] = df.message.str.findall(r'https?://\S+')
    df["web_pages"] = [''.join(map(str, lists)).lower() for lists in df['web_pages']]
    pattern_url = r'https?://\S+'
    subs_url = r'url-web'
    df['post'] = df['web_pages'].replace(to_replace = pattern_url, value = subs_url, regex = True)

    return(df)
http_extractor(df).head()
df["postid"] = ['Yes' if url != '' else 'No' for url in df["post"]]
post_id = pd.DataFrame(df.groupby('sentiment')['postid'].value_counts())
post_id
# Assign feature and response variables
X = df['clean_tweet']
y = df['sentiment']
heights = [len(y[y == label]) for label in [0, 1, 2, -1]]
bars = pd.DataFrame(zip(heights, [0,1,2,-1]), columns=['heights','labels'])
bars = bars.sort_values(by='heights',ascending=True)
# Let's pick a class size of roughly half the size of the largest size
class_size = 3500

bar_label_df = bars.set_index('labels')

resampled_classes = []

for label in [0, 1, 2, -1]:
    # Get number of observations from this class
    label_size = bar_label_df.loc[label]['heights']

    # If label_size < class size the upsample, else downsample
    if label_size < class_size:
        # Upsample
        label_data = df[['clean_tweet', 'sentiment']][df['sentiment'] == label]
        label_resampled = resample(label_data,
                                   # sample with replacement
                                   # (we need to duplicate observations)
                                   replace=True,
                                   # number of desired samples
                                   n_samples=class_size,
                                   random_state=27)
    else:
        # Downsample
        label_data = df[['clean_tweet', 'sentiment']][df['sentiment'] == label]
        label_resampled = resample(label_data,
                                   # sample without replacement
                                   # (no need for duplicate observations)
                                   replace=False,
                                   # number of desired samples
                                   n_samples=class_size,
                                   random_state=27)

    resampled_classes.append(label_resampled)
# Assign feature and response variables from resampled data
resampled_data = np.concatenate(resampled_classes, axis=0)

X_resampled = resampled_data[:, :-1]
y_resampled = resampled_data[:, -1]
# Plot original original data with resampled data
heights = [len(y_resampled[y_resampled == label]) for label in [0, 1, 2, -1]]
bars_resampled = pd.DataFrame(zip(heights, [0, 1, 2, -1]),
                              columns=['heights', 'labels'])
bars_resampled = bars_resampled.sort_values(by='heights', ascending=True)

fig = go.Figure(data=[
    go.Bar(name='Original', x=[-1, 0, 2, 1], y=bars['heights']),
    go.Bar(name='Resampled', x=[-1, 0, 2, 1], y=bars_resampled['heights'])
])
fig.update_layout(xaxis_title="Sentiment", yaxis_title="Sample size")
fig.show()
df_resampled = pd.DataFrame(X_resampled.reshape(-1,1))
df_resampled.columns = ['tweet']
df_resampled['sentiment'] = y_resampled
df_resampled['sentiment'] = df_resampled['sentiment'].astype('int')
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
                                   df_resampled['tweet'].values,
                                   df_resampled['sentiment'].values,
                                   test_size=0.1, random_state=42)
# Check shape of the split data
print('Training Data Shape:', X_train.shape)
print('Testing Data Shape: ', X_test.shape)
# Create a spaCy tokenizer
spacy.load('en')
lemmatizer = spacy.lang.en.English()


def tokenize(text):
    """
    Takes in text, tokenizes words, then lemmatizes tokens.

    Parameters:
    ---------
    text (str):String text

    Returns:
    ---------
    list of str:Lammitzed tokens

    """
    tokens = lemmatizer(text)
    return [token.lemma_ for token in tokens]
# Define multiple vectorizers to test which one gives us the best accuracy
vectorizer_dict = {'CV_1': CountVectorizer(max_df=0.8, min_df=3,
                                           tokenizer=tokenize,
                                           stop_words=stop_words),
                   'CV_2': CountVectorizer(ngram_range=(1, 3), max_df=0.8,
                                           min_df=3, tokenizer=tokenize,
                                           stop_words=stop_words),
                   'CV_3': CountVectorizer(ngram_range=(2, 3), max_df=0.8,
                                           min_df=3, tokenizer=tokenize,
                                           stop_words=stop_words),
                   'TF_1': TfidfVectorizer(max_df=0.8, min_df=3,
                                           tokenizer=tokenize,
                                           stop_words=stop_words),
                   'TF_2': TfidfVectorizer(ngram_range=(1, 3), max_df=0.8,
                                           min_df=3, tokenizer=tokenize,
                                           stop_words=stop_words),
                   'TF_3': TfidfVectorizer(ngram_range=(2, 3), max_df=0.8,
                                           min_df=3, tokenizer=tokenize,
                                           stop_words=stop_words)}
# Define multiple models to test which one gives us the best accuracy
model_dict = {'Logistic Regression': LogisticRegression(max_iter=1000),
              'Naive Bayes': MultinomialNB(),
              'LinearSVM': SGDClassifier(random_state=42),
              'Non-linear SVM': SVC(gamma="scale"),
              'Neural Network': MLPClassifier(),
              'Decision Tree': DecisionTreeClassifier(max_depth=6),
              'XGBoost': XGBClassifier(max_depth=6)
              }
# Run each classifier for each vectorizer
classifier_results_dict = defaultdict(list)
for vec_name, vectorizer in vectorizer_dict.items():

    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)
    print(vec_name)  # keep track of progress

    for mod_name, model in model_dict.items():
        model.fit(X_train_v, y_train)
        y_pred_v = model.predict(X_test_v)

        precision_v = round(100*precision_score(y_test, y_pred_v,
                            average='weighted'), 3)
        recall_v = round(100*recall_score(y_test, y_pred_v,
                         average='weighted'), 3)
        f1_v = round(2*(precision_v*recall_v) / (precision_v+recall_v), 3)

        classifier_results_dict['Vectorizer Type'].append(vec_name)
        classifier_results_dict['Model Name'].append(mod_name)
        classifier_results_dict[('Precision')].append(precision_v)
        classifier_results_dict[('Recall')].append(recall_v)
        classifier_results_dict[('F1-score')].append(f1_v)

classifier_results_df = pd.DataFrame(classifier_results_dict)
# Checking result
classifier_results_df.sort_values(by='F1-score',
                                  ascending=False).reset_index(drop=True)
fig = px.bar(classifier_results_df, x="Model Name", y="F1-score", color='Vectorizer Type',
             barmode='group', height=400)
fig.show()
# Best Model

# Best Model: TF_1    Non-linear SVM
model_svc = SVC(gamma="scale")

# Vectorization
vectorizer = vectorizer_dict['TF_1']
X_train_TF_1 = vectorizer.fit_transform(X_train)
X_test_cv = vectorizer.transform(X_test)
#log metrics on Comet,where 'metrics' is a dictionary of metrics
metrics = classifier_results_df.to_dict('index')
experiment.log_metrics(metrics)
model_svc.fit(X_train_TF_1, y_train)
y_pred_cv = model_svc.predict(X_test_cv)

precision_cv = round(100*precision_score(y_test, y_pred_cv,
                     average='weighted'), 3)
recall_cv = round(100*recall_score(y_test, y_pred_cv, average='weighted'), 3)
f1_cv = round(2*(precision_cv * recall_cv) / (precision_cv + recall_cv), 3)
print(classification_report(y_test, y_pred_cv))
# Print metrics scores
print('Precision Score:', precision_cv)
print('Recall Score:', recall_cv)
print('f1_Score:', f1_cv)
# Confusion Matrix
confusion_matrix(y_test, y_pred_cv)
# Enhance confusion matrix using a heatmap
cm =confusion_matrix(y_test, y_pred_cv)

categories = ['Anti','Neutral','Pro','News']
fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(cm, ax = ax, annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cbar=False,
            cmap='Greens')
plt.suptitle('Confusion Matrix of Support Vector Classifier used with a TF-IDF Vectorizer')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.show()
#Creating csv for submission

# Vectorizing and Normalizing the test data
test_data = vectorizer.transform(test['clean_tweet'])
test_data_norm = preprocessing.normalize(test_data)

# Making a submission dataframe
df_submission = pd.DataFrame()
df_submission['tweetid'] = test['tweetid']

y_test_data = model_svc.predict(test_data_norm) 
df_submission['sentiment'] = y_test_data

# Creating a csv file
df_submission.to_csv('Submission_file_SVC.csv', index=False)
experiment.end()
experiment.display()