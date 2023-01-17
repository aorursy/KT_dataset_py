import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
from matplotlib_venn import venn2
import re
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
eng_stopwords = stopwords.words('english')
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import os
print(os.listdir("../input/google-quest-challenge"))
train_data = pd.read_csv('../input/google-quest-challenge/train.csv')
test_data = pd.read_csv('../input/google-quest-challenge/test.csv')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

print('Size of train_data', train_data.shape)
print('Size of test_data', test_data.shape)
print('Size of sample_submission', sample_submission.shape)
train_data.head()
train_data.columns
test_data.head()
test_data.columns
sample_submission.head()
# Target variables
targets = list(sample_submission.columns[1:])
print(targets)
# Statistical overview of the Data
train_data[targets].describe()
# checking missing data for train_data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum() / train_data.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
# checking missing data for test_data
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum() / test_data.isnull().count()*100).sort_values(ascending=False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()
train_data["host"].value_counts()
# Distribution of Host(from which website Question & Answers collected)
temp = train_data["host"].value_counts()
df = pd.DataFrame({'labels': temp.index, 'values': temp.values})
df.iplot(kind='pie', labels='labels', values='values', title='Distribution of hosts in Training data')
temp = test_data["host"].value_counts()
print("Total number of states : ",len(temp))
df = pd.DataFrame({'labels': temp.index,'values': temp.values})
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of hosts in test data')
# Train data
temp = train_data["category"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of categories in training data in % ",
    xaxis=dict(
        title='category',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='test')
# Test data
temp = test_data["category"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of categories in test data in % ",
    xaxis=dict(
        title='category',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='test')
# Distribution of Target variables
fig, axes = plt.subplots(6, 5, figsize=(18, 15))
axes = axes.ravel()
bins = np.linspace(0, 1, 20)

for i, col in enumerate(targets):
	ax = axes[i]
	sns.distplot(train_data[col], label=col, kde=False, bins=bins, ax=ax)
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 6079])
plt.tight_layout()
plt.show()
plt.close()
# Venn Diagram(Common Features values in training and test data)
plt.figure(figsize=(23, 13))
plt.subplot(321)

venn2([set(train_data.question_user_name.unique()), set(test_data.question_user_name.unique())], set_labels=('Train set', 'Test set'))
plt.title("Common question_user_name in training and test data", fontsize=15)

plt.subplot(322)
venn2([set(train_data.answer_user_name.unique()), set(test_data.answer_user_name.unique())], set_labels=('Train set', 'Test set'))
plt.title("Common answer_user_name in training and test data", fontsize=15)

plt.subplot(323)
venn2([set(train_data.question_title.unique()), set(test_data.question_title.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common question_title in training and test data", fontsize=15)

plt.subplot(324)
venn2([set(train_data.question_user_name.unique()), set(train_data.answer_user_name.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common users in both question & answer in train data", fontsize=15)

plt.subplot(325)
venn2([set(test_data.question_user_name.unique()), set(test_data.answer_user_name.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common users in both question & answer in test data", fontsize=15)

plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9)
plt.show()
# Distribution for Question Title
train_question_title = train_data['question_title'].str.len()
test_question_title = test_data['question_title'].str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
sns.distplot(train_question_title, ax=ax1, color='blue')
sns.distplot(test_question_title, ax=ax2, color='green')
ax2.set_title('Distribution for Question Title in test data')
ax1.set_title('Distribution for Question Title in Training data')
plt.show()
# Distribution for Question body
train_question_title = train_data['question_body'].str.len()
test_question_title = test_data['question_body'].str.len()

fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,6))
sns.distplot(train_question_title, ax=ax1, color='blue')
sns.distplot(test_question_title, ax=ax2, color='green')
ax2.set_title('Distribution for Question Body in test data')
ax1.set_title('Distribution for Question Body in Training data')
plt.show()
# Distribution for Answers
train_question_title = train_data['answer'].str.len()
test_question_title = test_data['answer'].str.len()

fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,6))
sns.distplot(train_question_title, ax=ax1, color='blue')
sns.distplot(test_question_title, ax=ax2, color='green')
ax2.set_title('Distribution for Answers in test data')
ax1.set_title('Distribution for Answers in Training data')
plt.show()
# Duplicate Questions Title
print("Number of duplicate questions in descending order")
print("------------------------------------------------------")
train_data.groupby('question_title').count()['qa_id'].sort_values(ascending = False).head(25)
# Most popular Questions
train_data[train_data['question_title'] == 'What is the best introductory Bayesian statistics textbook?']
puncts = [
',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
'·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√'
]

misspell_dict = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}
def clean_text(text):
	# take the txt only no symbols[how are you?? => how are you]
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	# ['how', 'are', 'you']
	text = text.lower().split()
	stops = set(stopwords.words("english"))
	text = [word for word in text if not word in stops]
	text = " ".join(text)
	return(text)
def _get_misspell(misspell_dict):
	# target -> find all misspell_dict.keys() from a given txt
	misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
	# misspell_re => re.compile("(she'd|shouldn't|haven't|shouldnt|theres|hadn't|what're|who's|it's|she'll|weren't|
	# 				 you've|i'm|where's|that's|he'd|don't|they've|there's|what've|i'd|who'll|you're|can't|it'll|mustn't|he'll|who'd|i've|')  
	return misspell_dict, misspell_re
def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_misspell(misspell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
    return df
columns = ['question_title','question_body','answer']
train_data = clean_data(train_data, columns)
test_data = clean_data(test_data, columns)
print('Done cleaning done!!')
# training data
freq_dist = FreqDist([word for text in train_data['question_title'] for word in text.split()])

plt.figure(figsize=(20, 7))
plt.title('Word frequency on question title (Training Data)').set_fontsize(25)
plt.xlabel('').set_fontsize(25)
plt.ylabel('').set_fontsize(25)
freq_dist.plot(60,cumulative=False)
plt.show()
# test data
freq_dist = FreqDist([word for text in test_data['question_title'] for word in text.split()])

plt.figure(figsize=(20, 7))
plt.title('Word frequency on question title (Test Data)').set_fontsize(25)
plt.xlabel('').set_fontsize(25)
plt.ylabel('').set_fontsize(25)
freq_dist.plot(60,cumulative=False)
plt.show()
# training data
freq_dist = FreqDist([word for text in train_data['question_body'].str.replace('[^a-za-z0-9^,!.\/+-=]',' ') for word in text.split()])

plt.figure(figsize=(20, 7))
plt.title('Word frequency on question body (Training Data)').set_fontsize(25)
plt.xlabel('').set_fontsize(25)
plt.ylabel('').set_fontsize(25)
freq_dist.plot(60,cumulative=False)
plt.show()
# test data
freq_dist = FreqDist([word for text in test_data['question_body'] for word in text.split()])

plt.figure(figsize=(20, 7))
plt.title('Word frequency on question body (Test Data)').set_fontsize(25)
plt.xlabel('').set_fontsize(25)
plt.ylabel('').set_fontsize(25)
freq_dist.plot(60,cumulative=False)
plt.show()
# Number of characters in the text
train_data["question_title_num_chars"] = train_data["question_title"].apply(lambda x: len(str(x)))
train_data["question_body_num_chars"] = train_data["question_body"].apply(lambda x: len(str(x)))
train_data["answer_num_chars"] = train_data["answer"].apply(lambda x: len(str(x)))

test_data["question_title_num_chars"] = test_data["question_title"].apply(lambda x: len(str(x)))
test_data["question_body_num_chars"] = test_data["question_body"].apply(lambda x: len(str(x)))
test_data["answer_num_chars"] = test_data["answer"].apply(lambda x: len(str(x)))

# Number of words in the text
train_data["question_title_num_words"] = train_data["question_title"].apply(lambda x: len(str(x).split()))
train_data["question_body_num_words"] = train_data["question_body"].apply(lambda x: len(str(x).split()))
train_data["answer_num_words"] = train_data["answer"].apply(lambda x: len(str(x).split()))

test_data["question_title_num_words"] = test_data["question_title"].apply(lambda x: len(str(x).split()))
test_data["question_body_num_words"] = test_data["question_body"].apply(lambda x: len(str(x).split()))
test_data["answer_num_words"] = test_data["answer"].apply(lambda x: len(str(x).split()))

# Number of unique words in the text
train_data["question_title_num_unique_words"] = train_data["question_title"].apply(lambda x: len(set(str(x).split())))
train_data["question_body_num_unique_words"] = train_data["question_body"].apply(lambda x: len(set(str(x).split())))
train_data["answer_num_unique_words"] = train_data["answer"].apply(lambda x: len(set(str(x).split())))

test_data["question_title_num_unique_words"] = test_data["question_title"].apply(lambda x: len(set(str(x).split())))
test_data["question_body_num_unique_words"] = test_data["question_body"].apply(lambda x: len(set(str(x).split())))
test_data["answer_num_unique_words"] = test_data["answer"].apply(lambda x: len(set(str(x).split())))
# TF-IDF Features
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=128, n_iter=5)

tfquestion_title = tfidf.fit_transform(train_data["question_title"].values)
tfquestion_title_test = tfidf.transform(test_data["question_title"].values)

tfquestion_title = tsvd.fit_transform(tfquestion_title)
tfquestion_title_test = tsvd.fit_transform(tfquestion_title_test)

tfquestion_body = tfidf.fit_transform(train_data["question_body"].values)
tfquestion_body_test = tfidf.transform(test_data["question_body"].values)

tfquestion_body = tsvd.fit_transform(tfquestion_body)
tfquestion_body_test = tsvd.transform(tfquestion_body_test)

tfanswer = tfidf.fit_transform(train_data["answer"].values)
tfanswer_test = tfidf.transform(test_data["answer"].values)

tfanswer = tsvd.fit_transform(tfanswer)
tfanswer_test = tsvd.transform(tfanswer_test)

train_data["tfquestion_title"] = list(tfquestion_title)
test_data["tfquestion_title_test"] = list(tfquestion_title_test)

train_data["tfquestion_body"] = list(tfquestion_body)
test_data["tfquestion_body_test"] = list(tfquestion_body_test)

train_data["tfanswer"] = list(tfanswer)
test_data["tfanswer_test"] = list(tfanswer_test)