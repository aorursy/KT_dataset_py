import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import multiprocessing

news = pd.read_csv("../input/uci-news-aggregator.csv")

# Convert date
news['TIMESTAMP'] = pd.to_datetime(news['TIMESTAMP'], unit='ms')

# Let's take a look at our interesting columns.
news[["TITLE", "PUBLISHER", "CATEGORY", "HOSTNAME", "TIMESTAMP"]].head()
def date_printer(date):
    return "{}/{}/{}".format(date.day, date.month, date.year)

start, end = news['TIMESTAMP'].min(), news['TIMESTAMP'].max() 
print("Our dataset timeline starts at {} and ends at {}".format(date_printer(start), date_printer(end)))
news['MONTH'] = news['TIMESTAMP'].apply(lambda date: date.month)
news['DAY'] = news['TIMESTAMP'].apply(lambda date: date.day)

# Some months have 30 and others have 30 days. The first and last months in our dataset and not whole.
month_days = {
    3: 21,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 27
}
articles_per_day = {}
for month in month_days:
    n_articles = len(news[news['MONTH'] == month])
    articles_per_day[month] = n_articles / month_days[month]

ax = sns.barplot(x=list(articles_per_day.keys()), y=list(articles_per_day.values()))
ax.set_title("Normalized Counts")
ax.set_xlabel("Month")
ax.set_ylabel("Articles Per Day")
plt.show()
cat_map = {
    'b': 'Business',
    't': 'Science',
    'e': 'Entertainment',
    'm': 'Health'
}
ax = sns.countplot(news['CATEGORY'])
ax.set_title("Category Counts")
ax.set_xlabel("Category")
# Manipulate the labels to make them more readable
ax.set_xticklabels([cat_map[x.get_text()] for x in ax.get_xticklabels()], rotation=45)
plt.show()
from collections import Counter

# Byte magic to style print output
def emphasize(s):
    """Bold the string to help get the print reader's attention.
    
    Parameters
    ----------
    s : str
        String to be decorated with bold.
    
    Returns
    -------
    str
        The string in bold.
    """
    red = '\x1b[1;31m'
    stop = '\x1b[0m'
    return red + str(s) + stop

nunique = news['PUBLISHER'].nunique()
print("There are {} different publishers. Below some of the most common:".format(emphasize(nunique)))
for key, value in Counter(news['PUBLISHER']).most_common(5):
    print("   {} posted {} articles".format(emphasize(key), emphasize(value)))
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(s, lemmatize=True, decode=False):
    """ Split a sentence into it's words by removing case sensitivity and stopwords, as well as taking punctuation 
    and other special characters into account.
    
    Parameters
    ----------
    s : str
        The raw string
    lemmatize : bool, optional
        Optionally lemmatize the provided text
    decode : bool, optional
        Whether or not Unicode input that needs to be decoded is expected.
    
    Returns
    -------
    list of str
        The cleaned tokens
        
    """
    # Make sure the NLTK data are downloaded.
    try:
        if decode:
            s = s.decode("utf-8")
        tokens = word_tokenize(s.lower())
    except LookupError:
        nltk.download('punkt')
        tokenize(s)
    

    # Exclude punctuation only after NLTK tokenizer to ensure part of word punctuation is not removed.
    # For example only the second "." should be removed in the below string
    # "Mr. X correctly diagnozed his patient."
    ignored = stopwords.words("english") + [punct for punct in string.punctuation]
    clean_tokens = [token for token in tokens if token not in ignored]
    
    # Optionally lemmatize the output to reduce the number of unique words and address overfitting.
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in clean_tokens]
    return clean_tokens


def test_tokenize():
    """Unit test the tokenizer. """
    
    # With lemmatization
    text = "Mr. X correctly diagnosed his patients."
    expected_result = ['mr.', 'x', 'correctly', 'diagnosed', 'patient']
    assert tokenize(text) == expected_result
    
    # Without lemmatization
    expected_result = ['mr.', 'x', 'correctly', 'diagnosed', 'patients']
    assert tokenize(text, lemmatize=False) == expected_result
    
test_tokenize()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Bag of Words Representation using our own tokenizer.
vectorizer = CountVectorizer(lowercase=False, tokenizer=tokenize)
x = vectorizer.fit_transform(news['TITLE'])

# Create numerical labels.
encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# Let's keep this in order to interpret our results later,
encoder_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

# Split into a training and test set. Classifiers will be trained on the former and the final
# results will be reported on the latter.
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
from sklearn.ensemble import RandomForestClassifier

def report_accuracy(trained_clf):
    train_score = trained_clf.score(x_train, y_train)
    test_score = trained_clf.score(x_test, y_test)
    print("Training set accuracy score is: {}".format(emphasize(train_score)))
    print("Test set accuracy score is: {}".format(emphasize(test_score)))
    
# Let's use all our cores to speed things up.
n_cores = max(multiprocessing.cpu_count(), 1)

rf = RandomForestClassifier(n_jobs=n_cores)
rf.fit(x_train, y_train)
report_accuracy(rf)
from sklearn.metrics import confusion_matrix
import numpy as np

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=False):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap. Based on
    shaypal5's gist: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.set_ylabel('True label')
    heatmap.set_xlabel('Predicted label')
    heatmap.set_title(title)
    return fig

conf_mat = confusion_matrix(y_test, rf.predict(x_test))

# Get some readable labels
labels = [cat_map[encoder_mapping[label]] for label in sorted(encoder_mapping.keys())]
ax = print_confusion_matrix(conf_mat, labels, normalize=True)
plt.show()
news['WEEK'] = news['TIMESTAMP'].apply(lambda date: date.week)

# Aggregate by week
aggregated = news[['WEEK', 'MONTH', 'CATEGORY']]
grouped = aggregated.groupby(['WEEK', 'CATEGORY']).size().reset_index(name='article_count')
grouped['CATEGORY'] = grouped['CATEGORY'].apply(lambda x: cat_map[x])

plt.figure(figsize=(10, 7))
ax = sns.lineplot(x='WEEK', y='article_count', hue='CATEGORY', data=grouped, ci=None)
ax.set_xticks(list(range(11, 36)))
ax.set_xlabel("Week of the year")
ax.set_ylabel("Number of articles")
ax.set_title("Articles published per week")
plt.show()
from gensim import corpora
import numpy
import random

# Let's reuse the tokenizer we wrote before to clean the text.
clean_text = news['TITLE'].apply(tokenize)

# Reproducible topics.
numpy.random.seed(seed)
random.seed(seed)

# Create Dictionary and a Corpus (basic Gensim structures)
id2word = corpora.Dictionary(clean_text)
id2word.filter_extremes(no_below=5, no_above=0.05)
print(id2word)
corpus = [id2word.doc2bow(text) for text in clean_text]
import gensim
import re

num_topics = 4
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=seed)

# Let's inspect at the discovered topics.
def print_model(model):
    """Print a readable representation of a topic model. 
    
    Parameters
    ----------
    model : gensim.models.ldamodel.LdaModel
        A trained model.  
    """
    def print_topic(topic):
        topic_no, topic_repr = topic
        parts = topic_repr.split("+")
        words = [re.search('"(.*)"', part).group(1) for part in parts]
        return "{}: {}".format(topic_no, words)
    
    for topic in model.print_topics():
        print(print_topic(topic))
        
    
print_model(lda_model)
num_topics = 3
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=seed)
print_model(lda_model)
topic_mapping = {
    0: "Health",
    1: "Business & Technology",
    2: "Entertainment"
}

unseen_headlines = [
    "Beyonce won a music award",
    "Samsung releases new product - stock rises",
    "New cow disease outbreaks with multiple symptoms endagers humans"
]

def rank_headline(headline):
    bow_vector = id2word.doc2bow(tokenize(headline))
    lda_vector = lda_model[bow_vector]
    top_topic = topic_mapping[max(lda_vector, key=lambda item: item[1])[0]]
    distribution = {topic_mapping[topic_no]: proportion for topic_no, proportion in lda_vector}
    return top_topic, distribution

for headline in unseen_headlines:
    top_topic, distribution = rank_headline(headline)
    print("{}: {} \n Topic Mix: {}\n".format(headline, emphasize(top_topic), distribution))
unseen_headlines = [
    "Pop star Justin Bieber to start clothes company - stock expected to skyrocket",
    "Startup develops new sustainable vaccination against Ebola developed."
]

for headline in unseen_headlines:
    top_topic, distribution = rank_headline(headline)
    print("{}: {} - {}".format(headline, emphasize(top_topic), distribution))