# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Warnings

import warnings

warnings.filterwarnings('ignore')



# Styles

plt.style.use('ggplot')

sns.set_style('whitegrid')



plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.titlesize'] = 12

plt.rcParams['patch.force_edgecolor'] = True



# Text Preprocessing

import nltk

# nltk.download("all")

from nltk.corpus import stopwords

import string

from nltk.tokenize import word_tokenize



import spacy



nlp = spacy.load("en")
spam_folder = '/kaggle/input/ham-and-spam-dataset/spam'

ham_folder = '/kaggle/input/ham-and-spam-dataset/ham'



ham_filenames = [name for name in sorted(os.listdir(ham_folder)) if len(name) > 20]

spam_filenames = [name for name in sorted(os.listdir(spam_folder)) if len(name) > 20]



print('Number of non-spam samples:', len(ham_filenames))

print('Number of spam samples:', len(spam_filenames))

print('Ratio of non-spam to spam samples:', len(ham_filenames)/len(spam_filenames))
import email

import email.policy



def load_email(is_spam, filename):

    directory = spam_folder if is_spam else ham_folder

    

    with open(os.path.join(directory, filename), "rb") as f:

        return email.parser.BytesParser(policy=email.policy.default).parse(f)

    

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]

spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
from collections import Counter



def get_email_structure(email):

    if isinstance(email, str):

        return email

    payload = email.get_payload()

    if isinstance(payload, list):

        return "multipart({})".format(", ".join([

            get_email_structure(sub_email)

            for sub_email in payload

        ]))

    else:

        return email.get_content_type()



def structures_counter(emails):

    structures = Counter()

    for email in emails:

        structure = get_email_structure(email)

        structures[structure] += 1

    return structures



ham_structure = structures_counter(ham_emails)

spam_structure = structures_counter(spam_emails)
def html_to_plain(email):

    try:

        soup = BeautifulSoup(email.get_content(), 'html.parser')

        return soup.text.replace('\n\n','')

    except:

        return "empty"
def email_to_plain(email):

    struct = get_email_structure(email)

    for part in email.walk():

        partContentType = part.get_content_type()

        if partContentType not in ['text/plain','text/html']:

            continue

        try:

            partContent = part.get_content()

        except: # in case of encoding issues

            partContent = str(part.get_payload())

        if partContentType == 'text/plain':

            return partContent

        else:

            return html_to_plain(part)

        

#print(email_to_plain(ham_emails[42]))

print(email_to_plain(spam_emails[45]))
# structure all emails into plain text

ham_emails_plain = [email_to_plain(email) for email in ham_emails if len(ham_emails) > 100]

spam_emails_plain = [email_to_plain(email) for email in spam_emails if len(spam_emails) > 100]



# ham_structure = structures_counter(ham_emails_plain)



# ham_structure.most_common()
# some data conversion to get it into pandas

ham_dic = {}

spam_dic = {}

ham_dic['text'] = ham_emails_plain

spam_dic['text'] = spam_emails_plain



ham_df = pd.DataFrame(ham_dic, columns = ['text', 'category'])

spam_df = pd.DataFrame(spam_dic, columns = ['text', 'category'])



# setting labels

ham_df['category'] = 0

spam_df['category'] = 1



frames = [ham_df, spam_df]



# dataframe of messages with proper labels for spam and non-spam

messages = pd.concat(frames).reset_index(drop=True)
# Dropping rows with NA values

messages.dropna(inplace=True)



messages["category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)

plt.ylabel("Spam vs Ham")

plt.legend(["Ham", "Spam"])

plt.show()
spam_messages = messages[messages["category"] == 1]["text"]

ham_messages = messages[messages["category"] == 0]["text"]





spam_words = []

ham_words = []



# # Since this is just classifying the message as spam or ham, we can use isalpha(). 

# # This will also remove the not word in something like can't etc. 

# # In a sentiment analysis setting, its better to use 

# # sentence.translate(string.maketrans("", "", ), chars_to_remove)



def extractSpamWords(spamMessages):

    global spam_words, spam_exception_count

    spam_exception_count = 0

    try:

        word_tokenized = word_tokenize(spamMessages)

        words = [word.lower() for word in word_tokenized if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

        spam_words = spam_words + words

    except:

        spam_exception_count += 1

    

    

    

def extractHamWords(hamMessages):

    global ham_words, ham_exception_count

    ham_exception_count = 0

    try:

        word_tokenized = word_tokenize(hamMessages)

        words = [word.lower() for word in  word_tokenized if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

        ham_words = ham_words + words

    except:

        ham_exception_count += 1

        

    

# Checking tokenization errors. At some point I had to troubleshoot the code

spam_messages.apply(extractSpamWords)

print('spam exception count: ', spam_exception_count)

ham_messages.apply(extractHamWords)

print('ham exception count: ', ham_exception_count)

from wordcloud import WordCloud
#Ham word cloud



ham_wordcloud = WordCloud(width=600, height=400).generate(" ".join(ham_words))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Spam Word cloud



spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam_words))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()

# Top 10 spam words



spam_words = np.array(spam_words)

print("Top 10 Spam words are :\n")

print(pd.Series(spam_words).value_counts().head(n = 10))



ham_words = np.array(ham_words)

print("\nTop 10 Ham words are :\n")

print(pd.Series(ham_words).value_counts().head(n = 10))

messages["messageLength"] = messages["text"].apply(len)

messages["messageLength"].describe()
f, ax = plt.subplots(1, 2, figsize = (20, 6))



sns.distplot(messages[messages["category"] == 1]["messageLength"], bins = 20, ax = ax[0])

ax[0].set_xlabel("Spam Message Word Length")



sns.distplot(messages[messages["category"] == 0]["messageLength"], bins = 20, ax = ax[1])

ax[0].set_xlabel("Ham Message Word Length")



plt.show()

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



def cleanText(message):

    

    try:

        message = message.translate(str.maketrans('', '', string.punctuation))

        words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]

        return " ".join(words)

    except:

        print(message)

        

    

messages["text"] = messages["text"].apply(cleanText)

messages.head(n = 10)    

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

features = vec.fit_transform(messages["text"])

print(features.shape)

from sklearn.model_selection import train_test_split

print(features.shape)

print(messages["category"].shape)

X_train, X_test, y_train, y_test = train_test_split(features, messages["category"], stratify = messages["category"], test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import confusion_matrix



names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",

         "Naive Bayes", "SVM Linear"]



classifiers = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LogisticRegression(),

    SGDClassifier(max_iter = 100),

    MultinomialNB(),

    SVC(kernel = 'linear')

]



models = zip(names, classifiers)



for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + name + ":")

    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))

    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

    print("Confusion Matrix:\n") 

    confusion_m = confusion_matrix(y_test, y_pred)

    print(confusion_m)
