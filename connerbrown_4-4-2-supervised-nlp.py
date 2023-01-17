%matplotlib inline
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import gutenberg, stopwords
from collections import Counter
# Utility function for standard text cleaning.
def text_cleaner(text):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = re.sub(r'--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.split())
    return text
    
# Load and clean the data.
doc_length = 100000
persuasion = gutenberg.raw('austen-persuasion.txt')[:doc_length]
alice = gutenberg.raw('carroll-alice.txt')[:doc_length]

# The Chapter indicator is idiosyncratic
persuasion = re.sub(r'Chapter \d+', '', persuasion)
alice = re.sub(r'CHAPTER .*', '', alice)
    
alice = text_cleaner(alice)
persuasion = text_cleaner(persuasion)
# Parse the cleaned novels. This can take a bit.
nlp = spacy.load('en')
alice_doc = nlp(alice)
persuasion_doc = nlp(persuasion)
# Group into sentences.
alice_sents = [[sent, "Carroll Alice"] for sent in alice_doc.sents]
persuasion_sents = [[sent, "Austen Persuasion"] for sent in persuasion_doc.sents]

# Combine the sentences from the two novels into one data frame.
sentences = pd.DataFrame(alice_sents + persuasion_sents)
sentences.head()
# Utility function to create a list of the 2000 most common words.
def bag_of_words(text):
    
    # Filter out punctuation and stop words.
    allwords = [token.lemma_
                for token in text
                if not token.is_punct
                and not token.is_stop]
    
    # Return the most common words.
    return [item[0] for item in Counter(allwords).most_common(500)]
    

# Creates a data frame with features for each word in our common word set.
# Each value is the count of the times the word appears in each sentence.
def bow_features(sentences, common_words):
    
    # Scaffold the data frame and initialize counts to zero.
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:,'word_count'] = 0 # add word count
    df.loc[:,'entity_count'] = 0 # add entity count
    df.loc[:, common_words] = 0
    
    # Process each row, counting the occurrence of words in each sentence.
    for i, sentence in enumerate(df['text_sentence']):
        
        # Convert the sentence to lemmas, then filter out punctuation,
        # stop words, and uncommon words.
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words
                 )]
        df.loc[i,'entity_count'] = len(sentence.as_doc().ents)
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i,'word_count'] += 1
            df.loc[i, word] += 1
        
        # This counter is just to make sure the kernel didn't hang.
        if i % 500 == 0:
            print("Processing row {}".format(i))
            
    return df

# Set up the bags.
alicewords = bag_of_words(alice_doc)
persuasionwords = bag_of_words(persuasion_doc)

# Combine bags to create a set of unique words.
common_words = set(alicewords + persuasionwords)
# Create our data frame with features. This can take a while to run.
word_counts = bow_features(sentences, common_words)
word_counts.head()
from sklearn import ensemble
from sklearn.model_selection import train_test_split

rfc = ensemble.RandomForestClassifier()
Y = word_counts['text_source']
X = word_counts.drop(['text_sentence','text_source'], 1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.4,
                                                    random_state=0)
train = rfc.fit(X_train, y_train)

print('Training set score:', rfc.score(X_train, y_train))
print('\nTest set score:', rfc.score(X_test, y_test))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
train = lr.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', lr.score(X_train, y_train))
print('\nTest set score:', lr.score(X_test, y_test))
clf = ensemble.GradientBoostingClassifier()
train = clf.fit(X_train, y_train)

print('Training set score:', clf.score(X_train, y_train))
print('\nTest set score:', clf.score(X_test, y_test))
# Clean the Emma data.
emma = gutenberg.raw('austen-emma.txt')[:doc_length]
emma = re.sub(r'VOLUME \w+', '', emma)
emma = re.sub(r'CHAPTER \w+', '', emma)
emma = text_cleaner(emma)
print(emma[:100])
# Parse our cleaned data.
emma_doc = nlp(emma)
# Group into sentences.
emma_sents = [[sent, "Austen Emma"] for sent in emma_doc.sents]

# Emma is quite long, let's cut it down to the same length as Alice.
emma_sents = emma_sents[0:len(alice_sents)]
# Build a new Bag of Words data frame for Emma word counts.
# We'll use the same common words from Alice and Persuasion.
emma_sentences = pd.DataFrame(emma_sents)
emma_bow = bow_features(emma_sentences, common_words)

print('done')
# Now we can model it!
# Let's use logistic regression again.

# Combine the Emma sentence data with the Alice data from the test set.
X_Emma_test = np.concatenate((
    X_train.loc[y_train[y_train=='Carroll Alice'].index],
    emma_bow.drop(['text_sentence','text_source'], 1)
), axis=0)
y_Emma_test = pd.concat([y_train[y_train=='Carroll Alice'],
                         pd.Series(['Austen Emma'] * emma_bow.shape[0])])

# Model.
print('\nTest set score:', lr.score(X_Emma_test, y_Emma_test))
lr_Emma_predicted = lr.predict(X_Emma_test)
pd.crosstab(y_Emma_test, lr_Emma_predicted)
svc = SVC()
svc_train = svc.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', svc.score(X_train, y_train))
print('\nTest set score:', svc.score(X_test, y_test))
gnb = GaussianNB()
gnb_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Training score: ",gnb.score(X_train,y_train))
print("Testing score:",gnb.score(X_test,y_test))
print(pd.crosstab(y_test,gnb_pred))
# add another work
moby = gutenberg.raw('melville-moby_dick.txt')[:doc_length]
moby = re.sub(r'VOLUME \w+', '', moby)
moby = re.sub(r'CHAPTER \w+', '', moby)
moby = text_cleaner(moby)
print(moby[:100])
moby_doc = nlp(moby)
# Group into sentences.
moby_sents = [[sent, "Melville Moby"] for sent in moby_doc.sents]

# Build a new Bag of Words data frame for Emma word counts.
# We'll use the same common words from Alice and Persuasion.
moby_sentences = pd.DataFrame(moby_sents)
moby_bow = bow_features(moby_sentences, common_words)

print('done')
# add emma and moby bow to word_counts
df_works = pd.concat([word_counts,emma_bow,moby_bow],ignore_index=True)
Y_works = df_works['text_source']
X_works = df_works.drop(['text_sentence','text_source'], 1)
X_works_train, X_works_test, Y_works_train, Y_works_test = train_test_split(X_works,Y_works, test_size = 0.4, random_state=42)
# Now we can model it!
# Let's use logistic regression again.

works = ["Carroll Alice", "Austen Persuasion", "Austen Emma"]
for work in works:
    # Combine the Moby sentence data with the Alice data from the test set.
    work_train_index = list(Y_works_train[Y_works_train==work].index) + list(Y_works_train[Y_works_train=="Melville Moby"].index)
    X_moby_train = X_works_train.loc[work_train_index]
    Y_moby_train = Y_works_train.loc[work_train_index]
    work_test_index = list(Y_works_test[Y_works_test==work].index) + list(Y_works_test[Y_works_test=="Melville Moby"].index)
    X_moby_test = X_works_test.loc[work_test_index]
    Y_moby_test = Y_works_test.loc[work_test_index]
    print("===========",work,"===========")
# Model.
    gnb.fit(X_moby_train,Y_moby_train)
    display("Test set score:", gnb.score(X_moby_test, Y_moby_test))
    gnb_moby_predicted = gnb.predict(X_moby_test)
    print(pd.crosstab(Y_moby_test, gnb_moby_predicted))

