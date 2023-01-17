import pandas as pd
from textatistic import Textatistic
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import string
import re
# IMDB Movie Review
imdb_movie_review = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."

# Tweet
tweet = "As requested by PM Modi, lets make these promises as #IndiaFightsCOVID19, In battle against #COVID19, lets #TakeThePledge to take care of the elderly people at home. @PPBhaishri takes the pledge"
num_char = len(imdb_movie_review)
print(num_char)
# Split the string into words
words = imdb_movie_review.split()

# Print the list containing words
print(words)

# print the length of words
print(len(words))
# avg_word_len
def avg_word_len(text):
    # split the string into words
    words = text.split()
    
    # compute the length each word and store in a seperated list
    word_len = [len(word) for word in words]
    
    # compute avg_word_len
    avg_word_len = sum(word_len) / len(word_len)
    
    return avg_word_len


# call the avg_word_len and pass imdb_movie review and print
print(avg_word_len(imdb_movie_review))
# Return the numbe of Hashtags

def hashtag_count(text):
    # split the  string into words
    words = text.split()
    
    # create a list of hashtags
    hashtags = [word for word in words if word.startswith('#')]
    
    # print the hashtags
    print(hashtags)
    
    return len(hashtags)

# Call the hashtag_count function and pass the argument tweet
hashtag_count(tweet)
# create remove_spaces function and pass text as an argument and remove white spaces and return
def remove_spaces(text):
    return " ".join(text.split())

# Call the above function and print
print(remove_spaces("This   notebook       is all       about the feature             engineering for text                 data."))
def remove_punctuation(text):
    text = "".join(ch for ch in text if not ch in string.punctuation)
    return text

# Call the above function and print
remove_punctuation('@Welcome to the Kaggle and ------Good Bye!')
def remove_num(text):
    text = filter(str.isalpha, text)
    
    return "".join(text)

#  Call the above function and print
remove_num('!1k2;a3gg23456le3?')
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
#  Call the above function and print
print(remove_emoji("Happy😂"))
def remove_stop_word(text):
    text = " ".join(word for word in text.lower().split() if not word in  STOP_WORDS)
    return text

#  Call the above function and print
print(remove_stop_word("I was very active in Pubg mobile"))
# Load Spacy model
nlp = spacy.load('en_core_web_lg')

def tokenizing_string(text):
    doc = nlp(text)
    
    return [token.text for token in doc]

print(tokenizing_string("Hello! I'm here and I'm doing coding on Kaggle"))
def lemmatizing_string(text):
    doc = nlp(text)
    
    return [token.lemma_ for token in doc]

print(lemmatizing_string("Hello! I'm here and I'm doing coding on Kaggle"))
fake = "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media."

# Generate redability score
readability_scores = Textatistic(fake).scores

print('Flesch Score: ', readability_scores['flesch_score'])
print('Gunning fog index score: ', readability_scores['gunningfog_score'])
def pos_tagging(text):
    doc = nlp(text)
    
    return [(token.text, token.pos_) for token in doc]

pd.DataFrame(pos_tagging("Hello! How are you doing?"), columns=['words', 'pos tagging'])
def get_ner(text):
    doc = nlp(text)
    
    return [(ent.text, ent.label_) for ent in doc.ents]

get_ner("John Deo is a software engineer working at Kaggle. He lives in New York")
# Visualize NER
doc = nlp("John Deo is a software engineer working at Kaggle. He lives in New York")

displacy.serve(doc, style="ent")
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]

vectorizer = CountVectorizer()
# fit and transform corpus
count_vectorizer = vectorizer.fit_transform(corpus)

# Create Dataframe with the feature name
pd.DataFrame(count_vectorizer.toarray(), columns=vectorizer.get_feature_names())
vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

n_gram_vectorizer = vectorizer.fit_transform(corpus)

# Create Dataframe with the feature name
pd.DataFrame(n_gram_vectorizer.toarray(), columns=vectorizer.get_feature_names())
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(corpus)

# Create Dataframe with the feature name
pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
doc1 = nlp("I love to code in Jupyter Notebook.")
doc2 = nlp("I hate to code in Jupyter Notebook")

doc1.similarity(doc2)
doc1 = nlp("I am using Kaggle.")
doc2 = nlp("I am using Kaggle.")

doc1.similarity(doc2)