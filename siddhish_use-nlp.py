name='siddhish'
print('His name is {var}'.format(var=name))
print(f"his name is {name}")
print(f'His name is {name!r}')
d={'a':123,'b':456}
print(f"Address: {d['a']} Main street")
library=[('Author','Topic', 'Pages'),('Twain', 'Rafting', 601), ('Feynman', 'Physics', 95),('Hamilton', 'Mythology',144)]

for book in library:
  print(f'{book[0]:{10}} {book[1]:{8}} {book[2]:{7}}')
from datetime import datetime
today=datetime(year=2018, month=1, day=27)
print(f'{today:%B %d %Y}')
text = "The agent's phone number is 408-555-1234. Call soon!"
'phone' in text
import re
pattern = 'phone'
re.search(pattern,text)
pattern = "NOT IN TEXT"
re.search(pattern,text)
pattern = 'phone'
match = re.search(pattern,text)
match
match.span()
match.start()
match.end()
text1 = "my phone is a new phone"
match = re.search("phone",text1)
match.span()
matches = re.findall("phone",text1)
matches
len(matches)
for match in re.finditer("phone", text):
  print(match.span())
match.group()
text= "My telephone number is 408-555-1234"
phone = re.search(r'\d\d\d-\d\d\d-\d\d\d\d',text)
phone.group()
re.search(r'\d{3}-\d{3}-\d{4}',text)
re.search(r"man|woman","this man is searching something")
re.search(r"man|woman","This woman was here.")
re.findall(r".at", "The cat in the hat sat here")
re.findall(r"...at", "the bat went splat")
re.findall(r'\S+at', "the bat went splat")
re.findall(r'\d$','This ends with a number 2')
phrase = "there are 3 numbers 34 inside 5 this sentence."
re.findall(r'[^\d]',phrase)
re.findall(r'[^\d]+',phrase)
text_phrase = 'This is a string! But it has punctuation. How can we remove it?'
re.findall('[^!.?]+', text_phrase)
clean=' '.join(re.findall('[^!.?]+', text_phrase))
clean
text = 'Only find the hypen-words in this sentence. But you do not know how long-ish they are'
re.findall(r'[\w]+-[\w]+',text)
# Find words that start with cat and end with one of these options: 'fish','nap', or 'claw'
text = 'Hello, would you like some catfish?'
texttwo = "Hello, would you like to take a catnap?"
textthree = "Hello, have you seen this caterpillar?"
re.search(r'cat(fish|nap|claw)',text)
re.search(r'cat(fish|nap|claw)',texttwo)
abbr = 'NLP'
full_text = 'Natural Language Processing'
print(f'{abbr} stands for {full_text}')
import spacy
nlp = spacy.load('en_core_web_sm')
doc=nlp(u'Tesla is looking at buying U.S. startup for $6 million')
for token in doc:
  print(token.text, token.pos_, token.dep_)
nlp.pipeline
nlp.pipe_names
doc2 = nlp(u"Tesla isn't   looking into startups anymore.")

for token in doc2:
    print(token.text, token.pos_, token.dep_)
doc2[0].pos_
spacy.explain('PROPN')
doc2[0].dep_
spacy.explain('nsubj')
print(doc2[4].text)
print('lemma : ', doc2[4].lemma_)
# Boolean Values:
print(doc2[0].is_alpha)
print(doc2[0].is_stop)
doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')
life_quote = doc3[16:30]
print(life_quote)
doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
for sent in doc4.sents:
    print(sent)
# Import the toolkit and the full Porter Stemmer library
import nltk

from nltk.stem.porter import *
p_stemmer = PorterStemmer()
words = ['run','runner','running','ran','runs','easily','fairly']
for word in words:
  print(word+ '----->' + p_stemmer.stem(word))
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')
words = ['run','runner','running','ran','runs','easily','fairly']
for word in words:
  print(word+'------->'+s_stemmer.stem(word))
phrase = 'I am meeting him tomorrow at the meeting'
for word in phrase.split():
  print(word+'------->'+p_stemmer.stem(word))
doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")

for token in doc1:
    print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
def show_lemmas(text):
  for token in text:
    print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
doc2 = nlp(u"I saw eighteen mice today!")
show_lemmas(doc2)
doc3 = nlp(u"I am meeting him tomorrow at the meeting.")
show_lemmas(doc3)
doc4 = nlp(u"That's an enormous automobile")
show_lemmas(doc4)
print(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)
nlp.vocab['myself'].is_stop
# Add the word to the set of stop words. Use lowercase!
nlp.Defaults.stop_words.add('btw')

# Set the stop_word tag on the lexeme
nlp.vocab['btw'].is_stop = True
# Remove the word from the set of stop words
nlp.Defaults.stop_words.remove('beyond')

# Remove the stop_word tag from the lexeme
nlp.vocab['beyond'].is_stop = False
nlp.vocab['beyond'].is_stop
# Import the Matcher library
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
matcher.add('SolarPower', None, pattern1, pattern2, pattern3)
doc = nlp(u'The Solar Power industry continues to grow as demand \
for solarpower increases. Solar-power cars are gaining popularity.')
found_matches = matcher(doc)
print(found_matches)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start:end]                    # get the matched span
    print(match_id, string_id, start, end, span.text)
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)
found_matches = matcher(doc)
print(found_matches)
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}] # CHANGE THIS PATTERN

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)
doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')
found_matches = matcher(doc2)
print(found_matches)
# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
# Print the full text:
print(doc.text)
print(doc[4].text, doc[4].pos_, doc[4].tag_, spacy.explain(doc[4].tag_))
for token in doc:
  print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')
doc = nlp(u'I read books on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# Count the frequencies of different coarse-grained POS tags:
POS_counts = doc.count_by(spacy.attrs.POS)
POS_counts
for k,v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')
doc = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')

show_ents(doc)
doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')

for ent in doc.ents:
    print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)
doc = nlp(u'Tesla to build a U.K. factory for $6 million')

show_ents(doc)
from spacy.tokens import Span

# Get the hash value of the ORG entity label
ORG = doc.vocab.strings[u'ORG']  

# Create a Span for the new entity
new_ent = Span(doc, 0, 1, label=ORG)

# Add the entity to the existing Doc object
doc.ents = list(doc.ents) + [new_ent]
show_ents(doc)
doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')

show_ents(doc)
len([ent for ent in doc.ents if ent.label_=='MONEY'])
# From Spacy Basics:
doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc.sents:
    print(sent)
# Parsing the segmentation start tokens happens during the nlp pipeline
doc2 = nlp(u'This is a sentence. This is a sentence. This is a sentence.')

for token in doc2:
    print(token.is_sent_start, ' '+token.text)
doc3 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc3.sents:
    print(sent)
import numpy as np
import pandas as pd

df = pd.read_csv('/content/datasets_483_982_spam.csv',engine='python')

df=df[['v1', 'v2']]
df.head()
import seaborn as sns
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
len(df)
df.isnull().sum()
df['v1'].unique()
df['v1'].value_counts()
df['label'] = df.v1.map({'ham':0, 'spam':1})
df.head()
df['message_len'] = df.v2.apply(len)
df.head()
import matplotlib.pyplot as plt
plt.xscale('log')
bins=1.15**(np.arange(0,50))
plt.hist(df[df['v1']=='ham']['message_len'], bins=bins, alpha=0.8 )
plt.hist(df[df['v1']=='spam']['message_len'], bins=bins, alpha=0.8)
plt.legend(('ham', 'spam'))
plt.show()
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
def text_preprocess(message):
  stopword=stopwords.words('english') 
  punc=[char for char in message if char not in string.punctuation]
  punc=''.join(punc)
  return ' '.join([word for word in punc.split() if word.lower() not in stopword])
df['clean']=df.v2.apply(text_preprocess)
df.head()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
X = df.clean
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
max_words = 2000
max_len = 200
tokens = Tokenizer(num_words=max_words)
tokens.fit_on_texts(X_train)
sequences = tokens.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='layer')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='output_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
from keras.callbacks import EarlyStopping
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tokens.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accuracy = model.evaluate(test_sequences_matrix,Y_test)
print(accuracy[0], accuracy[1])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
Count_vec=CountVectorizer()
X=Count_vec.fit_transform(X)
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.33,random_state=42)
from sklearn.linear_model import LogisticRegression
logisticreg=LogisticRegression()
logisticreg.fit(X_train,Y_train)
logisticreg.score(X_test,Y_test)
y_pred=logisticreg.predict(X_test)
print(classification_report(Y_test,y_pred))
confusion_matrix(Y_test,y_pred)
from sklearn.naive_bayes import MultinomialNB
naive_bayes=MultinomialNB()
naive_bayes.fit(X_train,Y_train)
naive_bayes.score(X_test,Y_test)
y_pred=naive_bayes.predict(X_test)
print(classification_report(Y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_jobs=-1)
random_forest.fit(X_train,Y_train)
random_forest.score(X_test,Y_test)
y_pred=random_forest.predict(X_test)
print(classification_report(Y_test,y_pred))
confusion_matrix(Y_test,y_pred)
from sklearn.ensemble import AdaBoostClassifier
AdaBoost=AdaBoostClassifier()
AdaBoost.fit(X_train,Y_train)
AdaBoost.score(X_test,Y_test)
y_pred=AdaBoost.predict(X_test)
print(classification_report(Y_test,y_pred))
confusion_matrix(Y_test,y_pred)