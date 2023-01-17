import pandas as pd

import os

import string

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

import seaborn as sns

rand_state = 42

np.random.seed(rand_state)



warnings.simplefilter('ignore')

%matplotlib inline
path = '../input/sotu'

dirs = os.listdir(path)



df = pd.DataFrame(columns=['year', 'president', 'party', 'text'])



for i in range(len(dirs)):

    components = dirs[i].split('_')

    name = components[0]

    year = components[1].split('.')[0]

    df.loc[i,'year'] = year

    df.loc[i,'president'] = name   

    

    filename = os.path.join(path, dirs[i])

    text_file = open(filename, "r")

    

    lines = text_file.read()

    lines = lines.replace('\n', ' ')

    df.loc[i, 'text'] = lines.lower()

    

df.year = df.year.astype(int) 

df.president = df.president.astype(str)

df.text = df.text.astype(str)

print('Shape: ', df.shape)
# need to distinuish between Theodore Roosevelt and Franklin D. Roosevelt

indices = df.query("president =='Roosevelt' & year <= 1909").index

df.loc[indices,'president'] = 'Theodore Roosevelt'



indices = df.query("president == 'Roosevelt'").index

df.loc[indices,'president'] = 'Franklin D. Roosevelt'



indices = df.query("president =='Bush' & year <= 1992").index

df.loc[indices,'president'] = 'George H. W. Bush'



indices = df.query("president == 'Bush'").index

df.loc[indices,'president'] = 'George W. Bush'



indices = df.query("president =='Johnson' & year <= 1869").index

df.loc[indices,'president'] = 'Andrew Johnson'



indices = df.query("president == 'Johnson'").index

df.loc[indices,'president'] = 'Lyndon B. Johnson'



indices = df.query("president =='Adams' & year <= 1801").index

df.loc[indices,'president'] = 'John Adams'



indices = df.query("president == 'Adams'").index

df.loc[indices,'president'] = 'John Quincy Adams'





indices = df.query("president =='Harrison' & year <= 1841").index

df.loc[indices,'president'] = 'William Henry Harrison'



indices = df.query("president == 'Harrison'").index

df.loc[indices,'president'] = 'Benjamin Harrison'
def pres_to_party(name):

    republican = ['Lincoln', 'Grant', 'Hayes', 'Garfield', 'Arthur', 

                  'Benjamin Harrison', 'McKinley', 'Theodore Roosevelt', 

                  'Taft', 'Harding', 'Coolidge', 'Hoover', 'Eisenhower', 

                  'Nixon', 'Ford', 'Reagan', 'George H. W. Bush', 

                  'George W. Bush', 'Trump']

    if name in republican:

        return 'Republican'

    

    democratic = ['Jackson', 'Buren', 'Polk', 'Pierce', 

                  'Buchanan', 'Cleveland', 'Wilson', 'Franklin D. Roosevelt', 

                  'Truman', 'Kennedy', 'Lyndon B. Johnson', 'Carter', 'Clinton', 'Obama']

    if name in democratic:

        return 'Democratic'

    

    whig = ['William Henry Harrison', 'Taylor', 'Fillmore']

    if name in whig:

        return 'Whig'

    

    national_union = ['Andrew Johnson']

    if name in national_union:

        return 'National Union'

    

    

    unaffiliated = ['Washington', 'Tyler']

    if name in unaffiliated:

        return 'Unaffiliated'

    

    federalist = ['John Adams']

    if name in federalist:

        return 'Federalist'

    

    democratic_republican = ['Jefferson', 'Madison', 'Monroe', 'John Quincy Adams']

    if name in democratic_republican:

        return 'Democratic-Republican'

    

df.party = df.president.apply(pres_to_party)
df.set_index('year', inplace=True)

df.sort_index(inplace=True)



# need to drop George Washington's 1790 address as the file is empty

df = df.iloc[1:,:]

df.head()
df.groupby('party').size()
df = df[df.party.isin(['Republican', 'Democratic'])]
from nltk import sent_tokenize



sentences = [sent_tokenize(text) for text in df.text]



# remove the first and last sentences (meaningless intro/closing statements)

for i in range(len(sentences)):

    del sentences[i][0]

    del sentences[i][-1]

    

    

sentence_lengths = [len(sent) for sent in sentences]

df['sentences'] = sentences

df['sentence_length'] = [len(sent) for sent in sentences]



# now need to "unstack" the above list of lists of sentences

sentences_all = []

for sentences in sentences:

    for sent in sentences:

        sentences_all.append(sent)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(10,6))

sns.lineplot(x='year',y='sentence_length',hue='party',data=df.reset_index())

plt.xlabel('')

plt.ylabel('Number of Sentences')

sns.despine()

plt.show()
keys = df.president.unique()

values = np.arange(keys.shape[0])

pres_to_num = dict(zip(keys, values))



df['president_num'] = df.president.map(pres_to_num)

# we will use Democratic as the positive class

df['party_num'] = (df.party == 'Democratic').astype(int)



target = [] # np.zeros((len(sentences_all,)))



for i in range(df.shape[0]):

    target.append(np.ones((df.iloc[i,4],)) * df.iloc[i,5])

    

target = np.concatenate(target, axis=0)
from keras.utils import to_categorical



target = to_categorical(target)
from sklearn.model_selection import train_test_split



sentences_train, sentences_test, y_train, y_test = train_test_split(sentences_all, target, test_size=0.2, random_state=42)
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_all)



x_train = tokenizer.texts_to_sequences(sentences_train)

x_test = tokenizer.texts_to_sequences(sentences_test)
sentences_words = [len(sequence) for sequence in x_train]

print("99% quantile: ", pd.Series(sentences_words).quantile(.99))

sns.distplot(sentences_words)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



print('Vocabulary size: ', vocab_size)

print(sentences_train[2])

print(x_train[2])
from keras.preprocessing.sequence import pad_sequences



maxlen = 91



x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)

x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
from keras.models import Sequential

from keras import layers



embedding_dim = 100



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(target.shape[1], activation='softmax'))



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



estop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

mcp_save = ModelCheckpoint('embeddings_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1) #, epsilon=1e-4, mode='min')



history = model.fit(x_train, y_train,

                    epochs=10,

                    verbose=True,

                    validation_data=(x_test, y_test),

                    callbacks=[estop, mcp_save, reduce_lr_loss],

                    batch_size=128)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
import re

from gensim import models, corpora

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk import pos_tag

from nltk.stem import WordNetLemmatizer



NUM_TOPICS = 20

STOPWORDS = stopwords.words('english')



wnl = WordNetLemmatizer()



def penn2morphy(penntag):

    """ Converts Penn Treebank tags to WordNet. """

    morphy_tag = {'NN':'n', 'JJ':'a',

                  'VB':'v', 'RB':'r'}

    try:

        return morphy_tag[penntag[:2]]

    except:

        return 'n' 



def lemmatize_sent(text): 

    # Text input is string, returns lowercased strings.

    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 

            for word, tag in pos_tag(word_tokenize(text))]



def clean_text(text):

    tokenized_text = word_tokenize(text.lower())

    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]

    return lemmatize_sent(' '.join(cleaned_text))
df['tokens'] = df.text.apply(clean_text)

df.head()
#WordCloud

from wordcloud import WordCloud, STOPWORDS



cleaned_text = ' '.join(list(df.text))



wordcloud = WordCloud(stopwords=STOPWORDS,max_words=100,

                      background_color='black',min_font_size=6,

                      width=3000,collocations=False,

                      height=2500

                     ).generate(cleaned_text)



plt.figure(1,figsize=(20, 20))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('wordcloud_1.png')

plt.show()
token_string = ''

for tokens in df.tokens:

    token_string += ' '.join(tokens) + ' '



wordcloud_tokens = WordCloud(stopwords=STOPWORDS,max_words=100,

                      background_color='black',min_font_size=6,

                      width=3000,collocations=False,

                      height=2500

                     ).generate(token_string)



plt.figure(1,figsize=(20, 20))

plt.imshow(wordcloud_tokens)

plt.axis('off')

plt.savefig('wordcloud_2.png')

plt.show()
# Build a Dictionary - association word to numeric id

dictionary = corpora.Dictionary(df.tokens)



'''

We can "control" the level in which we extract topics from:

  * We can filter out tokens that show up in x% of all SOTU's, in effect

    uncovering more hidden topics (only present in (1-x)% of the SOTUs).

  * A similar strategy can be used to filter out very rare tokens by setting

    no_below



Initially we will not do this, but it follows that this strategy might be helpful



in classification models. In this case we might also want to increase

  the number of latent topics to discover: less frequenct topics could be 

  quite powerful in prediction

'''

dictionary.filter_extremes(no_below=3, no_above=.03)



# Transform the collection of texts to a numerical form

corpus = [dictionary.doc2bow(text) for text in df.tokens]



# Build the LDA model

lda_model = models.LdaModel(corpus=corpus, 

                            random_state=rand_state, 

                            iterations=200,

                            num_topics=20, 

                            id2word=dictionary)



print("LDA Model:")

 

for idx in range(NUM_TOPICS):

    # Print the first 10 most representative topics

    print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))
import pyLDAvis.gensim as gensimvis

import pyLDAvis





vis_data_lda = gensimvis.prepare(lda_model, corpus, dictionary)

#pyLDAvis.save_html(vis_data_lda,'lda_all.html')

pyLDAvis.display(vis_data_lda)
lda_scores = [] #np.array((len(corpus), NUM_TOPICS))



for i in range(len(corpus)):

    y = lda_model[corpus[i]]

    #lda_scores.append([score[1] for score in y])

    lda_scores.append({score[0]:score[1] for score in y})

    

lda_df = pd.DataFrame(lda_scores)

lda_df.index = df.index

lda_df.fillna(0.0, inplace=True)

lda_df.head()
lda_df['party'] = df.party

lda_df.groupby('party').mean()
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict





lda_new = lda_df[lda_df.party.isin(['Republican', 'Democratic'])]



X_train, X_test, y_train, y_test = train_test_split(lda_new.drop('party', axis=1), 

                                                    lda_new['party'], 

                                                    test_size=0.2, 

                                                    random_state=rand_state)



model = RidgeClassifier(class_weight='balanced', random_state=rand_state)

model.fit(X_train, y_train)

print("Train score: ", model.score(X_train, y_train))

print("Test score: ", model.score(X_test, y_test))
path = '../input/sotu'

dirs = os.listdir(path)



df = pd.DataFrame(columns=['year', 'president', 'text'])



for i in range(len(dirs)):

    components = dirs[i].split('_')

    name = components[0]

    year = components[1].split('.')[0]

    df.loc[i,'year'] = year

    df.loc[i,'president'] = name   

    

    filename = os.path.join(path, dirs[i])

    text_file = open(filename, "r")

    

    lines = text_file.read()

    df.loc[i, 'text'] = lines.replace('\n', ' ')

    

df.year = df.year.astype(int) 

df.president = df.president.astype(str)

df.text = df.text.astype(str)

df.set_index('year', inplace=True)

df.sort_index(inplace=True)



# need to drop George Washington's 1790 address as the file is empty

df = df.iloc[1:,:]
def clean(text):

    # remove \

    text = text.strip('\\')

    # replace -- with space

    text = text.replace('--',' ')

    

    return text
from sumy.summarizers.lex_rank import LexRankSummarizer

from sumy.summarizers.luhn import LuhnSummarizer

from sumy.summarizers.lsa import LsaSummarizer

from sumy.summarizers.text_rank import TextRankSummarizer



def summarize_text(text, should_print=True):

    #Summarize the document with 2 sentences

    summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, 2) 



    summarizer_1 = LuhnSummarizer()

    summary_1 = summarizer_1(parser.document,2)



    summarizer_2 = LsaSummarizer()

    summary_2 =summarizer_2(parser.document,2)



    summarizer_3 = TextRankSummarizer()

    summary_3 =summarizer_3(parser.document,2)

    

    if should_print:

        print('Lex Rank\n===========')

        for sentence in summary:

            print('-', sentence)

        

        print('\nLuhn\n===========')

        for sentence in summary_1:

            print('-', sentence)

            

        print('\nLSA\n===========')

        for sentence in summary_2:

            print('-', sentence)

            

        print('\nText Rank\n===========')

        for sentence in summary_3:

            print('-', sentence)

            

    return summary + summary_1 + summary_2 + summary_3
#Plain text parsers since we are parsing through text

from sumy.parsers.plaintext import PlaintextParser

from sumy.nlp.tokenizers import Tokenizer



text = clean(df.loc[2017,'text'])

parser = PlaintextParser.from_string(text, Tokenizer('english'))



print('-----------------------------------------')

print('Donald J. Trump, 2017')

print('-----------------------------------------')

summaries_2005 = summarize_text(text)
text = clean(df.loc[2012,'text'])

parser = PlaintextParser.from_string(text, Tokenizer('english'))



print('-----------------------------------------')

print('Barack Obama, 2012')

print('-----------------------------------------')

summaries_2012 = summarize_text(text)
text = clean(df.loc[2005,'text'])

parser = PlaintextParser.from_string(text, Tokenizer('english'))



print('-----------------------------------------')

print('George W. Bush, 2012')

print('-----------------------------------------')

summaries_2005 = summarize_text(text)