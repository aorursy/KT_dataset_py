# import libraries that we  might need

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from nltk.corpus import stopwords

import nltk

import re



from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras import layers

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
# Load the data set from csv file

df = pd.read_csv('../input/IMDB Dataset.csv')

print('Shape of  Data is {} rows  and {} columns'.format(df.shape[0],df.shape[1]))
print('A Sample Input & Output')

print('-----------------------------------------------------------')

print(df['review'][5])

print('-----------------------------------------------------------')

print(df['sentiment'][5])
# We need to preprocess the text to remove the punctuations and tags and any other junk characters.

def preprocessing(sen):

    #Remove Tags

    # Remove any html tags that might have

    sen = remove_tags(sen)

    

    sen = sen.lower()

    

    #  we only want to keep text and not words or punctuations.

    sen = re.sub(r'[^a-zA-Z]',' ',sen)

    

    # We will remove any single characters

    sen = re.sub(r'\s+',' ',sen)

    

    #  We will remove any extra spaces we have added

    sen = re.sub(r'\s+[a-zA-Z]\s+',' ',sen)

    

    return sen



tags = re.compile(r'<[^>]+>')

def remove_tags(sen):

    return tags.sub('',sen)
#  lets  pre-process the the data and put it  into our Input variables

X = df['review'].apply(preprocessing)

y = df['sentiment'].map({'positive':1,'negative':0})
#  Lets look at the processed data

print('--------------------------------------------')

X[5]
# We will split the data into two sets, train data set would later be split to be used for validation as well.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100



X_train = pad_sequences(X_train,padding='post',maxlen=maxlen)

X_test = pad_sequences(X_test,padding='post',maxlen=maxlen)
vocab_size
glove_file = open('../input/glove.6B.100d.txt',encoding='utf-8')

word_embedding = dict()



for line in glove_file:

    records = line.split()

    word = records[0]

    word_embedding[word] = np.asarray(records[1:],dtype='float32')
len(word_embedding)
glove_file.close()
word_matrix = np.zeros((vocab_size,100))

for word,index in tokenizer.word_index.items():

    embed_vector = word_embedding.get(word)

    if  embed_vector is not None:

        word_matrix[index] = embed_vector
import matplotlib.pyplot as plt

import seaborn as sns



def performance_plot(data):

    fig,ax = plt.subplots(figsize=(16,5),ncols=2,nrows=1)

    val_loss = data['val_loss']

    val_acc = data['val_acc']

    loss = data['loss']

    acc = data['acc']

    epochs = range(1,len(acc)+1)



    sns.lineplot(epochs,val_loss,ax=ax[0],label='Validation Loss')

    sns.lineplot(epochs,loss,ax=ax[0],label='Training Loss')



    sns.lineplot(epochs,val_acc,ax=ax[1],label='Validation acc')

    sns.lineplot(epochs,acc,ax=ax[1],label='Training acc')
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[word_matrix], input_length=maxlen , trainable=False)

model.add(embedding_layer)

model.add(layers.LSTM(128))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
performance_plot(history.history)
model.evaluate(X_test,y_test)
def sentiment_prediction(instance):

    instance = tokenizer.texts_to_sequences(instance)

    instance = pad_sequences(instance, padding='post', maxlen=maxlen)

    prob = model.predict(instance)

    if prob < 0.5: return 'Negative sentiment'

    else: return 'Positive sentiment'
review = "You probably already know the answer to the above question, but writer-director Sujeeth takes you through a maze of twists and turns before he gets there – none of them engaging. Pegged as India’s biggest action thriller, Saaho gets into the action mode pretty early on. The film begins with signature wide angles of massive structures and grim looking men who mean serious business. All through the first half, the film travels through cities trying to connect high stake robberies in Mumbai and the search for a missing black box that holds the key to a fortune. But by the time the ‘interval bang’ rolls around, you kind of already know where this is heading, thanks to on-the-nose dialogues. Then there’s Prabhas, with an entry so subtle it quickly takes a turn, leading to a loud, high-octane fight scene that sets the stage for many more such confrontations. While he does fit the bill perfectly for the larger-than-life role, his dialogue delivery is deliberately slow, almost like a drawl, and doesn’t always work. His one-liners and humour falls flat, none of the jokes somehow land. However, the way his character unfolds does keep the viewer guessing. Shraddha Kapoor looks glamorous but delivers a lifeless and expressionless performance for a character that’s poorly sketched to begin with. Introduced as a tough-talking cop, it doesn’t take long before she’s turned into a damsel-in-distress who often needs saving. She always seems to be the last one to know what’s up. Even the chemistry between the lead pair is a touch and go with even the hyped up ‘romantic fight scene’ not working completely. Among the many villains, Chunkey Pandey as Devraj stands out with a very convincing portrayal of his character. He oozes menace and seethes with anger, if only he had gotten better lines to match that acting prowess. The rest, despite being stupendous actors, somehow come off as mere caricatures who fail to make an impact. The way Mandira Bedi’s character develops is laughable. The songs are so oddly placed in the narrative; they only manage to add on to an already long runtime and add to a choppy narrative. Saaho surely delivers well as an action extravaganza with a climax that attempts to compensate for its many flaws. The film’s second half picks up pace, but is marred by a weak narrative that needs constant suspension of disbelief. The hyped up special effects and CGI too lack finesse for a film mounted on such a grand scale. Sujeeth attempts at a potboiler that fires in any and all directions to entertain the audience. You can also see how the story, despite being predictable, might have sounded good on paper, but the many twists and turns just leave you exhausted than excited. Saaho is an attempt at reinventing a story as old as time, if only the numerous ‘bangs’ managed to land."

review
review = preprocessing(review)

review = pd.Series(review)

sentiment_prediction(review)