from IPython.display import Image

import os

!ls ../input/
Image("/kaggle/input/example/example.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import spacy 

import matplotlib.pyplot as plt

from keras.layers import Dense

from keras.utils import to_categorical

from keras.models import Model, Input,Sequential

from keras import layers 

import nltk #Using for getting phonemes

import string #To handle special characters

import gensim #Library to do the doc2vec https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

from sklearn.feature_extraction.text import CountVectorizer #Used to create bag of phonemes

from sklearn.decomposition import PCA

import random

import warnings

warnings.filterwarnings("ignore")
#nltk.download('cmudict')

arpabet = nltk.corpus.cmudict.dict() #Instance of the object to be used to obtain the phonemes
#!python -m spacy download en_core_web_sm  # Download the spacy model to use

nlp = spacy.load('en_core_web_sm')
#Opening the file 

with open('/kaggle/input/kanyewestverses/kanye_verses.txt','r',encoding='ascii',errors='ignore') as txt:

  data=txt.read()
#Obtaining all the characters to compare them with the words that will get their phonemes and these (words) are not special characters

all_punct=[punct for punct in string.punctuation] 
def gettingFeatures(nlp,sentence,arpabet,num_words):

  """Function to obtain the characteristics of the last words using Spacy and the dictionary for phonemes.



  Parameters: 

              - nlp:Model downloaded from Spacy

              - sentence:Sentence that we want to get its characteristics

              - arpabet:Model downloaded from NLTK to obtain phonemes

              - num_words:Number of last words from which you want to obtain their characteristics

  Return: A tuple with the characteristics obtained.

  """

  doc=nlp(sentence)

  last_words,phonemes,tag,dep=([] for _ in range(4))

  for token in reversed(list(doc)): #We revert to go from the last words to the first

    if token.text not in all_punct: #We verify if the word does not belong to a special character

      last_words.append(token.text.lower()) #Convert to lowercase because it will be a key in the phoneme dictionary

      #Because arpabet is a dictionary (key: word, value: phonemes) it is possible that the word entered is not found in it,

      #that's why we verify

      try:

        phon=arpabet[token.text.lower()][0]

      except Exception as e:

        phon='NA'

      phonemes.append(phon)

    if len(last_words) == num_words: #number of last words

      break





  #The variable phonemes is a list of lists, we will flatten this

  flatten_phonemes=[phon for phons in phonemes for phon in phons]



  return last_words," ".join(flatten_phonemes)



  
gettingFeatures(nlp,"""(Ball so hard) That shit cray, ain’t it Jay? What she order, fish Filet?""",arpabet,4)
def creatingDf(data):

  """Function that will be used to create a dataFrame from the data read in the txt.



  Function that receives txt data and runs it by dividing it into verses and later into sentences (rhymes)

   it makes use of the gettingFeatures function to obtain the parameters and from this create a dataframe.



  Parameters:

              -data: Data previously read from txt.

  

  Return: A dataframe with the characteristics that were obtained by the gettingFeatures function

  """

  

  verses=list(filter(None,data.split('\n\n'))) #Split data in verses

  

  df=pd.DataFrame(columns=['Verses','Rhymes','VersesRhymes','Rhyme','Last_Words','Phonemes']) #Dataframe instance

  

  for index,verse in enumerate(verses): 

    

    rhymes=verse.split('\n') # Split rhymes in sentences

    j=1

    for i,rhyme in enumerate(rhymes):

      if (i+1) % 2 == 0:

        j-=1

      #The special characters of the rhymes will not be removed because these will be the answers later.

      

      features=gettingFeatures(nlp,rhyme,arpabet,1) #Calling the function to obtain the characteristics of the sentences

      df=df.append({'Verses':f'Verse {index+1}',

                 'Rhymes': f'Rhyme {j}',

                 'VersesRhymes': f'Verse {index+1} Rhyme {j}',

                 'Rhyme' : rhyme,

                 'Last_Words': features[0],

                 'Phonemes': features[1]

                }, ignore_index=True) 

        

      j+=1

  return df    
rpt=creatingDf(data)
rpt[rpt.Verses == 'Verse 1'].head()
def usingGensim(doc,tag=None,test=False):

  """Function to tokenize and add a tag to the statement using Gensim.



  Function for train and test data, makes use of the simple_preprocess function for tokenization and TaggedDocument to tagear them.

   If it is the training data, it will tokenize and tagerate the sentence for this the parameter must be passed tag, if it is the data

   test will then only be tokenized and the test parameter must be changed to True.

  Parameters:

              - doc: Sentence that you want to tokenize and tagear (training).

              - tag: Tag that will be assigned to the sentence if on the training data.

              - test: Flag to know if it is about the test or training data.

  Return: A list with the tokens of the statement if it is for the test and a TaggedDocument object with the tokens and the tag if it is for test data.

  """

  if test:

    return gensim.utils.simple_preprocess(doc) 

  else:

    return gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [tag])
usingGensim('(Ball so hard) That shit cray, ain’t it Jay? What she order, fish Filet?',tag=1)
taggedDocuments=[]

for index in range(len(rpt.Rhyme)):

  taggedDocuments.append(usingGensim(rpt.Rhyme[index],tag=index)) #The tag will be the index of each row to be able to locate it later
taggedDocuments[:5]
modelDoc2Vec = gensim.models.doc2vec.Doc2Vec(vector_size=50, epochs=40)

modelDoc2Vec.build_vocab(taggedDocuments)

modelDoc2Vec.train(taggedDocuments, total_examples=modelDoc2Vec.corpus_count, epochs=modelDoc2Vec.epochs)
cv=CountVectorizer()

X=cv.fit_transform(rpt['Phonemes']).toarray()
#Representation of phonemes in a sparse matrix

X[:2]
pca=PCA(n_components=2)

Xpca=pca.fit_transform(X) # X reducido a 2 dimensiones
plt.scatter(Xpca[:,0],Xpca[:,1],s=10) 
pca.explained_variance_ratio_.sum() 
from sklearn.mixture import BayesianGaussianMixture



bgm = BayesianGaussianMixture(n_components=5, n_init=10, random_state=42,covariance_type='tied') #100

bgm.fit(Xpca)
np.round(bgm.weights_, 2)
y_pred=bgm.predict(Xpca)
dfPrueba=pd.DataFrame(Xpca)

dfPrueba['y_pred']=y_pred
import matplotlib.cm as cm

grups=len(set(y_pred))

x = np.arange(grups)

ys = [i+x+(i*x)**2 for i in range(grups)]

colors = cm.rainbow(np.linspace(0, 1, len(ys)));

for i in range(grups):

  plt.scatter(dfPrueba[0][dfPrueba.y_pred == i],dfPrueba[1][dfPrueba.y_pred == i],s=10,c=[colors[i]])
#Incrementaremos la varianza explicada aumentando la cantidad de componentes

pca=PCA(n_components=15)

Xpca=pca.fit_transform(X) 

pca.explained_variance_ratio_.sum() 
bgm = BayesianGaussianMixture(n_components=100, n_init=10, random_state=42,covariance_type='tied') 

bgm.fit(Xpca)
y_pred=bgm.predict(Xpca)

rpt['y_pred_bgm']=y_pred

rpt.head(10)
df_X=pd.DataFrame(X)

df_X.head()
#Getting the target

y=rpt.y_pred_bgm.copy()

y.shape
df_X.shape,y.shape
y_not_categorical=y

y=to_categorical(y) 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(df_X,y,test_size=0.1,random_state=2019,shuffle=True)
def build_model(len_row):



  model=Sequential()

  

  model.add(layers.Dropout(0.1,input_shape=(len_row,)))



  model.add(Dense(120,init='uniform',activation='tanh'))

  

  model.add(Dense(100,init='uniform',activation='softmax'))

  

  

  

  model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

  

  return model
model = build_model(X_train.shape[1])

model.summary()
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test)) 
def plot_history(title, train, test, y_title, x_title):

  plt.title(title)

  plt.plot(train)

  plt.plot(test)

  plt.ylabel(y_title)

  plt.xlabel(x_title)

  plt.legend(['Train', 'Test'])

  plt.show()
plot_history("Accuracy Plot", history.history['acc'], history.history['val_acc'], "Accuracy", "Epocas")
plot_history("Loss Plot", history.history['loss'], history.history['val_loss'], "Loss", "Epocas")
def makeResponse(input_sentence):

  """Function that responds to the message entered with a dataset rhyme.

  """

  df_test=creatingDf(input_sentence)#Getting features

  bag_of_word_input=cv.transform(df_test.Phonemes).toarray() #Representation of your phonemes through an arrangement

  input_prediction=model.predict_classes(bag_of_word_input)  #Class prediction of the input phrase

  #We obtain its indexes to filter our search in the doc2vec

  index=rpt[rpt.y_pred_bgm == input_prediction[0]].index



  #tokenize the sentence

  tokens_gensim=usingGensim(input_sentence,test=True)

  #getting its vector representation

  input_vector = modelDoc2Vec.infer_vector(tokens_gensim)

  #The similarities of the phrase entered with all the rhymes, gives us a list of tuples, the tuple is composed of the index and its similarity with the phrase

  sims=modelDoc2Vec.docvecs.most_similar([input_vector], topn=len(taggedDocuments))

  #We go through each tuple until we find the first 10 that most resemble each other and have the same class

  index_most_simi=[]

  for tupla in sims:

    if tupla[0] in list(index): 

      index_most_simi.append(tupla)

      if len(index_most_simi) == 15:

        break

  

  rand_index=random.randrange(0, 15)



  return rpt.Rhyme[rpt.index == index_most_simi[rand_index][0]].values[0]

print(makeResponse("How are you?"))
dataset = """

*greet

Hi

Hello

Hey there



*affirm

Yes

Love it

Yeah

Yep



*negation

No

No, I don't 

Not at the moment

Not really



*stop

stop

quit

"""
def filter_intents_and_their_examples(dataset, intent_character = "*"):

  filter_list = list(filter(None, dataset.split("\n")))

  intents_examples = {}

  intent = ""

  for element in filter_list:

    if element[0] == intent_character:

      intent = element[1:]

    else:

      if intent not in intents_examples:

        intents_examples[intent] = [element]

      else:

        intents_examples[intent].append(element)

  return intents_examples



def transform_examples_in_vectors(intents_examples, nlp):

  intents_vector_examples = {}

  for key, values in intents_examples.items():

    vector_examples = []

    for example in values:

      vector_examples.append(nlp(example).vector)

    

    intents_vector_examples[key] = vector_examples

  return intents_vector_examples



def create_x_and_y(intents_vector_examples):

  x = []

  y = []

  for label, vector_examples in intents_vector_examples.items():

    x.extend(vector_examples)

    y.extend([label] * len(vector_examples))

  

  return x, y
#Getting a dic from intents with their sentences

intents_examples = filter_intents_and_their_examples(dataset)



#Transforming the sentences into vecs in the dic

intents_vector_examples = transform_examples_in_vectors(intents_examples, nlp)



#Getting intents' names

intents = list(intents_vector_examples.keys())



x, y = create_x_and_y(intents_vector_examples)
intents_examples
from sklearn.tree import DecisionTreeClassifier



decision_tree_classifier = DecisionTreeClassifier()

decision_tree_classifier.fit(x, y)



#Evaluacion del accuracy

accuracy = decision_tree_classifier.score(x, y)

accuracy
def predict_intent(sentence, nlp, model):

  vector = nlp(sentence).vector

  

  intent = model.predict([vector])

  return intent[0]
dataset = """

*greet

Hey, what's up buddy ? Do you wanna rhyme ?

How is it goin' bud ? Do you wanna play with some rhymes? 

Zup my nigga ? Would you like rap with me ?



*affirm

Alright, so just gimme a sentence and I'll continue with a rhyme .

Awesome, show me what you got and I'll rhyme that.

K, let's see what you got.



*negation

Hahaha it seems there's a coward around here. So come back as soon as possible.

Alright, go to get warm and come back soon.

K, no worries. Hope see ya soon.



*goodbye

That was exciting, I had fun dude. Take care man.

You did really good my homie, Hope see ya soon.

That was funny, I'll wait for your rhymes man.

"""
def filter_utterances_and_their_examples(dataset, utterance_character = "*"):

  filter_list = list(filter(None, dataset.split("\n")))



  utterances_examples = {}

  utterance = ""

  for element in filter_list:

    if element[0] == utterance_character:

      utterance = element[1:]

    else:

      if utterance not in utterances_examples:

        utterances_examples[utterance] = [element]

      else:

        utterances_examples[utterance].append(element)

  return utterances_examples
#Getting a dictionary of utterances with their sentences

utterances_examples = filter_utterances_and_their_examples(dataset)

utterances_examples
rules = {

    "greet": "greet",

    "negation": "negation",

    "affirm": "affirm",

    "stop": "goodbye"

}
from random import choice



def generate_answer(utterance, utterances_examples):

  answers = utterances_examples[utterance]

  answer = choice(answers)

  return answer
def return_answer(sentence, nlp, model, rules, utterances_examples):

  intent = predict_intent(sentence, nlp, decision_tree_classifier)

  

  utterance = rules[intent]

  

  answer = generate_answer(utterance, utterances_examples)

  return answer, intent
intent=None

while True:

  sentence = input()

  answer, intent = return_answer(sentence, nlp, decision_tree_classifier, rules, utterances_examples)

  print("bot says: {}\n".format(answer))

  if intent == 'affirm':

    while True: 

       sentence=input()

       rhyme=makeResponse(sentence)

       if sentence == 'stop':

         intent='stop'

         answer, intent = return_answer(sentence, nlp, decision_tree_classifier, rules, utterances_examples)

         print("bot says: {}\n".format(answer))

         break

       print("bot says: {}\n".format(rhyme))



  if intent == 'negation' or intent=='stop':

    break