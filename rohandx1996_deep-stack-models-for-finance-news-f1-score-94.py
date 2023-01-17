import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn
from gensim.summarization.summarizer import summarize
df1=pd.read_csv('../input/commodities/commodities_feb_1k.csv',encoding="latin-1")
df2=pd.read_csv('../input/commodities/commodities_feb_3.6k.csv',encoding="latin-1")
df3=pd.read_csv('../input/commodities/commodities_feb_4.9k.csv',encoding="latin-1")
df4=pd.read_csv('../input/commodities/commodities_jan_10k.csv',encoding="latin-1")
df5=pd.read_csv('../input/commodities/commodities_jan_5k.csv',encoding="latin-1")
df6=pd.read_csv('../input/commodities/commodities_jan_2k.csv',encoding="latin-1")

df4=pd.concat([df1,df2,df3,df5,df6,df4])
final_df= df4.sample(frac=1).reset_index(drop=True)
from gensim.summarization.textcleaner import split_sentences
#appos list 
appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
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
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
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
"didn't": "did not"
}
final_df=final_df.dropna()
final_df.isnull().sum()
final_df['Text'] = final_df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
final_df['Text'].head()
# removing punctuation
final_df['Text']=final_df['Text'].str.replace('[^\w\s]','')
#removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
final_df['Text']=final_df['Text'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))
from textblob import Word
final_df['Text'] = final_df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
sns.countplot(final_df.Sentiment)
plt.show()
sns.distplot(final_df.Sentiment)
plt.show()
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(final_df["Text"])
#tempdf.loc[tempdf.num_hrefs <= 4.0,'num_hrefs']=1
final_df.loc[(final_df.Sentiment <=10) & (final_df.Sentiment >=6 ),'Bins']=1
final_df.loc[(final_df.Sentiment <=5) ,'Bins']=0
final_df.shape
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import pandas as pd
from nltk.corpus import stopwords
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, 'lxml').get_text() 
    
    # 2. Remove non-letters with regex
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                           
    
    # 4. Create set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
final_df['Text']=final_df['Text'].apply(review_to_words)
final_df.shape
from sklearn.model_selection import train_test_split
postive=final_df.loc[final_df['Bins'] == 1.0]
negative= final_df.loc[final_df["Bins"]== 0.0]
from nltk import sent_tokenize
import random
def tokenize(text):
    '''text: list of text documents'''
    tokenized =  sent_tokenize(text)
    return tokenized

def shuffle_tokenized(text):
    random.shuffle(text)
    newl=list(text)
    shuffled.append(newl)
    return text

aug_pos= []
aug_neg=[]
reps=[]
for ng_rev in postive['Text']:
    tok = tokenize(ng_rev)
    shuffled= [tok]
    #print(ng_rev)
    for i in range(17):

        shuffle_tokenized(shuffled[-1])
    for k in shuffled:
        '''create new review by joining the shuffled sentences'''
        s = ' '
        new_rev = s.join(k)
        if new_rev not in aug_pos:
            aug_pos.append(new_rev)
        else:
            reps.append(new_rev)
for ng_rev in negative['Text']:
    tok = tokenize(ng_rev)
    shuffled= [tok]
    #print(ng_rev)
    for i in range(17):

        shuffle_tokenized(shuffled[-1])
    for k in shuffled:
        '''create new review by joining the shuffled sentences'''
        s = ' '
        new_rev = s.join(k)
        if new_rev not in aug_neg:
            aug_neg.append(new_rev)
        else:
            reps.append(new_rev)
    
new_positive=pd.DataFrame(data=aug_pos,columns=['Text'])
new_positive["Bins"]=1.0
new_negative=pd.DataFrame(data=aug_neg, columns=["Text"])
new_negative['Bins']=0.0
new_positive.shape ,new_negative.shape
#from nltk.corpus import wordnet
#from nltk.tokenize import word_tokenize
#from random import randint
#import nltk.data

# Load a text file if required
#def syno_aug(text):
 #   output=""
  #  new_review = []

# Load the pretrained neural net
   # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize the text
    #tokenized = tokenizer.tokenize(text)

# Get the list of words from the entire text
    #words = word_tokenize(text)

# Identify the parts of speech
    #tagged = nltk.pos_tag(words)

    #for i in range(0,len(words)):
        #replacements = []

    # Only replace nouns with nouns, vowels with vowels etc.
     #   for syn in wordnet.synsets(words[i]):

        # Do not attempt to replace proper nouns or determiners
      #      if tagged[i][1] =='NNP' or tagged[i][1] =='DT':
        #        break
       # 
        # The tokenizer returns strings like NNP, VBP etc
        # but the wordnet synonyms has tags like .n.
        # So we extract the first character from NNP ie n
        # then we check if the dictionary word has a .n. or not 
         #   word_type = tagged[i][1][0].lower()
          #  if syn.name().find("."+word_type+"."):
            # extract the word only
           #     r = syn.name()[0:syn.name().find(".")]
            #    replacements.append(r)
        
       # if len(replacements) > 0:
            # Choose a random replacement
        #    replacement = replacements[randint(0,len(replacements)-1)]
         #   output = output + " " + replacement
            
        #else:
            # If no replacement could be found, then just use the
            # original word
         #   output = output + " " + words[i]
    #new_review.append(output)
    #return(new_review)
# Load a text file if required
#def anto_aug(text):
 #   output=""
  #  new_review = []

# Load the pretrained neural net
   # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize the text
    #tokenized = tokenizer.tokenize(text)

# Get the list of words from the entire text
    #words = word_tokenize(text)

# Identify the parts of speech
    #tagged = nltk.pos_tag(words)

    #for i in range(0,len(words)):
     #   replacements = []

    # Only replace nouns with nouns, vowels with vowels etc.
      #  for syn in wordnet.synsets(words[i]):
       #     for lemma in syn.lemmas():
        #        for antonym in lemma.antonyms():

        # Do not attempt to replace proper nouns or determiners
         #           if tagged[i][1] =='NNP' or tagged[i][1] =='DT':
          #              break
        
        # The tokenizer returns strings like NNP, VBP etc
        # but the wordnet synonyms has tags like .n.
        # So we extract the first character from NNP ie n
        # then we check if the dictionary word has a .n. or not 
           #         word_type = tagged[i][1][0].lower()
            #        if antonym.name().find("."+word_type+"."):
            # extract the word only
             #           r = antonym.name()[0:syn.name().find(".")]
              #          replacements.append(r)
        
      #  if len(replacements) > 0:
            # Choose a random replacement
       #     replacement = replacements[randint(0,len(replacements)-1)]
        #    output = output + " " + replacement
            
        #else:
            # If no replacement could be found, then just use the
        # original word
         #   output = output + " " + words[i]
 #   new_review.append(output)
  #  return(new_review)
#import numpy as np
#from newspaper import Article, ArticleException
#import requests
#links=[]
#def text_extractor2(link):
 #   article = Article(link)
  #  try:
   #     article.download()
    #    article.parse()
     #   article = article.text[:]
      #  text=summarize(article,ratio=0.7)
    #except (ArticleException,ValueError):
     #   print(link)
      #  links.append(link)
       # return np.nan
   # return text
    



#final_df["summary"]=final_df[15001:].URL.apply(text_extractor2)
    
final_df.isnull().sum()
summary_df1=pd.read_csv('../input/summary-finance/summary_3314.csv',encoding='latin-1')
summary_df2=pd.read_csv('../input/summary-finance/summary_15001.csv',encoding='latin-1')
summary_df3=pd.read_csv('../input/summary-finance/summary_17191.csv',encoding='latin-1')
summary_df4=pd.read_csv('../input/summary-finance/summary_6000.csv',encoding='latin-1')
summary=pd.concat([summary_df1,summary_df2,summary_df3,summary_df4])
summary= summary.sample(frac=1).reset_index(drop=True)
summary
#summary['summary'] = summary['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#summary['summary']=summary['summary'].str.replace('[^\w\s]','')
#summary['Text']=summary['Text'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))
#freq = pd.Series(' '.join(summary['summary']).split()).value_counts()[:20] # removing top 20 common words
#rare = pd.Series(' '.join(summary['summary']).split()).value_counts()[-20:] #removing top 20 rare words
#freq=list(freq.index)
#rare=list(rare.index)
#summary['summary']=summary['summary'].apply(lambda x :" ".join(x for x in x.split() if x not in freq))
#summary['summary']=summary['summary'].apply(lambda x :" ".join(x for x in x.split() if x not in rare))
#summary['summary'] = summary['summary'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#summary=summary.loc[:,['Bins','summary']]
#summary=summary.rename(columns={'Bins':'Target','summary':'Text'})
new_negative=new_negative.rename(columns={'Bins':'Target','Text':'Text'})
new_positive=new_positive.rename(columns={'Bins':'Target','Text':'Text'})
sentences=pd.concat([new_positive,new_negative])
final_new = final_df.loc[:,['Text','Bins']]
final_new=final_new.rename(columns={'Bins':'Target','Text':'Text'})
final_update=pd.concat([final_new])
final_update= final_update.sample(frac=1).reset_index(drop=True)
test_data=final_update.sample(3000)
final_update=final_update.drop(test_data.index)
x = final_update.Text
y = final_update.Target.astype(int)
final_update.shape
from IPython.display import Image
Image("../input/cnnlstm/System-architecture-of-the-proposed-regional-CNN-LSTM-model.png")
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import MaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU,AveragePooling1D,MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam ,SGD
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation,GRU,Bidirectional
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Conv1D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D,Flatten
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.advanced_activations import LeakyReLU ,ELU 
list_classes = ["Target"]
list_sentences_train = final_update["Text"]
list_sentences_test = test_data["Text"]
y_test=test_data["Target"]
max_features = 5000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
import numpy as np
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.xlabel("Distribution of comment")
plt.ylabel("no of comments")
plt.title("no of comments vs no of words distribution ")
plt.show()
maxlen = 400
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te=pad_sequences(list_tokenized_test,maxlen=maxlen)
inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(98, activation='relu',return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = LeakyReLU(0.12)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, 
                              save_best_only=False)
model.compile(loss='binary_crossentropy', optimizer='adam', 
                              metrics=['accuracy'])

model.summary()
# Start to train model 
history1 = model.fit(X_t, y, 
                    batch_size=32, 
                    epochs=1, 
                    verbose=1, 
                    validation_data=[X_te,y_test],
                    callbacks=[checkpointer],
                    shuffle=True)
final_df.shape
final_new = final_df.loc[:,['Text','Bins']]
final_new=final_new.rename(columns={'Bins':'Target','Text':'Text'})
final_update=pd.concat([final_new,new_positive,new_negative])
final_update= final_update.sample(frac=1).reset_index(drop=True)
test_data=final_update.sample(3000)
final_update=final_update.drop(test_data.index)
x = final_update.Text
y = final_update.Target.astype(int)
final_update.shape
list_classes = ["Target"]
list_sentences_train = final_update["Text"]
max_features = 6000
maxlen = 400
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
list_sentences_test=test_data["Text"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test=test_data["Target"]
inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = Conv1D(64, 3, activation='relu')(x)
x = Conv1D(128,3,activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = LSTM(98, activation='relu',return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = ELU(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)
model1 = Model(inputs=inp, outputs=x)
checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, 
                              save_best_only=False)
model1.compile(loss='binary_crossentropy', optimizer='adam', 
                              metrics=['accuracy'])
model1.summary()
# Start to train model 
history1 = model1.fit(X_t, y, 
                    batch_size=32, 
                    epochs=7, 
                    verbose=1, 
                    validation_data=[X_te,y_test],
                    callbacks=[checkpointer],
                    shuffle=True)
import matplotlib.pyplot as plt
%matplotlib inline
# summarize history for accuracy
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("model_accuracy_100_salinas.svg")
plt.show()

# summarize history for loss 
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("model_loss_100_salinas.svg")
plt.show()
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
prediction = model1.predict(X_te)
y_pred = (prediction > 0.5)
print(accuracy_score(y_pred,y_test))
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
def define_model(length,vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 256)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=5, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 256)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 256)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1= Dropout(0.3)(merged)
    dense1 = Dense(50, activation='relu')(merged)  
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model
model = define_model(400, max_features)
checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, 
                              save_best_only=True)
history1=model.fit([X_t,X_t,X_t], y,batch_size=32, 
                    epochs=10, 
                    verbose=1, 
                    validation_data=([X_te,X_te,X_te],[y_test]),
                    callbacks=[checkpointer],
                    shuffle=True)

# So we reached to the the highest accuracy with less log loss score 
# summarize history for accuracy
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("model_accuracy_100_salinas.svg")
plt.show()

# summarize history for loss 
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("model_loss_100_salinas.svg")
plt.show()
from sklearn.metrics import accuracy_score ,classification_report
preds = model.predict([X_te,X_te,X_te])
preds = (preds[:,0] > 0.5).astype(np.int)
print(accuracy_score(preds,y_test))
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(preds, y_test)))
print('Confusion matrix:')
confusion_matrix(preds, y_test)
def model_predict(news):
    list_sentences_train = news
    max_features = 5000
    maxlen = 400
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    output=model.predict([X_t,X_t,X_t])
    return output

# postive news it is  # investment in tesla in electromotive cars
news=["""nvestors interested in electric cars have a variety of options. Automakers such as Tesla Motors exclusively manufacture electric vehicles and may be directly invested in by purchasing stock. Companies within the automotive sector that manufacture vehicle parts or supply raw materials used in producing electric cars are another means of gaining portfolio exposure to electric cars. Another slightly less risky option is to invest in exchange-traded funds (ETFs) with holdings in securities related to electric vehicle production or electric vehicle parts.

Some major automakers, such as Toyota, are investing heavily in electric vehicles and allow investors to choose both traditional and electric vehicles for their investments. Chevrolet and Nissan have also made notable electric car models available in the U.S. market. Investors should carefully consider available investment opportunities and evaluate the potential risk-return tradeoff offered by electric vehicles and the automotive industry.

Many manufacturers develop auto parts for traditional and electric vehicles. Polypore International (PPO) produces lead-acid batteries used in both conventional and electric vehicles. This stock offers investors the opportunity to invest generally in the production of vehicle batteries. As electric vehicle and conventional vehicle usage grows, more batteries will be needed, and this company will likely benefit from increased global car demand.

Another battery company, Plug Power (PLUG), manufactures hydrogen fuel cell batteries used in electric vehicles and many other types of electronic equipment. These batteries may replace lead-acid batteries in fork lifts. Plug Power batteries are also used outside of the automotive industry, giving the company a large market.

Sociedad Quimica y Minera (SQM) is a major supplier of lithium, an element used in many batteries powering electric vehicles and other clean technologies. Investment in companies such as Polypore International, Plug Power, and SQM offers portfolio exposure to electric vehicles while also maintaining diverse holdings outside the automotive industry.

Electric Vehicles Exchange-Traded Funds
Exchange-traded funds that track electric vehicles are another possible opportunity for investors. These funds allow investors to purchase shares in funds that track electric vehicle industry development. Investments are spread across multiple companies, reducing investment risk and offering returns similar to the average returns of the entire sector. ETFs track gains and losses of stock indexes and are traded directly on the stock market in a means similar to stock trading. Just as in traditional stock trading, stop-loss limits may be placed, and dividends are paid to brokerage accounts.

Significant ETFs that include electric vehicle stock and supplier stock include QCLN and LIT. The First Trust NASDAQ Clean Edge Green Energy Index Fund (QCLN) has Tesla among its holdings and includes other companies with green technology offerings. Global X Lithium (LIT) tracks lithium suppliers and battery companies. This fund's most significant holdings include FMC Corporation, Avalon Rare Metals Incorporated, and Rockwood.
"""]
output=model_predict(news)
print("probability for this news",output)

