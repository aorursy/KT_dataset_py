import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
import string
import seaborn as sns
import matplotlib.pyplot as plt
import re
# VISUALIZATION
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
pd.read_csv('/kaggle/input/yelp-reviews/test.csv')
train = pd.read_csv('/kaggle/input/yelp-reviews/train.csv')
#test = pd.read_csv('/kaggle/input/test.csv')
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have","sbse":"biggest","chor":"thief","paisa":"money","barbaad":"waste",
                            "grmi":"summer","chori":"steal","paise":"money","sbse bada":"biggest","bada":"big","❤":"love"}
#test["Polarity"]=test["Polarity"].astype("str")
#test["Polarity"]=test["Polarity"].str.replace("1","negative")
#test["Polarity"]=test["Polarity"].str.replace("2","positive")

train["Polarity"]=train["Polarity"].astype("str")
train["Polarity"]=train["Polarity"].str.replace("1","negative")
train["Polarity"]=train["Polarity"].str.replace("2","positive")
labels = train["Polarity"].values.tolist()
def process(sen):
    processed = []
    for i in range(0, len(sen)):
        processed_feature = re.sub(r'\([^)]*\)', '', str(sen[i]))
        processed_feature = re.sub('"','', processed_feature)
        processed_feature= ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in processed_feature.split(" ")])
        
    # Converting to Lowercase
        processed_feature = processed_feature.lower()
        
        processed_feature=re.sub(r'^https?:\/\/.*[\r\n]*', '', processed_feature, flags=re.MULTILINE)
        
    # Remove all the special characters
        
        #processed_feature = re.sub(r'\W', ' ', processed_feature)

    # remove all single characters
        processed_feature= re.sub("[^a-z0-9❤, ]", "", processed_feature)
        processed_feature= re.sub(r'\s+[a-z]\s+', ' ', processed_feature)
        

    # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)


        processed.append(processed_feature.strip())
    return processed
train[:100000]
process_train=process(train["Comments"][:200000])
#process_test=process(test["Comments"])
input_sentences = [text.split(" ") for text in process_train]
#Creating Vocabulary (word index)
# Initialize word2id and label2id dictionaries that will be used to encode words and labels
word2id = dict()
label2id = dict()

max_words = 0 # maximum number of words in a sentence

# Construction of word2id dict
for sentence in input_sentences:
    for word in sentence:
        # Add words to word2id dict if not exist
        if word not in word2id:
            word2id[word] = len(word2id)
    # If length of the sentence is greater than max_words, update max_words
    if len(sentence) > max_words:
        max_words = len(sentence)
    
# Construction of label2id and id2label dicts
label2id = {j: i for i, j in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}
id2label
label2id
# Encode input words and labels
X=[]
Y=[]

X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    
Y = [label2id[label] for label in labels[:200000]]
X
# Apply Padding to X
import keras
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, max_words)

# Convert Y to numpy array
Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

# Print shapes
print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))
max_words
embedding_dim = 100 # The dimension of word embeddings

# Define input tensor
sequence_input = keras.Input(shape=(max_words,), dtype='int32')

# Word embedding layer
embedded_inputs =keras.layers.Embedding(len(word2id) + 1,
                                        embedding_dim,
                                        input_length=max_words)(sequence_input)

# Apply dropout to prevent overfitting
embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

# Apply Bidirectional LSTM over embedded inputs
lstm_outs = keras.layers.wrappers.Bidirectional(
    keras.layers.LSTM(embedding_dim, return_sequences=True)
)(embedded_inputs)

# Apply dropout to LSTM outputs to prevent overfitting
lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

# Attention Mechanism - Generate attention vectors
input_dim = int(lstm_outs.shape[2])
permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

# Last layer: fully connected with softmax activation
fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

# Finally building model
model = keras.Model(inputs=[sequence_input], outputs=output)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

# Print model summary
model.summary()
# Train model 2 iterations
model.fit(X, Y, epochs=2, batch_size=512, validation_split=0.20, shuffle=True)
# Re-create the model to get attention vectors as well as label prediction
model_with_attentions = keras.Model(inputs=model.input, 
                                    outputs=[model.output, 
                                             model.get_layer('attention_vec').output])
!pip install googletrans
from googletrans import Translator
translator = Translator(service_urls=['translate.google.co.in'])


def translate(word):
	return translator.translate(word,src='hi' , dest='en')
import random
import math
#sample_text='Sbse bada chor Quikr paise ceo "Pranay" usko email kro to b koi rply ni ata kyu paisa to ussi ki jeb me jata h.. I hope Quikr abi bankruptcy ni h aur tmko notice mil gya hoga meri side se'
#len(sample_text.split())
#translate(sample_text[:20]).text

from nltk.tokenize import sent_tokenize,word_tokenize
  
#text = "Hello everyone. Welcome to GeeksforGeeks. You are studying NLP article"
#sent_tokenize(text) 
#translate(sent_tokenize(sample_text)[0]).text 
df=pd.read_excel('/kaggle/input/companies/hometriangle.xlsx')
#quikr=pd.read_excel('/kaggle/input/companies/quikr_official.xlsx')urbancompany.xlsx
df["comments"]=df["comments"].replace("[]","")
#quikr["comments"]=quikr["comments"].replace("[]","")
com=list(df["comments"])
for i in range(len(com)):
    if com[i]!="":
        com[i]=com[i][1:len(com[i])-1]
        com[i]=com[i].split("',")
df["comments"].tolist()
#processed_com=[]
for i in range(len(com)):
    if com[i]!="":
        com[i]=process(com[i])
    
new=[]
for i in range(len(com)):
    if com[i]!="" and com[i]!=[]:
        new1=[]
        for j in range(len(com[i])):
            if com[i][j]!="":
                com[i][j]=translate(com[i][j]).text
                #s=""
                #for word in com[i][j].split():
                #    s+=translate(word).text+" "
                new1.append(com[i][j].strip())
            else:
                new1.append("")
        new.append(new1)
    else:
        new.append("")
for i in range(len(new)):
    if new[i]!="":
        if "" in new[i]:
            new[i].remove("") 
# Encode samples
preds=[]
for i in range(len(new)):
    
    if new[i]=='' or new[i]==[]:
        preds.append("No comments")
    else:
        pred_score=[]
        #emo=[]
        for j in range(len(new[i])):
            
            tokenized_sample = new[i][j].split(" ")
            encoded_samples = [[word2id[word] for word in tokenized_sample if word in word2id]]
            encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)
               
            label_probs, attentions = model_with_attentions.predict(encoded_samples)
            label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0])}
            pred_score.append(label_probs)
            
        preds.append(pred_score) 
                 
   
#translate(new[4][1]).text
new
preds
overall=[]
pos_total=[]
neg_total=[]
total_pol=[]
for i in range(len(preds)):
    if preds[i]!="No comments":
        for emo in preds[i]:
            pos=0.0
            neg=0.0
            for k, v in emo.items():
                print('{}: {}'.format(k, v))
                if str(k)=='positive':
                    pos+=v
                    
                else:
                    neg+=v
        if pos>0.6:
            overall.append("positive")
        elif neg>0.6:
            overall.append("negative")
        else:
            overall.append("neutral")
            
        pos_total.append(pos)
        neg_total.append(neg)
        total_pol.append(pos-neg)
    else:
        pos_total.append("No comments")
        neg_total.append("No comments")
        overall.append("No comments")
        total_pol.append("No comments")
pos_total[110]
neg_total[110]
overall
preds
total_pol
df["total_polarity"]=total_pol
df["overall"]=overall
df.to_excel("hometriangle_sentiment.xlsx")

