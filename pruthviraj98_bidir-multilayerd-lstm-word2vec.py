import spacy

import nltk

from nltk.corpus import stopwords

from spacy.lang.en.stop_words import STOP_WORDS

from spacy import lemmatizer

from nltk.tokenize import RegexpTokenizer

import pandas as pd

import gensim

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

import torch 

import torch.nn as nn

import torch.optim as optim

import numpy as np
nlp=spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
tokenizer=RegexpTokenizer("(\w+\'?\w+?)")
# nltk.download('punkt')

# nltk.download('stopwords')
sw1=stopwords.words("english")

sw2=STOP_WORDS

stop_words=set(sw1).union(sw2)
def tokenize(rev):

    return(tokenizer.tokenize(str(rev).lower()))
def remove_stop_words(rev_tokens):

    return([tok for tok in rev_tokens if tok not in stop_words])
def lemmatize(rev_tokens):

    result=[]

    for tok in rev_tokens:

        temp=nlp(tok)

        for tok in temp:

            result.append(tok.lemma_)

    return result
def preprocess_pipeline(review):

    review=tokenize(review)

    review=remove_stop_words(review)

    review=lemmatize(review)

    return review
df=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

df.head(2)
reviews=list(df['review'])

sentiments=list(df['sentiment'])
reviews=list(map(lambda x: preprocess_pipeline(x), reviews))
dimension=100 # will also be used in the RNN unit part 

model=Word2Vec(reviews, size=dimension, window=3, min_count=3, workers=4)
model.sg#its CBOW not skipgram
# tip: as soon as we train our model, we have to delete the model unless there is further training or updation required in the model because it consumes lots of memory

# but, to use it even after deleting the model, we can use key'd vector model that holds all the info about the embedding model

word_vec=model.wv

del(model)
#save the vocabulary of the model

len(word_vec.vocab)
word_vec.similar_by_word(word="bad", topn=10)
word_vec.similarity("good", "be")
#now apply contextual relation with the word

#ex: king - man + woman = queen

#example: 

word_vec.most_similar(negative=["bad"], positive=["decent"], topn=5)
# from gensim.models import KeyedVectors #to store the loaded pretrained model
# model=KeyedVectors.load_word2vec_format('pretrained embedding path in .bin format', binary=True)
#to ensure results are reproducable set these

SEED=2031

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic=True
def convWordInd(embedding_model, review):

    indice_rev=[]

    for word in review:

        try:

            indice_rev.append(embedding_model.vocab[word].index)

        except:

            pass

    return torch.tensor(indice_rev)
review_indexes=list(map(lambda x: convWordInd(word_vec, x), reviews))
pad_value=len(word_vec.index2word) #used later during RNN param initialization

pad_value #this is the length of the longest review in the batch
embed_weights=torch.Tensor(word_vec.vectors)
class RNN(nn.Module):

    def __init__(self, inp_dim, embed_dim, hidden_dim, out_dim, n_layers, bidirectional, dropout, embed_weights):

        super().__init__()

        self.embedding_layer=nn.Embedding.from_pretrained(embed_weights)

        self.rnn=nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)

        self.dense=nn.Linear(hidden_dim*2, out_dim)#ip layer is hid_layer*2 because in case of bidirectional RNN, there are 2 hidden layer o/p

        self.dropout=nn.Dropout(dropout)#never use dropout in the ip or op layers but in the intermediate layers

        

    def forward(self, x, text_lens):

        embedded=self.embedding_layer(x)

        packed_embed=nn.utils.rnn.pack_padded_sequence(embedded, text_lens)

        packed_out, (hidden, cell)=self.rnn(packed_embed) #output size=[text_len, batch_size, hiddendim*num of dim]

        #output, output_lens=nn.utils.rnn.pad_packed_sequence(packed_out) commented as output is not used here 

        #bdirlstm consists [f0, b0, f1, b1, ..... fn, bn]

        #so, concatinating the last two hidden state (forward and backward) from the last layer, it is passed to linear layer 

        hidden =self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.dense(hidden.squeeze(0))        
#updating the hyperparameters

inp_dim=pad_value

embed_dim=dimension

hidden_dim=256

output_dim=1

n_layers=2

bidirectional=True

dropout=0.5
model=RNN(inp_dim, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, embed_weights)

model
optimizer=optim.Adam(model.parameters())

loss_function=nn.BCEWithLogitsLoss()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
#binary encoding the y vals

sentiments=[0 if label == 'negative' else 1 for label in sentiments]

sentiments[:5]
X_train, X_test, Y_train, Y_test=train_test_split(review_indexes, sentiments, test_size=0.25)

X_train, X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.2)
#now batches out of entire dataset is prepared

batch_size=128

def iterate_func(x, y):

    size=len(x)

    permute=np.random.permutation(size) #for creating the random blocks from the list of reviews

    iterator=[]

    for i in range(0, size, batch_size): #from 0 to size stepping from batch size

        indices=permute[i: i+batch_size]

        batch={}

        batch["text"]=[x[i] for i in indices]

        batch["label"]=[y[i] for i in indices]

        #sort the texts based on their lengths

        batch["text"], batch["label"]=zip(*sorted(zip(batch["text"], batch["label"]), key=lambda x: len(x[0]), reverse=True))

        batch["length"]=[len(rev) for rev in batch["text"]]

        batch["length"]=torch.IntTensor(batch["length"])

        #Now, reviews are padded in each batch and are passed into the model. 

        #For padding, pytorch offers the method within its nn module

        batch["text"]=torch.nn.utils.rnn.pad_sequence(batch["text"], batch_first=True).t()#transpose is performed so that it is accepted into rnn

        batch["label"]=torch.Tensor(batch["label"])

        

        #now pushing all to the gpu

        batch["text"]=batch["text"].to(device)

        batch["label"]=batch["label"].to(device)

        batch["length"]=batch["length"].to(device)

        

        iterator.append(batch)

        

    return iterator
train_iter=iterate_func(X_train, Y_train)

val_iter=iterate_func(X_val, Y_val)

test_iter=iterate_func(X_test, Y_test)
model=model.to(device)

criterion=loss_function.to(device)
def binary_acc(preds, y):

    round_preds=torch.round(torch.sigmoid(preds))

    pos=(round_preds==y).float()

    accuracy=pos.sum()/len(pos)

    return accuracy
def train(model, iterator, optimizer, criterion):

    epoch_loss=0

    epoch_accuracy=0

    model.train()

    

    for batch in iterator:

        optimizer.zero_grad()

        predictions=model(batch["text"], batch["length"]).squeeze(1)

        loss=criterion(predictions, batch["label"])

        accuracy=binary_acc(predictions, batch["label"])

        loss.backward()

        optimizer.step() #updates the weight for the model

        epoch_loss+=loss.item()

        epoch_accuracy+=accuracy.item()

    

    #return the average epoch loss and iterator

    return(epoch_loss/len(iterator), epoch_accuracy/len(iterator))
def evaluator(model, iterator, criterion):

    epoch_loss=0

    epoch_accuracy=0

    model.eval()

    #to prevent any gradient calculation with nograd is used

    with torch.no_grad():

        for batch in iterator:

            predictions=model(batch["text"], batch["length"]).squeeze(1)

            loss=criterion(predictions, batch["label"])

            accuracy=binary_acc(predictions, batch["label"])

            epoch_loss+=loss.item()

            epoch_accuracy+=accuracy.item()



    #return the average epoch loss and iterator

    return(epoch_loss/len(iterator), epoch_accuracy/len(iterator))
epochs=7



for epoch in range(epochs):

    train_loss, train_accuracy=train(model, train_iter, optimizer, criterion)

    valid_loss, valid_accuracy=evaluator(model, val_iter, criterion)

    

    print("Epoch number: ", epoch)

    print("Train Loss = ", train_loss, " Train Accuracy = ", train_accuracy)

    print("Validation Loss = ", valid_loss, " Validation Accuracy = ", valid_accuracy)
test_loss, test_accuracy=evaluator(model, test_iter, criterion)

print("Epoch number: ", epoch)

print("Test Loss = ", test_loss, " Test Accuracy = ", test_accuracy)