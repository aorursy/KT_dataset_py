
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == "toxic_test.csv":
            test_df = pd.read_csv(os.path.join(dirname,filename))
        elif filename == "toxic_train.csv":
            train_df = pd.read_csv(os.path.join(dirname,filename))


print(train_df[train_df["toxic"] == 0]["comment_text"].values[0])
print("\n")
print(train_df[train_df["toxic"] == 0]["comment_text"].values[1])
print("\n")
print(train_df[train_df["toxic"] == 0]["comment_text"].values[2])
print("\n")
print(train_df[train_df["toxic"] == 0]["comment_text"].values[3])
print("\n")
print(train_df[train_df["toxic"] == 0]["comment_text"].values[4])
print(train_df[train_df["toxic"] == 1]["comment_text"].values[0])
print("\n")
print(train_df[train_df["toxic"] == 1]["comment_text"].values[1])
print("\n")
print(train_df[train_df["toxic"] == 1]["comment_text"].values[2])
print("\n")
print(train_df[train_df["toxic"] == 1]["comment_text"].values[3])
print("\n")
print(train_df[train_df["toxic"] == 1]["comment_text"].values[4])
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

tot = train_df.shape[0]
num_toxic = train_df[train_df.toxic == 1].shape[0]

slices = [num_toxic/tot,(tot - num_toxic)/tot]
labeling = ['Toxic','Non-Toxic']
explode = [0.2,0]
plt.pie(slices,explode=explode,shadow=True,autopct='%1.1f%%',labels=labeling,wedgeprops={'edgecolor':'black'})
plt.title('Number of Toxic vs. Non-Toxic Text Samples')
plt.tight_layout()
plt.show()


from sklearn import feature_extraction, linear_model, model_selection, preprocessing

cv1 = feature_extraction.text.CountVectorizer(max_df=0.25)
cv1.fit_transform(train_df["comment_text"])

print("%i words discarded for occurring too frequently:" %(len(cv1.stop_words_)))
print(cv1.stop_words_)

cv2 = feature_extraction.text.CountVectorizer(min_df=5)
cv2.fit_transform(train_df["comment_text"])

print("%i words discarded for occurring too infrequently:" %(len(cv2.stop_words_)))
print(list(cv2.stop_words_)[0:20])

count_vectorizer = feature_extraction.text.CountVectorizer(max_df=0.25, min_df=5)
train_vectors = count_vectorizer.fit_transform(train_df["comment_text"])

print("There are %i words in this corpus" %(train_vectors.shape[1]))

from sklearn.metrics import f1_score

clf = linear_model.RidgeClassifier().fit(train_vectors, train_df["toxic"])
print("Percent correctly labeled comments by Ridge Classifier:")
print(clf.score(train_vectors, train_df["toxic"]))

print("f1 score for non-toxic comments:")
print(f1_score(train_df["toxic"],clf.predict(train_vectors),pos_label=0))

print("f1 score for toxic comments:")
print(f1_score(train_df["toxic"],clf.predict(train_vectors),pos_label=1))

predict_toxic = clf.predict(train_vectors)
error1 = []
error2 = []
for i in range(len(predict_toxic)):
    prediction = predict_toxic[i]
    actual = train_df.iloc[i, 2]
    if prediction == 0 and actual == 1 and len(error1) < 5:
        error1.append(train_df.iloc[i,1])
    elif prediction == 1 and actual == 0 and len(error2) < 5:
        error2.append(train_df.iloc[i,1])

print("Toxic comments incorrectly labeled as Non-Toxic comments: ")
print("\n")
for comment in error1:
    print(comment)
    print("-"*120)
    
print("Non-Toxic comments incorrectly labeled as Toxic comments: ")
print("\n")
for comment in error2:
    print(comment)
    print("-"*120)
    
n = 25
idx_max = (-clf.coef_).argsort()

print("Most 'toxic' words:")
for i in range(n):
    num = idx_max[0][i]
    print(count_vectorizer.get_feature_names()[num])

    
idx_min = (clf.coef_).argsort()
print("\n")
print("Most 'non-toxic' words:")
for i in range(n):
    num = idx_min[0][i]
    print(count_vectorizer.get_feature_names()[num])
    

plt.hist(clf.coef_[0,:],bins=500,range=[-0.5,0.5])
plt.title("Distribution of Ridge Classifier Coefficient Values")
plt.show()

pos = len(clf.coef_[clf.coef_ > 0])

tot = len(clf.coef_[0,:])

toxic_leaning = pos/tot
non_toxic_leaning = 1 - toxic_leaning

print("Percentage of words with a positive (leaning towards toxic) coefficient:")
print(toxic_leaning)

print("Percentage of words with a negative (leaning towards non-toxic) coefficient:")
print(non_toxic_leaning)

count_vectorizer = feature_extraction.text.TfidfVectorizer(max_df=0.25, min_df=5)
train_vectors = count_vectorizer.fit_transform(train_df["comment_text"])

print("There are %i words in this corpus" %(train_vectors.shape[1]))

clf = linear_model.RidgeClassifier().fit(train_vectors, train_df["toxic"])
print("Percent correctly labeled comments by Ridge Classifier:")
print(clf.score(train_vectors, train_df["toxic"]))

print("f1 score for non-toxic comments:")
print(f1_score(train_df["toxic"],clf.predict(train_vectors),pos_label=0))

print("f1 score for toxic comments:")
print(f1_score(train_df["toxic"],clf.predict(train_vectors),pos_label=1))
predict_toxic = clf.predict(train_vectors)
error1 = []
error2 = []
for i in range(len(predict_toxic)):
    prediction = predict_toxic[i]
    actual = train_df.iloc[i, 2]
    if prediction == 0 and actual == 1 and len(error1) < 5:
        error1.append(train_df.iloc[i,1])
    elif prediction == 1 and actual == 0 and len(error2) < 5:
        error2.append(train_df.iloc[i,1])

print("Toxic comments incorrectly labeled as Non-Toxic comments: ")
print("\n")
for comment in error1:
    print(comment)
    print("-"*120)
    
print("Non-Toxic comments incorrectly labeled as Toxic comments: ")
print("\n")
for comment in error2:
    print(comment)
    print("-"*120)
n = 25
idx_max = (-clf.coef_).argsort()

print("Most 'toxic' words:")
for i in range(n):
    num = idx_max[0][i]
    print(count_vectorizer.get_feature_names()[num])

    
idx_min = (clf.coef_).argsort()
print("\n")
print("Most 'non-toxic' words:")
for i in range(n):
    num = idx_min[0][i]
    print(count_vectorizer.get_feature_names()[num])

plt.hist(clf.coef_[0,:],bins=500,range=[-0.5,0.5])
plt.title("Distribution of Ridge Classifier Coefficient Values")
plt.show()

pos = len(clf.coef_[clf.coef_ > 0])

tot = len(clf.coef_[0,:])

toxic_leaning = pos/tot
non_toxic_leaning = 1 - toxic_leaning

print("Percentage of words with a positive (leaning towards toxic) coefficient:")
print(toxic_leaning)

print("Percentage of words with a negative (leaning towards non-toxic) coefficient:")
print(non_toxic_leaning)
import torch   
from torchtext import data

train_path = '/kaggle/input/hate-speech-detection/toxic_train.csv'
test_path = '/kaggle/input/hate-speech-detection/toxic_test.csv'

TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

fields = [('', None), ('comment_text',TEXT),('toxic', LABEL)]

training_data=data.TabularDataset(path = train_path,format = 'csv',fields = fields,skip_header = True)
print(vars(training_data.examples[0]))

import random
train_data, valid_data = training_data.split(split_ratio=0.7)

#initialize glove embeddings
TEXT.build_vocab(train_data,min_freq=3)  
LABEL.build_vocab(train_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(TEXT.vocab.freqs.most_common(10))  

#Word dictionary
#print(TEXT.vocab.stoi)  
#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(device)

#set batch size
BATCH_SIZE = 400

#Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.comment_text),
    sort_within_batch=True,
    device = device)
import torch.nn as nn

class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
        
        return outputs
#define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)
import torch.optim as optim

#define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

#define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)
from tqdm import tqdm_notebook as tqdm

def train(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    accs = []
    losses = []
    batches = []
    
    #set the model in training phase
    model.train()  
    
    batch_num = 0
    
    for batch in tqdm(iterator):
        
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text, text_lengths = batch.comment_text   
        
        #convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()  
        
        #compute the loss
        loss = criterion(predictions, batch.toxic)        
        
        #compute the binary accuracy
        acc = binary_accuracy(predictions, batch.toxic)   
        
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()
        
        losses.append(loss.item())
        accs.append(acc.item())
        batches.append(batch_num)
        
        batch_num += 1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), losses, accs, batches
def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    
    losses = []
    accs = []
    batches = []

    #deactivating dropout layers
    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
        
        batch_num = 0
    
        for batch in iterator:
            
        
            #retrieve text and no. of words
            text, text_lengths = batch.comment_text
            
            #convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()
            
            #compute loss and accuracy
            loss = criterion(predictions, batch.toxic)
            acc = binary_accuracy(predictions, batch.toxic)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            losses.append(loss.item())
            accs.append(acc.item())
            batches.append(batch_num)
            batch_num += 1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), losses, accs, batches
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
     
    #train the model
    train_loss, train_acc, lossess, accs, batches = train(model, train_iterator, optimizer, criterion)
    
    plt.plot(batches, losses)
    plt.title("Loss during epoch %i" %(epoch))
    plt.show()
    
    plt.plot(batches, accs)
    plt.title("Accuracy during epoch %i" %(epoch))
    plt.show()
    
    #evaluate the model
    valid_loss, valid_acc, losses, accs, batches = evaluate(model, valid_iterator, criterion)
    
    plt.plot(batches, losses)
    plt.title("Loss during epoch %i" %(epoch))
    plt.show()
    
    plt.plot(batches, accs)
    plt.title("Accuracy during epoch %i" %(epoch))
    plt.show()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        #torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
