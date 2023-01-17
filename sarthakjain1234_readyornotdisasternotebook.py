import pandas as pd
import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud, STOPWORDS
from string import punctuation
from collections import Counter
from nltk.stem.porter import *
import itertools
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("GPU is available")
else:
    print("GPU is not available, CPU is being used instead")
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

print("The csv shape: ", str(train_df.shape))
train_df.head()

test_df.head()
print(" Example text: ", str(train_df["text"][1]), "\n", "Target: ", str(train_df["target"][1]))
total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum())/(train_df.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ["total", "percent"])
missing_data

train_df = train_df.drop(["keyword", "location", "id"], axis = 1)
test_df = test_df.drop(["keyword", "location", "id"], axis = 1)
print("Keyword, Location, and id are all dropped successfully")


phrase = "Hello,,?? This is my Tweet number 17!!!. I also wanted or been wanting you to try reading this tweet https://www.youtube.com and \
also be writing or seeing https://relentless.com. This is my 28th and and maybe 32nd time seeing this. #ThisisFun"
punct_list = set(punctuation)


def remove_punct(text):
    
    new_text = "".join(ch for ch in text if ch not in punct_list)
    return new_text

def remove_stopwords(text):
    
    text_split = text.split(" ")
    text = [word for word in text_split if word not in STOPWORDS]
    return text

def remove_http(text_list):
    
    new_text = [word for word in text_list if word.find("http") == -1]
    return new_text

def stem_porter(text_list):
    
    stemmer = PorterStemmer()
    
    new_text = [stemmer.stem(word) for word in text_list]
    return new_text

def change_number(text_list):
    
    new_text = []
    for word in text_list:
        if (bool(re.search(r'\d', word)) == False):
            new_text.append(word)
        else:
            new_text.append("||Numeric||")
    
    return new_text

change_number(stem_porter(remove_http(remove_stopwords(remove_punct(phrase.lower())))))
#Explains what stem does
print("Before stem: ")
print(remove_http(remove_stopwords(remove_punct(phrase.lower()))))

print("After stem: ")
print(stem_porter(remove_http(remove_stopwords(remove_punct(phrase.lower())))))
def preprocess_text(text):
    
    text = change_number(stem_porter(remove_http(remove_stopwords(remove_punct(text.lower())))))
    return text

preprocess_text(phrase)
train_df["text"] = train_df["text"].apply(lambda x: preprocess_text(x))
test_df["text"] = test_df["text"].apply(lambda x: preprocess_text(x))
#The new preprocessed data
print(train_df["text"])
train_df.head()
##### Turn the text into numbers

freq = {}
for row in train_df["text"]:
    
    for word in row:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1


sorted_freq = sorted(freq, key = freq.get, reverse = True)
sorted_freq.remove('')

#Now we have to tokenize
vocab2int = {word: ii + 1 for ii, word in enumerate(sorted_freq)}
int2vocab = {ii + 1: word for ii, word in enumerate(sorted_freq)}

print(dict(itertools.islice(vocab2int.items(), 100)))
phrase_list = change_number(stem_porter(remove_http(remove_stopwords(remove_punct(phrase.lower())))))
print("Before: ", str(phrase_list))
def tokenize_text(text_list):
    
    int_text = []
    for word in text_list:
        try:
            int_text.append(vocab2int[word])
        except:
            int_text.append(0)
            
    return int_text

print("After: ", str(tokenize_text(phrase_list)))

print("Second test: ")
common_list = ["fire", "bomb", "somemorerandom,,!!", "im", "first", "jibberish", "the", "||Numeric||"]
print(tokenize_text(common_list))
train_df["text"] = train_df["text"].apply(lambda x: tokenize_text(x))
train_df.head(10)
max_length = 0
for tweet in train_df["text"]:
    if len(tweet) >= max_length:
        max_length = len(tweet)

        
print("Max Length: ", str(max_length))    
    
def pad_text(int_list, sequence_length = 52):
    
    padded_list = np.zeros((sequence_length), dtype = int)
    padded_list[-len(int_list):] = np.array(int_list)[:sequence_length]
    
    return padded_list


int_list = tokenize_text(phrase_list)
print(pad_text(int_list))

#We have some rows we need to drop or else the pad won't work
list_of_empty_rows = []
for ii, tweet in enumerate(train_df["text"]):
    if len(tweet) == 0:
        list_of_empty_rows.append(ii)

print(list_of_empty_rows)     
train_df = train_df.drop(list_of_empty_rows)
print("Rows have been dropped")
train_df["text"] = train_df["text"].apply(lambda x: pad_text(x))
print("First Review")
print(train_df["text"][0][:])

print("Next Review")
print(train_df["text"][1028][:])

print("Next Review")
print(train_df["text"][456][:])

print("Length of review: ", str(len(train_df["text"][906])))
def get_text(csv_column):
    listed_data = []
    for row in csv_column:
        listed_data.append(row)
        
    return listed_data

def get_target(csv_column):
    listed_data = []
    for row in csv_column:
        listed_row = [row]
        listed_data.append(listed_row)
    return listed_data
train_x = get_text(train_df["text"])
train_y = get_target(train_df["target"])

assert len(train_y) == len(train_x)
print(len(train_x))
print(len(train_y))
assert len(train_x) == len(train_y)
print(len(train_x))
#Uncomment these lines to create datalaoder for validation
# assert len(valid_x) == len(valid_y)
# print(len(valid_x))
def create_dataloader(train_x, train_y, batch_size = 30):
    
    #Make sure to convert from Numpy to Torch Tensor
    train_dataset = TensorDataset(torch.LongTensor(train_x), torch.FloatTensor(train_y))
    
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    
    return train_loader

#Lets create a dataloader
train_loader = create_dataloader(train_x[:7595], train_y[:7595])
#Lets look at the batches of data
train_loader_iter = iter(train_loader)
next(train_loader_iter)[0].shape
class LSTM(nn.Module):
    
    def __init__(self,vocab_size, embedding_dim, hidden_size, n_layers):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, self.hidden_size, num_layers = self.n_layers, dropout = 0.2, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        
        batch_size = x.size(0)
        sequence_len = x.size(1)
        
        embeddings = self.embedding(x)
        lstm_out, hidden = self.lstm1(embeddings)
        output = lstm_out.contiguous().view(-1, self.hidden_size)
        
        output = self.sigmoid(self.fc1(output))
        output = output.reshape(batch_size, sequence_len, -1)
        output = output[:, -1]
        
        
        return hidden, output
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda())
            
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
            
        return hidden
        
        
        
def forward_and_backprop(rnn, optimizer, tweet, target, criterion, hidden):
    
    if train_on_gpu:
        rnn.cuda()
    
    hidden = ([each.data for each in hidden])
    optimizer.zero_grad()
    
    hidden, output = rnn(tweet, hidden)
    loss = criterion(output, target)
    
    loss.backward()
    
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    
    loss_for_batch = loss.item()
    
    return loss_for_batch, hidden
    
    
    
def train(rnn, epochs, optimizer, criterion, batch_size, train_loader):
    
    rnn.train()
    if train_on_gpu:
        rnn = rnn.cuda()
        
    for epoch in range(1, epochs + 1):
        
        hidden = rnn.init_hidden(batch_size)
        train_loss = 0
        
        for batch_i, (tweet, target) in enumerate(train_loader):
            
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            if train_on_gpu:
                tweet = tweet.cuda()
                target = target.cuda() 
            
            batch_loss, batch_hidden = forward_and_backprop(rnn, optimizer, tweet, target, criterion, hidden)
            train_loss += batch_loss
        
        
        print("Epoch Number: ", str(epoch)) 
        print("Train Loss: ", str(train_loss))
        
    
    
        
    
    
epochs = 20
batch_size = 30
lr = 0.001

vocab_size = len(vocab2int)
embedding_dim = 230
hidden_size = 250
num_layers = 2

rnn = LSTM(vocab_size, embedding_dim, hidden_size, num_layers) 

optimizer = optim.Adam(rnn.parameters(), lr = lr)
criterion = nn.BCELoss()
train(rnn, epochs, optimizer, criterion, batch_size, train_loader)
def predict(net, test_review):
    
    assert len(test_review) > 0
    
    net.eval()
    preprocessed_data = tokenize_text(preprocess_text(test_review))
    
    assert len(test_review) > 0
    
    padded_data = pad_text(preprocessed_data)
    padded_data = padded_data.reshape(1, -1)
    padded_data = torch.from_numpy(padded_data)
   
    
    batch_size = padded_data.size(0)
    hidden = net.init_hidden(batch_size)
    
    if train_on_gpu:
        padded_data = padded_data.cuda()
    
    hidden, output = net(padded_data, hidden)
    
    print("Unrounded Answer: ", output)
    
    answer = np.round(output.cpu().detach().numpy())
    if answer == 1:
        print("Call in immediate emergency at location")
    else:
        print("General Commentary")
    
    

predict(rnn, "Breaking News: Flooding on streets")
predict(rnn, "Fire ravaged houses next to me and are approaching me")
predict(rnn, "We the best music. We just chillin #Ballin")
predict(rnn, "Smoke in the air. It smells like smoke #Fire")
predict(rnn, "Smoke in the air. It smells like smoke #SaySikeRightNow")
predict(rnn, "I hear strange noises. The wall is shaking")
predict(rnn, "I am on fire with playing this game #TheGOAT")
predict(rnn, "High winds very high winds the ground is shaking")
predict(rnn, "I dont do domestic violence")
predict(rnn, "The houses next to us have burned to pure ash")
predict(rnn, "Breaking News: High Water levels threatening Silicon Valley")
predict(rnn, "Oh no what is happening. Flooding is affected my House")
predict(rnn, "Oh no what is happening. Flooding is affected my House")
predict(rnn, "there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all")
predict(rnn, "This fire is huge. How I am I supposed to put this out myself bruh")
predict(rnn, "I can a broken car set ablaze on the side of street calling 911")

    
predict(rnn, "ACCIDENT - HIT AND RUN - COLD at 500 BLOCK OF SE VISTA TER GRESHAM OR [Gresham Police #PG15000044357...")
predict(rnn, "@DaveOshry @Soembie So if I say that I met her by accident this week- would you be super jelly Dave?...")
predict(rnn, "We're shaking...It's an earthquake")
predict(rnn, "We are still living in the aftershock of Hiroshima people are still the scars of history.' - Edward...")
predict(rnn, "320 [IR] ICEMOON [AFTERSHOCK] | http://t.co/THyzOMVWU0 | @djicemoon | #Dubstep #TrapMusic #DnB #EDM ...")
predict(rnn, "#UPDATE: Picture from the Penn Twp. airplane accident. http://t.co/6JfgDnZRlC")
predict(rnn, "@thugIauren I had myself on airplane mode by accident ??")
predict(rnn, "Typhoon Soudelor kills 28 in China and Taiwan")
predict(rnn, "No I don't like cold!")
predict(rnn, "Not a diss song. People will take 1 thing and run with it. Smh it's an eye opener though. He is abou...")
predict(rnn, "Just got to love burning your self on a damn curling wand... I swear someone needs to take it away f...")
predict(rnn, "I hate badging shit in accident")
predict(rnn, "Horrible Accident Man Died In Wings of AirplaneåÊ(29-07-2015) http://t.co/5ZRKZdhODe")

