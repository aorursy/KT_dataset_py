import numpy as np

import pandas as pd



import unicodedata, re, string

import nltk



import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader



import seaborn as sns

sns.set(color_codes=True)



import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.tsv", sep="\t")

df_test = pd.read_csv("../input/test.tsv", sep="\t")
df_train.info()
df_train.head()
df_train['Phrase'][0]
df_train.loc[df_train['SentenceId'] == 1]
dist = df_train.groupby(["Sentiment"]).size()



fig, ax = plt.subplots(figsize=(12,8))

sns.barplot(dist.keys(), dist.values);
def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words



def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words



def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



def remove_numbers(words):

    """Remove all interger occurrences in list of tokenized words with textual representation"""

    new_words = []

    for word in words:

        new_word = re.sub("\d+", "", word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



def stem_words(words):

    """Stem words in list of tokenized words"""

    stemmer = LancasterStemmer()

    stems = []

    for word in words:

        stem = stemmer.stem(word)

        stems.append(stem)

    return stems



def lemmatize_verbs(words):

    """Lemmatize verbs in list of tokenized words"""

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    for word in words:

        lemma = lemmatizer.lemmatize(word, pos='v')

        lemmas.append(lemma)

    return lemmas



def normalize(words):

    words = remove_non_ascii(words)

    words = to_lowercase(words)

    words = remove_punctuation(words)

    words = remove_numbers(words)

#    words = remove_stopwords(words)

    return words
# First step - tokenizing phrases

df_train['Words'] = df_train['Phrase'].apply(nltk.word_tokenize)



# Second step - passing through prep functions

df_train['Words'] = df_train['Words'].apply(normalize) 

df_train['Words'].head()
# Third step - creating a list of unique words to be used as dictionary for encoding

word_set = set()

for l in df_train['Words']:

    for e in l:

        word_set.add(e)

        

word_to_int = {word: ii for ii, word in enumerate(word_set, 1)}



# Check if they are still the same lenght

print(len(word_set))

print(len(word_to_int))

# Now the dict to tokenize each phrase

df_train['Tokens'] = df_train['Words'].apply(lambda l: [word_to_int[word] for word in l])

df_train['Tokens'].head()
# Step four - get the len of longest phrase

max_len = df_train['Tokens'].str.len().max()

print(max_len)
# Pad each phrase representation with zeroes, starting from the beginning of sequence

# Will use a combined list of phrases as np array for further work. This is expected format for the Pytorch utils to be used later



all_tokens = np.array([t for t in df_train['Tokens']])

encoded_labels = np.array([l for l in df_train['Sentiment']])



# Create blank rows

features = np.zeros((len(all_tokens), max_len), dtype=int)

# for each phrase, add zeros at the end 

for i, row in enumerate(all_tokens):

    features[i, :len(row)] = row



#print first 3 values of the feature matrix 

print(features[:3])



 
split_frac = 0.8



## split data into training, validation, and test data (features and labels, x and y)



split_idx = int(len(features)*0.8)

train_x, remaining_x = features[:split_idx], features[split_idx:]

train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]



test_idx = int(len(remaining_x)*0.5)

val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]

val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]



## print out the shapes of  resultant feature data

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(train_x.shape), 

      "\nValidation set: \t{}".format(val_x.shape),

      "\nTest set: \t\t{}".format(test_x.shape))
# create Tensor datasets

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))



# dataloaders

batch_size = 54



# make sure the SHUFFLE your training data

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)



# Check the size of the loaders (how many batches inside)

print(len(train_loader))

print(len(valid_loader))

print(len(test_loader))
# First checking if GPU is available

train_on_gpu=torch.cuda.is_available()



if(train_on_gpu):

    print('Training on GPU.')

else:

    print('No GPU available, training on CPU.')
class SentimentRNN(nn.Module):

    """

    The RNN model that will be used to perform Sentiment analysis.

    """



    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        """

        Initialize the model by setting up the layers.

        """

        super(SentimentRNN, self).__init__()



        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        # embedding and LSTM layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 

                            dropout=drop_prob, batch_first=True)

        

        # dropout layer

        self.dropout = nn.Dropout(0.3)

        

        # linear

        self.fc = nn.Linear(hidden_dim, output_size)

        

    def forward(self, x, hidden):

        """

        Perform a forward pass of our model on some input and hidden state.

        """

        batch_size = x.size(0)



        # embeddings and lstm_out

        embeds = self.embedding(x)



        lstm_out, hidden = self.lstm(embeds, hidden)



        # transform lstm output to input size of linear layers

        lstm_out = lstm_out.transpose(0,1)

        lstm_out = lstm_out[-1]



        out = self.dropout(lstm_out)

        out = self.fc(out)        



        return out, hidden

    

    def init_hidden(self, batch_size):

        ''' Initializes hidden state '''

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,

        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden
# Instantiate the model w/ hyperparams

vocab_size = len(word_to_int)+1 # +1 for the 0 padding

output_size = 5

embedding_dim = 400

hidden_dim = 256

n_layers = 2



net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)



print(net)
# loss and optimization functions

lr=0.003



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# training params

epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing



counter = 0

print_every = 100

clip=5 # gradient clipping



# move model to GPU, if available

if(train_on_gpu):

    net.cuda()



net.train()

# train for some number of epochs

for e in range(epochs):

    # initialize hidden state

    h = net.init_hidden(batch_size)



    # batch loop

    for inputs, labels in train_loader:

        counter += 1



        if(train_on_gpu):

            inputs, labels = inputs.cuda(), labels.cuda()



        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])



        # zero accumulated gradients

        net.zero_grad()



        # get the output from the model

        output, h = net(inputs, h)

        # calculate the loss and perform backprop

        loss = criterion(output, labels)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()



        # loss stats

        if counter % print_every == 0:

            # Get validation loss

            val_h = net.init_hidden(batch_size)

            val_losses = []

            net.eval()

            for inputs, labels in valid_loader:



                # Creating new variables for the hidden state, otherwise

                # we'd backprop through the entire training history

                val_h = tuple([each.data for each in val_h])



                if(train_on_gpu):

                    inputs, labels = inputs.cuda(), labels.cuda()



                output, val_h = net(inputs, val_h)

                val_loss = criterion(output, labels)



                val_losses.append(val_loss.item())



            net.train()

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
# Get test data loss and accuracy



test_losses = [] # track loss

num_correct = 0



# init hidden state

h = net.init_hidden(batch_size)



net.eval()

# iterate over test data

for inputs, labels in test_loader:



    # Creating new variables for the hidden state, otherwise

    # we'd backprop through the entire training history

    h = tuple([each.data for each in h])



    if(train_on_gpu):

        inputs, labels = inputs.cuda(), labels.cuda()

    

    # get predicted outputs

    output, h = net(inputs, h)

    

    # calculate loss

    test_loss = criterion(output, labels)

    test_losses.append(test_loss.item())

    

    # convert output probabilities to predicted class

    _, pred = torch.max(output,1)

    

    # compare predictions to true label

    correct_tensor = pred.eq(labels.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    num_correct += np.sum(correct)



# -- stats! -- ##

# avg test loss

print("Test loss: {:.3f}".format(np.mean(test_losses)))



# accuracy over all test data

test_acc = num_correct/len(test_loader.dataset)

print("Test accuracy: {:.3f}".format(test_acc))