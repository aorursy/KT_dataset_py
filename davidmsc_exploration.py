## Imports

import torch

import torch.nn as nn

import torch.nn.functional as F



import pandas as pd

import numpy as np

import pickle

import string

import random



from keras.preprocessing.text import Tokenizer

import keras.preprocessing.sequence as sq



from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

import seaborn as sn



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = 20, 10

plt.rcParams.update({'font.size': 25})



from IPython.display import clear_output
# Load data and embedding

path_in = '../input/quora-insincere-questions-classification/'



train = pd.read_csv(path_in + 'train.csv')



## Limit data to prevent kernel from crashing in the from scratch part

train = train[:50000]
print("Example sentence: \"{}\"\nLabel: {}".format(train['question_text'][8], train['target'][8]))
print("Ratio of positives to total: {:.2f}%".format(len(train.loc[train['target'] == 1])*100 / len(train)))
def getNumberofWordsFound():

    ## Generate random

    population = list(np.arange(0, len(train)))

    population = random.sample(population, 100)

    

    number_of_words = []

    

    for sample in population:

        sentence = train['question_text'].iloc[int(sample)]



        ## Convert to lowercase and split sentence

        sentence = sentence.lower()



        ## Sepearate punctuation

        chs = string.punctuation

        for ch in chs:

            idx = sentence.find(ch)



            if idx != -1:

                sentence = sentence.replace(ch, " " + ch)



        sentence = sentence.split(' ') 

        number_of_words.append(len(sentence))

        

    return number_of_words





num_of_words = getNumberofWordsFound()
plt.hist(num_of_words, range=(0, 30), color="green")

plt.title("Number of characters per sentence")

plt.xlabel("# characters")

plt.ylabel("# sentences")

plt.show()
def load_embedding(embedding_path, embedding_dim, word_index):

    print('Loading word embeddings...')

    

    vocab_size = len(word_index)    

    embedding_index = pickle.load(open(embedding_path, 'rb'))

    words = embedding_index.keys()

    coef = embedding_index.values()

    

    embedding_matrix = np.zeros((vocab_size+1, embedding_dim));

    for word, i in word_index.items():

        embedding_vector = embedding_index.get(word);

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector;  

            

    print('Finished loading word embeddings. {:.0f} words loaded.'.format(len(embedding_matrix)))

    return embedding_matrix
def make_vocabulary(data):

    '''

    Input:

    data: series of sentences for which to build the vocabulary

    

    Output:

    Dictionary of words and their frequency

    '''

    vocab = {}

    for sentence_number in range(len(data)):

        clear_output(wait=True)

        print("Processing sentence {} / {}".format(sentence_number+1 ,len(data)))

        sentence = data[sentence_number]

                

        ## Sepearate punctuation

        chs = string.punctuation

        

        for ch in chs:

            idx = sentence.find(ch)



            if idx != -1:

                sentence = sentence.replace(ch, " " + ch)

        

        ## Split into words

        sentence = sentence.split(' ')   

        

        for word in sentence:

            word = word.lower()

            try:

                vocab[word] += 1

            except KeyError:

                    vocab[word] = 1

        

    print("Done")

        

    return vocab



vocab = make_vocabulary(train['question_text'])
## Take vocabulary

def tokenizer_fit(sentences, vocabulary):

    '''

    Inputs: 

    sentences: series object

    vocabulary: dictionary with words as keys

    

    Output:

    word_index: dictionary with words as keys and tokens as columns

    '''

    word_index = {}    

    token = 1

    

    for i in range(len(sentences)):

        

        sentence = sentences[i]

        

        ## Sepearate punctuation

        chs = string.punctuation

        

        for ch in chs:

            idx = sentence.find(ch)



            if idx != -1:

                sentence = sentence.replace(ch, " " + ch)

        

        ## Split into characters

        sentence = sentence.split(' ')   

        

        for word in sentence:            

            if word not in word_index:

                word_index[word] = token

                token += 1

    

    return word_index
def tokenizer_apply(word_index, sentences, max_dim):

    '''

    Inputs:

    word_index: dictionary with tokens

    sentences: series with sentences

    

    Output:

    sequences: array with post-padded tokens

    '''

    ## Initialize array    

    sequences = np.zeros((len(sentences), max_dim))

    

    for i in range(len(sentences)):

        sentence = sentences[i]

        

        ## Sepearate punctuation

        chs = string.punctuation

        

        for ch in chs:

            idx = sentence.find(ch)



            if idx != -1:

                sentence = sentence.replace(ch, " " + ch)

        

        ## Split into characters

        sentence = sentence.split(' ')

        

        ## Truncate at max_dim

        sentence = sentence[0:max_dim]

                        

        for j in range(len(sentence)):

            word = sentence[j]

            



            try:

                sequences[i,j] = int(word_index[word])

            

            except KeyError:

                sequences[i, j] = 0

                

    return sequences
max_len = 50



# Build the tokenizer dictionary in the tokenizer class

word_index = tokenizer_fit(train['question_text'], vocab) 



# Split train set into train and validation sets

train, validation = train_test_split(train, test_size=0.2, shuffle = True)

train.reset_index(inplace=True)

validation.reset_index(inplace=True)



# Tokenize the questions

train_sequences = tokenizer_apply(word_index, train['question_text'], max_len)

validation_sequences = tokenizer_apply(word_index, validation['question_text'], max_len)



## Convert data to tensors

training_dataset = torch.utils.data.TensorDataset(torch.LongTensor(np.array(train_sequences)), torch.FloatTensor(np.array(train['target'])))

validation_dataset = torch.utils.data.TensorDataset(torch.LongTensor(np.array(validation_sequences)), torch.FloatTensor(np.array(validation['target'])))
## Load embedding

embedding_dim = 300

embedding_path = "../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl"





embedding_matrix = load_embedding(embedding_path, embedding_dim, word_index)

weights = torch.FloatTensor(embedding_matrix)
# Build the tokenizer dictionary in the tokenizer class

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

tokenizer.fit_on_texts(train['question_text'])



# Split train set into train and validation sets

train, validation = train_test_split(train, test_size=0.1, shuffle = True)

#train.reset_index(inplace=True)

#validation.reset_index(inplace=True)



# Tokenize the questions

train_sequences = tokenizer.texts_to_sequences(train['question_text'])

validation_sequences = tokenizer.texts_to_sequences(validation['question_text'])



# Extract the dictionary witht the tokens

word_index = tokenizer.word_index
## Pad sequences with 0s so that each sequence is of the same length respresented by the max_length

max_length = 50

padding_type = 'pre'

trunc_type = 'pre'



train_sequences_padded = sq.pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences_padded = sq.pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
## Convert data to tensors

training_dataset = torch.utils.data.TensorDataset(torch.LongTensor(np.array(train_sequences_padded)), torch.FloatTensor(np.array(train['target'])))

validation_dataset = torch.utils.data.TensorDataset(torch.LongTensor(np.array(validation_sequences_padded)), torch.FloatTensor(np.array(validation['target'])))
## Load embedding

embedding_dim = 300

embedding_path = "../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl"





embedding_matrix = load_embedding(embedding_path, embedding_dim, word_index)

weights = torch.FloatTensor(embedding_matrix)
class Net(nn.Module):

    def __init__(self, seq_length, hidden_layer):

        super(Net, self).__init__()

        

        ## Embedding layer

        self.embd = nn.Embedding.from_pretrained(weights)

        

        ## LSTM

        self.lstm = nn.LSTM(input_size=len(weights[0,:]), hidden_size=hidden_layer, batch_first=True)

        

        ## Linear layers

        self.fc1 = nn.Linear(hidden_layer, hidden_layer)

        self.dropout1 = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(hidden_layer, 1)

        self.tanh = nn.Tanh()

        self.out = nn.Sigmoid()

        

        self.initialize()

        

    def initialize(self):

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.fc1.bias.data.zero_()

      



    def forward(self, x):

        embd_out = self.embd(x)       

        lstm_out, (h_out, _) = self.lstm(embd_out)

        x = self.tanh(self.dropout1(self.fc1(h_out)))

        x = self.fc2(x)

        x = self.out(x)

        

        return x
### Function for training

epochs = 10

learning_rate = 0.001



seq_length = 50

hidden_layer_1 = 300



batch_size = 1000





## Define dataloader

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset), shuffle=False)



## Initialize net and define loss function and optimizer

model = Net(seq_length, hidden_layer_1)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





##### Iterate through the data

loss_acc = {'Loss': [], 'Train accuracy': [], 'Validation accuracy':[]}



for epoch in range(epochs):

    print("Epoch {} / {}".format(epoch+1, epochs))



    train_correct = 0    

    model.train()

    

    for x, y in train_loader:

        # Set gradients to zero

        optimizer.zero_grad()

        

        # Forward pass

        y_hat = model(x)

        

        # Evaluate output / compute loss

        y_hat = y_hat.view(-1)        

        loss = criterion(y_hat, y)       

        

        # Backward pass / optimize

        loss.backward()

        

        # Update weights

        optimizer.step()

                

        ## Evaluate train result

        y_hat = np.where(y_hat.detach().numpy() > 0.5, 1, 0)

        train_correct += (y_hat == y.numpy()).sum()

    

    train_acc = train_correct / len(train_loader.dataset)

    loss_acc['Loss'].append(loss.item())

      

        

    # Get validation accuracy

    val_correct = 0

    with torch.no_grad():

        for x, y in validation_loader:

            y_hat = model(x)

            y_hat = y_hat.view(-1)



            ## Evaluate validation result

            y_hat = np.where(y_hat.numpy() > 0.5, 1, 0)

            val_correct += (y_hat == y.numpy()).sum()

    

    val_acc = val_correct / len(validation_loader.dataset)

    

    

    # Append scores to dictionary

    loss_acc['Train accuracy'].append(train_acc)

    loss_acc['Validation accuracy'].append(val_acc)

    

    print("Loss: {:.4f}".format(loss_acc['Loss'][-1]))

    print("Training accuracy: {:.4f} | Validation accuracy: {:.4f}".format(loss_acc['Train accuracy'][-1], loss_acc['Validation accuracy'][-1]))



## Save the model

#torch.save(model.state_dict(), './quora_questions_classifier.pt')

    

print("\nTraining completed!")
## Plots

fig = plt.figure(1)

plt.plot(loss_acc['Loss'], color="red")

plt.title("Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss [-]")



fig = plt.figure(2)

plt.plot(loss_acc['Train accuracy'], "o-", color="black", label="Train")

plt.plot(loss_acc['Validation accuracy'], "^-",color="blue", label="Validation")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.legend()

plt.ylabel("Accuracy [%]")
## Use trained model for validation

model.eval()



for x, y in validation_loader:

    validation_data = x

    validation_labels = y



pred = model(validation_data)

pred = pred.squeeze()

predicted_labels = np.where(pred > 0.5, 1, 0)



accuracy_score(np.array(validation_labels), predicted_labels)
def Compute_Metrics():

    precision = precision_score(validation_labels, predicted_labels)

    print("Precision: {:.4f}".format(precision))

    

    recall = recall_score(validation_labels, predicted_labels)

    print("Recall: {:.4f}".format(recall))

    

    f1 = f1_score(validation_labels, predicted_labels)

    print("F1 Score: {:.4f}".format(f1))

    

    return precision, recall, f1



precision, recall, f1 = Compute_Metrics()
## Confusion matrix

def Confusion_Matrix(con_mat):

    

    plt.title("Confusion Matrix", fontsize=50)

    sn.set(font_scale=2.5)

    sn.heatmap(con_mat, annot=True, fmt='g', annot_kws={"size":30}, xticklabels=["Negatives", "Positives"], yticklabels=["Negatives", "Positives"], 

               cmap="Blues")

    plt.xlabel("Prediction", fontsize=35)

    plt.ylabel("Truth", fontsize=35)



    

con_mat = confusion_matrix(validation_labels, predicted_labels)

Confusion_Matrix(con_mat)