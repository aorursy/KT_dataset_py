import os

import time

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix,mean_absolute_error,mean_squared_error

from tqdm import tqdm



# Pytorch Imports

import torch

import torch.nn as nn

import torch.utils.data

from torch.autograd import Variable



# Deterministic behaviour when using GPUs

def fixing_seed(seed=1326):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    print(f">> Set Numpy/Torch/Cuda Seeds\n>> Deterministic CUDNN : {torch.backends.cudnn.deterministic}")

fixing_seed()
X_array = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 

                    [9.779], [6.182], [7.59], [2.167], [7.042], 

                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)



y_array = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 

                    [3.366], [2.596], [2.53], [1.221], [2.827], 

                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
class LinearRegression(nn.Module):

    

    def __init__(self, input_dim, output_dim):

        # Calling Super Class's constructor

        super(LinearRegression, self).__init__() 

        # We can add more inbuilt layers or initialize custom layers here

        self.linear = nn.Linear(input_dim, output_dim)

        

    def forward(self, x):

        # Forward decides flow of data in the pytorch models

        #  We can tweak and fine architecture in forward pass

        out = self.linear(x)

        return out
input_dim = X_array.shape[1]

output_dim = 1

learning_rate = 0.01

num_epochs = 500



# Linear regression model

model = LinearRegression(input_dim, output_dim).cuda()

criterion = nn.MSELoss() # Mean Squared Loss

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



# Train the model

for epoch in range(num_epochs):

    # Convert numpy arrays to torch tensors

    inputs = torch.from_numpy(X_array).cuda()

    targets = torch.from_numpy(y_array).cuda()



    # Forward pass

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    

    # Backward and optimize

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    

    if (epoch+1) % 100 == 0:

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



# Plot the graph

predicted = model(inputs).detach().cpu().numpy()

plt.plot(X_array, y_array, 'ro', label='Original data')

plt.plot(X_array, predicted, label='Fitted line')

plt.legend()

plt.show()

# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
print(f"Mean Absolute Error {mean_absolute_error(y_array,predicted)**.5}")

print(f"Mean Squared Error {mean_squared_error(y_array,predicted)}")

print(f"Root Mean Squared Error {mean_squared_error(y_array,predicted)**.5}")
# DEBUG 

nrows  = 10000
embed_size = 300 

max_features = 5000 

maxlen = 70



train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv", nrows= nrows)

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv", nrows= nrows)

print('Train data dimension: ', train_df.shape)

print('Test data dimension: ', test_df.shape)
def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix
train_df["question_text"] = train_df["question_text"].str.lower()

test_df["question_text"] = test_df["question_text"].str.lower()



# fill up the missing values

x_train = train_df["question_text"].fillna("_##_").values

x_test = test_df["question_text"].fillna("_##_").values



# Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

# Pad the sentences 

x_train = pad_sequences(x_train, maxlen=maxlen)

x_test = pad_sequences(x_test, maxlen=maxlen)

# Get the target values

y_train = train_df['target'].values
fastText_embeddings = load_fasttext(tokenizer.word_index)

embedding_matrix = fastText_embeddings
def sigmoid(x):

    return 1 / (1 + np.exp(-x))    



class BiGRUClassifier(nn.Module):  

    def __init__(self):

        hidden_size = 256

        self.hidden_size = 300

        self.batch_size = 512

        super(BiGRUClassifier, self).__init__()

        self.embedding = nn.Embedding(max_features, 300)

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.gru = nn.GRU(input_size=300, hidden_size=hidden_size, 

                          bidirectional=True, batch_first=True)

        #out 

        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)

        self.linear2 = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.1)

        

    def forward(self, question, train=True):

        batch = question.size(0)

        question_embed = self.embedding(question) 



        gru_output, hidden = self.gru(question_embed) 

        hidden = hidden.transpose(0, 1).contiguous().view(batch, -1) 

        hidden = self.dropout(hidden)

        hidden = torch.relu(self.linear1(hidden))  #batch x hidden_size

        hidden = self.dropout(hidden)

        return (self.linear2(hidden))  

    

    def init_hidden(self, batch_size):

        return cuda_available(torch.zeros(2, batch_size, self.hidden_size))

def TrainingInference(model, batch_size=512, epochs=5):

    n_epochs = epochs # how many times to iterate over all samples

    # matrix for the out-of-fold predictions

    train_preds = np.zeros((len(train_df)))

    # matrix for the predictions on the test set

    test_preds = np.zeros((len(test_df)))

    x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()

    test = torch.utils.data.TensorDataset(x_test_cuda)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=10).split(x_train, y_train))

    for i, (train_idx, valid_idx) in enumerate(splits):    

        # split data in train / validation according to the KFold indeces

        # also, convert them to a torch tensor and store them on the GPU (done with .cuda())

        x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()

        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()

        x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()

        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()



        # make sure everything in the model is running on the GPU

        model.cuda()

        # define binary cross entropy loss

        # note that the model returns logit to take advantage of the log-sum-exp trick 

        # for numerical stability in the loss

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        optimizer = torch.optim.Adam(model.parameters())



        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)



        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)



        print(f'Fold {i + 1}')



        for epoch in range(n_epochs):

            # set train mode of the model. This enables operations which are only applied during training like dropout

            start_time = time.time()

            model.train()

            avg_loss = 0.  

            for x_batch, y_batch in tqdm(train_loader, disable=True):

                # Forward pass: compute predicted y by passing x to the model.

                y_pred = model(x_batch)



                # Compute and print loss.

                loss = loss_fn(y_pred, y_batch)



                # Before the backward pass, use the optimizer object to zero all of the

                # gradients for the Tensors it will update (which are the learnable weights

                # of the model)

                optimizer.zero_grad()



                # Backward pass: compute gradient of the loss with respect to model parameters

                loss.backward()



                # Calling the step function on an Optimizer makes an update to its parameters

                optimizer.step()

                avg_loss += loss.item() / len(train_loader)



            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout

            model.eval()



            # predict all the samples in y_val_fold batch per batch

            valid_preds_fold = np.zeros((x_val_fold.size(0)))

            test_preds_fold = np.zeros((len(test_df)))



            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(valid_loader):

                y_pred = model(x_batch).detach()



                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]



            elapsed_time = time.time() - start_time 

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))



        # predict all samples in the test set batch per batch

        for i, (x_batch,) in enumerate(test_loader):

            y_pred = model(x_batch).detach()



            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]



        train_preds[valid_idx] = valid_preds_fold

        test_preds += test_preds_fold / len(splits)

    return train_preds, test_preds
model = BiGRUClassifier()

train_oofpreds, test_oofpreds = TrainingInference(model)

predictions = (train_oofpreds > .5).astype('uint')

print(f"Confusion Matrix \n{classification_report(y_train, predictions)}")

print(f"Accuracy {accuracy_score(y_train, predictions)}")
# Custom Layers

class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

        super(Attention, self).__init__(**kwargs)

        

        self.supports_masking = True



        self.bias = bias

        self.feature_dim = feature_dim

        self.step_dim = step_dim

        self.features_dim = 0

        

        weight = torch.zeros(feature_dim, 1)

        nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

        

        if bias:

            self.b = nn.Parameter(torch.zeros(step_dim))

        

    def forward(self, x, mask=None):

        feature_dim = self.feature_dim

        step_dim = self.step_dim



        eij = torch.mm(

            x.contiguous().view(-1, feature_dim), 

            self.weight

        ).view(-1, step_dim)

        

        if self.bias:

            eij = eij + self.b

            

        eij = torch.tanh(eij)

        a = torch.exp(eij)

        

        if mask is not None:

            a = a * mask



        a = a / torch.sum(a, 1, keepdim=True) + 1e-10



        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)
class BiLSTMGRUAttention(nn.Module):

    def __init__(self):

        super(BiLSTMGRUAttention, self).__init__()

        hidden_size = 512

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False      

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)

        self.gru = nn.GRU(256, 256*2, bidirectional=True, batch_first=True)  

        self.lstm_attention = Attention(128 * 2, maxlen)

        self.gru_attention = Attention(256 * 4, maxlen)

        self.linear = nn.Linear(3328,512)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.linear2 = nn.Linear(512, 256)

        self.linear3 = nn.Linear(256, 64)

        self.out = nn.Linear(64, 1)



    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.squeeze(

            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        

        h_lstm, _ = self.lstm(h_embedding)

        h_gru, _ = self.gru(h_lstm)

        

        h_lstm_atten = self.lstm_attention(h_lstm)

        h_gru_atten = self.gru_attention(h_gru)

        

        # global average pooling

        avg_pool = torch.mean(h_gru, 1)

        # global max pooling

        max_pool, _ = torch.max(h_gru, 1)

        

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)

        conc = self.relu(self.linear(conc))

        

        conc = self.relu(self.linear2(conc))

        conc = self.relu(self.linear3(conc))

#         conc = self.dropout(conc)



        out = self.out(conc)

        return out
model = BiLSTMGRUAttention()

train_oofpreds, test_oofpreds = TrainingInference(model, epochs=15)
predictions = (train_oofpreds > .5).astype('uint')

print(f"Confusion Matrix \n{classification_report(y_train, predictions)}")

print(f"Accuracy {accuracy_score(y_train, predictions)}")
embed_size = 300 

max_features = 10000

maxlen = 500





train_df = pd.read_csv("../input/spooky-author-identification/train.csv", nrows= nrows)

test_df = pd.read_csv("../input/spooky-author-identification/test.csv", nrows=nrows)

print('Train data dimension: ', train_df.shape)

print('Test data dimension: ', test_df.shape)
train_df["text"] = train_df["text"].str.lower()

test_df["text"] = test_df["text"].str.lower()

# fill up the missing values

x_train = train_df["text"].fillna("_##_").values

x_test = test_df["text"].fillna("_##_").values

# Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

# Pad the sentences 

x_train = pad_sequences(x_train, maxlen=maxlen)

x_test = pad_sequences(x_test, maxlen=maxlen)

# Get the target values

y_train = train_df['author'].values

y_dumm = pd.get_dummies(train_df['author'].values).values
train_df['author'].value_counts()
y_train
fastText_embeddings = load_fasttext(tokenizer.word_index)

embedding_matrix = fastText_embeddings
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train = le.fit_transform(y_train)
class BiLSTMGRUAttention(nn.Module):

    def __init__(self):

        super(BiLSTMGRUAttention, self).__init__()

        hidden_size = 256

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False      

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)  

        self.lstm_attention = Attention(hidden_size * 2, maxlen)

        self.gru_attention =  Attention(hidden_size * 2, maxlen)

        self.linear = nn.Linear(1024, 16)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(16, 3)

    

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.squeeze(

            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        

        h_lstm, _ = self.lstm(h_embedding)

        h_gru, _ = self.gru(h_lstm)

        

        h_lstm_atten = self.lstm_attention(h_lstm)

        h_gru_atten = self.gru_attention(h_gru)

        

        # global average pooling

        avg_pool = torch.mean(h_gru, 1)

        # global max pooling

        max_pool, _ = torch.max(h_gru, 1)

        

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)

        conc = self.relu(self.linear(conc))

        conc = self.dropout(conc)

        out = self.out(conc)

        return out
def MultiClassTrainingInference(model, batch_size=512, epochs=5):

    n_epochs = epochs # how many times to iterate over all samples

    # matrix for the out-of-fold predictions

    train_preds = np.zeros((len(train_df)))

    # matrix for the predictions on the test set

    test_preds = np.zeros((len(test_df)))

    x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()

    test = torch.utils.data.TensorDataset(x_test_cuda)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=10).split(x_train, y_train))

    for i, (train_idx, valid_idx) in enumerate(splits):    

        # split data in train / validation according to the KFold indeces

        # also, convert them to a torch tensor and store them on the GPU (done with .cuda())

        x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()

        y_train_fold = torch.tensor(y_dumm[train_idx, np.newaxis], dtype=torch.float32).cuda()

        x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()

        y_val_fold = torch.tensor(y_dumm[valid_idx, np.newaxis], dtype=torch.float32).cuda()



        # make sure everything in the model is running on the GPU

        model.cuda()

        # define cross entropy loss

        # for numerical stability in the loss

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        optimizer = torch.optim.Adam(model.parameters())



        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)



        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)



        print(f'Fold {i + 1}')



        for epoch in range(n_epochs):

            # set train mode of the model. This enables operations which are only applied during training like dropout

            start_time = time.time()

            model.train()

            avg_loss = 0.  

            for x_batch, y_batch in tqdm(train_loader, disable=True):

                # Forward pass: compute predicted y by passing x to the model.

                y_pred = model(x_batch)



                # Compute and print loss.

                loss = loss_fn(y_pred, y_batch.view(-1,3).max(1)[1])



                # Before the backward pass, use the optimizer object to zero all of the

                # gradients for the Tensors it will update (which are the learnable weights

                # of the model)

                optimizer.zero_grad()



                # Backward pass: compute gradient of the loss with respect to model parameters

                loss.backward()



                # Calling the step function on an Optimizer makes an update to its parameters

                optimizer.step()

                avg_loss += loss.item() / len(train_loader)



            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout

            model.eval()



            # predict all the samples in y_val_fold batch per batch

            valid_preds_fold = np.zeros((x_val_fold.size(0)))

            test_preds_fold = np.zeros((len(test_df)))



            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(valid_loader):

                y_pred = model(x_batch).detach()



                avg_val_loss += loss_fn(y_pred, y_batch.view(-1,3).max(1)[1]).item() / len(valid_loader)

                

                _, pred_label = torch.max(y_pred.cpu().data, 1)

                

                valid_preds_fold[i * batch_size:(i+1) * batch_size] = pred_label



            elapsed_time = time.time() - start_time 

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))



        # predict all samples in the test set batch per batch

        for i, (x_batch,) in enumerate(test_loader):

            y_pred = model(x_batch).detach()

            

            _, pred_label = torch.max(y_pred.cpu().data, 1)





            test_preds_fold[i * batch_size:(i+1) * batch_size] = pred_label



        train_preds[valid_idx] = valid_preds_fold

        test_preds += test_preds_fold / len(splits)

    return train_preds, test_preds
# del loss_fn

torch.cuda.empty_cache() 
model = BiGRUClassifier()

train_oofpreds, test_oofpreds = MultiClassTrainingInference(model)
predictions = (train_oofpreds)

print(f"Confusion Matrix \n{classification_report(y_train, predictions)}")

print(f"Accuracy {accuracy_score(y_train, predictions)}")