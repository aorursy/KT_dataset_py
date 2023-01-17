!pip install transformers
import random

import numpy as np

import os

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,

                              TensorDataset)

from sklearn.metrics import accuracy_score, f1_score



from transformers import *
# Load dataframes

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
# We'll use 80% of the training data set to 

# actually train the model, and use the rest as validation set

msk = np.random.rand(len(train_df)) < 0.8

train = train_df[msk]

valid = train_df[~msk]
# InputExample is one unit of data, with an id, text, and label

class InputExample(object):

    def __init__(self, id, text, label=None):

        self.id = id

        self.text = text

        self.label = label



# InputData is one unit of features that we'll actually feed into BERT,

# with an ID, input_ids, input_mask, segment_ids, and label (target value)

class InputData(object):

    def __init__(self,

                 example_id,

                 choices_features,

                 label



                 ):

        self.example_id = example_id

        _, input_ids, input_mask, segment_ids = choices_features[0]

        self.choices_features = {

            'input_ids': input_ids,

            'input_mask': input_mask,

            'segment_ids': segment_ids

        }

        self.label = label



# Generate examples from the dataframe

def read_examples(df, is_training):

    if not is_training:

        df['target'] = np.zeros(len(df), dtype=np.int64)

    examples = []

    for val in df[['id', 'text', 'target']].values:

        examples.append(InputExample(id=val[0], text=val[1], label=val[2]))

    return examples, df



def select_field(features, field):

    return [

        feature.choices_features[field] for feature in features

    ]
# Generate the input_ids, input_mask, and segment_ids as described above

# Note that, for now, we're considering every tweet to be only one sentence.

def convert_examples_to_features(examples, tokenizer, max_seq_length,

                                 is_training):

    features = []

    for example_index, example in enumerate(examples):



        text = tokenizer.tokenize(example.text)

        MAX_TEXT_LEN = max_seq_length - 2 

        text = text[:MAX_TEXT_LEN]



        choices_features = []



        tokens = ["[CLS]"] + text + ["[SEP]"]  

        segment_ids = [0] * (len(text) + 2) 

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)



        padding_length = max_seq_length - len(input_ids)

        input_ids += ([0] * padding_length)

        input_mask += ([0] * padding_length)

        segment_ids += ([0] * padding_length)

        choices_features.append((tokens, input_ids, input_mask, segment_ids))



        label = example.label



        features.append(

            InputData(

                example_id=example.id,

                choices_features=choices_features,

                label=label

            )

        )

    return features
# The BERT tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Hyperparameters

max_seq_length = 512  

learning_rate = 1e-5  

num_epochs = 2  

batch_size = 8 
# Training

train_examples, train_df = read_examples(train, is_training=True)

labels = train_df['target'].astype(int).values

train_features = convert_examples_to_features(

    train_examples, tokenizer, max_seq_length, True)

train_input_ids = torch.tensor(select_field(train_features, 'input_ids'))

train_input_mask = torch.tensor(select_field(train_features, 'input_mask'))

train_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'))

train_label = torch.tensor([f.label for f in train_features])



# Validation

valid_examples, valid_df = read_examples(valid, is_training=True)

labels = valid_df['target'].astype(int).values

valid_features = convert_examples_to_features(

    valid_examples, tokenizer, max_seq_length, True)

valid_input_ids = torch.tensor(select_field(valid_features, 'input_ids'))

valid_input_mask = torch.tensor(select_field(valid_features, 'input_mask'))

valid_segment_ids = torch.tensor(select_field(valid_features, 'segment_ids'))

valid_label = torch.tensor([f.label for f in valid_features])



# Test

test_examples, test_df = read_examples(test, is_training=False)

test_features = convert_examples_to_features(

    test_examples, tokenizer, max_seq_length, True)

test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)

test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)

test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)



# Torch datasets

train = torch.utils.data.TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)

valid = torch.utils.data.TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)

test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)



# Torch dataloaders

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):

    def __init__(self, hidden_size = 768, num_classes = 2):

        super(NeuralNet, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased',  

                                        output_hidden_states=True,

                                        output_attentions=True)



        for param in self.bert.parameters():

            param.requires_grad = True

        

        self.drop_out = nn.Dropout() # dropout layer to prevent overfitting

        self.fc = nn.Linear(hidden_size, num_classes) # fully connected layer

        

    def forward(self, input_ids, input_mask, segment_ids):

        

        last_hidden_state, pooler_output, all_hidden_states, all_attentions = self.bert(input_ids, 

                                                                                        token_type_ids = segment_ids,

                                                                                        attention_mask = input_mask)

        last_hidden_state = last_hidden_state[:, 0,:]                                                       

        

        # Linear layer expects a tensor of size [batch size, input size]

        out = self.drop_out(last_hidden_state) 

        out = self.fc(out) 

        return F.log_softmax(out)
model = NeuralNet()

model.cuda() # Send the model to the GPU
# Training the model

model.train()

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_step = len(train_loader)

for epoch in range(num_epochs):

    train_loss = 0.

    

    for i, batch in enumerate(train_loader):

        batch = tuple(t.cuda() for t in batch)

        x_ids, x_mask, x_sids, y_truth = batch

        y_pred = model(x_ids, x_mask, x_sids)

        loss = loss_fn(y_pred, y_truth)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item() / len(train_loader)



        total = len(y_truth)

        _, predicted = torch.max(y_pred.data, 1)

        correct = (predicted == y_truth).sum().item()



        # Uncomment to get status updates during training

        '''

        if (i + 1) % 50 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),

                          (correct / total) * 100))

        '''     
# Test the model on the validation set

model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for i, batch in enumerate(valid_loader):

        batch = tuple(t.cuda() for t in batch)

        #batch = tuple(t for t in batch)

        x_ids, x_mask, x_sids, y_truth = batch

        y_pred = model(x_ids, x_mask, x_sids)

        _, predicted = torch.max(y_pred.data, 1)

        total += len(labels)

        correct += (predicted == y_truth).sum().item()



    print('Test Accuracy of the model on the validation set: {} %'.format((correct / total) * 100))

test_preds = []

with torch.no_grad():

        for i, batch in enumerate(test_loader):

            batch = tuple(t.cuda() for t in batch)

            x_ids, x_mask, x_sids = batch

            y_pred = model(x_ids, x_mask, x_sids).detach()

            test_preds[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()  

      
sample['target'] = np.argmax(test_preds, axis = 1)

sample.to_csv('submission.csv', index = False)
from sklearn.metrics import confusion_matrix



# Load dataframes

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



# We'll use 80% of the training data set to 

# actually train the model, and use the rest as validation set

msk = np.random.rand(len(train_df)) < 0.8

train = train_df[msk]

valid = train_df[~msk]
batch_size = 1



# Training

train_examples, train_df = read_examples(train, is_training=True)

labels = train_df['target'].astype(int).values

train_features = convert_examples_to_features(

    train_examples, tokenizer, max_seq_length, True)

train_input_ids = torch.tensor(select_field(train_features, 'input_ids'))

train_input_mask = torch.tensor(select_field(train_features, 'input_mask'))

train_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'))

train_label = torch.tensor([f.label for f in train_features])



# Validation

valid_examples, valid_df = read_examples(valid, is_training=True)

labels = valid_df['target'].astype(int).values

valid_features = convert_examples_to_features(

    valid_examples, tokenizer, max_seq_length, True)

valid_input_ids = torch.tensor(select_field(valid_features, 'input_ids'))

valid_input_mask = torch.tensor(select_field(valid_features, 'input_mask'))

valid_segment_ids = torch.tensor(select_field(valid_features, 'segment_ids'))

valid_label = torch.tensor([f.label for f in valid_features])



# Test

test_examples, test_df = read_examples(test, is_training=False)

test_features = convert_examples_to_features(

    test_examples, tokenizer, max_seq_length, True)

test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)

test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)

test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)



# Torch datasets

train = torch.utils.data.TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)

valid = torch.utils.data.TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)

test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)



# Torch dataloaders

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
Bert = BertModel.from_pretrained('bert-base-uncased',  

                                    output_hidden_states=True,

                                    output_attentions=True)

Bert.cuda()



# These will be used to generate training and testing dataframes

# to train the logistic regression model on

total_step = len(train_loader)

train_rows = []



with torch.no_grad():

        for i, batch in enumerate(train_loader):

            batch = tuple(t.cuda() for t in batch)

            x_ids, x_mask, x_sids, y_truth = batch

            last_hidden_state, _, _, _ = Bert(x_ids, 

                                                token_type_ids = x_sids,

                                              attention_mask = x_mask)

            train_rows.append(np.array(last_hidden_state[:, 0, :].cpu()))

            

            # Uncomment to get status updates

            '''

            if (i + 1) % 50 == 0:

                 print('Step [{}/{}]'.format( i + 1, total_step))

            '''          
total_step = len(valid_loader)

valid_rows = []



with torch.no_grad():

        for i, batch in enumerate(valid_loader):

            batch = tuple(t.cuda() for t in batch)

            x_ids, x_mask, x_sids, y_truth = batch

            last_hidden_state, _, _, _ = Bert(x_ids, 

                                                token_type_ids = x_sids,

                                              attention_mask = x_mask)

            valid_rows.append(np.array(last_hidden_state[:, 0, :].cpu()))

            

            # Uncomment to get status updates

            '''

            if (i + 1) % 50 == 0:

                print('Step [{}/{}]'.format( i + 1, total_step))  

            '''
total_step = len(test_loader)

test_rows = []



with torch.no_grad():

        for i, batch in enumerate(test_loader):

            batch = tuple(t.cuda() for t in batch)

            x_ids, x_mask, x_sids = batch

            last_hidden_state, _, _, _ = Bert(x_ids, 

                                                token_type_ids = x_sids,

                                              attention_mask = x_mask)

            test_rows.append(np.array(last_hidden_state[:, 0, :].cpu()))

            

            # Uncomment to get status updates

            '''

            if (i + 1) % 50 == 0:

                print('Step [{}/{}]'.format( i + 1, total_step))  

            '''
# Train dataframe

train_df2 = pd.DataFrame(np.concatenate(train_rows))

train_df2['target'] = np.array(train_label)



# Validation dataframe

valid_df2 = pd.DataFrame(np.concatenate(valid_rows))

valid_df2['target'] = np.array(valid_label)



# Test dataframe

test_df2 = pd.DataFrame(np.concatenate(test_rows))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

logreg = LogisticRegression()
X = train_df2.drop('target', axis = 1)

y = train_df2['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg.fit(X_train, y_train)
X_val = valid_df2.drop('target', axis = 1)

y_val = valid_df2['target']

y_val_pred = logreg.predict(X_val)

confusion_matrix(y_val, y_val_pred)
# Generate the submission

y_sub = logreg.predict(test_df2)
sample['target'] = y_sub

#sample.to_csv('submission.csv', index = False)