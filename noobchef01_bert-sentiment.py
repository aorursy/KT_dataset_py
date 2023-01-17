# imports

import torch



import random

import numpy as np
# seed

SEED = 1234



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
# pre-trained model

from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# special tokens

init_token = tokenizer.cls_token

eos_token = tokenizer.sep_token

pad_token = tokenizer.pad_token

unk_token = tokenizer.unk_token



# special tokens numericalized

init_token_idx = tokenizer.cls_token_id

eos_token_idx = tokenizer.sep_token_id

pad_token_idx = tokenizer.pad_token_id

unk_token_idx = tokenizer.unk_token_id



# max sequence length

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']



# tokenize and cut to max length

def tokenize_and_cut(sentence):

    tokens = tokenizer.tokenize(sentence) 

    tokens = tokens[:max_input_length-2]

    return tokens
# set fields

from torchtext import data



# transformer expects in batch first

TEXT = data.Field(batch_first = True,

                  use_vocab = False,

                  tokenize = tokenize_and_cut,

                  preprocessing = tokenizer.convert_tokens_to_ids,

                  init_token = init_token_idx,

                  eos_token = eos_token_idx,

                  pad_token = pad_token_idx,

                  unk_token = unk_token_idx)



LABEL = data.LabelField(dtype = torch.long)



from torchtext import datasets



train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)



train_data, valid_data = train_data.split(random_state = random.seed(SEED))
LABEL.build_vocab(train_data)



BATCH_SIZE = 10



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data), 

    batch_size = BATCH_SIZE, 

    device = device)
from transformers import BertForSequenceClassification, AdamW, BertConfig



model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.

    num_labels = 2, # The number of output labels--2 for binary classification.

                    # You can increase this for multi-class tasks.   

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)
# model.to(device)

# batch = next(iter(train_iterator))

# mask = (batch.text != tokenizer.pad_token_id)

        

# outputs = model(batch.text, token_type_ids=None, attention_mask=mask, labels=batch.label)[:2]
# (outputs[0].shape)

# outputs[0]



# outputs[1].shape

# outputs[1]
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = AdamW(model.parameters(),

                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5

                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                )
from transformers import get_linear_schedule_with_warmup



# Number of training epochs (authors recommend between 2 and 4)

epochs = 4



# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_iterator) * epochs

# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)

model = model.to(device)



def binary_accuracy(pred_labels, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """

    correct = (pred_labels == y).float() #cimport torch.nn as nnimport torch.nn as nnonvert into float for division 

    acc = correct.sum() / len(correct)

    return acc



def train(model, iterator, optimizer, scheduler):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        

        optimizer.zero_grad()

        

        # pad masking

        mask = (batch.text != tokenizer.pad_token_id)

        

        outputs = model(batch.text, token_type_ids=None, attention_mask=mask, labels=batch.label)[:2]

        

        loss = outputs[0]

        

        acc = binary_accuracy(outputs[1].argmax(1), batch.label)

        

        loss.backward()

        

        optimizer.step()

        scheduler.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    with torch.no_grad():

    

        for batch in iterator:

            

            mask = (batch.text != tokenizer.pad_token_id)

            outputs = model(batch.text, token_type_ids=None, attention_mask=mask, labels=batch.label)

        

            loss = outputs[0]

        

            acc = binary_accuracy(outputs[1].argmax(1), batch.label)



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 2



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    

    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iterator, optimizer, scheduler)

    valid_loss, valid_acc = evaluate(model, valid_iterator)

        

    end_time = time.time()

        

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'bert-imdb-sent.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
model.load_state_dict(torch.load('bert-imdb-sent.pt'))



test_loss, test_acc = evaluate(model, test_iterator)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
def predict_sentiment(model, tokenizer, sentence):

    model.eval()

    tokens = tokenizer.tokenize(sentence)

    tokens = tokens[:max_input_length-2]

    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(0)

    prediction = model(tensor)[0].argmax(1)

    return "positive" if prediction.item() == 1 else "negative"
#print(predict_sentiment(model, tokenizer, "Why sad brother"))



print(predict_sentiment(model, tokenizer, "this movie had bad acting and the director was incompetent"))
import os
os.listdir()