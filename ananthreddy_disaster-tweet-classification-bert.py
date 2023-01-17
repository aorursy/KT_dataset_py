# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print (train_data.shape)

train_data.head()
train_data.isnull().sum()
train_data.fillna(value = 'None', inplace = True)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print (sample_submission.shape)

sample_submission.head()
import torch



if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

    

print (device)
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
from tqdm import tqdm, trange



train_values = []

for i in trange(len(train_data)):

    keyword = train_data['keyword'][i]

    location = train_data['location'][i]

    text = train_data['text'][i]

    

    sentence = keyword+' '+location+' '+text

    train_values.append(sentence)
train_values = ['[CLS] '+sent+' [SEP]' for sent in train_values]

tokenized_statements = [tokenizer.tokenize(sent) for sent in train_values]

print (tokenized_statements[0])
from keras.preprocessing.sequence import pad_sequences



input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_statements]

input_ids = pad_sequences(input_ids, maxlen = 512, dtype = 'long', truncating = 'post', padding = 'post')
attention_masks = []



for sent in input_ids:

    att_mask = [int(token_id>0) for token_id in sent]

    attention_masks.append(att_mask)
labels = train_data['target'].values
from sklearn.model_selection import  train_test_split



train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state = 42, test_size = 0.15)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state = 42, test_size = 0.15)
train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)



train_labels = torch.tensor(train_labels)

validation_labels = torch.tensor(validation_labels)



train_masks = torch.tensor(train_masks)

validation_masks = torch.tensor(validation_masks)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



batch_size = 8



train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)



validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = RandomSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler = validation_sampler, batch_size = batch_size)
from transformers import BertForSequenceClassification, AdamW, BertConfig



model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2, output_attentions = False, 

                                                      output_hidden_states = False)

model.cuda()
params = list(model.named_parameters())



print ('The BERT model has {:} different named parameters.\n'.format(len(params)))

print ('===== Embedding Layer =======\n')



for p in params[0:5]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



print('\n==== First Transformer ====\n')



for p in params[5:21]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



print('\n==== Output Layer ====\n')



for p in params[-4:]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters(),

                  lr = 2e-5, # args.learning_rate

                  eps = 1e-8 # args.adam_epsilon

                )
from transformers import get_linear_schedule_with_warmup



epochs = 5



total_steps = len(train_dataloader)*epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
import numpy as np



def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis = 1).flatten()

    labels_flat = labels.flatten()

    return (np.sum(pred_flat == labels_flat)/len(labels_flat))
import time

import datetime



def format_time(elapsed):

    '''

    Takes a time in seconds and returns a string hh:mm:ss

    '''

    # Round to the nearest second.

    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))
import time

import random



seed_val = 42



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)



loss_values = []



for epoch_i in range(0, epochs):

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    print('Training...')

    

    t0 = time.time()

    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.

        if step%40 == 0 and not step == 0:

            elapsed = format_time(time.time())

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            

        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)



        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)



        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()



        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        

    avg_train_loss = total_loss/len(train_dataloader)

    loss_values.append(avg_train_loss)

    

    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    

    print("")

    print("Running Validation...")

    

    t0 = time.time()

    model.eval()

    

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    

    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        with torch.no_grad():

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    

print("")

print("Training complete!")
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Use plot styling from seaborn.

sns.set(style='darkgrid')



# Increase the plot size and font size.

sns.set(font_scale=1.5)

plt.rcParams["figure.figsize"] = (12,6)



# Plot the learning curve.

plt.plot(loss_values, 'b-o')



# Label the plot.

plt.title("Training loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")



plt.show()
real_test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print (real_test_data.shape)

real_test_data.head()
real_test_data.isnull().sum()
real_test_data.fillna(value = 'None', inplace = True)
from tqdm import tqdm, trange



test_values = []

for i in trange(len(real_test_data)):

    keyword = real_test_data['keyword'][i]

    location = real_test_data['location'][i]

    text = real_test_data['text'][i]

    

    sentence = keyword+' '+location+' '+text

    test_values.append(sentence)
test_values = ['[CLS] '+sent+' [SEP]' for sent in test_values]

test_tokenized_statements = [tokenizer.tokenize(sent) for sent in test_values]

print (test_tokenized_statements[0])
from keras.preprocessing.sequence import pad_sequences



test_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in test_tokenized_statements]

test_input_ids = pad_sequences(test_input_ids, maxlen = 512, dtype = 'long', truncating = 'post', padding = 'post')
test_attention_masks = []



for sent in test_input_ids:

    att_mask = [int(token_id>0) for token_id in sent]

    test_attention_masks.append(att_mask)
test_inputs = torch.tensor(test_input_ids)

test_masks = torch.tensor(test_attention_masks)
batch_size = 8



test_data = TensorDataset(test_inputs, test_masks)

test_sampler = SequentialSampler(test_data)

test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)
# print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))



model.eval()



predictions , true_labels = [], []



# Predict 

for batch in test_dataloader:

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask = batch



    with torch.no_grad():

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)



    logits = outputs[0]

    logits = logits.detach().cpu().numpy()

    

    predictions.append(logits)

print('DONE.')
preds = []

for i in range(len(predictions)):

    for j in range(len(predictions[i])):

        preds.append(np.argmax(predictions[i][j]))
submission = pd.DataFrame(columns = ['id', 'target'])

submission['id'] = real_test_data['id']

submission['target'] = preds
print (submission.shape)

submission.head()
submission.to_csv('submission.csv', index = False)