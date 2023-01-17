# Setting UP torch with GPU

import torch

# If there's a GPU available...

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import torch.nn.functional as F

import seaborn as sns

from collections import Counter

from sklearn.preprocessing import OrdinalEncoder

from fancyimpute import KNN

from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig

from transformers import get_linear_schedule_with_warmup

import time

import datetime

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("../input/nlp-getting-started/train.csv")

test=pd.read_csv("../input/nlp-getting-started/test.csv")

sub=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train.head(10)
test.head(10)
sub.head(10)
# Checking the Missing Values

sns.heatmap(train.isnull())
train.isnull().sum()/len(train)*100
train.describe()
train.shape
test.shape
# Checking the Distribution of the Target

train['target'].value_counts().plot(kind='bar',color=['G','R'])

plt.xlabel('Target')

plt.ylabel('Total Count')

plt.title('Target Distribution')

plt.show()
# Checking the Target Percentages

train['target'].value_counts(normalize=True)*100
# Checking the Most Common Keywords

y=Counter(train['keyword']).most_common(25)

keyword=[i[0] for i in y]

keyword_count=[i[1] for i in y]

sns.barplot(keyword_count,keyword).set_title('KeyWord Distribution')
# Checking the Distribution of Location

y=Counter(train['location']).most_common(20)

location=[i[0] for i in y]

location_count=[i[1] for i in y]

sns.barplot(location_count,location).set_title("Location Distribution")
# Checking the Correlation of Features

sns.pairplot(train)
# Encoding the Train Data

#instantiate both packages to use

encoder = OrdinalEncoder()

# create a list of categorical columns to iterate over

cat_cols = ['keyword','location']



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data



#create a for loop to iterate through each column in the data

for columns in cat_cols:

    encode(train[columns])
# Encoding the Test Columns

#create a for loop to iterate through each column in the data

for columns in cat_cols:

    encode(test[columns])
# impute data and convert

imputer = KNN(3) #use 2 nearest rows which have a feature to fill in each row’s missing features

encode_data_train = pd.DataFrame(np.round(imputer.fit_transform(train[['keyword','location']])),columns = ['keyword','location'])
# impute data and convert

imputer = KNN(3) #use 2 nearest rows which have a feature to fill in each row’s missing features

encode_data_test = pd.DataFrame(np.round(imputer.fit_transform(test[['keyword','location']])),columns = ['keyword','location'])
# Dropping the Unncessary Columns from Both Train and Test

train.drop(columns=['keyword','location'],inplace=True)

test.drop(columns=['keyword','location'],inplace=True)
# Concatenating the Encoded and Train Features

train=pd.concat([train,encode_data_train],axis=1)

test=pd.concat([test,encode_data_test],axis=1)
train.head(10)
test.head(10)
# Dropping the Id column Feature

train.drop(columns=['id'],inplace=True)

test.drop(columns=['id'],inplace=True)
# Getting the list of sentences and Corresponding Labels

sentences=train.text.values

labels=train.target.values
# Load the BERT tokenizer.

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Tokenize all of the sentences and map the tokens to thier word IDs.

input_ids = []

# For every sentence...

for sent in sentences:

    # `encode` will:

    #   (1) Tokenize the sentence.

    #   (2) Prepend the `[CLS]` token to the start.

    #   (3) Append the `[SEP]` token to the end.

    #   (4) Map tokens to their IDs.

    encoded_sent = tokenizer.encode(

                        sent,                      # Sentence to encode.

                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        # This function also supports truncation and conversion

                        # to pytorch tensors, but we need to do padding, so we

                        # can't use these features :( .

                        #max_length = 128,          # Truncate all sentences.

                        #return_tensors = 'pt',     # Return pytorch tensors.

                   )

    

    # Add the encoded sentence to the list.

    input_ids.append(encoded_sent)

# Print sentence 0, now as a list of IDs.

print('Original: ', sentences[0])

print('Token IDs:', input_ids[0])
print('Max sentence length: ', max([len(sen) for sen in input_ids]))
# Set the maximum sequence length.

# I've chosen 64 somewhat arbitrarily. It's slightly larger than the

# maximum training sentence length of 47...

MAX_LEN = 104

print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

# Pad our input tokens with value 0.

# "post" indicates that we want to pad and truncate at the end of the sequence,

# as opposed to the beginning.

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 

                          value=0, truncating="post", padding="post")

print('\Done.')
# Create attention masks

attention_masks = []

# For each sentence...

for sent in input_ids:

    

    # Create the attention mask.

    #   - If a token ID is 0, then it's padding, set the mask to 0.

    #   - If a token ID is > 0, then it's a real token, set the mask to 1.

    att_mask = [int(token_id > 0) for token_id in sent]

    

    # Store the attention mask for this sentence.

    attention_masks.append(att_mask)
# Use train_test_split to split our data into train and validation sets for

# training

# Use 90% for training and 10% for validation.

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=2018, test_size=0.1)

# Do the same for the masks.

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,

                                             random_state=2018, test_size=0.1)
# Convert all inputs and labels into torch tensors, the required datatype 

# for our model.

train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)

validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)

validation_masks = torch.tensor(validation_masks)
# The DataLoader needs to know our batch size for training, so we specify it 

# here.

# For fine-tuning BERT on a specific task, the authors recommend a batch size of

# 16 or 32.

batch_size = 32

# Create the DataLoader for our training set.

train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



# Load BertForSequenceClassification, the pretrained BERT model with a single 

# linear classification layer on top. 

model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.

    num_labels = 2, # The number of output labels--2 for binary classification.

                    # You can increase this for multi-class tasks.   

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)

# Tell pytorch to run this model on the GPU.

model.cuda()
# Get all of the model's parameters as a list of tuples.

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 

# I believe the 'W' stands for 'Weight Decay fix"

optimizer = AdamW(model.parameters(),

                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5

                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                )

# Number of training epochs (authors recommend between 2 and 4)

epochs = 4

# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):

    '''

    Takes a time in seconds and returns a string hh:mm:ss

    '''

    # Round to the nearest second.

    elapsed_rounded = int(round((elapsed)))

    

    # Format as hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))
# This training code is based on the `run_glue.py` script here:

# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.

seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.

loss_values = []

# For each epoch...

for epoch_i in range(0, epochs):

    

    # ========================================

    #               Training

    # ========================================

    

    # Perform one full pass over the training set.

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    print('Training...')

    # Measure how long the training epoch takes.

    t0 = time.time()

    # Reset the total loss for this epoch.

    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 

    # `train` just changes the *mode*, it doesn't *perform* the training.

    # `dropout` and `batchnorm` layers behave differently during training

    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)

    model.train()

    # For each batch of training data...

    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.

        if step % 40 == 0 and not step == 0:

            # Calculate elapsed time in minutes.

            elapsed = format_time(time.time() - t0)

            

            # Report progress.

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 

        #

        # As we unpack the batch, we'll also copy each tensor to the GPU using the 

        # `to` method.

        #

        # `batch` contains three pytorch tensors:

        #   [0]: input ids 

        #   [1]: attention masks

        #   [2]: labels 

        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a

        # backward pass. PyTorch doesn't do this automatically because 

        # accumulating the gradients is "convenient while training RNNs". 

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)

        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).

        # This will return the loss (rather than the model output) because we

        # have provided the `labels`.

        # The documentation for this `model` function is here: 

        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

        outputs = model(b_input_ids, 

                    token_type_ids=None, 

                    attention_mask=b_input_mask, 

                    labels=b_labels)

        

        # The call to `model` always returns a tuple, so we need to pull the 

        # loss value out of the tuple.

        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can

        # calculate the average loss at the end. `loss` is a Tensor containing a

        # single value; the `.item()` function just returns the Python value 

        # from the tensor.

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.

        loss.backward()

        # Clip the norm of the gradients to 1.0.

        # This is to help prevent the "exploding gradients" problem.

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.

        # The optimizer dictates the "update rule"--how the parameters are

        # modified based on their gradients, the learning rate, etc.

        optimizer.step()

        # Update the learning rate.

        scheduler.step()

    # Calculate the average loss over the training data.

    avg_train_loss = total_loss / len(train_dataloader)            

    

    # Store the loss value for plotting the learning curve.

    loss_values.append(avg_train_loss)

    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        

    # ========================================

    #               Validation

    # ========================================

    # After the completion of each training epoch, measure our performance on

    # our validation set.

    print("")

    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently

    # during evaluation.

    model.eval()

    # Tracking variables 

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch

    for batch in validation_dataloader:

        

        # Add batch to GPU

        batch = tuple(t.to(device) for t in batch)

        

        # Unpack the inputs from our dataloader

        b_input_ids, b_input_mask, b_labels = batch

        

        # Telling the model not to compute or store gradients, saving memory and

        # speeding up validation

        with torch.no_grad():        

            # Forward pass, calculate logit predictions.

            # This will return the logits rather than the loss because we have

            # not provided labels.

            # token_type_ids is the same as the "segment ids", which 

            # differentiates sentence 1 and 2 in 2-sentence tasks.

            # The documentation for this `model` function is here: 

            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            outputs = model(b_input_ids, 

                            token_type_ids=None, 

                            attention_mask=b_input_mask)

        

        # Get the "logits" output by the model. The "logits" are the output

        # values prior to applying an activation function like the softmax.

        logits = outputs[0]

        # Move logits and labels to CPU

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        

        # Calculate the accuracy for this batch of test sentences.

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        

        # Accumulate the total accuracy.

        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")

print("Training complete!")
# Saving the Model

torch.save(model.state_dict(),'model.bin')
import plotly.express as px

f = pd.DataFrame(loss_values)

f.columns=['Loss']

fig = px.line(f, x=f.index, y=f.Loss)

fig.update_layout(title='Training loss of the Model',

                   xaxis_title='Epoch',

                   yaxis_title='Loss')

fig.show()
# Preparing the Testing Dataset

# Create sentence and label lists

MAX_LEN=104

sentences = test.text.values

# Tokenize all of the sentences and map the tokens to thier word IDs.

input_ids = []

# For every sentence...

for sent in sentences:

    # `encode` will:

    #   (1) Tokenize the sentence.

    #   (2) Prepend the `[CLS]` token to the start.

    #   (3) Append the `[SEP]` token to the end.

    #   (4) Map tokens to their IDs.

    encoded_sent = tokenizer.encode(

                        sent,                      # Sentence to encode.

                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                   )

    

    input_ids.append(encoded_sent)

# Pad our input tokens

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 

                          dtype="long", truncating="post", padding="post")

# Create attention masks

attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask) 

# Convert to tensors.

prediction_inputs = torch.tensor(input_ids)

prediction_masks = torch.tensor(attention_masks)

# Set the batch size.  

batch_size = 32  

# Create the DataLoader.

prediction_data = TensorDataset(prediction_inputs, prediction_masks)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode

model.eval()

# Tracking variables 

predictions= []

# Predict

for batch in prediction_dataloader:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  

  # Unpack the inputs from our dataloader

  b_input_ids, b_input_mask = batch

  

  # Telling the model not to compute or store gradients, saving memory and 

  # speeding up prediction

  with torch.no_grad():

      # Forward pass, calculate logit predictions

      outputs = model(b_input_ids, token_type_ids=None, 

                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU

  logits = logits.detach().cpu().numpy()

  

  output=F.softmax(torch.tensor(logits))

  # Store predictions

  

  predictions.append(output)

    

print('DONE.')
test_y=[]

for prediction in predictions:

    prediction=prediction.detach().cpu().numpy()

    for pred in prediction:

        test_y.append(np.argmax(pred))
sub['target']=test_y
sub.to_csv("submission.csv",index=False)