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
# read dataset

df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

# first five rows

df_train.head()

import tensorflow as tf



# Get the GPU device name.

device_name = tf.test.gpu_device_name()



# The device name should look like the following:

if device_name == '/device:GPU:0':

    print('Found GPU at: {}'.format(device_name))

else:

    raise SystemError('GPU device not found')
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
# Import relevant packages

import matplotlib.pyplot as plt

import nltk

import numpy as np

import re 



import string

import seaborn as sns



from nltk.corpus import stopwords  # Remove useless words

from nltk.stem.lancaster import LancasterStemmer  # Convert words to base form; aggressive

from sklearn.metrics import accuracy_score

from sklearn.metrics import matthews_corrcoef



# Import packages that help us to create document-term matrix

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# In case there is a bug, just download stopwords again.

nltk.download('stopwords')
set(stopwords.words('english'))
# Check percentage of disaster tweets

df_train.target.value_counts(normalize=True)
# Text preprocessing



# remove all numbers with letters

alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)



# '[%s]' % re.escape(string.punctuation),' ' - replace punctuation with white space

# .lower() - convert all strings to lowercase 

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())



# Remove all '\n' in the string and replace it with a space

remove_n = lambda x: re.sub("\n", " ", x)



# Remove all non-ascii characters 

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)



# Apply all the lambda functions wrote previously through .map on the comments column

df_train['text'] = df_train['text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

df_test['text'] = df_test['text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

df_train['text'][3]
!pip install transformers
# Setting source_text & target label

sentences = df_train.text.values

sentences_test  = df_test.text.values

labels = df_train.target.values
from transformers import BertTokenizer



# Load the BERT tokenizer.

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Check working of bert

# Print the original Text.

print(' Original: ', sentences[100])



# Print the text split into tokens.

print('Tokenized: ', tokenizer.tokenize(sentences[100]))



# Print the text mapped to token ids.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[100])))
def tocknize(text_input):

    # Tokenize all of the sentences and map the tokens to thier word IDs.

    input_ids = []



    # For every sentence...

    for sent in text_input:

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

    

    return input_ids
print('Max length of text in Train Set: ', max([len(sen) for sen in tocknize(sentences)]))

print('Max length of text in Test Set: ', max([len(sen) for sen in tocknize(sentences_test)]))
input_ids = tocknize(sentences)
# We'll borrow the `pad_sequences` utility function to do this.

from keras.preprocessing.sequence import pad_sequences



# Set the maximum sequence length.

# I've chosen 64 somewhat arbitrarily. It's slightly larger than the

# maximum training sentence length of 47...

MAX_LEN = 64



print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)



print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))



# Pad our input tokens with value 0.

# "post" indicates that we want to pad and truncate at the end of the sequence,

# as opposed to the beginning.

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 

                          value=0, truncating="post", padding="post")



print('\nDone.')
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
#print(input_ids)
# Use train_test_split to split our data into train and validation sets for

# training

from sklearn.model_selection import train_test_split



# Use 90% for training and 10% for validation. stratify=labels not used as value with 1 contains 90% in split

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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



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

from transformers import BertForSequenceClassification, AdamW, BertConfig



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

from transformers import get_linear_schedule_with_warmup



# Number of training epochs (authors recommend between 2 and 4)

epochs = 1



# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_dataloader) * epochs



# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)
# # Define a helper function for calculating accuracy.

# import numpy as np



# # Function to calculate the accuracy of our predictions vs labels

# def flat_accuracy(preds, labels):

#     pred_flat = np.argmax(preds, axis=1).flatten()

#     labels_flat = labels.flatten()

#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
# Helper function for formatting elapsed times.

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

# We're ready to kick off the training!

import random



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

    eval_loss, eval_accuracy,eval_mcc_accuracy = 0, 0, 0

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

        

        

        pred_flat = np.argmax(logits, axis=1).flatten()

        labels_flat = label_ids.flatten()

        tmp_eval_accuracy = accuracy_score(pred_flat, labels_flat)

        

# you are looking at a metric to measure and maximize the overall accuracy of the classification model,

# MCC seems to the best bet since it is not only easily interpretable but also robust

# to changes in the prediction goal.



        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

    

        eval_accuracy += tmp_eval_accuracy

        eval_mcc_accuracy += tmp_eval_mcc_accuracy

        nb_eval_steps += 1

    

    print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')

    print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

        

#         # Calculate the accuracy for this batch of test sentences.

#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        

#         # Accumulate the total accuracy.

#         eval_accuracy += tmp_eval_accuracy



        # Track the number of batches

#         nb_eval_steps += 1



#     # Report the final accuracy for this validation run.

#     print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

#     print("  Validation took: {:}".format(format_time(time.time() - t0)))



print("")

print("Training complete!")
import matplotlib.pyplot as plt





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
given_labels = np.array(df_submission['target'])

tensor_labels = torch.tensor(given_labels)

# tocken_generation

input_ids_test = tocknize(sentences_test)



# pad sequence function

def padding_seq(input_token):

    out_seq = pad_sequences(input_token, 

                            maxlen=MAX_LEN,

                            dtype="long", 

                            truncating="post", 

                            padding="post")

    return out_seq



# masking sequence function

def mask_atten(input_seq):

    # Create attention masks

    attention_masks = []



    



    # Create a mask of 1s for each token followed by 0s for padding

    for seq in input_seq:

        out_mask = [float(i>0) for i in seq]

        attention_masks.append(out_mask) 



    return attention_masks



test_pad = padding_seq(input_ids_test)

test_mask = mask_atten(test_pad)



# test dataset and mask to tensor

tensor_input = torch.tensor(test_pad)

tensor_mask = torch.tensor(test_mask)



BATCH_SIZE = 32 

prediction_data = TensorDataset(tensor_input, tensor_mask, tensor_labels)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = BATCH_SIZE)





print('Predicting labels for {:,} test sentences...'.format(len(tensor_input)))



# Put model in evaluation mode

model.eval()



# Tracking variables 

predictions , true_labels = [], []



# Predict 

for batch in prediction_dataloader:

    b_input_ids = batch[0].to(device)

    b_input_mask = batch[1].to(device)

    b_labels = batch[2].to(device)

    

    # Telling the model not to compute or store gradients, saving memory and 

    # speeding up prediction

    with torch.no_grad():

      # Forward pass, calculate logit predictions

        outputs = model(b_input_ids, token_type_ids=None,

                        attention_mask=b_input_mask)



    logits = outputs[0]



    # Move logits and labels to CPU

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()

  

    # Store predictions and true labels

    predictions.append(logits)

    true_labels.append(label_ids)



print('DONE.')



flat_predictions = [item for sublist in predictions for item in sublist]

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

df_submission['target'] = flat_predictions

df_submission.to_csv('submission.csv', index = False)

df_submission.head()