# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
!pip install transformers
import pandas as pd

df=pd.read_csv('/kaggle/input/new-addition/indonesian_train.csv')

df.head()
df.drop(['Unnamed: 0'],axis=1,inplace=True)
emotion_dict={'joy':4,'sadness':5,'fear':3,'trust':7,'anger':0,'anticipation':1,'disgust':2,'surprise':6}

df.head()
df['len']=df['content'].astype(str).apply(len)

df.head()
df.emotion.value_counts()
df.quantile(0.95,axis=0)
# Get the lists of sentences and their labels.

sentences = df.content.values

labels = df.emotion.values
from transformers import XLMRobertaTokenizer



# Load the XLMRoberta tokenizer.

print('Loading XLMRoberta tokenizer...')

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# Print the original sentence.

print(' Original: ', sentences[0])

print('\n')

# Print the sentence split into tokens.

print('Tokenized: ', tokenizer.tokenize(sentences[0]))

print('\n')

# Print the sentence mapped to token ids.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
ls=[]

for i in range(len(df)):

    ls.append(len(tokenizer.tokenize(sentences[i])))

temp=pd.DataFrame()

temp['len_token']=ls

temp.quantile(0.99,axis=0)
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

print('\n')

h=[len(i) for i in input_ids]

h.sort(reverse=True)

print('Total no lengths',len(h))

print('\n')

print('Top 70 tokens having highest lenght',h[:50])

# We'll borrow the `pad_sequences` utility function to do this.

from keras.preprocessing.sequence import pad_sequences



# Set the maximum sequence length.

# I've chosen 300 somewhat arbitrarily.

MAX_LEN = 300



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
# Use train_test_split to split our data into train and validation sets for

# training

from sklearn.model_selection import train_test_split



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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



# The DataLoader needs to know our batch size for training, so we specify it 

# here.

# For fine-tuning BERT on a specific task, the authors recommend a batch size of

# 16 or 32 or less



batch_size = 4



# Create the DataLoader for our training set.

train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



# Create the DataLoader for our validation set.

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

from transformers import BertForSequenceClassification, AdamW, BertConfig,XLMRobertaForSequenceClassification



# Load BertForSequenceClassification, the pretrained BERT model with a single 

# linear classification layer on top. 

model = XLMRobertaForSequenceClassification.from_pretrained(

    "xlm-roberta-base", # Use the 12-layer XLMRoberta,

    num_labels = 8, # The number of output labels--8 for multiclass classification.

                       

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)



# Tell pytorch to run this model on the GPU.

model.cuda()
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 

# I believe the 'W' stands for 'Weight Decay fix"

optimizer = AdamW(model.parameters(),

                  lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 1e-5

                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                )

from transformers import get_linear_schedule_with_warmup





epochs = 20



# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_dataloader) * epochs



# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)
import numpy as np



# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
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
%matplotlib inline

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
import pandas as pd



# Load the dataset into a pandas dataframe.

df = pd.read_csv("/kaggle/input/new-addition/indonesian_test.csv" )



# Report the number of sentences.

print('Number of test sentences: {:,}\n'.format(df.shape[0]))



# Create sentence and label lists

sentences = df.content.values

labels = df.emotion.values



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

prediction_labels = torch.tensor(labels)



# Set the batch size.  

batch_size = 2



# Create the DataLoader.

prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# Prediction on test set



print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))



# Put model in evaluation mode

model.eval()



# Tracking variables 

predictions , true_labels = [], []



# Predict 

for batch in prediction_dataloader:

    # Add batch to GPU

    batch = tuple(t.to(device) for t in batch)

  

  # Unpack the inputs from our dataloader

    b_input_ids, b_input_mask, b_labels = batch

  

  

  # Telling the model not to compute or store gradients, saving memory and 

  # speeding up prediction

    with torch.no_grad():# Forward pass, calculate logit predictions

        outputs = model(b_input_ids, token_type_ids=None, 

                      attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels

    predictions.append(logits)

    true_labels.append(label_ids)



print('    DONE.')

    

from sklearn.metrics import f1_score,precision_score,recall_score



f1_set = []

re_set=[]

pre_set=[]



# Evaluate each test batch 

print('Calculating  F1 ,Precision, Recall score  for each batch...')



# For each input batch...

for i in range(len(true_labels)):

  

  # The predictions for this batch are a 2-column ndarray (one column for "0" 

  # and one column for "1"). Pick the label with the highest value and turn this

  # in to a list of 0s and 1s.

    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

    # Calculate and store the coef for this batch.  

    justice = f1_score(true_labels[i], pred_labels_i,average='macro')    

    justice_re=recall_score(true_labels[i], pred_labels_i,average='macro')

    justice_pre=precision_score(true_labels[i], pred_labels_i,average='macro')

    

    f1_set.append(justice)

    re_set.append(justice_re)

    pre_set.append(justice_pre)
# Combine the predictions for each batch into a single list of 0s and 1s.

flat_predictions = [item for sublist in predictions for item in sublist]

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()



# Combine the correct labels for each batch into a single list.

flat_true_labels = [item for sublist in true_labels for item in sublist]



# Calculate the f1

f1 = f1_score(flat_true_labels, flat_predictions,average='macro')



rec = recall_score(flat_true_labels, flat_predictions,average='macro')



pre = precision_score(flat_true_labels, flat_predictions,average='macro')



print('F1 score for XLM robrta: %.3f' % f1)

print('Recall score for XLM robrta: %.3f' % rec)

print('Precision score for XLM robrta: %.3f' % pre)

import os



# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()



output_dir = '/model_save/'



# Create output directory if needed

if not os.path.exists(output_dir):

    os.makedirs(output_dir)



print("Saving model to %s" % output_dir)



# Save a trained model, configuration and tokenizer using `save_pretrained()`.

# They can then be reloaded using `from_pretrained()`

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

model_to_save.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)



# Good practice: save your training arguments together with the trained model

# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
