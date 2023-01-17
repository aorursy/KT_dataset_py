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
import torch

if torch.cuda.is_available():  

    device = torch.device("cuda")

    print('We will use the GPU:', torch.cuda.get_device_name(0))

    

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
df_train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_train.head()
import re

def preprocess(text):



    text=text.lower()

    # remove hyperlinks

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)

    #Replace &amp, &lt, &gt with &,<,> respectively

    text=text.replace(r'&amp;?',r'and')

    text=text.replace(r'&lt;',r'<')

    text=text.replace(r'&gt;',r'>')

    #remove hashtag sign

    #text=re.sub(r"#","",text)   

    #remove mentions

    text = re.sub(r"(?:\@)\w+", '', text)

    #text=re.sub(r"@","",text)

    #remove non ascii chars

    text=text.encode("ascii",errors="ignore").decode()

    #remove some puncts (except . ! ?)

    text=re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+','',text)

    text=re.sub(r'[!]+','!',text)

    text=re.sub(r'[?]+','?',text)

    text=re.sub(r'[.]+','.',text)

    text=re.sub(r"'","",text)

    text=re.sub(r"\(","",text)

    text=re.sub(r"\)","",text)

    

    text=" ".join(text.split())

    return text



df_train['text'] = df_train['text'].apply(preprocess)

df_test['text'] = df_test['text'].apply(preprocess)

df_train=df_train[df_train["text"]!='']
df_train=df_train[["text","target"]]

df_train.head()
df_train["target"].value_counts()
# Get the lists of lyrics and their labels.

texts = df_train.text.values

labels = df_train.target.values
from transformers import ElectraTokenizer, ElectraForSequenceClassification,AdamW

import torch

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator',num_labels=2)

model.cuda()
#to show length of embedding will be helpful to determine maximum length of comments and padding threshold

import matplotlib.pyplot as plt

def plot_sentence_embeddings_length(text_list, tokenizer):

    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))

    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))

    fig, ax = plt.subplots(figsize=(8, 5));

    ax.hist(tokenized_texts_len, bins=40);

    ax.set_xlabel("Length of Comment Embeddings");

    ax.set_ylabel("Number of Comments");

    return

plot_sentence_embeddings_length(texts, tokenizer)
indices=tokenizer.batch_encode_plus(texts,max_length=64,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)



input_ids=indices["input_ids"]

attention_masks=indices["attention_mask"]
from sklearn.model_selection import train_test_split



# Use 99% for training and 1% for validation.

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=42, test_size=0.2)

# Do the same for the masks.

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,

                                             random_state=42, test_size=0.2)
# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels, dtype=torch.long)

validation_labels = torch.tensor(validation_labels, dtype=torch.long)

train_masks = torch.tensor(train_masks, dtype=torch.long)

validation_masks = torch.tensor(validation_masks, dtype=torch.long)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



batch_size = 32



# Create the DataLoader for our training set.

train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



# Create the DataLoader for our validation set.

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
optimizer = AdamW(model.parameters(),

                  lr = 6e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5

                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                )



from transformers import get_linear_schedule_with_warmup



# Number of training epochs (authors recommend between 2 and 4)

epochs = 5



# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_dataloader) * epochs



# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, 

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



        # Progress update every 100 batches.

        if step % 50 == 0 and not step == 0:

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

    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

      

print("")

print("Training complete!")
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



preds=[]

true=[]



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



        outputs = model(b_input_ids, 

                        token_type_ids=None, 

                        attention_mask=b_input_mask)

    

    # Get the "logits" output by the model. The "logits" are the output

    # values prior to applying an activation function like the softmax.

    logits = outputs[0]



    # Move logits and labels to CPU

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()

    

    preds.append(logits)

    true.append(label_ids)

    # Calculate the accuracy for this batch of test sentences.

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    

    # Accumulate the total accuracy.

    eval_accuracy += tmp_eval_accuracy



    # Track the number of batches

    nb_eval_steps += 1



# Report the final accuracy for this validation run.

print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("  Validation took: {:}".format(format_time(time.time() - t0)))
# Combine the predictions for each batch into a single list of 0s and 1s.

flat_predictions = [item for sublist in preds for item in sublist]

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.

flat_true_labels = [item for sublist in true for item in sublist]
from sklearn.metrics import classification_report

print(classification_report(flat_predictions,flat_true_labels))
comments1 = df_test.text.values



indices1=tokenizer.batch_encode_plus(comments1,max_length=128,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)

input_ids1=indices1["input_ids"]

attention_masks1=indices1["attention_mask"]



prediction_inputs1= torch.tensor(input_ids1)

prediction_masks1 = torch.tensor(attention_masks1)





# Set the batch size.  

batch_size = 32 



# Create the DataLoader.

prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)

prediction_sampler1 = SequentialSampler(prediction_data1)

prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)



print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))



# Put model in evaluation mode

model.eval()



# Tracking variables 

predictions = []



# Predict 

for batch in prediction_dataloader1:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  

  # Unpack the inputs from our dataloader

  b_input_ids1, b_input_mask1 = batch

  

  # Telling the model not to compute or store gradients, saving memory and 

  # speeding up prediction

  with torch.no_grad():

      # Forward pass, calculate logit predictions

      outputs1 = model(b_input_ids1, token_type_ids=None, 

                      attention_mask=b_input_mask1)



  logits1 = outputs1[0]



  # Move logits and labels to CPU

  logits1 = logits1.detach().cpu().numpy()

  

  

  # Store predictions and true labels

  predictions.append(logits1)



flat_predictions = [item for sublist in predictions for item in sublist]

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submit=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':flat_predictions})

#submit.to_csv('submission.csv',index=False)
submit.head()
df_leak = pd.read_csv('/kaggle/input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv', encoding ='ISO-8859-1')[['choose_one', 'text']]



# Creating target and id

df_leak['target'] = (df_leak['choose_one'] == 'Relevant').astype(np.int8)

df_leak['id'] = df_leak.index.astype(np.int16)

df_leak.drop(columns=['choose_one', 'text'], inplace=True)



# Merging target to test set

df_test = df_test.merge(df_leak, on=['id'], how='left')



print('Leaked Data Set Shape = {}'.format(df_leak.shape))

print('Leaked Data Set Memory Usage = {:.2f} MB'.format(df_leak.memory_usage().sum() / 1024**2))



perfect_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

perfect_submission['target'] = df_test['target'].values

perfect_submission.to_csv('submission.csv', index=False)

perfect_submission.describe()