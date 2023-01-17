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
from transformers import BertTokenizer



# Load the BERT tokenizer.

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# pd.set_option('display.max_colwidth', -1)

# df1 = pd.read_csv('/kaggle/input/comcastcomplaints/comcast_consumeraffairs_complaints.csv')

# print(df1.shape)

# df1 = df1.dropna()

# print(df1.shape)

# print(df1.columns)

# sentences = df1.text.values

# labels = df1.rating.values
# df1['BERT_tokenizer_len'] = df1['text'].apply(lambda x: len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))))
# def break_text(text, length=32):

#     all_text = []

#     for i in range(0,len(text),length//2):

#         all_text.append(' '.join(text.split(' ')[i:i+length]))

    

#     return(list(filter(None, all_text)))
# df2 = df1[df1['BERT_tokenizer_len']>64].iloc[:,-3:-1]
# initial_shape = df2.shape[0]

# for i in df2.index:

    

#     try:

#         all_text = break_text(df2.loc[i,'text'])

#         rating = df2.loc[i,'rating']

#         for j in all_text:

#             review = j

#             s = {'rating':rating, 'text':review}

#             df2 = pd.concat([df2,pd.DataFrame(s,index=[i])])

#     except:

#         print('Can not break row with index {} using window method'.format(i))


# df2.to_csv('window_out.csv')  
df2 = pd.read_csv('window_out.csv')
df2.shape
df2.loc[122,['rating','text']]
sentences = df2.text.values

labels = df2.rating.values
# Print the original sentence.

print(' Original: ', sentences[0])



# Print the sentence split into tokens.

print('Tokenized: ', tokenizer.tokenize(sentences[0]))



# Print the sentence mapped to token ids.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
max_len = 0

len_counts = []

# For every sentence...

for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.

        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        #print(input_ids)

        len_counts.append(len(input_ids))

        

            

    



    #Update the maximum sentence length.

        max_len = max(max_len, len(input_ids))



        





print('Max sentence length: ', max_len)
input_ids = []

attention_masks = []



# For every sentence...

for sent in sentences:

    # `encode_plus` will:

    #   (1) Tokenize the sentence.

    #   (2) Prepend the `[CLS]` token to the start.

    #   (3) Append the `[SEP]` token to the end.

    #   (4) Map tokens to their IDs.

    #   (5) Pad or truncate the sentence to `max_length`

    #   (6) Create attention masks for [PAD] tokens.

    encoded_dict = tokenizer.encode_plus(

                        sent,                      # Sentence to encode.

                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        max_length = 64,           # Pad & truncate all sentences.

                        pad_to_max_length = True,

                        return_attention_mask = True,   # Construct attn. masks.

                        return_tensors = 'pt',     # Return pytorch tensors.

                   )

    

    # Add the encoded sentence to the list.    

    input_ids.append(encoded_dict['input_ids'])

    

    # And its attention mask (simply differentiates padding from non-padding).

    attention_masks.append(encoded_dict['attention_mask'])



# Convert the lists into tensors.

input_ids = torch.cat(input_ids, dim=0)

attention_masks = torch.cat(attention_masks, dim=0)

labels = torch.tensor(labels)



# Print sentence 0, now as a list of IDs.

print('Original: ', sentences[0])

print('Token IDs:', input_ids[0])
print('Original: ', sentences[10])

print('Token IDs:', input_ids[10])
from torch.utils.data import TensorDataset, random_split



# Combine the training inputs into a TensorDataset.

dataset = TensorDataset(input_ids, attention_masks, labels)



# Create a 90-10 train-validation split.



# Calculate the number of samples to include in each set.

train_size = int(0.9 * len(dataset))

val_size = len(dataset) - train_size



# Divide the dataset by randomly selecting samples.

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



print('{:>5,} training samples'.format(train_size))

print('{:>5,} validation samples'.format(val_size))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



# The DataLoader needs to know our batch size for training, so we specify it 

# here. For fine-tuning BERT on a specific task, the authors recommend a batch 

# size of 16 or 32.

batch_size = 16



# Create the DataLoaders for our training and validation sets.

# We'll take training samples in random order. 



train_dataloader = DataLoader(

            train_dataset,  # The training samples.

            sampler = RandomSampler(train_dataset), # Select batches randomly

            batch_size = batch_size # Trains with this batch size.

        )



# For validation the order doesn't matter, so we'll just read them sequentially.

validation_dataloader = DataLoader(

            val_dataset, # The validation samples.

            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.

            batch_size = batch_size # Evaluate with this batch size.

        )
from transformers import BertForSequenceClassification, AdamW, BertConfig

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# Load BertForSequenceClassification, the pretrained BERT model with a single 

# linear classification layer on top. 

model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.

    num_labels = len(df1['rating'].unique()), # The number of output labels--2 for binary classification.

                    # You can increase this for multi-class tasks.   

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)



# Tell pytorch to run this model on the GPU.

model.cuda()
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 

# I believe the 'W' stands for 'Weight Decay fix"

optimizer = AdamW(model.parameters(),

                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5

                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                )
from transformers import get_linear_schedule_with_warmup



# Number of training epochs. The BERT authors recommend between 2 and 4. 

# We chose to run for 4, but we'll see later that this may be over-fitting the

# training data.

epochs = 3



# Total number of training steps is [number of batches] x [number of epochs]. 

# (Note that this is not the same as the number of training samples).

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

import numpy as np



# This training code is based on the `run_glue.py` script here:

# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128



# Set the seed value all over the place to make this reproducible.

seed_val = 42



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)



# We'll store a number of quantities such as training and validation loss, 

# validation accuracy, and timings.

training_stats = []



# Measure the total training time for the whole run.

total_t0 = time.time()



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

    total_train_loss = 0



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

        # The documentation for this `model` function is here: 

        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

        # It returns different numbers of parameters depending on what arguments

        # arge given and what flags are set. For our useage here, it returns

        # the loss (because we provided labels) and the "logits"--the model

        # outputs prior to activation.

        loss, logits = model(b_input_ids, 

                             token_type_ids=None, 

                             attention_mask=b_input_mask, 

                             labels=b_labels)



        # Accumulate the training loss over all of the batches so that we can

        # calculate the average loss at the end. `loss` is a Tensor containing a

        # single value; the `.item()` function just returns the Python value 

        # from the tensor.

        total_train_loss += loss.item()



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



    # Calculate the average loss over all of the batches.

    avg_train_loss = total_train_loss / len(train_dataloader)            

    

    # Measure how long this epoch took.

    training_time = format_time(time.time() - t0)



    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(training_time))

        

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

    total_eval_accuracy = 0

    total_eval_loss = 0

    nb_eval_steps = 0



    # Evaluate data for one epoch

    for batch in validation_dataloader:

        

        # Unpack this training batch from our dataloader. 

        #

        # As we unpack the batch, we'll also copy each tensor to the GPU using 

        # the `to` method.

        #

        # `batch` contains three pytorch tensors:

        #   [0]: input ids 

        #   [1]: attention masks

        #   [2]: labels 

        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)

        

        # Tell pytorch not to bother with constructing the compute graph during

        # the forward pass, since this is only needed for backprop (training).

        with torch.no_grad():        



            # Forward pass, calculate logit predictions.

            # token_type_ids is the same as the "segment ids", which 

            # differentiates sentence 1 and 2 in 2-sentence tasks.

            # The documentation for this `model` function is here: 

            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            # Get the "logits" output by the model. The "logits" are the output

            # values prior to applying an activation function like the softmax.

            (loss, logits) = model(b_input_ids, 

                                   token_type_ids=None, 

                                   attention_mask=b_input_mask,

                                   labels=b_labels)

            

        # Accumulate the validation loss.

        total_eval_loss += loss.item()



        # Move logits and labels to CPU

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        # Calculate the accuracy for this batch of test sentences, and

        # accumulate it over all batches.

        total_eval_accuracy += flat_accuracy(logits, label_ids)

        



    # Report the final accuracy for this validation run.

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))



    # Calculate the average loss over all of the batches.

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    

    # Measure how long the validation run took.

    validation_time = format_time(time.time() - t0)

    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    print("  Validation took: {:}".format(validation_time))



    # Record all statistics from this epoch.

    training_stats.append(

        {

            'epoch': epoch_i + 1,

            'Training Loss': avg_train_loss,

            'Valid. Loss': avg_val_loss,

            'Valid. Accur.': avg_val_accuracy,

            'Training Time': training_time,

            'Validation Time': validation_time

        }

    )



print("")

print("Training complete!")



print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
import pandas as pd



# Display floats with two decimal places.

pd.set_option('precision', 2)



# Create a DataFrame from our training statistics.

df_stats = pd.DataFrame(data=training_stats)



# Use the 'epoch' as the row index.

df_stats = df_stats.set_index('epoch')



# A hack to force the column headers to wrap.

#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])



# Display the table.

df_stats

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Use plot styling from seaborn.

sns.set(style='darkgrid')



# Increase the plot size and font size.

sns.set(font_scale=1.5)

plt.rcParams["figure.figsize"] = (12,6)



# Plot the learning curve.

plt.plot(df_stats['Training Loss'], 'b-o', label="Training")

plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

plt.plot(df_stats['Valid. Accur.'], 'r-o', label="Validation accuracy")



# Label the plot.

plt.title("Training & Validation Loss & validation accuracy")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.xticks([1, 2, 3, 4])



plt.show()