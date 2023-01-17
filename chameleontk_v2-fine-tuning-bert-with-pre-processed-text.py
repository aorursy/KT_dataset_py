import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
%matplotlib inline
# plt.figure(figsize=(16, 6))

import warnings
warnings.filterwarnings("ignore")

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
data_dir = "/kaggle/input/shopee-sentiment-analysis/"
dftest = pd.read_csv(data_dir+"test.csv")
dftrain = pd.read_csv(data_dir+"train.csv")
print(dftrain.shape)
dftrain.head()
print(dftrain.groupby("rating")["rating"].value_counts())
dftrain.groupby("rating")["rating"].hist();

# recommend only 2–4 epochs of training for fine-tuning BERT
train_df = dftrain
test_df = dftest
import emoji  # https://pypi.org/project/emoji/

have_emoji_train_idx = []
have_emoji_test_idx = []

for idx, review in enumerate(dftrain['review']):
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_train_idx.append(idx)
        
for idx, review in enumerate(dftest['review']):
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_test_idx.append(idx)
train_emoji_percentage = round(len(have_emoji_train_idx) / train_df.shape[0] * 100, 2)
print(f'Train data has {len(have_emoji_train_idx)} rows that used emoji, that means {train_emoji_percentage} percent of the total')

test_emoji_percentage = round(len(have_emoji_test_idx) / test_df.shape[0] * 100, 2)
print(f'Test data has {len(have_emoji_test_idx)} rows that used emoji, that means {test_emoji_percentage} percent of the total')
def emoji_cleaning(text):
    
    # Change emoji to text
    text = emoji.demojize(text).replace(":", " ")
    
#     # Delete repeated emoji
#     tokenizer = text.split()
#     repeated_list = []
    
#     for word in tokenizer:
#         if word not in repeated_list:
#             repeated_list.append(word)
    
#     text = ' '.join(text for text in repeated_list)
    text = text.replace("_", " ").replace("-", " ")
    return text
train_df['demojize_review'] = train_df['review'].apply(emoji_cleaning)
test_df['demojize_review'] = test_df['review'].apply(emoji_cleaning)
import re
def review_cleaning(text):
    
    # delete lowercase and newline
    text = text.lower()
    text = re.sub(r'\n', '', text)
    
    # change emoticon to text
    text = re.sub(r':\(', 'dislike', text)
    text = re.sub(r': \(\(', 'dislike', text)
    text = re.sub(r':, \(', 'dislike', text)
    text = re.sub(r':\)', 'smile', text)
    text = re.sub(r';\)', 'smile', text)
    text = re.sub(r':\)\)\)', 'smile', text)
    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)
    text = re.sub(r'=\)\)\)\)', 'smile', text)
    text = re.sub(r'=\)\)\)\)', 'smile', text)
    
#     # delete punctuation
    text = re.sub('[^a-z0-9 ]', '', text)
    
    tokenizer = text.split()
    
    return ' '.join([text for text in tokenizer]).strip()

train_df['clean_review'] = train_df['demojize_review'].apply(review_cleaning)
test_df['clean_review'] = test_df['demojize_review'].apply(review_cleaning)
def delete_repeated_char(text):
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    return text

train_df['clean_review'] = train_df['clean_review'].apply(delete_repeated_char)
test_df['clean_review'] = test_df['clean_review'].apply(delete_repeated_char)
print('Before: ', train_df.loc[92129, 'review'])
print('After: ', train_df.loc[92129, 'clean_review'])

print('\nBefore: ', train_df.loc[56938, 'review'])
print('After: ', train_df.loc[56938, 'clean_review'])

print('\nBefore: ', train_df.loc[72677, 'review'])
print('After: ', train_df.loc[72677, 'clean_review'])

print('\nBefore: ', train_df.loc[36558, 'review'])
print('After: ', train_df.loc[36558, 'clean_review'])
# how many too short sentences
train_df[train_df['clean_review'].str.len() < 2].shape
# fill empty review with "good" << average word in rating 3
test_df.loc[test_df['clean_review'].str.len() <=2, 'clean_review'] = "good"
train_df = train_df[train_df['clean_review'].str.len() >= 2]

# !pip install pyenchant
# !apt-get install python-enchant --yes
train_tokens = train_df['clean_review']
test_tokens = test_df['clean_review']

tokens = ' '.join([text for text in train_tokens] + [text for text in test_tokens]).split(" ")
tokens = set(tokens)

import enchant
d = enchant.Dict("en_US")

unknown_words = []
for idx, t in enumerate(tokens):
    if len(t)<=1:
        continue
        
#     if idx >= 10:
#         break
        
    if not d.check(t):
#         print(t, d.suggest(t))
        unknown_words.append(t)
    
print(len(unknown_words), len(tokens), len(unknown_words)*100.0/len(tokens))
# !pip install googletrans
# freq_unknown_words = []
# for idx, t in enumerate(unknown_words):
#     if len(t)<=1:
#         continue
        
#     if idx%100==0:
#         print(f"{idx}/{len(unknown_words)} => ", len(freq_unknown_words))
        
#     n1 = len(train_df[train_df['clean_review'].str.contains(t)])
#     n2 = len(test_df[test_df['clean_review'].str.contains(t)])
    
#     n = n1+n2
    
#     if n>=10:
#         freq_unknown_words.append(t)


# print(len(freq_unknown_words), len(unknown_words), len(tokens))
# import googletrans
# from googletrans import Translator
# translator = Translator()

# translated_words = {}


# for idx, t in enumerate(freq_unknown_words):
#     if len(t)<=1:
#         continue
    
#     if idx%100==0:
#         print(f"{idx}/{len(freq_unknown_words)}")
        
#     translated_words[t] = []
#     for scr_lang in ['id', "ms", "tl", "th"]:
#         # vi: Vietnam
        
#         if scr_lang=="th":
#             th_char = re.sub('[ก-ฮ]', '', t)
#             if th_char==t:
#                 continue
        
#         w = translator.translate(t, src=scr_lang, dest='en').text
#         if w!=t:
#             translated_words[t].append((scr_lang, w))
#             break
# translated_words
# import json
# with open('translated_words.json', 'w') as outfile:
#     json.dump(translated_words, outfile)
# with open('translated_words.json', 'r') as outfile:
#     translated_words = json.load(outfile)
def recover_shortened_words(text):
    
    slang_words = [
        (r'\bbgus\b', 'awesome'),


        (r'\btq\b', 'thanks'),
        (r'\btks\b', 'thanks'),
        
        (r'\bsis\b', 'sister'),
        (r'\bsuka\b', 'love'),
        (r'\bssuka\b', 'love'),
    ]
    
    for w in slang_words:
        text = re.sub(w[0], w[1], text)
    
    for w in translated_words:
        if len(translated_words[w])==0:
            continue
        _, s = translated_words[w][0]
        text = re.sub(r'\b'+w+r'\b', s, text)
        
    return text

print("suka", recover_shortened_words("suka"))
print("sukaaxasxa", recover_shortened_words("sukaaxasxa"))

rows = []
for idx, row in train_df.iterrows():
    row["translated_clean_review"] = recover_shortened_words(row["clean_review"])
    rows.append(row.values)
    
    if idx%100==0:
        print(f"{idx}/{len(train_df)}")


columns = list(train_df.columns)+["translated_clean_review"]
new_train = pd.DataFrame(rows, columns=columns)

rows = []
for idx, row in test_df.iterrows():
    row["translated_clean_review"] = recover_shortened_words(row["clean_review"])
    rows.append(row.values)
    
    if idx%100==0:
        print(f"{idx}/{len(test_df)}")


columns = list(test_df.columns)+["translated_clean_review"]
new_test = pd.DataFrame(rows, columns=columns)



new_train.head()
# train_df['translated_clean_review'] = train_df['clean_review'].apply(recover_shortened_words)
# test_df['translated_clean_review'] = test_df['clean_review'].apply(recover_shortened_words)
new_test.to_csv('preprocess_test.csv', index=False)
new_train.to_csv('preprocess_train.csv', index=False)
new_train["n"] = new_train['translated_clean_review'].str.len()
new_train[new_train['n']<100]["n"].hist(bins=100)
new_train.head()

import pandas as pd
data_dir = "/kaggle/input/shopee-sentiment-analysis/"
dftest = pd.read_csv(data_dir+"test.csv")
dftrain = pd.read_csv(data_dir+"train.csv")
import tensorflow as tf
import torch
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print("Device: CUDA")
else: 
    device = torch.device("cpu")
!pip install transformers
# MODEL_CLASSES = {
#     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
#     'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
# }

dftrain.shape
sentences = dftrain.review.values
labels = dftrain.rating.values

# 146811

# import random
# idx = random.sample(range(len(sentences)), 1000)
# sentences = dftrain.review.values[idx]
# labels = dftrain.rating.values[idx]

labels = [l-1 for l in labels]
sentences[0:10], labels[0:10]
from transformers import BertTokenizer
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to thier word IDs.
def sentencesToIds(sentences):
  input_ids = []
  for sent in sentences:
      # `encode` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.

      # This function also supports truncation and conversion
      encoded_sent = tokenizer.encode(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True) # Add '[CLS]' and '[SEP]'
                          
      # Add the encoded sentence to the list.
      input_ids.append(encoded_sent)
  return input_ids

# input_ids = sentencesToIds(sentences)
# print('Original: ', sentences[0])
# print('Token IDs:', input_ids[0])

from keras.preprocessing.sequence import pad_sequences
# Set the maximum sequence length.
MAX_LEN = 256

print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

def getAttentionMask(input_ids):
  # Create attention masks
  attention_masks = []
  # For each sentence...
  nUnkToken = 0
  for sent in input_ids:
      
      # Create the attention mask.
      #   - If a token ID is 0, then it's padding, set the mask to 0.
      #   - If a token ID is > 0, then it's a real token, set the mask to 1.
      att_mask = [int(token_id > 0) for token_id in sent]
      
      n = sum([int(token_id==tokenizer.unk_token_id) for token_id in sent])
      nUnkToken += n
      # Store the attention mask for this sentence.
      attention_masks.append(att_mask)

  print("#UnknownTokens:", nUnkToken)
  return attention_masks

# attention_masks = getAttentionMask(input_ids)
input_ids = sentencesToIds(sentences)
print('Max sentence length: ', max([len(sen) for sen in input_ids]))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
attention_masks = getAttentionMask(input_ids)
import pickle

# Save to file in the current working directory
pkl_filename = "input_ids.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(input_ids, file)
    
pkl_filename = "attention_masks.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(attention_masks, file)

# # Load from file
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)
 
print("DONE")
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# The DataLoader needs to know our batch size for training, so we specify it  here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of

batch_size = 45

def preprocess(sentences, labels, test=False):
  # input_ids = sentencesToIds(sentences)
  # print('Max sentence length: ', max([len(sen) for sen in input_ids]))

  # input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
  # attention_masks = getAttentionMask(input_ids)

  if not test:
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)
  else:
    train_inputs = input_ids
    train_labels = labels
    train_masks = attention_masks

    validation_inputs = []
    validation_labels = []
    validation_masks = []

  # Convert all inputs and labels into torch tensors
  train_inputs = torch.tensor(train_inputs)
  validation_inputs = torch.tensor(validation_inputs)

  train_labels = torch.tensor(train_labels)
  validation_labels = torch.tensor(validation_labels)

  train_masks = torch.tensor(train_masks)
  validation_masks = torch.tensor(validation_masks)


  
  # Create the DataLoader for our training set.
  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  # Create the DataLoader for our validation set.
  validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

  return train_dataloader, validation_dataloader
train_dataloader, validation_dataloader = preprocess(sentences, labels)
import os
CUDA_LAUNCH_BLOCKING=1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
gc.collect()
from transformers import BertForSequenceClassification, AdamW, BertConfig
# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 5,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
# Tell pytorch to run this model on the GPU.
model.cuda();

optimizer = AdamW(model.parameters(), lr = 2e-5,eps = 1e-8)
from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)

import numpy as np
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
import time
import datetime
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random

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
    
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        # print(device)
        # print(batch[0].shape, batch[1].shape, batch[2].shape, )
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
      
        # Always clear any previously calculated gradients before performing a
        model.zero_grad()        
        
        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we have provided the `labels`.
        
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
        # calculate the average loss at the end. 
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
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

print("DONE")
import plotly.express as px
f = pd.DataFrame(loss_values)
f.columns=['Loss']
fig = px.line(f, x=f.index, y=f.Loss)
fig.update_layout(title='Training loss of the Model',
                   xaxis_title='Epoch',
                   yaxis_title='Loss')
fig.show()

pkl_filename = "v1BERT.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# model.save_pretrained(data_dir+"/v1_BERT/")


# model = BertForSequenceClassification.from_pretrained(data_dir+"/v1_BERT/")  
# dftest = dftest.head(100)
print(dftest.shape)
dftest.head()
# dftest = dftest.head()
print('Predicting labels for {:,} test sentences...'.format(len(dftest.review)))
fake_labels = [1 for i in range(len(dftest))]

sentences = dftest.review.values
input_ids = sentencesToIds(sentences)
print('Max sentence length: ', max([len(sen) for sen in input_ids]))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
attention_masks = getAttentionMask(input_ids)

test_dataloader, _ = preprocess(dftest.review.values, fake_labels, test=True)
# Put model in evaluation mode
model.eval()
# Tracking variables 
predictions , true_labels = [], []
# Predict 
for batch in test_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
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

# predictions[0].shape
def get_pred_class(preds):
  pred_classes = []
  for p in preds:
    # print(p.shape)
    b_pred_class = np.argmax(p, axis=1).flatten() + 1
    pred_classes = pred_classes + list(b_pred_class)
  return pred_classes
dftest["rating"] = get_pred_class(predictions)
dftest[["review_id", "rating"]].to_csv("v1_BERT.csv", index=False)
print("DONE")