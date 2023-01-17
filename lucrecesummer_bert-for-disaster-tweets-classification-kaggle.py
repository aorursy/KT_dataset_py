!nvidia-smi
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
!pip install transformers
import pandas as pd

df = pd.read_csv("train.csv")
print('Number of training sentences: ', len(df))
df.sample(5)
# Print some negative sample tweets
for txt in df[df.target==0].text.sample(5).values:
  print(txt)
# Print some positive sample tweets
for txt in df[df.target==1].text.sample(5).values:
  print(txt)
df.text.isna().sum()
print("Positive data: {:.2f}%".format(len(df[df.target==1])*100/len(df)))
tweets = df.text.values
labels = df.target.values
print("{} out of {} tweets have a http://... link within itself. ({:.2f}%)".format(len([t for t in tweets if "http://" in t]), len(df), len([t for t in tweets if "http://" in t])*100/len(df)))
[t for t in tweets if "http://" in t][:5]
# Print some tweets with URL that does NOT have URL at the end
[t for t in [t for t in tweets if "http://" in t] if "http://" not in t.split()[-1]][:5]
print("percentage of POSITIVE samples containing http URLs at the end: {:.2f}%".format(len([t for t in df[df['target']==1]['text'] if "http://" in t])*100/len(df[df['target']==1])))
print("percentage of NEGATIVE samples containing http URLs at the end: {:.2f}%".format(len([t for t in df[df['target']==0]['text'] if "http://" in t])*100/len(df[df['target']==0])))
print("{} out of {} tweets have a @user_id tag within itself. ({:.2f}%)".format(len([t for t in tweets if "@" in t]), len(df), len([t for t in tweets if "@" in t])*100/len(df)))
[t for t in tweets if "@" in t][:5]
print("percentage of POSITIVE samples containing @user_id tag: {:.2f}%".format(len([t for t in df[df['target']==1]['text'] if "@" in t])*100/len(df[df['target']==1])))
print("percentage of NEGATIVE samples containing @user_id tag: {:.2f}%".format(len([t for t in df[df['target']==0]['text'] if "@" in t])*100/len(df[df['target']==0])))
print("{} out of {} tweets have a # tag within itself. ({:.2f}%)".format(len([t for t in tweets if "#" in t]), len(df), len([t for t in tweets if "#" in t])*100/len(df)))
print("percentage of POSITIVE samples containing # tag: {:.2f}%".format(len([t for t in df[df['target']==1]['text'] if "#" in t])*100/len(df[df['target']==1])))
print("percentage of NEGATIVE samples containing # tag: {:.2f}%".format(len([t for t in df[df['target']==0]['text'] if "#" in t])*100/len(df[df['target']==0])))
from transformers import BertTokenizer

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print(' Original: ', tweets[1], labels[1])
print('Tokenized: ', tokenizer.tokenize(tweets[1]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[1])))
print(' Original: ', tweets[-1]) # a tweet with http URL
print('   Target: ', labels[-1])
print('Tokenized: ', tokenizer.tokenize(tweets[-1]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[-1])))
tweets = [" ".join([word if 'http://' not in word else "http" for word in t.split()]) for t in tweets]
tweets[-1]
print(' Original: ', tweets[-4])
print('   Target: ', labels[-4])
print('Tokenized: ', tokenizer.tokenize(tweets[-4]))
print(' Original: ', tweets[-17])
print('   Target: ', labels[-17])
print('Tokenized: ', tokenizer.tokenize(tweets[-17]))
tokenizer.tokenize("Living safely")
tweets = [" ".join([word if '@' not in word else "@" for word in t.split()]) for t in tweets]
tweets[-4]
import numpy as np

encoded_tweets = [tokenizer.encode(t) for t in tweets]
lens = np.array([len(t) for t in encoded_tweets])

print('# of sentences:', len(tweets))
print('Max sentence length: ', max(lens))
print('Avg sentence length: ', np.mean(lens))
print('Median sentence length: ', np.median(lens))
import matplotlib.pyplot as plt

unique = list(set(lens))
unique.sort()
cnt = [sum([1 if l==u else 0 for l in lens]) for u in unique]
plt.bar(unique, cnt)
# `encode_plus` will:
#   (1) Tokenize the sentence.
#   (2) Prepend the `[CLS]` token to the start.
#   (3) Append the `[SEP]` token to the end.
#   (4) Map tokens to their IDs.
#   (5) Pad or truncate the sentence to `max_length`
#   (6) Create attention masks for [PAD] tokens.

def encode(sentences, labels, tokenizer, max_len):
    encoded_dicts = [tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                     ) for sent in sentences]
    input_ids = [d['input_ids'] for d in encoded_dicts]  
    attention_masks = [d['attention_mask'] for d in encoded_dicts]  

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels
input_ids, attention_masks, labels = encode(tweets, labels, tokenizer, max_len=100)
print('Original: ', tweets[0])
print('\nToken IDs:', input_ids[0])
print(len(input_ids[0]))
tokenizer.convert_ids_to_tokens(input_ids[0][:20])
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

def make_dataloader(input_ids, attention_masks, labels, split=1):  
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if split:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print('{} training samples'.format(train_size))
        print('{} validation samples'.format(val_size))
    else: 
        train_dataset = dataset
        print(print('{} training samples (no validation)'.format(len(dataset))))

    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    batch_size = 32

    train_dataloader = DataLoader(
                          train_dataset,  # training samples.
                          sampler = RandomSampler(train_dataset), # Select batches randomly
                          batch_size = batch_size # Trains with this batch size.
                      )

    if split:
        # For validation the order doesn't matter, so just read them sequentially.
        validation_dataloader = DataLoader(
                                    val_dataset, # The validation samples.
                                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                                    batch_size = batch_size # Evaluate with this batch size.
                                )
    
        return train_dataloader, validation_dataloader

    return train_dataloader
train_dataloader, validation_dataloader = make_dataloader(input_ids, attention_masks, labels)
from transformers import BertForSequenceClassification, BertConfig

# Load BertForSequenceClassification (pretrained BERT model + a single linear classification layer on top) 
model = BertForSequenceClassification.from_pretrained(
              "bert-base-uncased",          # 12-layer BERT base model w/ uncased vocab
              num_labels = 2,               # number of output labels (2 for binary classification)  
              output_attentions = False,    # returns attentions weights?
              output_hidden_states = False, # return all hidden-states?
        )
model.cuda()
!nvidia-smi
next(model.parameters()).is_cuda
from transformers import AdamW 

# Note: AdamW is a class from the huggingface library (not pytorch)- 'W'= 'Weight Decay fix"
optimizer = AdamW(
                    model.parameters(),
                    lr = 5e-5,         # default 
                    eps = 1e-8         # default 
                )
import numpy as np
import time, datetime

# Helper functions
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''Takes time in seconds and returns a string hh:mm:ss'''

    elapsed_rounded = int(round((elapsed)))  # Round to the nearest second
    return str(datetime.timedelta(seconds=elapsed_rounded))  # Format as hh:mm:ss

def set_random_seed(seed):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
len(train_dataloader), len(validation_dataloader)
import random
from transformers import get_linear_schedule_with_warmup

def train_BERT(train_dataloader, validation_dataloader, model, optimizer, n_epochs, output_hidden=0):   
    set_random_seed(seed=42)  # Set seed to make this reproducible.
    
    total_t0 = time.time()   # Measure the total training time for the whole run.
    training_stats = []   # Store training and valid loss, valid accuracy, and timings.
    hidden_states = []

    # lr scheduler
    n_batches_train = len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(  optimizer, 
                                                  num_warmup_steps = 0, # Default value in run_glue.py
                                                  num_training_steps = n_batches_train * n_epochs  )

    for epoch_i in range(n_epochs):   
        # =================== Training =================== #       
        t0 = time.time()   
        total_train_loss, total_train_accuracy = 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            input_ids, att_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.zero_grad()        
            if output_hidden:
                loss, logits, h = model(input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=att_mask, 
                                        labels=labels)
                
                h = [layer.detach().cpu().numpy() for layer in h]
                if epoch_i == n_epochs - 1: # store the last epoch's hidden states
                    hidden_states.append(h[-1]) # only save last layer's h           
            else:
                loss, logits = model(input_ids, 
                                    token_type_ids=None, # Not required since training on a SINGLE sentence, not a pair
                                    attention_mask=att_mask, 
                                    labels=labels)
                
            total_train_loss += loss.item()  
            total_train_accuracy += flat_accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()      
            scheduler.step()   # Update learning rate
        
        print("Epoch: {}/{}".format((epoch_i+1), n_epochs),
              "  Train loss: {0:.4f}".format(total_train_loss/n_batches_train),
              "  Train Acc: {0:.4f}".format(total_train_accuracy/n_batches_train),
              "  ({:})".format(format_time(time.time() - t0)))
        
        training_stats.append( {'epoch':           epoch_i + 1,
                                'Training Loss':   total_train_loss/n_batches_train,
                                'Training Acc' :   total_train_accuracy/n_batches_train,
                                'Training Time':   format_time(time.time() - t0)} )

        if validation_dataloader is not None:
            # =================== Validation =================== #
            n_batches_valid = len(validation_dataloader)
            t0 = time.time()
            model.eval()

            total_eval_accuracy, total_eval_loss = 0, 0
            for batch in validation_dataloader:
                v_input_ids, v_att_mask, v_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                with torch.no_grad(): 
                    if output_hidden:       
                        loss, logits, val_h = model(v_input_ids, 
                                                    token_type_ids=None, 
                                                    attention_mask=v_att_mask,
                                                    labels=v_labels)
                    
                        val_h = [layer.detach().cpu().numpy() for layer in val_h] # save GPU memory
                    else:
                        loss, logits = model(v_input_ids, 
                                             token_type_ids=None, 
                                             attention_mask=v_att_mask,
                                             labels=v_labels)
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = v_labels.detach().cpu().numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            print("  Valid Loss: {0:.4f}".format(total_eval_loss/n_batches_valid),
                  "  Valid Acc: {0:.4f}".format(total_eval_accuracy/n_batches_valid),
                  "  ({:})".format(format_time(time.time()-t0)))

            training_stats.append( {'            Valid. Loss':     total_eval_loss/n_batches_valid,
                                    'Valid. Acc':   total_eval_accuracy/n_batches_valid,
                                    'Validation Time': format_time(time.time()-t0)} )

    print("\nTraining complete.")
    print("Duration: {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    if output_hidden:
        return training_stats, hidden_states
    else:
        return training_stats
training_stats = train_BERT(train_dataloader, validation_dataloader, 
                            model=model, optimizer=optimizer, 
                            n_epochs=2)
train_dataloader = make_dataloader(input_ids, attention_masks, labels, split=0)
model = BertForSequenceClassification.from_pretrained(
              "bert-base-uncased",          # 12-layer BERT base model, w/ uncased vocab
              num_labels = 2,               # number of output labels (2 for binary classification)  
              output_attentions = False,    # Whether the model returns attentions weights.
              output_hidden_states = False, # Whether the model returns all hidden-states.
        )
model.cuda()
optimizer = AdamW(  model.parameters(), lr = 5e-5, eps = 1e-8)
training_stats = train_BERT(train_dataloader, None,
                            model=model, optimizer=optimizer, 
                            n_epochs=2)
test_df = pd.read_csv("test.csv")
test_sentences = test_df.text.values
test_sentences = [" ".join([word if 'http://' not in word else "http" for word in t.split()]) for t in test_sentences]
test_sentences = [" ".join([word for word in t.split() if '@' not in word]) for t in test_sentences]
test_encoded_sentences = [tokenizer.encode(s) for s in test_sentences]
test_sent_lens = np.array([len(s) for s in test_encoded_sentences])

print('# of sentences:', len(test_sentences))
print('Max sentence length: ', max(test_sent_lens))
print('Avg sentence length: ', np.mean(test_sent_lens))
print('Median sentence length: ', np.median(test_sent_lens))
encoded_dicts = [tokenizer.encode_plus(  sent,                      
                                         add_special_tokens = True, 
                                         max_length = 100,          
                                         pad_to_max_length = True,
                                         return_attention_mask = True,   
                                         return_tensors = 'pt'  ) for sent in test_sentences]
input_ids = [d['input_ids'] for d in encoded_dicts]  
input_ids = torch.cat(input_ids, dim=0)
attention_masks = [d['attention_mask'] for d in encoded_dicts]  
attention_masks = torch.cat(attention_masks, dim=0)

prediction_data = TensorDataset(input_ids, attention_masks)
prediction_dataloader = DataLoader(dataset = prediction_data, 
                                   sampler = SequentialSampler(prediction_data), # doesn't need to be sampled randomly
                                   batch_size = 32)
len(prediction_dataloader)
print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
model.eval()
predictions, true_labels = [], []
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        logits = model(  b_input_ids, 
                         token_type_ids=None, 
                         attention_mask=b_input_mask  ) # no loss, since "labels" not provided

    logits = logits[0].detach().cpu().numpy() # extract x from [[x]]
    predictions.append(logits)
print('    DONE.')
flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
submission = pd.DataFrame(test_df.id)
submission['target'] = flat_predictions
submission.to_csv('submission_6_23_17_10.csv', index=False)
