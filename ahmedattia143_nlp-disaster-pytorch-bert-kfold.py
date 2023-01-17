!pip install transformers
import torch

import pandas as pd 

import random 

import time

import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import f1_score

import numpy as np 

from torch.utils.data import TensorDataset,Subset

from transformers import BertTokenizer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from torch.nn import functional as F
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
# Load the BERT tokenizer.

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train = pd.read_csv('../input/nlp-getting-started/train.csv')

sentences = train.text.values

labels = train.target.values
# Tokenize all of the sentences and map the tokens to thier word IDs.

input_ids = []

attention_masks = []



# For every sentence...

for sent in sentences:

  

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
batch_size = 16 

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#helper function to get train and val data loaders for each fold 

def get_data_loaders(dataset,train_indexes,val_indexes):

    train_tensor = Subset(dataset,train_indexes)

    val_tensor = Subset(dataset,val_indexes)

    train_dataloader = DataLoader(

            train_tensor, 

            sampler = RandomSampler(train_tensor), 

            batch_size = batch_size

        )



    val_dataloader = DataLoader(

            val_tensor, 

            sampler = SequentialSampler(val_tensor), 

            batch_size = batch_size 

        )

    return train_dataloader,val_dataloader

# Combine the training inputs into a TensorDataset.

dataset = TensorDataset(input_ids, attention_masks, labels)
df = pd.read_csv("../input/nlp-getting-started/test.csv")

sentences = df.text.values

input_ids = []

attention_masks = []

for sent in sentences:



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

# Set the batch size.  

batch_size = 16  

# Create the DataLoader.

prediction_data = TensorDataset(input_ids, attention_masks)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
from transformers import BertForSequenceClassification, AdamW, BertConfig

def get_bert_model():

    model = BertForSequenceClassification.from_pretrained(

      "bert-base-uncased", 

      num_labels = 2,           

      output_attentions = False, 

      output_hidden_states = False, 

    )

    # Tell pytorch to run this model on the GPU.

    model.cuda()

    return model
import numpy as np

# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):

    '''

    Takes a time in seconds and returns a string hh:mm:ss

    '''

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set the seed value all over the place to make this reproducible.

seed_val = 1000

random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)
total_folds = 15

current_fold = 0

all_folds_preds = []

epochs = 1

fold=StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=1000)



training_stats = []
# Measure the total training time for the whole run.

total_t0 = time.time()

#for each fold..

for train_index, test_index in fold.split(train,train['target']):

    model = get_bert_model()

    optimizer = AdamW(model.parameters(),lr = 5e-5,eps = 1e-8)

    current_fold = current_fold+1

    train_dataloader,validation_dataloader = get_data_loaders(dataset,train_index,test_index)

    print("")

    print('================= Fold {:} / {:} ================='.format(current_fold,total_folds))

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

        model.train()

        # For each batch of training data...

        for step, batch in enumerate(train_dataloader):



            b_input_ids = batch[0].to(device)

            b_input_mask = batch[1].to(device)

            b_labels = batch[2].to(device)

            model.zero_grad()        



            loss, logits = model(b_input_ids, 

                              token_type_ids=None, 

                              attention_mask=b_input_mask, 

                              labels=b_labels)





            total_train_loss += loss.item()



            # Perform a backward pass to calculate the gradients.

            loss.backward()



            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #update weights

            optimizer.step()





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

        total_f1_score = 0

        total_eval_accuracy = 0

        total_eval_loss = 0

        nb_eval_steps = 0



        # Evaluate data for one epoch

        for batch in validation_dataloader:





            b_input_ids = batch[0].to(device)

            b_input_mask = batch[1].to(device)

            b_labels = batch[2].to(device)



            with torch.no_grad():        

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

            total_f1_score += f1_score(np.argmax(logits,axis=1),label_ids)



        # Report the final accuracy and f1_score for this validation run.

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        

        avg_f1_score = total_f1_score / len(validation_dataloader)

        print("  F1_score: {0:.2f}".format(avg_f1_score))



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

              'f1_score' : avg_f1_score,

              'Training Time': training_time,

              'Validation Time': validation_time,

              'fold' : current_fold

              

          }

        )



    print("")

    print("Training complete!")



    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



    # ========================================

    # Predicting and saving predictions for all folds

    # ========================================



    print("")

    print("now predicting for this fold")



    # Put model in evaluation mode

    model.eval()

    # Tracking variables 

    predictions  = []

    # Predict 

    for batch in prediction_dataloader:

        # Add batch to GPU

        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader

        b_input_ids, b_input_mask = batch

        # speeding up prediction

        with torch.no_grad():

            # Forward pass, calculate logit predictions

            outputs = model(b_input_ids, token_type_ids=None, 

                            attention_mask=b_input_mask)



        logits = outputs[0]



        # Move logits and labels to CPU

        logits = logits.detach().cpu().numpy()





        predictions.append(logits)



    stack = np.vstack(predictions)

    final_preds = F.softmax(torch.from_numpy(stack))[:,1].numpy()

    all_folds_preds.append(final_preds)

print("Completed")

pd.set_option('precision', 2)

df_stats = pd.DataFrame(data=training_stats)

df_stats = df_stats.set_index('fold')

df_stats
to_submit =np.mean(all_folds_preds,0)
sub=pd.DataFrame()

sub['id'] = df['id']

sub['target'] = to_submit

sub['target'] = sub['target'].apply(lambda x: 1 if x>0.5 else 0)

sub.head()
sub.to_csv('bert_base_15_2e-5_64',index=False)