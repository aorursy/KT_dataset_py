!pip install transformers
import pandas as pd

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
train_df.head(10)
sentences = train_df.text.values

labels = train_df.target.values
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
print("original " , sentences[0])



print("Tokenize" , tokenizer.tokenize(sentences[0]))



print("Token Id" , tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
length = []



for sent in sentences:

    length.append(len(tokenizer.encode(sent)))

    

max(length)
max_len = 86

input_ids = []

attention_masks = []

for sent in sentences:

    encode = tokenizer.encode_plus(sent,max_length=max_len,pad_to_max_length=True )

    input_ids.append(encode['input_ids'])

    attention_masks.append(encode['attention_mask'])
# Use train_test_split to split our data into train and validation sets for

# training

from sklearn.model_selection import train_test_split



# Use 90% for training and 10% for validation.

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=2018, test_size=0.1)

# Do the same for the masks.

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,

                                             random_state=2018, test_size=0.1)
import torch



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

from transformers import get_linear_schedule_with_warmup



model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.

    num_labels = 2, # The number of output labels--2 for binary classification.

                    # You can increase this for multi-class tasks.   

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)



# Tell pytorch to run this model on the GPU.

model.cuda()



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
import numpy as np



import time

import datetime



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

import torch



if torch.cuda.is_available():

  device = torch.device("cuda")

  print(torch.cuda.get_device_name(0))

device
import random 



seed_val = 42



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed(seed_val)



loss_value = []



for epoch_i in range(0 , epochs):

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    print('Training...')

    

    t0 = time.time()

    

    total_loss = 0

    

    model.train()

    

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and step != 0:

            elapsed = format_time(time.time() - t0)

            

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        

        input_ids = batch[0].to(device)

        input_mask = batch[1].to(device)

        label = batch[2].to(device) 

        

        model.zero_grad()

        

        output = model(input_ids,

                      token_type_ids = None,

                      attention_mask = input_mask,

                      labels = label)

        

        loss = output[0]

        

        total_loss += loss.item()

        

        loss.backward()

        

        torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0)

        

        optimizer.step()

        

        scheduler.step()

    

    avg_train_loss = total_loss / len(train_dataloader)

    

    loss_value.append(avg_train_loss)

    

    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    

    # ========================================

    #               Validation

    # ========================================

    

    t0 = time.time()

    

    model.eval()

    

    # Tracking variables 

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    

    for batch in validation_dataloader:

        batch = (t.to(device) for t in batch)

        

        input_id , input_mask , label = batch

        

        with torch.no_grad():

            outputs = model(input_id, 

                           token_type_ids = None,

                           attention_mask = input_mask)

        logits = outputs[0]

        

        logits = logits.detach().cpu().numpy()

        label = label.to('cpu').numpy()

        

        tmp_eval_accuracy = flat_accuracy(logits, label)

        

        # Accumulate the total accuracy.

        eval_accuracy += tmp_eval_accuracy

        

        nb_eval_steps += 1

        # Report the final accuracy for this validation run.

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

plt.plot(loss_value, 'b-o')



# Label the plot.

plt.title("Training loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")



plt.show()
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")



sentences = test_df.text.values

    

max_len = 86

input_ids = []

attention_masks = []

for sent in sentences:

    encode = tokenizer.encode_plus(sent,max_length=max_len,pad_to_max_length=True )

    input_ids.append(encode['input_ids'])

    attention_masks.append(encode['attention_mask'])
prediction_inputs = torch.tensor(input_ids)

prediction_masks = torch.tensor(attention_masks)
prediction_data = TensorDataset(prediction_inputs, prediction_masks)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
prediction_sampler
model.eval()



predictions = []

i  = 0

for batch in prediction_dataloader:

    batch = tuple(t.to(device) for t in batch)

    

    input_id , input_mask = batch

    

    with torch.no_grad():

        output = model(input_id,

                      token_type_ids = None,

                      attention_mask = input_mask)

        

    logits = output[0]

    

    logits = logits.detach().cpu().numpy()

    predictions.extend(np.argmax(logits, axis=1))

    
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':predictions})

sub.to_csv('submission.csv',index=False)
df = pd.read_csv('submission.csv')

df.head(20)