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

from transformers import BertTokenizer,BertForSequenceClassification,AdamW,get_linear_schedule_with_warmup

import tensorflow as tf

import os

import pandas as pd

import numpy as np
#Get GPU name

gpu=tf.test.gpu_device_name()

print(gpu)
# If GPU available

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    

    device=torch.device("cuda")

    print('There are %d GPU(s) available.' %torch.cuda.device_count())

    print('We will use the GPU:',torch.cuda.get_device_name(0))

else:

    print('No GPU available, using the CPU.')

    device = torch.device("cpu")
#Get training and validation data

train=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")

test=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")
test.id
train.groupby('language')['id'].count()
#Initialize BertTokenzier

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#Data Preprocessing and tensor generation

seed=2



#train=train.sample(n=10000,random_state=seed)

tokenized_train=tokenizer.batch_encode_plus(train.iloc[:,1:3].to_numpy().tolist(),max_length=128,pad_to_max_length=True,return_tensors='pt')

labels_train=torch.tensor(train.label.values[:])



#val=val.sample(n=2000,random_state=seed)

tokenized_val=tokenizer.batch_encode_plus(train.iloc[-1000:,1:3].to_numpy().tolist(),max_length=128,pad_to_max_length=True,return_tensors='pt')

labels_val=torch.tensor(train.label.values[-1000:])



tokenized_test=tokenizer.batch_encode_plus(test.iloc[:,1:3].to_numpy().tolist(),max_length=128,pad_to_max_length=True,return_tensors='pt')
from torch.utils.data import TensorDataset, DataLoader, RandomSampler



batch_size=32



train_data=TensorDataset(tokenized_train['input_ids'],tokenized_train['attention_mask'],tokenized_train['token_type_ids'],labels_train)

train_sampler=RandomSampler(train_data)

train_dataloader=DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



val_data=TensorDataset(tokenized_val['input_ids'],tokenized_val['attention_mask'],tokenized_val['token_type_ids'],labels_val)

val_sampler=RandomSampler(val_data)

val_dataloader=DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



test_data=TensorDataset(tokenized_test['input_ids'],tokenized_test['attention_mask'],tokenized_test['token_type_ids'])

# test_sampler=RandomSampler(test_data)

test_dataloader=DataLoader(test_data, batch_size=batch_size)
#Bert Model transformer with a single sequence classification layer on top

model=BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=3,output_attentions=False,output_hidden_states=False)

model.cuda()
#Set learning rate

optimizer=AdamW(model.parameters(),lr=2e-5)

epochs=8



#Training steps is no_of_batches*no_of_epochs

total_steps=len(train_dataloader)*epochs



#Learning rate scheduler

scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps = total_steps)
# Function to calculate the accuracy of predictions

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
import random



seed_val = 12



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)



loss_values = []



for epoch_i in range(0, epochs):

    

    #Put model into training mode

    model.train()



    total_loss=0



    for step, batch in enumerate(train_dataloader):



        #Unpack the training batch

        b_input_ids = batch[0].to(device)

        b_attention_mask=batch[1].to(device)

        b_token_type = batch[2].to(device)

        b_labels = batch[3].to(device)



        #Clear previously calculated gradients before performing a backward pass

        #model.zero_grad()        //Not sure if useful or not



        #Perform a forward pass and get the loss

        outputs=model(b_input_ids,token_type_ids=b_token_type,attention_mask=b_attention_mask,labels=b_labels)

        

        loss = outputs[0]

        total_loss += loss.item()



        #Perform backward pass to calculate gradients

        loss.backward()



        #Clip the norm of the gradients to 1.0.

        #Used to prevent the "exploding gradients" problem.

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



        #Update weights

        optimizer.step()



        #Update learning rate

        scheduler.step()



    #Avg loss over training data for an epoch

    avg_train_loss = total_loss / len(train_dataloader)



    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))



    #### Validation



    #Evaluation mode

    model.eval()



    #Tracking variables 

    val_accuracy=0

    nb_val_steps=0



    #Evaluate data for one epoch

    for batch in val_dataloader:

        

        #Add batch to GPU

        batch = tuple(t.to(device) for t in batch)

        

        #Unpack the inputs from dataloader

        b_input_ids,b_attention_mask, b_token_type, b_labels = batch

        

        #Telling the model not to compute or store gradients, saving memory and speeding up validation

        with torch.no_grad():        

            outputs = model(b_input_ids,token_type_ids=b_token_type,attention_mask=b_attention_mask)

        

        #Get the "logits" output by the model. The "logits" are output values prior to applying an activation function like the softmax.

        logits = outputs[0]

        

        #Move logits and labels to CPU

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        

        #Calculate the accuracy for this batch of test sentences.

        tmp_val_accuracy = flat_accuracy(logits, label_ids)

        

        #Accumulate the total accuracy.

        val_accuracy += tmp_val_accuracy



        #Track the number of batches

        nb_val_steps += 1



    # Report the final accuracy for this validation run.

    print("  Accuracy: {0:.2f}".format(val_accuracy/nb_val_steps))

final_output = []

for batch in test_dataloader:

  #Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  

  #Unpack the inputs from dataloader

  b_input_ids,b_attention_mask, b_token_type= batch

  

  #Telling the model not to compute or store gradients, saving memory and speeding up validation

  with torch.no_grad():        

      outputs = model(b_input_ids,token_type_ids=b_token_type,attention_mask=b_attention_mask)

  

  #Get the "logits" output by the model. The "logits" are output values prior to applying an activation function like the softmax.

  logits = outputs[0]

  final_output.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).flatten())

  
output = pd.DataFrame({'id': test.id,

                       'prediction': final_output})

output.to_csv('submission.csv', index=False)