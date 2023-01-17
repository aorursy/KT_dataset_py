!ls '/kaggle/input/'
import pandas as pd
import random
import numpy as np
import re
from transformers import BertTokenizer
import torch
from torch.utils.data.dataset import Dataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
from string import punctuation
tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bert-base-uncased', do_lower_case=True)
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv',engine='python')
df.head()
df.columns
df['p_review'] = df['review']
df['p_review'].replace(to_replace = '.<br\s+\/'.format(punctuation),inplace=True,value='',regex=True)
df['p_review'].replace(to_replace = '[{}]'.format(punctuation),inplace=True,value='',regex=True)
df['p_review'].replace(to_replace = '\s+',inplace=True,value=' ',regex=True)


#converting polarity to numnber
df['p_polarity']=df['sentiment']
df['p_polarity'] = df['p_polarity'].map({'positive':0,'negative':1})
df.head()
data_all=[]
for i,row in df.iterrows():
    data_all.append((row['p_review'],row['p_polarity']))
id_list = list(range(0,len(data_all)))
np.random.seed(3)
np.random.shuffle(id_list)
train_data = [data_all[i] for i in id_list[:int(len(data_all)*.90)]]
test_data = [data_all[i] for i in id_list[int(len(data_all)*.90):]]
len(train_data),len(test_data)
train_data=train_data[:2000]
test_data=test_data[:1000]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

#Data Loader
class MyLoader(Dataset):
    def __init__(self,data_set):
        
        self.data=data_set
        
    def __getitem__(self,index):
        return self.data[index][0],self.data[index][1]
        
        
    def __len__(self):
        return len(self.data)
def my_collate_fn(batch):
    sent_list = [i for i,j in batch]
    lbl_list = [j for i,j in batch]
    t_encode = tokenizer.batch_encode_plus(sent_list,pad_to_max_length=True,max_length=512)
    return torch.tensor(t_encode['input_ids']),torch.tensor(t_encode['token_type_ids']),torch.tensor(t_encode['attention_mask']),torch.tensor(lbl_list)
batch_sz=4
train_obj = MyLoader(train_data)
train_loader=torch.utils.data.DataLoader(train_obj,batch_size=batch_sz,collate_fn=my_collate_fn)

val_obj = MyLoader(test_data)
val_loader=torch.utils.data.DataLoader(val_obj,batch_size=batch_sz,collate_fn=my_collate_fn)
for i,(input_ids,token_type_ids,attention_mask,lbl_list) in enumerate(train_loader): 
    print(type(input_ids[0]))
    break
#Load and Fine tune PreTrained Bert Model

model = BertForSequenceClassification.from_pretrained(\
                                                      
    "/kaggle/input/bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.\
                                                      
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.\
                                                      
    output_hidden_states = False, # Whether the model returns all hidden-states.
                                                      
)
model.to(device)
# The epsilon parameter eps = 1e-8 is “a very small number to prevent any division by zero in the implementation”
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
epochs=2
loss_values = []
# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Reset the total loss for this epoch.
    total_loss = 0
    # Put the model into training mode. Don't be mislead--the call to 
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_loader):
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)
        # Always clear any previously calculated gradients before performing a
        
        model.zero_grad()        
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple.
        loss = outputs[0]
        # Accumulate the training loss over all of the batches so that we can
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
#         scheduler.step()
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_loader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    #print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in val_loader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids,abc, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
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
   # print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")
# torch.save(model,'checkpoint.pth')
loaded_model = torch.load('checkpoint.pth')
loaded_model.eval()
loaded_model.to(device)
!ls 
val_obj1 = MyLoader([('I found this movie which is good.',0),('one of the rubbish movie ever watched',1)])
val_loader1=torch.utils.data.DataLoader(val_obj1,batch_size=batch_sz,collate_fn=my_collate_fn)
for (b_input_ids,abc,b_input_mask,lbl) in val_loader1:
    outputs = loaded_model(b_input_ids.to(device), 
                            token_type_ids=None, 
                            attention_mask=b_input_mask.to(device))
    
    print(outputs[0])
