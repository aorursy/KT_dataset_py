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
df = pd.read_csv('/kaggle/input/smile-annotations-final.csv')
# df.set_index('id', inplace=True)
df.head()
new_df = pd.read_csv('/kaggle/input/smile-annotations-final.csv',names=['id', 'texts', 'class'])
# df.set_index('id', inplace=True)
new_df.head()
new_df.head(10)
new_df.describe()
new_df.isnull().sum()
new_df['class'].value_counts()
new_df = new_df[new_df['class'] != 'nocode']
new_df = new_df[~new_df['class'].str.contains('\|')]
new_df['class'].value_counts()
def label(cl):
    if(cl=='happy'):
        return 0
    elif(cl=='not-relevant'):
        return 1
    elif(cl=='angry'):
        return 2
    elif(cl=='surprise'):
        return 3
    elif(cl=='sad'):
        return 4
    else:
        return 5
new_df['label']=new_df['class'].apply(label)
new_df
from sklearn.model_selection import train_test_split
X=new_df.index
Y=new_df['label']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=Y)
# x_train
new_df['data_type'] = ['not_set']*new_df.shape[0]
new_df.loc[x_train, 'data_type'] = 'train'
new_df.loc[x_test, 'data_type'] = 'test'
new_df.groupby(['class', 'label', 'data_type']).count()
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
#uncased here means that we are using all lower case data, and do_lower_case we will have to set it true as as we are using uncased bert base.
# it will ocnvert everything to lower case..
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')
encoded_data_train = tokenizer.batch_encode_plus(
    new_df[new_df['data_type']=='train'].texts.values, 
    add_special_tokens=True, # Setting this to true just to know where the sentence begin and where it ends
    return_attention_mask=True,  # equivalent to mask_zero=True in enbedding layer.To know where 
                                 # actual values are not there and it is just blank..
    pad_to_max_length=True, # padding the sequences to max length
    max_length=256, # the max length for padding
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    new_df[new_df['data_type']=='test'].texts.values, 
    add_special_tokens=True,     # Setting this to true just to know where the sentence begin and where it ends
    return_attention_mask=True,  # equivalent to mask_zero=True in enbedding layer.To know where 
                                # actual values are not there and it is just blank..
    pad_to_max_length=True, # padding the sequences to max length
    max_length=256,  # the max length for padding
    return_tensors='pt'
)
encoded_data_train.values
encoded_data_train.keys()
encoded_data_test.keys()
tokenizer.convert_ids_to_tokens(encoded_data_train['input_ids'][0])
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor( new_df[new_df['data_type']=='train'].label.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(new_df[new_df['data_type']=='test'].label.values)
input_ids_train
data_train=TensorDataset(input_ids_train, attention_masks_train, labels_train)
data_test=TensorDataset(input_ids_test, attention_masks_test, labels_test)
data_train
len(data_train)
len(data_test)
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels= 6, #number of output labels
                                                      output_attentions=False, # for ignoring unneccesary inputs from the model
                                                      output_hidden_states=False)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
dataloader_train = DataLoader(data_train, 
                              sampler=RandomSampler(data_train), 
                              batch_size=32)

dataloader_validation = DataLoader(data_test, 
                                   sampler=SequentialSampler(data_test), 
                                   batch_size=32)
from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(data_train)*epochs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
from tqdm.notebook import tqdm
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'Training Loss': '{:.2f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'Finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training Loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    tqdm.write(f'Validation loss: {val_loss}')
_, predictions, true_vals = evaluate(dataloader_validation)
predictions
predictions[0]
predictions[0].flatten()
possible_labels = new_df['class'].unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class Name: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
accuracy_per_class(predictions, true_vals)