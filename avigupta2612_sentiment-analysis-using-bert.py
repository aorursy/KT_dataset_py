import pandas as pd

import numpy as np

import torch
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_train.head()
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

df_test.head()
df_train.target.value_counts()
df_train = df_train[['text','target']]

df_train.head()
from sklearn.model_selection import train_test_split

xtrain,xval,ytrain,yval = train_test_split(df_train.index.values, df_train.target.values,

                                           test_size = 0.2, random_state=15, stratify = df_train.target.values)

print(len(xtrain),len(xval))
df_train['set_type'] = 'nil'*df_train.shape[0]

df_train.loc[xtrain, 'set_type'] = 'train'

df_train.loc[xval, 'set_type'] = 'val'

df_train.head(10)
df_train.groupby(['target', 'set_type']).count()
df_test=df_test[['text']]

df_test.head()
from transformers import BertTokenizer

from torch.utils.data import TensorDataset

tokenizer= BertTokenizer.from_pretrained('bert-base-uncased',

                                        do_lower_case=True)
encoded_train = tokenizer.batch_encode_plus(

    df_train[df_train.set_type=='train'].text.values,

    add_special_tokens=True,

    return_attention_masks=True,

    pad_to_max_length=True,

    max_length=256,

    return_tensors='pt'

)

encoded_val = tokenizer.batch_encode_plus(

    df_train[df_train.set_type=='val'].text.values,

    add_special_tokens=True,

    return_attention_masks=True,

    pad_to_max_length=True,

    max_length=256,

    return_tensors='pt'

)

encoded_test = tokenizer.batch_encode_plus(

    df_test.text.values,

    add_special_tokens=True,

    return_attention_masks=True,

    pad_to_max_length=True,

    max_length=256,

    return_tensors='pt'

)

input_ids_train = encoded_train['input_ids']

attention_masks_train = encoded_train['attention_mask']

labels_train = torch.tensor(df_train[df_train.set_type=='train'].target.values)



input_ids_val = encoded_val['input_ids']

attention_masks_val = encoded_val['attention_mask']

labels_val = torch.tensor(df_train[df_train.set_type=='val'].target.values)



input_ids_test = encoded_test['input_ids']

attention_masks_test = encoded_test['attention_mask']
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

dataset_test = TensorDataset(input_ids_test, attention_masks_test)
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',

                                     num_labels = 2,

                                     output_attentions = False,

                                     output_hidden_states = False

                                     )
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

dataloader_train = DataLoader(

    dataset_train,

    sampler= RandomSampler(dataset_train),

    batch_size=32

)

dataloader_val = DataLoader(

    dataset_val,

    sampler = SequentialSampler(dataset_val),

    batch_size=32

)

dataloader_test = DataLoader(

    dataset_test,

    sampler = SequentialSampler(dataset_test),

    batch_size=32

)
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),

                 lr=1e-5,

                 eps=1e-8)

epochs = 4

scheduler = get_linear_schedule_with_warmup(

    optimizer,

    num_warmup_steps=0,

    num_training_steps=len(dataloader_train)*epochs

)
from sklearn.metrics import f1_score

def f1_score_func(preds, labels):

    preds_flat = np.argmax(preds, axis =1).flatten()

    labels_flat = labels.flatten()

    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):

    preds_flat = np.argmax(preds, axis =1).flatten()

    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):

        y_preds = preds_flat[labels_flat==label]

        y_true = labels_flat[labels_flat==label]

        print(f'Class: {label}')

        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}')
import random



seed_val = 10

random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    training_loss=0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch),

                        leave=False,

                        disable=False

                       )

    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {

            'input_ids': batch[0],

            'attention_mask':batch[1],

            'labels':batch[2]

        }

        outputs = model(**inputs)

        loss=outputs[0]

        training_loss+= loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    

    

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = training_loss/len(dataloader_train)

    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_val)

    val_f1 = f1_score_func(predictions, true_vals)

    tqdm.write(f'Validation Loss: {val_loss}')

    tqdm.write(f'F1 score: {val_f1}')

    

        
_, predictions, true_vals = evaluate(dataloader_val)
accuracy_per_class(predictions, true_vals)
len(dataloader_test)
model.eval()

predictions=[]

for batch in dataloader_test:

    batch = tuple(b.to(device) for b in batch)

        

    inputs = {'input_ids':      batch[0],

              'attention_mask': batch[1]

             }



    with torch.no_grad():        

        outputs = model(**inputs)

    

    logits = outputs[0]



    logits = logits.detach().cpu().numpy()

    predictions.append(np.argmax(logits,axis=1))



    
from itertools import chain

prediction = list(chain.from_iterable(predictions))

sub= pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub.head()
sub['target']=prediction
sub.head()
sub.to_csv('submission.csv', index=False)