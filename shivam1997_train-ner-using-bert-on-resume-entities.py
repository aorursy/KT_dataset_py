!pip install --no-deps seqeval[gpu]
%who
import numpy as np
import pandas as pd

import spacy
from spacy.gold import biluo_tags_from_offsets
nlp = spacy.load("en_core_web_lg")

from tqdm import trange
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

from seqeval.metrics import classification_report, accuracy_score, f1_score
# Adding '\n' to the default spacy tokenizer

prefixes = ('\\n', ) + nlp.Defaults.prefixes
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search
# Personal Custom Tags Dictionary
entity_dict = {
    'Name': 'NAME', 
    'College Name': 'CLG',
    'Degree': 'DEG',
    'Graduation Year': 'GRADYEAR',
    'Years of Experience': 'YOE',
    'Companies worked at': 'COMPANY',
    'Designation': 'DESIG',
    'Skills': 'SKILLS',
    'Location': 'LOC',
    'Email Address': 'EMAIL'
}
# loading the dataset
df = pd.read_json('/kaggle/input/resume-entities-for-ner/Entity Recognition in Resumes.json', lines=True)
df.head()
# Checking for unique values present in 'extras' column
df['extras'].unique()
# Since, 'extras' column contains no information we can drop the column
df = df.drop(['extras'], axis=1)
df.head()
def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)

    return merged
# From 'annotation' column, we are extracting the starting index, ending index, entity label
# So that we can convert the content in BILOU format

def get_entities(df):
    
    entities = []
    
    for i in range(len(df)):
        entity = []
    
        for annot in df['annotation'][i]:
            try:
                ent = entity_dict[annot['label'][0]]
                start = annot['points'][0]['start']
                end = annot['points'][0]['end'] + 1
                entity.append((start, end, ent))
            except:
                pass
    
        entity = mergeIntervals(entity)
        entities.append(entity)
    
    return entities
# Adding a new column 'entities'
df['entities'] = get_entities(df)
df.head()
def get_train_data(df):
    tags = []
    sentences = []

    for i in range(len(df)):
        text = df['content'][i]
        entities = df['entities'][i]
    
        doc = nlp(text)
    
        tag = biluo_tags_from_offsets(doc, entities)
        tmp = pd.DataFrame([list(doc), tag]).T
        loc = []
        for i in range(len(tmp)):
            if tmp[0][i].text is '.' and tmp[1][i] is 'O':
                loc.append(i)
        loc.append(len(doc))
    
        last = 0
        data = []
        for pos in loc:
            data.append([list(doc)[last:pos], tag[last:pos]])
            last = pos
    
        for d in data:
            tag = ['O' if t is '-' else t for t in d[1]]
            if len(set(tag)) > 1:
                sentences.append(d[0])
                tags.append(tag)
    
    return sentences, tags
sentences, tags = get_train_data(df)
len(sentences), len(tags)
tag_vals = set(['X', '[CLS]', '[SEP]'])
for i in range(len(tags)):
    tag_vals = tag_vals.union(tags[i])
tag_vals
tag2idx = {t: i for i, t in enumerate(tag_vals)}
tag2idx
idx2tag = {tag2idx[key] : key for key in tag2idx.keys()}
idx2tag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
def get_tokenized_train_data(sentences, tags):

    tokenized_texts = []
    word_piece_labels = []

    for word_list, label in zip(sentences, tags):
    
        # Add [CLS] at the front
        temp_lable = ['[CLS]']
        temp_token = ['[CLS]']
    
        for word, lab in zip(word_list, label):
            token_list = tokenizer.tokenize(word.text)
            for m, token in enumerate(token_list):
                temp_token.append(token)
                if m == 0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append('X')  
                
        # Add [SEP] at the end
        temp_lable.append('[SEP]')
        temp_token.append('[SEP]')
    
        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_lable)
    
    return tokenized_texts, word_piece_labels
tokenized_texts, word_piece_labels = get_tokenized_train_data(sentences, tags)
print(tokenized_texts[0])
print(word_piece_labels[0])
MAX_LEN = 512
bs = 4
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print(len(input_ids[0]))
print(input_ids[0])
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels], maxlen=MAX_LEN, value=tag2idx["O"], 
                     padding="post", dtype="long", truncating="post")
print(len(tags[0]))
print(tags[0])
attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
print(attention_masks[0])
tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(input_ids, tags, attention_masks, random_state=2020, 
                                                                                 test_size=0.3)
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))
model.cuda();
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
epochs = 10
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
model.eval()

y_true = []
y_pred = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, label_ids = batch

    with torch.no_grad():
        logits = model(input_ids, token_type_ids=None, attention_mask=input_mask,)

    logits = logits.detach().cpu().numpy()
    logits = [list(p) for p in np.argmax(logits, axis=2)]
    
    label_ids = label_ids.to('cpu').numpy()
    input_mask = input_mask.to('cpu').numpy()
    
    for i,mask in enumerate(input_mask):
        temp_1 = [] # Real one
        temp_2 = [] # Predict one
        
        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare
            if m:
                if idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != "[CLS]" and idx2tag[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                    temp_1.append(idx2tag[label_ids[i][j]])
                    temp_2.append(idx2tag[logits[i][j]])
            else:
                break
        
            
        y_true.append(temp_1)
        y_pred.append(temp_2)
    
print("f1 socre: %f"%(f1_score(y_true, y_pred)))
print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

print(classification_report(y_true, y_pred,digits=4))

