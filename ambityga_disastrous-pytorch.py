import pandas as pd
import numpy as np
import torch
import transformers
import time 
import datetime
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, \
                RandomSampler, SequentialSampler
import torch.nn.functional as F
print("Transformers:",transformers.__version__)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train = train.sample(frac=1,random_state = 124).reset_index(drop=True)
print(np.std([len(x) for x in train['text']]))
print(np.std([len(x) for x in test['text']]))
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case = True)
tokenizer.encode_plus(train['text'][1],max_length=60).keys()
def encode(values):
    ids = []
    masks = []
    for keyword,t in values:
        encodes_dict = tokenizer.encode_plus(text = str(keyword),
                                             text_pair=str(t),
                                             truncation = True,
                                             max_len = 64,
                                             pad_to_max_length=True,
                                             return_token_type_ids=False)
        ids.append(encodes_dict['input_ids'])
        masks.append(encodes_dict['attention_mask'])
    return ids, masks
X_train,X_val,y_train,y_val = train_test_split(train.loc[:,['text','keyword']].values,
                                               train['target'].values,
                                               random_state = 128,
                                               test_size = 0.2,
                                               stratify = train['target'])
X_train = torch.tensor(encode(X_train))
X_val = torch.tensor(encode(X_val))

y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
id_train,mask_train = X_train
id_val,mask_val = X_val
BATCH_SIZE = 32
X_train = TensorDataset(id_train,mask_train, y_train)
train_sampler = RandomSampler(X_train)
train_dataloader = DataLoader(X_train,sampler = train_sampler,
                              batch_size = BATCH_SIZE)

X_val = TensorDataset(id_val,mask_val, y_val)
val_sampler = RandomSampler(X_val)
val_dataloader = DataLoader(X_val,sampler = val_sampler,
                              batch_size = BATCH_SIZE)
import torch
clf = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                   output_attentions = False,
                                                   output_hidden_states = False,
                                                                  num_labels = 2)
clf.trainable = False
clf.cuda()
del model
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased",
                                                   output_attentions = False,
                                                   output_hidden_states = False)
        self.bert.trainable = False
        self.drop = torch.nn.Dropout(0.4,inplace = True)
        self.linear = torch.nn.Linear(768,2)
        
        
    def forward(self,ids, masks,labels):
        x = self.bert(ids,masks)
        x = self.drop(x[1])
        x = self.linear(F.softmax(x))
        return x
clf = Classifier()
clf.cuda()        
optimizer = transformers.AdamW(clf.parameters(),
                              lr = 2e-5,
                              eps = 1e-8)
epochs = 1
total_steps = len(train_dataloader)*epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps = 0,
                                                         num_training_steps = total_steps)
def flat_accuracy(preds,labels):
    pred_flat = np.argmax(preds,axis =1).flatten()
    labels_flat = labels.flat()
    return np.sum(pred_flat==labels_flat)/len(labels_flat)
def format_time(elapsed):
    elapsed_roounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
random.seed(50)
np.random.seed(50)
torch.manual_seed(50)
torch.cuda.manual_seed_all(50)

loss_arr = []
for i in range(epochs):
    print("Epoch({:}/{:}) :".format(i+1,epochs))
    t0 = time.time()
    total_loss = 0
    clf.train()
    for step,batch in enumerate(train_dataloader):
        if step % 30 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        clf.zero_grad()
        ouptuts = clf(b_ids,b_masks,labels = b_labels)
        loss = outputs.item()
        total_loss += loss.item()
        out.backward()
        optimtizer.step()
        scheduler.step()
    avg_train_loss = train_loss/len(train_dataloader)
    
    loss_arr.append(avg_train_loss)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    # ========= Validation ==========
    
    print("")
    print("Running Validation...")
    t0 = time.time()
    # evaluation mode
    clf.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    for batch in val_dataloader:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        
        with torch.no_grad():
            
            outputs = clf(b_input_ids, 
                           token_type_ids = None, 
                           attention_mask = b_input_mask)
            
        logits = outputs[0]
        # move logits to cpu
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # get accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_steps += 1
    
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    
torch.cuda.get_device_name(0)

for _ in range(100):
    torch.cuda.empty_cache()
torch.cuda.memory_cached(0)/(1024*1024*1024)
del clf, b_ids, b_masks, b_labels
del outputs
