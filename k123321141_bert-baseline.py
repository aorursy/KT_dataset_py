import time

start_time = time.time()
import torch

pos_w = torch.FloatTensor([1.]).cuda()

a = torch.FloatTensor([1, 1]).cuda()

b = torch.FloatTensor([0.3, 0.5]).cuda()

torch.nn.functional.binary_cross_entropy_with_logits(b, a, pos_weight=pos_w)

import numpy as np 

import pandas as pd 

import os

import torch

import torch.utils.data

import torch.nn.functional as F

from tqdm import tqdm, tqdm_notebook

from sklearn import metrics

!cd ../input/apex-master/apex-master/apex-master/ && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')



from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel



from apex import amp



from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification,BertAdam

BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'



bert = BertModel.from_pretrained(BERT_FP).cuda()

tokenizer = BertTokenizer(vocab_file='../input/torch-bert-weights/bert-base-uncased-vocab.txt')
identity_words = set([

    'homosexual', 'gay', 'lesbian',

    'blacks', 'black',

    'whites', 'white',

    'Muslims', 'Muslim', 'islamic', 'islam', 

    'christian', 'jesus', 'bible', 'jewish',

    'psychiatric', 'mental_illness',

    'russian', 'palestinian',

    'mexico', 'india', 'canada',

    'muslim', 'black', 'white',

    

])

identity_words = set([w.lower() for w in identity_words] + [w.lower()+'s' for w in identity_words])

def mask_identity(identity_words, sample):

    return ' '.join(['[MASK]' if w.lower() in identity_words else w for w in sample.split(' ')])

MAX_SEQUENCE_LENGTH = 220

SEED = 1234



Data_dir="../input/data/jigsaw-unintended-bias-in-toxicity-classification"

TOXICITY_COLUMN = 'target'



np.random.seed(SEED)

torch.manual_seed(SEED)

device=torch.device('cuda')



def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        text = mask_identity(identity_words, text)

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a) > max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(longer)

    return np.array(all_tokens)



tokenizer = BertTokenizer(vocab_file='../input/torch-bert-weights/bert-base-uncased-vocab.txt', do_lower_case=True)



# total_df = pd.read_csv("./data/jigsaw-unintended-bias-in-toxicity-classification/train.csv").sample(1000,random_state=SEED)

total_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv").sample(1000000, random_state=SEED)

N = len(total_df)



num_to_load= (N // 10)*9                 #Train size to match time limit

valid_size= N//10 

print('loaded %d records' % len(total_df))

# Make sure all comment_text values are strings

total_df['comment_text'] = total_df['comment_text'].astype(str) 

# Make sure all comment_text values are strings

sequences = convert_lines(total_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)

total_df=total_df.fillna(0)

# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

y_columns=['target']

# convert target to 0,1

total_df['target']=(total_df['target']>=0.5).astype(float)





X = sequences[:num_to_load]                

y = total_df[y_columns].values[:num_to_load]

X_val = sequences[num_to_load:]                

y_val = total_df[y_columns].values[num_to_load:]

train_df = total_df.head(num_to_load)

val_df = total_df.tail(valid_size)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long), torch.tensor(y_val,dtype=torch.float))

lr=2e-5

batch_size = 32

accumulation_steps=2

EPOCHS = 1

model = BertForSequenceClassification.from_pretrained(BERT_FP, num_labels=1)

model.zero_grad()

model = model.to(device)

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]

train = train_dataset



num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)



optimizer = BertAdam(optimizer_grouped_parameters,

                     lr=lr,

                     warmup=0.05,

                     t_total=num_train_optimization_steps)



model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

model = model.train()

tq = tqdm_notebook(range(EPOCHS))

iters = 1

auc = 0

for epoch in tq:

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    avg_loss = 0.

    avg_accuracy = 0.

    lossf=None

    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)

    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic

    for i,(x_batch, y_batch) in tk0:

        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)

        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device), pos_weight=pos_w)

        accuracy = torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float))

        

        with amp.scale_loss(loss, optimizer) as scaled_loss:

            scaled_loss.backward()

        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps

            optimizer.step()                            # Now we can do an optimizer step

            optimizer.zero_grad()

        

        if lossf:

            lossf = 0.98*lossf+0.02*loss.item()

        else:

            lossf = loss.item()

        tk0.set_postfix(loss = lossf, accuracy=accuracy.item(), iters=iters)

        avg_loss += loss.item() / len(train_loader)

        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)

        

                

        iters += 1

        if iters % 2000 == 0:

            torch.save(model.state_dict(), './iters_%d.mdl' % (iters))

        if iters > 35000:

            break

        if (time.time() - start_time ) // 60 > 100:

            break

print('iter : %d'  % iters)
print('iter : %d'  % iters)
model.eval()

test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

print('loaded %d records' % len(test_df))

# Make sure all comment_text values are strings

test_df['comment_text'] = test_df['comment_text'].astype(str) 

# Make sure all comment_text values are strings

sequences = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)

X = sequences

y = np.zeros([len(X), 1])

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

tk = tqdm_notebook(enumerate(test_loader),total=len(test_loader),leave=False)

idx = 7000000

model.to(device)

with open('./submission.csv', 'w') as f:

    f.write('id,prediction\n')

    with torch.no_grad():

        for i,(x_batch, _) in tk:

            out = torch.sigmoid(model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)).cpu().detach().numpy()

            for o in out.flatten():

                f.write('%d,%.1f\n' % (idx, o))

                idx += 1

print('done')


