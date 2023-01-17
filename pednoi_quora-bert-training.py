import numpy as np

import pandas as pd



import os

import random



import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook, trange



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score
!pip install transformers
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW, WarmupLinearSchedule



import torch

from torch.utils.tensorboard import SummaryWriter
SEED = 2019



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(SEED)
data_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
print(len(data_df))

data_df.head()
print(len(test_df))

test_df.head()
N_FOLDS = 5



skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

splits = skf.split(data_df, data_df['target'])



def get_fold(fold):    

    for i, (train_index, valid_index) in enumerate(splits):

        if i == fold:

            return data_df.iloc[train_index], data_df.iloc[valid_index]

        

train_df, valid_df = get_fold(0)
pd.value_counts(train_df['target']).plot.bar()

plt.show()
pd.value_counts(valid_df['target']).plot.bar()

plt.show()
MAX_LENGTH = 320

BATCH_SIZE = 24

EPOCHS = 1

LEARNING_RATE = 2e-5

ACCUM_STEPS = 1

EVAL_STEPS = 10000

MAX_STEPS = 20000
pretrained_weights = 'bert-base-uncased'



config = BertConfig.from_pretrained(pretrained_weights, num_labels=2)

model = BertForSequenceClassification.from_pretrained(pretrained_weights, config=config)

tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
def encode_text(texts):

    

    # encoding

    X = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LENGTH) 

         for text in tqdm(texts)]

    

    # padding

    X = [x + [0 for _ in range(MAX_LENGTH-len(x))] for x in X]            

    

    return X



train_X = encode_text(train_df['question_text'])

train_y = train_df['target'].values



valid_X = encode_text(valid_df['question_text'])

valid_y = valid_df['target'].values



test_X = encode_text(test_df['question_text'])
def evaluate(model):    

    preds = None

    eval_loss = 0.0

    nb_eval_steps = 0

    

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_X, dtype=torch.long), torch.tensor(valid_y, dtype=torch.long))

    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)    

    

    for batch in tqdm_notebook(valid_loader, desc="Evaluating"):

        model.eval()

        batch = tuple(t.to(device) for t in batch)



        with torch.no_grad():

            inputs = {'input_ids': batch[0], 'labels': batch[1]}

            outputs = model(**inputs)

            

            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()



        nb_eval_steps += 1

        

        if preds is None:

            preds = logits.detach().cpu().numpy()

        else:

            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)



    eval_loss = eval_loss / nb_eval_steps

    

    preds = np.argmax(preds, axis=1)

    f1 = f1_score(valid_y, preds)

    

    return eval_loss, f1
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X, dtype=torch.long), torch.tensor(train_y, dtype=torch.long))

train_sampler = torch.utils.data.RandomSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
device = torch.device('cuda')

model = model.to(device)



model.zero_grad()



no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]



if MAX_STEPS > 0:

    t_total = MAX_STEPS

else:

    t_total = len(train_loader) // ACCUM_STEPS * EPOCHS



optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)
global_step = 0

tr_loss, logging_loss = 0.0, 0.0



tq = tqdm_notebook(range(EPOCHS))

tb_writer = SummaryWriter()



best_f1 = -1 #0.0

output_model_file = 'pytorch_model.bin'



for _ in tq:

    lossf = None

    eval_loss, eval_f1 = None, None

    

    tk0 = tqdm_notebook(enumerate(train_loader), total=len(train_loader), leave=False)

    

    for step, batch in tk0:

        model.train()

        batch = tuple(t.to(device) for t in batch)

        

        inputs = {'input_ids': batch[0], 'labels': batch[1]}



        outputs = model(**inputs)

        loss = outputs[0] # model outputs are always tuple in transformers (see doc)



        if ACCUM_STEPS > 1:

            loss = loss / ACCUM_STEPS



        loss.backward()

        tr_loss += loss.item()

        

        if (step + 1) % ACCUM_STEPS == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            

            optimizer.step()

            scheduler.step()

            model.zero_grad()

            

            global_step += 1

            

        if EVAL_STEPS > 0 and (step + 1) % EVAL_STEPS == 0:

            eval_loss, eval_f1 = evaluate(model)

            

            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/EVAL_STEPS, global_step)

            logging_loss = tr_loss

            

            tb_writer.add_scalar('eval_loss', eval_loss, global_step)

            tb_writer.add_scalar('eval_f1', eval_f1, global_step)

            

            if eval_f1 > best_f1:

                torch.save(model.state_dict(), output_model_file)

                best_f1 = eval_f1

                    

        lossf = 0.98*lossf + 0.02*loss.item() if lossf else loss.item()        

        tk0.set_postfix(loss=lossf, eval_loss=eval_loss, eval_f1=eval_f1)

        

        if MAX_STEPS > 0 and global_step > MAX_STEPS:

            tk0.close()

            break



    if MAX_STEPS > 0 and global_step > MAX_STEPS:

        tq.close()

        break        
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_X, dtype=torch.long))

test_sampler = torch.utils.data.SequentialSampler(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
model_state_dict = torch.load('pytorch_model.bin')

model = BertForSequenceClassification.from_pretrained(pretrained_weights, state_dict=model_state_dict)

model.to(device)



preds = None



for batch in tqdm_notebook(test_loader, desc="testing"):

    model.eval()

    batch = tuple(t.to(device) for t in batch)



    with torch.no_grad():

        inputs = {'input_ids': batch[0]}

        outputs = model(**inputs)

        logits = outputs[0]



    if preds is None:

        preds = logits.detach().cpu().numpy()

    else:

        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)



preds = np.argmax(preds, axis=1)



submission = test_df[['qid']]

submission['prediction'] = preds

submission.to_csv('submission.csv', index=False)
submission.head()