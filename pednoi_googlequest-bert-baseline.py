!pip install transformers
import os

import random

from tqdm.notebook import tqdm



import numpy as np

import pandas as pd



from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel, AdamW, WarmupLinearSchedule



import torch

import torch.nn as nn



from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

from scipy.stats import spearmanr
SEED = 2019



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(SEED)
ROOT = '../input/google-quest-challenge/'



data_df = pd.read_csv(ROOT+'train.csv')

test_df = pd.read_csv(ROOT+'test.csv')

submission_df = pd.read_csv(ROOT+'sample_submission.csv')
N_FOLDS = 5



kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

splits = kf.split(data_df)



def get_fold(fold):    

    for i, (train_index, valid_index) in enumerate(splits):

        if i == fold:

            return data_df.iloc[train_index], data_df.iloc[valid_index]

        

train_df, valid_df = get_fold(0)
pretrained_weights = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
QLEN = 150

ALEN = 359



NUM_LABEL = 30



def encode_text(questions, answers):

    

    X = []

    

    for q, a in tqdm(zip(questions, answers)):

        q_en = tokenizer.encode(q, add_special_tokens=False, max_length=QLEN)

        q_en = q_en + [0]*(QLEN-len(q_en))

        

        a_en = tokenizer.encode(a, add_special_tokens=False, max_length=ALEN)

        a_en = a_en + [0]*(ALEN-len(a_en))

        

        X.append(tokenizer.build_inputs_with_special_tokens(q_en, a_en))

        

    return np.array(X)



train_X = encode_text(train_df['question_body'], train_df['answer'])

train_y = train_df[train_df.columns[-NUM_LABEL:]].values



valid_X = encode_text(valid_df['question_body'], valid_df['answer'])

valid_y = valid_df[valid_df.columns[-NUM_LABEL:]].values



test_X = encode_text(test_df['question_body'], test_df['answer'])
BATCH_SIZE = 12

EPOCHS = 15

LEARNING_RATE = 2e-5

ACCUM_STEPS = 1

EVAL_STEPS = 500

MAX_STEPS = -1
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X, dtype=torch.long), torch.tensor(train_y, dtype=torch.float))

train_sampler = torch.utils.data.RandomSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)



valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_X, dtype=torch.long), torch.tensor(valid_y, dtype=torch.float))

valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)

valid_loader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)
class BertForMultiLabelClassification(BertPreTrainedModel):

    def __init__(self, config):

        super(BertForMultiLabelClassification, self).__init__(config)

        self.num_labels = config.num_labels



        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)



        self.init_weights()



    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,

                position_ids=None, head_mask=None, labels=None):                     



        outputs = self.bert(input_ids,

                            attention_mask=attention_mask,

                            token_type_ids=token_type_ids,

                            position_ids=position_ids,

                            head_mask=head_mask)



        pooled_output = outputs[1]



        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        

        if labels is not None:

            loss_fct = nn.BCEWithLogitsLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs



        return outputs # (loss), logits, (hidden_states), (attentions)
config = BertConfig.from_pretrained(pretrained_weights, num_labels=NUM_LABEL)

model = BertForMultiLabelClassification.from_pretrained(pretrained_weights, config=config)
def evaluate(model):

    preds = None

    eval_loss = 0.0

    nb_eval_steps = 0

        

    for batch in valid_loader:

        model.eval()

        batch = tuple(t.to(device) for t in batch)



        with torch.no_grad():

            inputs = {'input_ids': batch[0], 'labels': batch[1]}

            outputs = model(**inputs)

            

            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()



        nb_eval_steps += 1

        

        if preds is None:

            preds = torch.sigmoid(logits).detach().cpu().numpy()

        else:

            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)



    eval_loss = eval_loss / nb_eval_steps

    

    corr = 0

    for i in range(NUM_LABEL):

        corr += spearmanr(valid_y[:, i], preds[:, i]).correlation/NUM_LABEL



    return eval_loss, corr
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

scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1500, t_total=t_total)
global_step = 0

tr_loss, logging_loss = 0.0, 0.0



tq = tqdm(range(EPOCHS))

tb_writer = SummaryWriter()



best_corr = -1.0

output_model_file = 'pytorch_model.bin'



lossf = None

step_list = []

train_loss_list = []



valid_loss, valid_corr = None, None

valid_loss_list, valid_corr_list = [], []



for _ in tq:    

    tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    

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

            

            if EVAL_STEPS > 0 and (global_step + 1) % EVAL_STEPS == 0:

                valid_loss, valid_corr = evaluate(model)



                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss)/EVAL_STEPS, global_step)

                train_loss_list.append((tr_loss - logging_loss)/EVAL_STEPS)                        



                tb_writer.add_scalar('valid_loss', valid_loss, global_step)

                tb_writer.add_scalar('valid_corr', valid_corr, global_step)



                valid_loss_list.append(valid_loss)

                valid_corr_list.append(valid_corr)



                step_list.append(global_step)

                logging_loss = tr_loss            



                if valid_corr > best_corr:

                    torch.save(model.state_dict(), output_model_file)

                    best_corr = valid_corr

                    

        lossf = 0.98*lossf + 0.02*loss.item() if lossf else loss.item()        

        tk0.set_postfix(loss=lossf, valid_loss=valid_loss, valid_corr=valid_corr)

        

        if MAX_STEPS > 0 and global_step > MAX_STEPS:

            tk0.close()

            break

    

    tq.set_postfix(loss=lossf, valid_loss=valid_loss, valid_corr=valid_corr)



    if MAX_STEPS > 0 and global_step > MAX_STEPS:

        tq.close()

        break
plt.plot(step_list, train_loss_list, label='train_loss')

plt.plot(step_list, valid_loss_list, label='valid_loss')

plt.plot(step_list, valid_corr_list, label='valid_corr')

plt.legend()

plt.ylabel('loss, corr')

plt.xlabel('step')

plt.show()
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_X, dtype=torch.long))

test_sampler = torch.utils.data.SequentialSampler(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
model_state_dict = torch.load('pytorch_model.bin')

model = BertForMultiLabelClassification.from_pretrained(pretrained_weights, config=config, state_dict=model_state_dict)

model.to(device)



preds = None



for batch in tqdm(test_loader, desc="testing"):

    model.eval()

    batch = tuple(t.to(device) for t in batch)



    with torch.no_grad():

        inputs = {'input_ids': batch[0]}

        outputs = model(**inputs)

        logits = outputs[0]



    if preds is None:

        preds = torch.sigmoid(logits).detach().cpu().numpy()

    else:

        preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)



preds_df = pd.DataFrame(data=preds, columns=submission_df.columns[-NUM_LABEL:])

submission_df = pd.concat([test_df['qa_id'], preds_df], axis=1)



submission_df.to_csv('submission.csv', index=False)
submission_df