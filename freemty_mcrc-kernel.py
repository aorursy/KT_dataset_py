
import os , re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tokenizers
import transformers
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import sys
sys.path.append("/kaggle/input/functions/")
import utils
from torch.nn.modules.loss import _WeightedLoss
from IPython.display import FileLink
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
MAX_LEN = 486
VALID_BATCH_SIZE = 6
TRAIN_BATCH_SIZE = 6
EPOCHS = 5
ROBERTA_PATH = '/kaggle/input/roberta-zh'
TRAIN_PATH = '/kaggle/input/dataset/train.csv'
VALID_PATH = '/kaggle/input/dataset/dev.csv'
TEST_PATH = '/kaggle/input/dataset/test.csv'
MODEL_PATH = '/kaggle/input/baidu-mrc/model.bin'

TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(ROBERTA_PATH, "vocab.txt"),
    lowercase=True,
    handle_chinese_chars = False)

def process_data(context, question, tokenizer, seq_len ,  answer = None ,data_type = 'train'):
    
    context_token = tokenizer.encode(context)
    question_token = tokenizer.encode(question)
    len_context = len(context)
    #len_context1 = len(context_token.original_str)
    len_context1= len(context_token.normalized_str)
    if len_context != len_context1:
        context = str(context_token.normalized_str)

    context_ids = context_token.ids[1:-1]
    question_ids = question_token.ids[1:-1][0:48]
    offsets = context_token.offsets[1:-1]

    if data_type == 'train':
        answer_token = tokenizer.encode(answer)
        len_as = len(answer)
        char_target = [0] * len(context)
        for ind in (i for i, e in enumerate(context) if (e == answer[0] or e == answer[0].lower())):
            if (context[ind: ind+len_as] == answer or context[ind: ind+len_as] == answer.lower()):
                idx0 = ind
                idx1 = ind + len_as
                break
        #print(orig_context[idx0:idx1])
        for ct in range(idx0, idx1):
                char_target[ct] = 1
        target = [0] * len(context_ids)
        for i , (of1 , of2) in enumerate(offsets):
            if np.sum(char_target[of1 : of2]) > 0:
                target[i] = 1
        '''
        target_start = np.nonzero(target)[0][0]
        target_end = np.nonzero(target)[0][-1]
        #test the traget
        filtered_output  = ""
        for ix in range(target_start, target_end + 1):
            filtered_output += orig_context[offsets[ix][0]: offsets[ix][1]]
            if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                filtered_output += orig_context[offsets[ix][1] : offsets[ix+1][0]]
        print('orig_answer:' + answer)
        print('pred_output:' + filtered_output)
        '''
    ids = [101] + question_ids + [102] + context_ids + [102]
    ids_type = [0] * (len(question_ids) + 2) + [1] * len(offsets) + [1]
    offsets = [(0,0)] * (len(question_ids) + 2) + offsets + [(0,0)]
    if data_type == 'train':
        target = [0] * (len(question_ids) + 2) + target + [0]
        target_start = np.nonzero(target)[0][0]
        target_end = np.nonzero(target)[0][-1]

    if len(ids) > seq_len:
        ids = ids[:seq_len-1] + [102]
        ids_type = ids_type[:seq_len-1] + [1]
        offsets = offsets[:seq_len-1] + [(0,0)]
        if data_type == 'train':
            target = target[:seq_len-1] + [0]
            if target_end >= (seq_len - 1):
                target_end  = seq_len - 2
            if target_start >= (seq_len - 1):
                target_start = seq_len - 2

    mask = [1] * len(ids)
    padding_len = seq_len - len(ids)  
    if padding_len > 0:
        ids += [0] * padding_len
        offsets += [(0,0)] * padding_len
        mask +=  [0] * padding_len
        ids_type += [0] * padding_len
        if data_type == 'train':
            target += [0] * padding_len

    if data_type =='train':
        return{
            'ids':ids,
            'mask':mask,
            'ids_type':ids_type,
            'target':target,
            'target_start':target_start,
            'target_end':target_end,
            'offsets':offsets,
            'orig_context':context,
            'orig_question':question,
            'orig_answer':answer
        }
    else:
        return{
            'ids':ids,
            'mask':mask,
            'ids_type':ids_type,
            'offsets':offsets,
            'orig_context':context,
            'orig_question':question,
        }
class QA_dataset:
    def __init__(self , context , question , answer ):
        self.tokenizer = TOKENIZER
        self.seq_len = MAX_LEN
        self.orig_context = context
        self.orig_question = question
        self.orig_answer = answer

    def __len__(self):
        return len(self.orig_context)

    def __getitem__(self , item):
        context = str(self.orig_context[item])
        question = str(self.orig_question[item])
        answer = eval(self.orig_answer[item])

        data = process_data(
            context = context, 
            question = question,
            answer = answer[0], 
            tokenizer = self.tokenizer,
            seq_len = self.seq_len,
            data_type = 'train'
        )

        return{
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'ids_type': torch.tensor(data["ids_type"], dtype=torch.long),
            'targets_start': torch.tensor(data["target_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["target_end"], dtype=torch.long),
            'orig_context': data["orig_context"],
            'orig_question': data["orig_question"],
            'orig_answer': data["orig_answer"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }
class QA_dataset_valid:
    def __init__(self , context , question , answer = None):
        self.tokenizer = TOKENIZER
        self.seq_len = MAX_LEN
        self.orig_context = context
        self.orig_question = question
        self.orig_answer = answer

    def __len__(self):
        return len(self.orig_context)

    def __getitem__(self , item):
        context = str(self.orig_context[item])
        question = str(self.orig_question[item])
        answer = eval(self.orig_answer[item])
        

        data = process_data(
            context = context, 
            question = question,
            tokenizer = self.tokenizer,
            seq_len = self.seq_len,
            data_type = 'valid'
        )

        return{
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'ids_type': torch.tensor(data["ids_type"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'orig_context': data["orig_context"],
            'orig_question': data["orig_question"],
            'orig_answer': answer,
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }
class QA_dataset_test:
    def __init__(self , context , question ):
        self.tokenizer = TOKENIZER
        self.seq_len = MAX_LEN
        self.orig_context = context
        self.orig_question = question

    def __len__(self):
        return len(self.orig_context)

    def __getitem__(self , item):
        context = str(self.orig_context[item])
        question = str(self.orig_question[item])
        data = process_data(
            context = context, 
            question = question,
            tokenizer = self.tokenizer,
            seq_len = self.seq_len,
            data_type = 'valid'
        )

        return{
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'ids_type': torch.tensor(data["ids_type"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'orig_context': data["orig_context"],
            'orig_question': data["orig_question"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }
def collate_fn(batch):
	# 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
    orig_context = [item['orig_context'] for item in batch]
    orig_question = [item['orig_question'] for item in batch]
    offsets = [item['offsets'] for item in batch]
    ids = [item["ids"] for item in batch]
    mask = [item["mask"] for item in batch]
    ids_type= [item["ids_type"] for item in batch]

    orig_answer = [item['orig_answer'][:] for item in batch]

    return {
            'ids': torch.stack(ids , dim = 0) ,
            'ids_type': torch.stack(ids_type ,dim = 0),
            'mask': torch.stack(mask , dim = 0),
            'orig_context': orig_context,
            'orig_question': orig_question,
            'orig_answer': orig_answer,
            'offsets': torch.stack(offsets , dim = 0)
    }


class ModelV1(transformers.BertPreTrainedModel):
    def __init__(self):
        model_config = transformers.BertConfig.from_pretrained(ROBERTA_PATH)
        model_config.output_hidden_states = True
        super(ModelV1, self).__init__(model_config)
        self.roberta = transformers.BertModel.from_pretrained(ROBERTA_PATH, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.Cov1S = nn.Conv1d(768 * 2, 128 , kernel_size = 2 ,stride = 1 )
        self.Cov1E = nn.Conv1d(768 * 2, 128, kernel_size = 2 ,stride = 1 )
        self.Cov2S = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.Cov2E = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.lS = nn.Linear(64 , 1)
        self.lE = nn.Linear(64 , 1)
        torch.nn.init.normal_(self.lS.weight, std=0.02)
        torch.nn.init.normal_(self.lE.weight, std=0.02)
        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        out = out.permute(0,2,1)
        same_pad1 = torch.zeros(ids.shape[0], 768*2 , 1).cuda()
        same_pad2 = torch.zeros(ids.shape[0] , 128 , 1).cuda()
        out1 = torch.cat((same_pad1 , out), dim = 2)
        out1 = self.Cov1S(out1)
        out1 = torch.cat((same_pad2 , out1), dim = 2)
        out1 = self.Cov2S(out1)
        out1 = F.leaky_relu(out1)
        out1 = out1.permute(0,2,1)
        start_logits = self.lS(out1).squeeze(-1)

        out2 = torch.cat((same_pad1 , out), dim = 2)
        out2 = self.Cov1E(out2)
        out2 = torch.cat((same_pad2 , out2), dim = 2)
        out2 = self.Cov2E(out2)
        out2 = F.leaky_relu(out2)
        out2 = out2.permute(0,2,1)
        end_logits = self.lE(out2).squeeze(-1)
        #print(start_logits.shape)
        return start_logits, end_logits

def inference(orig_context ,offset ,idx_start ,idx_end ,verbose = False):
    #test the traget

    final_output  = ""

    for ix in range(idx_start, idx_end + 1):
        final_output += orig_context[offset[ix][0]: offset[ix][1]]
        if (ix+1) < len(offset) and offset[ix][1] < offset[ix+1][0]:
            final_output += orig_context[offset[ix][1] : offset[ix+1][0]]

    if verbose == True:
            print(f"Output= {final_output.strip()}")

    return final_output
    

def get_score(pred , answer , verbose = True):
    if verbose :
        print('answer:' + str(answer))
        print('pred :' + str(pred))
    f1 = utils.calc_f1_score(answer , pred)
    em = utils.calc_em_score(answer , pred)
    return f1, em
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.15):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss
def get_best_start_end_idxs(_start_logits, _end_logits):
    best_logit = -1000
    best_idxs = None
    for start_idx, start_logit in enumerate(_start_logits):
        for end_idx, end_logit in enumerate(_end_logits[start_idx:]):
            logit_sum = (start_logit + end_logit).item()
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (start_idx, start_idx+end_idx)
    return best_idxs
def loss_fn(o1, o2, t1, t2):
    dn = nn.CrossEntropyLoss()
    fn = SmoothCrossEntropyLoss()
    l1 = fn(o1, t1)
    l2 = fn(o2, t2)
    return l1 + l2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["ids_type"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]
        

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)  

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        #print(targets_start)
        #print(targets_end)
        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
def eval_fn(data_loader, model, device, scheduler):
    model.eval()
    EM = utils.AverageMeter()
    F1 = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["ids_type"]
        mask = d["mask"]
        orig_context = d['orig_context']
        orig_answer = d['orig_answer']
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        outputs_start = torch.softmax(o1 , axis = 1).cpu().detach().numpy()
        outputs_end = torch.softmax(o2 , axis = 1).cpu().detach().numpy()
        offsets = d["offsets"].cpu().numpy()
        ems , f1s = [] , []
        for px, context in enumerate(orig_context):
            answers = orig_answer[px]
            offset = offsets[px]
            idx_start , idx_end = get_best_start_end_idxs(outputs_start[px,:],outputs_end[px,:])
            final_output = inference(
                offset = offset,
                orig_context = context,
                idx_start = idx_start,
                idx_end = idx_end,
                )

            f1 , em = get_score(final_output , answers , verbose = False)
            f1s.append(f1)
            ems.append(em)
        F1.update(np.mean(f1s) , ids.size(0))
        EM.update(np.mean(ems) , ids.size(0))
        tk0.set_postfix(F1 = F1.avg , EM = EM.avg)
        scheduler.step()

    return F1.avg , EM.avg
def pred_fn(data_loader, model, device, scheduler):

    preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["ids_type"]
        mask = d["mask"]
        orig_context = d['orig_context']
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        outputs_start = torch.softmax(o1 , axis = 1).cpu().detach().numpy()
        outputs_end = torch.softmax(o2 , axis = 1).cpu().detach().numpy()
        offsets = d["offsets"].cpu().numpy()
        for px, context in enumerate(orig_context):
            offset = offsets[px]
            idx_start , idx_end = get_best_start_end_idxs(outputs_start[px,:],outputs_end[px,:])
            final_output = inference(
                offset = offset,
                orig_context = context,
                idx_start = idx_start,
                idx_end = idx_end,
                )
            preds.append(final_output)
    return preds
def run(action = 'eval' , fold = None):
    
    dfx = pd.read_csv(TRAIN_PATH)
    dfv = pd.read_csv(VALID_PATH)
    if action == 'train':
            df_train = dfx[dfx.fold_num != fold].reset_index(drop=True)
            df_valid = dfx[dfx.fold_num == fold].reset_index(drop=True)
    if action == 'valid':
            df_train = dfx.reset_index(drop=True)
            df_valid = dfv.reset_index(drop=True)
    
    df_test = pd.read_csv(TEST_PATH)

    train_dataset = QA_dataset(
        context=df_train.context.values,
        answer=df_train.text.values,
        question=df_train.question.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=0
    )
    valid_dataset = QA_dataset_valid(
        context=df_valid.context.values,
        answer=df_valid.text.values,
        question=df_valid.question.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=0,
        collate_fn= collate_fn
    )
    
    test_dataset = QA_dataset_test(
        context=df_test.context.values[:],
        question=df_test.question.values[:]
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=0,
    )

    device = torch.device("cuda")
    model = ModelV1()
    model.to(device)

    if action != 'train':
        model.load_state_dict(torch.load(MODEL_PATH , map_location=device))
        
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    f1_max = 0
    em_max = 0
    if action == 'train':
        for i in range(5):
            train_fn(train_data_loader, model, optimizer, device, scheduler)
            if i >=3 :
                f1 , em= eval_fn(valid_data_loader, model,device, scheduler)
                print(f"epoch{i+1}: F1 = {f1} , EM = {em}")
                if (f1 > f1_max ):
                    torch.save(model.state_dict(),f"model_{fold}.bin")
                    f1_max = f1
                    em_max = em
    if action == 'eval':
        eval_fn(valid_data_loader, model ,device, scheduler)
    if action == 'pred':
        preds = pred_fn(test_data_loader, model, device, scheduler)
        preds = pd.DataFrame(preds)
        preds.to_csv('submission.csv' , index = 0)
        
run('train',0)
'''
run('train',0)
run('train',1)
run('train',2)
run('train',3)
run('train',4)
run('eval')
'''
#run('eval')
#run('pred')
os.chdir(r'/kaggle/working')
FileLink(r'model.bin')