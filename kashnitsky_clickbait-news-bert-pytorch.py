import os, sys, shutil

import time

import gc

from contextlib import contextmanager

from pathlib import Path

import random

import numpy as np, pandas as pd

from tqdm import tqdm, tqdm_notebook



from sklearn.metrics import f1_score, classification_report

from sklearn.preprocessing import LabelBinarizer

import torch

import torch.nn as nn

import torch.utils.data



from matplotlib import pyplot as plt

%config InlineBackend.figure_formats = ['retina']
MAX_SEQUENCE_LENGTH = 512    # maximal possible sequence length is 512. Generally, the higher, the better, but more GPU memory consumed

BATCH_SIZE = 16              # refer to the table here https://github.com/google-research/bert to adjust batch size to seq length



SEED = 1234

EPOCHS = 2                                   

PATH_TO_DATA = Path("../input/clickbait-news-detection/")

WORK_DIR = Path("../working/")



LRATE = 2e-5             # hard to tune with BERT, but this shall be fine (improvements: LR schedules, fast.ai wrapper for LR adjustment)

ACCUM_STEPS = 2          # wait for several backward steps, then one optimization step, this allows to use larger batch size

                         # well explained here https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

WARMUP = 5e-2            # warmup helps to tackle instability in the initial phase of training with large learning rates. 

                         # During warmup, learning rate is gradually increased from 0 to LRATE.

                         # WARMUP is a proportion of total weight updates for which warmup is done. By default, it's linear warmup

USE_APEX = True         # using APEX shall speedup training (here we use mixed precision training), https://github.com/NVIDIA/apex
# nice way to report running times

@contextmanager

def timer(name):

    t0 = time.time()

    yield

    print(f'[{name}] done in {time.time() - t0:.0f} s')
# make results fully reproducible

def seed_everything(seed=123):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
if USE_APEX:

    with timer('install Nvidia apex'):

        # Installing Nvidia Apex

        os.system('git clone https://github.com/NVIDIA/apex; cd apex; pip install -v --no-cache-dir' + 

                  ' --global-option="--cpp_ext" --global-option="--cuda_ext" ./')

        os.system('rm -rf apex/.git') # too many files, Kaggle fails

        from apex import amp
device = torch.device('cuda')
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!unzip uncased_L-12_H-768_A-12.zip

BERT_MODEL_PATH = Path('uncased_L-12_H-768_A-12/')
# Add the Bert Pytorch repo to the PATH using files from: https://github.com/huggingface/pytorch-pretrained-BERT

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig



# Translate model from tensorflow to pytorch

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    str(BERT_MODEL_PATH / 'bert_model.ckpt'),

    str(BERT_MODEL_PATH / 'bert_config.json'),

    str(WORK_DIR / 'pytorch_model.bin')

)



shutil.copyfile(BERT_MODEL_PATH / 'bert_config.json', WORK_DIR / 'bert_config.json')

bert_config = BertConfig(str(BERT_MODEL_PATH / 'bert_config.json'))
# Converting the lines to BERT format

def convert_lines(example, max_seq_length, tokenizer):

    max_seq_length -= 2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a) > max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(f"There are {longer} lines longer than {max_seq_length}")

    return np.array(all_tokens)
with timer('Read data and convert to BERT format'):

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,

                                              do_lower_case=True)

    train_df = pd.read_csv(PATH_TO_DATA / "train.csv")



    valid_df = pd.read_csv(PATH_TO_DATA / "valid.csv")

    

    train_val_df = pd.concat([train_df, valid_df])

    

    test_df = pd.read_csv(PATH_TO_DATA / "test.csv")

    

    print('loaded {} train + validation and {} test records'.format(

        len(train_val_df), len(test_df)))



    # Make sure all text values are strings

    train_val_df['text'] = train_val_df['title'].astype(str).fillna("DUMMY_VALUE") + '_' + train_val_df['text'].astype(str).fillna("DUMMY_VALUE") 

    test_df['text'] = test_df['title'].astype(str).fillna("DUMMY_VALUE") + '_' + test_df['text'].astype(str).fillna("DUMMY_VALUE") 

    

    X_train = convert_lines(train_val_df["text"], MAX_SEQUENCE_LENGTH, tokenizer)

    X_test = convert_lines(test_df["text"], MAX_SEQUENCE_LENGTH, tokenizer)

    

    label_binarizer = LabelBinarizer()

    y_train = label_binarizer.fit_transform(train_val_df['label'])

    

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), 

                                               torch.tensor(y_train, dtype=torch.float))

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

    

#     del train_val_df, train_df, valid_df, test_df; gc.collect()
with timer('Setting up BERT'):

    output_model_file = "bert_pytorch.bin"

    seed_everything(SEED)

    model = BertForSequenceClassification.from_pretrained(WORK_DIR,

                                                          cache_dir=None,

                                                          num_labels=y_train.shape[1])

    model.zero_grad()

    model = model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 

         'weight_decay': 0.01},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 

         'weight_decay': 0.0}

        ]



    num_train_optimization_steps = int(EPOCHS * len(train_dataset) / BATCH_SIZE / ACCUM_STEPS)



    optimizer = BertAdam(optimizer_grouped_parameters,

                         lr=LRATE, warmup=WARMUP,

                         t_total=num_train_optimization_steps)

    if USE_APEX:

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    model = model.train()
with timer('Training'):

    loss_history = []

    tq = tqdm_notebook(range(EPOCHS))

    for epoch in tq:

        train_loader = torch.utils.data.DataLoader(train_dataset, 

                                                   batch_size=BATCH_SIZE, 

                                                   shuffle=True)

        avg_loss = 0.

        lossf = None

        tk0 = tqdm_notebook(enumerate(train_loader), total=len(train_loader), leave=False)

        optimizer.zero_grad()   

        for i,(x_batch, y_batch) in tk0:

            y_pred = model(x_batch.to(device), 

                           attention_mask=(x_batch>0).to(device), labels=None)

            loss = nn.BCEWithLogitsLoss()(y_pred, y_batch.to(device))

            if USE_APEX:

                with amp.scale_loss(loss, optimizer) as scaled_loss:

                    scaled_loss.backward()

            else:

                loss.backward()

            if (i+1) % ACCUM_STEPS == 0:                     # Wait for several backward steps

                optimizer.step()                             # Now we can do an optimizer step

                optimizer.zero_grad()

            if lossf:

                lossf = 0.98 * lossf + 0.02 * loss.item()

            else:

                lossf = loss.item()

            tk0.set_postfix(loss = lossf)

            avg_loss += loss.item() / len(train_loader)

            

            loss_history.append(loss.item())

            

        tq.set_postfix(avg_loss=avg_loss)

        

# torch.save(model.state_dict(), output_model_file)
plt.plot(range(len(loss_history)), loss_history);

plt.ylabel('Loss'); plt.xlabel('Batch number');
!nvidia-smi
with timer('Test predictions'):

    # The following 3 lines are not needed but show how to download the model for prediction

#     model = BertForSequenceClassification(bert_config, num_labels=y_train.shape[1])

#     model.load_state_dict(torch.load(output_model_file))

#     model.to(device)

    for param in model.parameters():

        param.requires_grad = False

    model.eval();

    

    test_pred_probs = np.zeros((len(X_test), y_train.shape[1]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        

    for i,(x_batch,) in enumerate(tqdm_notebook(test_loader)):

        pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)

        test_pred_probs[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred.detach().cpu().squeeze().numpy()
test_preds = label_binarizer.inverse_transform(test_pred_probs)



sub_df = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', index_col='id')

sub_df['label'] = test_preds 

sub_df.to_csv('submission.csv')
!head submission.csv