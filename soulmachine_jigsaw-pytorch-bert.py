from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import dask.bag as db

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

import torch.utils.data

from tqdm import tqdm, tqdm_notebook



import os

import random

import subprocess

import sys

import time
import platform

print(f'Python version: {platform.python_version()}')

print(f'PyTorch version: {torch.__version__}')
# This notebook runs on GPU

assert torch.cuda.is_available()



DEVICE = torch.device('cuda')

NUM_GPUS = torch.cuda.device_count()

assert NUM_GPUS > 0
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows

import logging

logging.basicConfig(level=logging.INFO)
import argparse



def define_args(str_list):

    '''

    A lite version of args at https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L565

    

    The following flags are set to constant values implicitly thus they're removed from args:

      * local_rank=-1

      * fp16=True

      * cache_dir=''

      

    '''

    parser = argparse.ArgumentParser()



    ## Required parameters

    parser.add_argument("--data_dir",

                        default=None,

                        type=str,

                        required=True,

                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--bert_model", default=None, type=str, required=True,

                        help="Bert pre-trained model name selected in the list: bert-base-uncased, "

                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "

                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--max_seq_length",

                        default=None,

                        type=int,

                        required=True,

                        help="The maximum total input sequence length after WordPiece tokenization. \n"

                             "Sequences longer than this will be truncated, and sequences shorter \n"

                             "than this will be padded.")



    ## Other parameters

    parser.add_argument("--do_lower_case",

                        action='store_true',

                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",

                        default=32,

                        type=int,

                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",

                        default=8,

                        type=int,

                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",

                        default=5e-5,

                        type=float,

                        help="The initial learning rate for Adam.")

    parser.add_argument("--begin_epoch",

                        default=0,

                        type=int,

                        help="The begin training epoch, starts from 0.")

    parser.add_argument("--end_epoch",

                        default=1,

                        type=int,

                        help="The end training epoch, excluded.")

    parser.add_argument("--warmup_proportion",

                        default=0.1,

                        type=float,

                        help="Proportion of training to perform linear learning rate warmup for. "

                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--seed',

                        type=int,

                        default=42,

                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',

                        type=int,

                        default=1,

                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',

                        action='store_true',

                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',

                        type=float, default=0,

                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"

                             "0 (default value): dynamic loss scaling.\n"

                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--verbose', '-v', action='count')



    args = parser.parse_args(str_list)



    args.do_lower_case = 'uncased' in args.bert_model

    # see dataset https://www.kaggle.com/soulmachine/bert-fine-tuned-for-jigsaw

    args.output_dir = f'../input/bert-fine-tuned-for-jigsaw/jigsaw-{args.bert_model}-len-{args.max_seq_length}-{"fp16" if args.fp16 else "fp32"}'



    return args
args = define_args([

    '--data_dir', '../input/jigsaw-unintended-bias-in-toxicity-classification',

    '--bert_model', 'bert-base-uncased',

    '--max_seq_length', '220',

    '--fp16',

    '--learning_rate', '2e-5',

    '--begin_epoch', '0',

    '--end_epoch', '4',

    '-v',

])

args
# If the last checkpoint exists, skip training

IS_TRAINING = not os.path.exists(f'{args.output_dir}/epoch-{args.end_epoch-1}')

IS_TRAINING
def check_args(args):

    assert args.begin_epoch < args.end_epoch

    if args.begin_epoch > 0:

        assert os.path.exists(f'{args.output_dir}/epoch-{args.begin_epoch-1}')

    if IS_TRAINING:

        for i in range(args.begin_epoch, args.end_epoch):

            assert not os.path.exists(f'{args.output_dir}/epoch-{i}')

    else:

        assert os.path.exists(f'{args.output_dir}/epoch-{args.end_epoch-1}')
check_args(args)
random.seed(args.seed)

np.random.seed(args.seed)

torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)

torch.cuda.manual_seed_all(args.seed)



torch.backends.cudnn.deterministic = True
def install_apex():

    try:

        import apex

    except ModuleNotFoundError:

        print('Installing NVIDIA Apex')

        if 'KAGGLE_URL_BASE' in os.environ:  # kaggle kernel

            APEX_SRC = '../input/nvidia-apex/apex-master/apex-master'

            assert os.path.exists(APEX_SRC)

            print(subprocess.check_output(

                f'{sys.executable} -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" {APEX_SRC}',

                shell=True).decode('utf-8'))

        else:

            APEX_SRC = '../input/apex'

            if not os.path.exists(APEX_SRC):

                os.makedirs(APEX_SRC)

                print(subprocess.check_output(f'git clone https://github.com/NVIDIA/apex {APEX_SRC}', shell=True).decode('utf-8'))

            else:

                print(f'{APEX_SRC} already exists')

            print(subprocess.check_output(

                f'sudo {sys.executable} -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" {APEX_SRC}',

                shell=True).decode('utf-8'))

        import apex

        print('Installed apex successfully')
if IS_TRAINING and args.fp16:

    install_apex()
try:

    from pytorch_pretrained_bert import BertTokenizer, BertModel

except ModuleNotFoundError:

    print('Installing Install pytorch-pretrained-bert ...')

    if 'KAGGLE_URL_BASE' in os.environ:  # kaggle kernel

        bert_lib = '../input/pytorchpretrainedbert/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master'

        assert os.path.exists(bert_lib)

        sys.path.insert(0, bert_lib)

    else:

        print(subprocess.check_output('sudo -u jupyter conda install -y -c conda-forge pytorch-pretrained-bert', shell=True).decode('utf-8'))

    print('Installed pytorch-pretrained-bert successfully')
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification, BertConfig, BertForMaskedLM

from pytorch_pretrained_bert.tokenization import BertTokenizer

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
y_columns=['target']
if 'KAGGLE_URL_BASE' in os.environ:  # kaggle kernel

    MODELS_ROOT_DIR = '../input/pretrained-bert-models-for-pytorch'

    VOCAB_FILE = f'{MODELS_ROOT_DIR}/{args.bert_model}-vocab.txt'

    tokenizer = BertTokenizer.from_pretrained(VOCAB_FILE, do_lower_case=args.do_lower_case, cache_dir=None)

else:

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
def convert_lines(lines, max_seq_length, tokenizer):

    '''

      Converting the lines to BERT format.

      

      Copied from https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming

    '''

    max_seq_length -= 2  # CLS, SEP

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(lines):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(f'longer: {longer}')

    return np.array(all_tokens)
def convert_lines_parallel(i):

    total_lines = len(X_train)

    num_lines_per_thread = total_lines // os.cpu_count() + 1

    lines = X_train[i * num_lines_per_thread : (i+1) * num_lines_per_thread]

    return convert_lines(lines, args.max_seq_length, tokenizer)
def preprocess_data(df, has_label=True):

    # Make sure all comment_text values are strings

    df['comment_text'] = df['comment_text'].astype(str).fillna("DUMMY_VALUE")

    if has_label:

        # convert target to 0,1

        df['target']=(df['target']>=0.5).astype(float)
if IS_TRAINING:

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))#.sample(num_to_load+valid_size,random_state=args.seed)

    preprocess_data(train_df)

    

    X_train = train_df["comment_text"]

    X_train = np.vstack(db.from_sequence(list(range(os.cpu_count()))).map(convert_lines_parallel).compute())

    Y_train = train_df[y_columns].values



    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(Y_train,dtype=torch.float))

    num_train_optimization_steps = int(

        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * (args.end_epoch-args.begin_epoch)

    

    assert Y_train.shape[1] == 1

    print(X_train.shape)

    print(Y_train.shape)

    print(X_train.dtype)

    print(Y_train.dtype)
NUM_VALID_SAMPLES = 100000



valid_df = pd.read_csv(os.path.join(args.data_dir, "train.csv")).sample(NUM_VALID_SAMPLES, random_state=args.seed)

preprocess_data(valid_df)



X_valid = convert_lines(valid_df['comment_text'], args.max_seq_length, tokenizer)

Y_valid = valid_df[y_columns].values

valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.long))



assert Y_valid.shape[1] == 1

print(X_valid.shape)

print(Y_valid.shape)

print(X_valid.dtype)

print(Y_valid.dtype)
test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

preprocess_data(test_df, has_label=False)
X_test = convert_lines(test_df["comment_text"], args.max_seq_length, tokenizer)
print(X_test.shape)

print(X_test.dtype)
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
def prepare_model(model):

    '''

      See https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

    '''

#     if args.fp16:

#         # Users should not manually cast their model or data to .half()

#         # see https://nvidia.github.io/apex/amp.html

#         model.half()

    model.zero_grad()

    model.to(DEVICE)

    if NUM_GPUS > 1:

        model = torch.nn.DataParallel(model)

    return model
def load_model():

    if args.begin_epoch == 0:  # load BERT model

        print('Load BERT model')

        if 'KAGGLE_URL_BASE' in os.environ:  # kaggle kernel

            MODELS_ROOT_DIR = '../input/pretrained-bert-models-for-pytorch'

            MODEL_PATH = f'{MODELS_ROOT_DIR}/{args.bert_model}'

            model = BertForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir=None, num_labels=len(y_columns))

        else:

            model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=None, num_labels=len(y_columns))

    else:  # args.begin_epoch > 0

        print('Load previous checkpoint')

        model_dir = f'{args.output_dir}/epoch-{args.begin_epoch-1}'

        assert os.path.exists(model_dir)

        model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=len(y_columns), cache_dir=None)



    return model
if IS_TRAINING:

    model = prepare_model(load_model())
def prepare_optimizer(model):

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]

    if False:  # args.fp16

        try:

            from apex.optimizers import FP16_Optimizer

            from apex.optimizers import FusedAdam

        except ImportError:

            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")



        optimizer = FusedAdam(optimizer_grouped_parameters,

                              lr=args.learning_rate,

                              bias_correction=False,

                              max_grad_norm=1.0)

        if args.loss_scale == 0:

            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

        else:

            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:

        optimizer = BertAdam(optimizer_grouped_parameters,

                             lr=args.learning_rate,

                             warmup=args.warmup_proportion,

                             t_total=num_train_optimization_steps)

    return optimizer
if IS_TRAINING:

    optimizer = prepare_optimizer(model)
def save_model(model, tokenizer, output_dir):

    '''

      Save a trained model and configuration.

    '''

    if os.path.exists(output_dir) and os.listdir(output_dir):

        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    

    # Save a trained model and configuration

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self



    # If we save using the predefined names, we can load using `from_pretrained`

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)

    output_config_file = os.path.join(output_dir, CONFIG_NAME)



    torch.save(model_to_save.state_dict(), output_model_file)

    model_to_save.config.to_json_file(output_config_file)

    tokenizer.save_vocabulary(output_dir)
def train(model, optimizer, train_dataset):

    if args.fp16:

        # Allow Amp to perform casts as required by the opt_level

        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")

        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,

                                             t_total=num_train_optimization_steps)

    global_step = 0



    model=model.train()



    start_time = time.time()

    outer_tq = tqdm_notebook(range(args.begin_epoch, args.end_epoch))

    for epoch in outer_tq:

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

        avg_loss = 0.

        avg_accuracy = 0.

        lossf=None

        epoch_start_time = time.time()



        #for step, batch in enumerate(tqdm(train_dataloader, desc= f'Iteration {epoch}')):

        #    batch = tuple(t.to(DEVICE) for t in batch)



        inner_tq = tqdm_notebook(enumerate(train_loader), total=len(train_loader),leave=False, desc= f'Iteration {epoch}')

        for step, (x_batch, y_batch) in inner_tq:

            optimizer.zero_grad()

            y_pred = model(x_batch.to(DEVICE), attention_mask=(x_batch>0).to(DEVICE), labels=None)

            loss =  F.binary_cross_entropy_with_logits(y_pred, y_batch.to(DEVICE))



            if NUM_GPUS > 1:

                loss = loss.mean() # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:

                loss = loss / args.gradient_accumulation_steps



            if args.fp16:

                # optimizer.backward(loss)

                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:

                    scaled_loss.backward()

            else:

                loss.backward()



            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()

                optimizer.zero_grad()

                global_step += 1



            if lossf:

                lossf = 0.98*lossf+0.02*loss.item()

            else:

                lossf = loss.item()

            inner_tq.set_postfix(loss = lossf)

            avg_loss += loss.item() / len(train_loader)

            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(DEVICE)).to(torch.float) ).item()/len(train_loader)

        outer_tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

        save_model(model, tokenizer, f'{args.output_dir}/epoch-{epoch}')

        epoch_end_time = time.time()

        print(f'Iteration {step} time elapsed {int(epoch_end_time-epoch_start_time)}s')



    end_time = time.time()

    print(f'Time elapsed {int(end_time-start_time)}s')
if not os.path.exists(f'{args.output_dir}/epoch-{args.end_epoch-1}'):

    train(model, optimizer, train_dataset)
def load_eval_model(model_dir):

    # Load a trained model and vocabulary that you have fine-tuned

    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=len(y_columns), cache_dir=None)

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case, cache_dir=None)

    model.to(DEVICE)

    model.eval()

    for param in model.parameters():

        param.requires_grad = False

    return model
def predict(model, valid_dataset):

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)

    batch_size=args.eval_batch_size

    valid_preds = np.zeros((len(valid_dataset)))

    

    for step, (x_batch, ) in tqdm_notebook(enumerate(valid_loader), total=len(valid_loader)):

        y_pred = model(x_batch.to(DEVICE), attention_mask=(x_batch>0).to(DEVICE), labels=None)

        valid_preds[step*batch_size:(step+1)*batch_size]=y_pred[:,0].detach().cpu().squeeze().numpy()

    return valid_preds
model = load_eval_model(f'{args.output_dir}/epoch-{args.end_epoch-1}')
valid_preds = predict(model, valid_dataset)
# From baseline kernel

from sklearn import metrics

from sklearn.metrics import roc_auc_score



def calculate_overall_auc(df, model_name):

    true_labels = df[TOXICITY_COLUMN]>0.5

    predicted_labels = df[model_name]

    return metrics.roc_auc_score(true_labels, predicted_labels)



def power_mean(series, p):

    total = sum(np.power(series, p))

    return np.power(total / len(series), 1 / p)



def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):

    bias_score = np.average([

        power_mean(bias_df[SUBGROUP_AUC], POWER),

        power_mean(bias_df[BPSN_AUC], POWER),

        power_mean(bias_df[BNSP_AUC], POWER)

    ])

    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)







SUBGROUP_AUC = 'subgroup_auc'

BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



def compute_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def compute_subgroup_auc(df, subgroup, label, model_name):

    subgroup_examples = df[df[subgroup]>0.5]

    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])



def compute_bpsn_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]

    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return compute_auc(examples[label]>0.5, examples[model_name])



def compute_bnsp_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]

    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return compute_auc(examples[label]>0.5, examples[model_name])



def compute_bias_metrics_for_model(dataset,

                                   subgroups,

                                   model,

                                   label_col,

                                   include_asegs=False):

    """Computes per-subgroup metrics for all subgroups and one model."""

    records = []

    for subgroup in subgroups:

        record = {

            'subgroup': subgroup,

            'subgroup_size': len(dataset[dataset[subgroup]>0.5])

        }

        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)

        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)

        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
MODEL_NAME = 'model1'

# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



valid_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()

TOXICITY_COLUMN = 'target'

bias_metrics_df = compute_bias_metrics_for_model(valid_df, identity_columns, MODEL_NAME, 'target')
bias_metrics_df
get_final_metric(bias_metrics_df, calculate_overall_auc(valid_df, MODEL_NAME))
y_test = predict(model, test_dataset)
y_test.shape
test_pred = torch.sigmoid(torch.tensor(y_test)).numpy().ravel()
test_pred.shape
submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': test_pred

})

submission.to_csv('submission.csv', index=False)
submission.head()