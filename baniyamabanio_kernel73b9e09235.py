%ls /kaggle/input/nlp-getting-started/
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

prefix = '/kaggle/input/nlp-getting-started/'
train_df = pd.read_csv(prefix + 'train.csv')

#train_df.describe
train_df.head()
train_df = pd.DataFrame({

    'id':range(len(train_df)),

    'label':train_df["target"],

    'alpha':['a']*train_df.shape[0],

    'text': train_df["text"].replace(r'\n', ' ', regex=True)

})



train_df.head()

len(train_df)
test_df = pd.read_csv(prefix + 'test.csv')

#test_df.describe
test_df.head()
test_df.isnull().sum()
test_df = pd.DataFrame({

    'id':range(len(test_df)),

    'label':[0]*test_df.shape[0],

    'alpha':['a']*test_df.shape[0],

    'text': test_df["text"].replace(r'\n', ' ', regex=True)

})



test_df.head()

len(test_df)
!mkdir data

!pip install contractions
import re

import contractions

import string

def fix_contractions(text):

    return contractions.fix(text)



def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_mark(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



print("tweet before contractions fix : ", train_df.iloc[1055]["text"])

print("-"*20)

train_df['text']=train_df['text'].apply(lambda x : fix_contractions(x))

test_df['text']=test_df['text'].apply(lambda x : fix_contractions(x))

train_df['text']=train_df['text'].apply(lambda x : remove_url(x))

test_df['text']=test_df['text'].apply(lambda x : remove_url(x))

train_df['text']=train_df['text'].apply(lambda x : remove_mark(x))

test_df['text']=test_df['text'].apply(lambda x : remove_mark(x))

print("tweet after contractions fix : ", train_df.iloc[1055]["text"])

train_df.to_csv('data/train.tsv', sep='\t', index=False, header=False)

test_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)


from __future__ import absolute_import, division, print_function



import csv

import logging

import os

import sys

from io import open



from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import matthews_corrcoef, f1_score



from multiprocessing import Pool, cpu_count

from tqdm import tqdm



logger = logging.getLogger(__name__)

csv.field_size_limit(2147483647)



class InputExample(object):

    """A single training/test example for simple sequence classification."""



    def __init__(self, guid, text_a, text_b=None, label=None):

        """Constructs a InputExample.

        Args:

            guid: Unique id for the example.

            text_a: string. The untokenized text of the first sequence. For single

            sequence tasks, only this sequence must be specified.

            text_b: (Optional) string. The untokenized text of the second sequence.

            Only must be specified for sequence pair tasks.

            label: (Optional) string. The label of the example. This should be

            specified for train and dev examples, but not for test examples.

        """

        self.guid = guid

        self.text_a = text_a

        self.text_b = text_b

        self.label = label





class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, input_mask, segment_ids, label_id):

        self.input_ids = input_ids

        self.input_mask = input_mask

        self.segment_ids = segment_ids

        self.label_id = label_id





class DataProcessor(object):

    """Base class for data converters for sequence classification data sets."""



    def get_train_examples(self, data_dir):

        """Gets a collection of `InputExample`s for the train set."""

        raise NotImplementedError()



    def get_dev_examples(self, data_dir):

        """Gets a collection of `InputExample`s for the dev set."""

        raise NotImplementedError()



    def get_labels(self):

        """Gets the list of labels for this data set."""

        raise NotImplementedError()



    @classmethod

    def _read_tsv(cls, input_file, quotechar=None):

        """Reads a tab separated value file."""

        with open(input_file, "r", encoding="utf-8-sig") as f:

            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)

            lines = []

            for line in reader:

                if sys.version_info[0] == 2:

                    line = list(unicode(cell, 'utf-8') for cell in line)

                lines.append(line)

            return lines





class BinaryProcessor(DataProcessor):

    """Processor for the binary data sets"""



    def get_train_examples(self, data_dir):

        """See base class."""

        return self._create_examples(

            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")



    def get_dev_examples(self, data_dir):

        """See base class."""

        return self._create_examples(

            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")



    def get_labels(self):

        """See base class."""

        return ["0", "1"]



    def _create_examples(self, lines, set_type):

        """Creates examples for the training and dev sets."""

        examples = []

        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)

            text_a = line[3]

            label = line[1]

            examples.append(

                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples





def convert_example_to_feature(example_row, pad_token=0,

sequence_a_segment_id=0, sequence_b_segment_id=1,

cls_token_segment_id=1, pad_token_segment_id=0,

mask_padding_with_zero=True):

    example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id = example_row



    tokens_a = tokenizer.tokenize(example.text_a)



    tokens_b = None

    if example.text_b:

        tokens_b = tokenizer.tokenize(example.text_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total

        # length is less than the specified length.

        # Account for [CLS], [SEP], [SEP] with "- 3"

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    else:

        # Account for [CLS] and [SEP] with "- 2"

        if len(tokens_a) > max_seq_length - 2:

            tokens_a = tokens_a[:(max_seq_length - 2)]



    # The convention in BERT is:

    # (a) For sequence pairs:

    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]

    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

    # (b) For single sequences:

    #  tokens:   [CLS] the dog is hairy . [SEP]

    #  type_ids:   0   0   0   0  0     0   0

    #

    # Where "type_ids" are used to indicate whether this is the first

    # sequence or the second sequence. The embedding vectors for `type=0` and

    # `type=1` were learned during pre-training and are added to the wordpiece

    # embedding vector (and position vector). This is not *strictly* necessary

    # since the [SEP] token unambiguously separates the sequences, but it makes

    # it easier for the model to learn the concept of sequences.

    #

    # For classification tasks, the first vector (corresponding to [CLS]) is

    # used as as the "sentence vector". Note that this only makes sense because

    # the entire model is fine-tuned.

    tokens = tokens_a + [sep_token]

    segment_ids = [sequence_a_segment_id] * len(tokens)



    if tokens_b:

        tokens += tokens_b + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)



    if cls_token_at_end:

        tokens = tokens + [cls_token]

        segment_ids = segment_ids + [cls_token_segment_id]

    else:

        tokens = [cls_token] + tokens

        segment_ids = [cls_token_segment_id] + segment_ids



    input_ids = tokenizer.convert_tokens_to_ids(tokens)



    # The mask has 1 for real tokens and 0 for padding tokens. Only real

    # tokens are attended to.

    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)



    # Zero-pad up to the sequence length.

    padding_length = max_seq_length - len(input_ids)

    if pad_on_left:

        input_ids = ([pad_token] * padding_length) + input_ids

        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask

        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

    else:

        input_ids = input_ids + ([pad_token] * padding_length)

        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)



    assert len(input_ids) == max_seq_length

    assert len(input_mask) == max_seq_length

    assert len(segment_ids) == max_seq_length



    if output_mode == "classification":

        label_id = label_map[example.label]

    elif output_mode == "regression":

        label_id = float(example.label)

    else:

        raise KeyError(output_mode)



    return InputFeatures(input_ids=input_ids,

                        input_mask=input_mask,

                        segment_ids=segment_ids,

                        label_id=label_id)

    



def convert_examples_to_features(examples, label_list, max_seq_length,

                                 tokenizer, output_mode,

                                 cls_token_at_end=False, pad_on_left=False,

                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,

                                 sequence_a_segment_id=0, sequence_b_segment_id=1,

                                 cls_token_segment_id=1, pad_token_segment_id=0,

                                 mask_padding_with_zero=True):

    """ Loads a data file into a list of `InputBatch`s

        `cls_token_at_end` define the location of the CLS token:

            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]

            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]

        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

    """



    label_map = {label : i for i, label in enumerate(label_list)}



    examples = [(example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]



    process_count = 1



    with Pool(process_count) as p:

        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=100), total=len(examples)))





    return features





def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    """Truncates a sequence pair in place to the maximum length."""



    # This is a simple heuristic which will always truncate the longer sequence

    # one token at a time. This makes more sense than truncating an equal percent

    # of tokens from each, since if one sequence is very short then each token

    # that's truncated likely contains more information than a longer sequence.

    while True:

        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length:

            break

        if len(tokens_a) > len(tokens_b):

            tokens_a.pop()

        else:

            tokens_b.pop()





processors = {

    "binary": BinaryProcessor

}



output_modes = {

    "binary": "classification"

}



GLUE_TASKS_NUM_LABELS = {

    "binary": 2

}
!pip install pytorch_transformers
from __future__ import absolute_import, division, print_function



import glob

import logging

import os

import random

import json



import numpy as np

import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,

                              TensorDataset)

import random

from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm_notebook, trange



from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,

                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 

                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,

                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)



from pytorch_transformers import AdamW, WarmupLinearSchedule



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
args = {

    'data_dir': './data/',

    'model_type':  'roberta',

    'model_name': 'roberta-base',

    'task_name': 'binary',

    'output_dir': 'outputs/',

    'cache_dir': 'cache/',

    'do_train': True,

    'do_eval': True,

    'fp16': False,

    'fp16_opt_level': 'O1',

    'max_seq_length': 128,

    'output_mode': 'classification',

    'train_batch_size': 8,

    'eval_batch_size': 8,



    'gradient_accumulation_steps': 1,

    'num_train_epochs': 3,

    'weight_decay': 0,

    'learning_rate': 4e-5,

    'adam_epsilon': 1e-8,

    'warmup_steps': 0,

    'max_grad_norm': 1.0,



    'logging_steps': 50,

    'evaluate_during_training': False,

    'save_steps': 2000,

    'eval_all_checkpoints': True,



    'overwrite_output_dir': False,

    'reprocess_input_data': True,

    'notes': 'Using Yelp Reviews dataset'

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args
with open('args.json', 'w') as f:

    json.dump(args, f)
if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:

    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))
MODEL_CLASSES = {

    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),

    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),

    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),

    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

}



config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])

tokenizer = tokenizer_class.from_pretrained(args['model_name'])
model = model_class.from_pretrained(args['model_name'])
model.to(device)
task = args['task_name']



processor = processors[task]()

label_list = processor.get_labels()

num_labels = len(label_list)
def load_and_cache_examples(task, tokenizer, evaluate=False):

    processor = processors[task]()

    output_mode = args['output_mode']

    

    mode = 'dev' if evaluate else 'train'

    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

    

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:

        logger.info("Loading features from cached file %s", cached_features_file)

        features = torch.load(cached_features_file)

               

    else:

        logger.info("Creating features from dataset file at %s", args['data_dir'])

        label_list = processor.get_labels()

        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])

        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,

            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end

            cls_token=tokenizer.cls_token,

            sep_token=tokenizer.sep_token,

            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,

            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet

            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)

        

        logger.info("Saving features into cached file %s", cached_features_file)

        torch.save(features, cached_features_file)

        

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    elif output_mode == "regression":

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)



    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset
from torch.utils.tensorboard import SummaryWriter

def train(train_dataset, model, tokenizer):

    tb_writer = SummaryWriter()

    

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    

    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},

        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)

    

    if args['fp16']:

        try:

            from apex import amp

        except ImportError:

            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

        

    logger.info("***** Running training *****")

    logger.info("  Num examples = %d", len(train_dataset))

    logger.info("  Num Epochs = %d", args['num_train_epochs'])

    logger.info("  Total train batch size  = %d", args['train_batch_size'])

    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])

    logger.info("  Total optimization steps = %d", t_total)



    global_step = 0

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")

    

    for _ in train_iterator:

        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            model.train()

            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids':      batch[0],

                      'attention_mask': batch[1],

                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids

                      'labels':         batch[3]}

            outputs = model(input_ids=batch[0],attention_mask=batch[1],token_type_ids=batch[2],labels=batch[3])

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            print("\r%f" % loss, end='')



            if args['gradient_accumulation_steps'] > 1:

                loss = loss / args['gradient_accumulation_steps']



            if args['fp16']:

                with amp.scale_loss(loss, optimizer) as scaled_loss:

                    scaled_loss.backward()

                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])

                

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])



            tr_loss += loss.item()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:

                scheduler.step()  # Update learning rate schedule

                optimizer.step()

                model.zero_grad()

                global_step += 1



                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:

                    # Log metrics

                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(model, tokenizer)

                        for key, value in results.items():

                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)

                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)

                    logging_loss = tr_loss



                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:

                    # Save model checkpoint

                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))

                    if not os.path.exists(output_dir):

                        os.makedirs(output_dir)

                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                    model_to_save.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)





    return global_step, tr_loss / global_step
if args['do_train']:

    train_dataset = load_and_cache_examples(task, tokenizer)

    #print(train_dataset[0])

    global_step, tr_loss = train(train_dataset, model, tokenizer)

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
def submit(pred_ids):

    sub = pd.read_csv(prefix+'sample_submission.csv')

    sub['target'] = list(map(int,pred_ids))

    sub.to_csv('submission.csv', index=False)
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix

from scipy.stats import pearsonr



def get_mismatched(labels, preds):

    mismatched = labels != preds

    examples = processor.get_dev_examples(args['data_dir'])

    wrong = [i for (i, v) in zip(examples, mismatched) if v]

    

    return wrong



def get_eval_report(labels, preds):

    mcc = matthews_corrcoef(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {

        "mcc": mcc,

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn

    }, get_mismatched(labels, preds)



def compute_metrics(task_name, preds, labels):

    assert len(preds) == len(labels)

    return get_eval_report(labels, preds)



def evaluate(model, tokenizer, prefix=""):

    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_output_dir = args['output_dir']



    results = {}

    EVAL_TASK = args['task_name']



    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):

        os.makedirs(eval_output_dir)





    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    preds_ = []

    # Eval!

    logger.info("***** Running evaluation {} *****".format(prefix))

    logger.info("  Num examples = %d", len(eval_dataset))

    logger.info("  Batch size = %d", args['eval_batch_size'])

    eval_loss = 0.0

    nb_eval_steps = 0

    preds = None

    out_label_ids = None

    preds_ = []

    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):

        model.eval()

        batch = tuple(t.to(device) for t in batch)



        with torch.no_grad():

            inputs = {'input_ids':      batch[0],

                      'attention_mask': batch[1],

                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids

                      'labels':         batch[3]}

            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[:2]

            for i in range(logits.size(0)):

                preds_.append(logits[i,:].cpu().max(0)[1].item())

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:

            preds = logits.detach().cpu().numpy()

            out_label_ids = inputs['labels'].detach().cpu().numpy()

        else:

            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    #print(len(preds_))

    #input()

    submit(preds_)

    #print(len(out_label_ids))

    #print(list(out_label_ids))

    eval_loss = eval_loss / nb_eval_steps

    if args['output_mode'] == "classification":

        preds = np.argmax(preds, axis=1)

    elif args['output_mode'] == "regression":

        preds = np.squeeze(preds)

    result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)

    results.update(result)



    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:

        logger.info("***** Eval results {} *****".format(prefix))

        for key in sorted(result.keys()):

            logger.info("  %s = %s", key, str(result[key]))

            writer.write("%s = %s\n" % (key, str(result[key])))

    

    return results, wrong
results = {}

if args['do_eval']:

    checkpoints = [args['output_dir']]

    if args['eval_all_checkpoints']:

        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))

        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:

        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

        model = model_class.from_pretrained(checkpoint)

        model.to(device)

        result, wrong_preds = evaluate(model, tokenizer, prefix=global_step)

        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())

        results.update(result)
%ls
# %ls ./outputs

# !cat ./outputs/eval_results.txt



# sub = pd.read_csv(prefix+'sample_submission.csv')

# sub.describe