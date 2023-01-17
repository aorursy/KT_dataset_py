import numpy as np

import pandas as pd

import os

import warnings

import random

import torch 

from torch import nn

import torch.optim as optim

import tokenizers

from transformers import RobertaModel, RobertaConfig

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 37

seed_everything(seed)
"""

Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).

GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned

using a masked language modeling (MLM) loss.

"""





import argparse

import glob

import logging

import os

import pickle

import random

import re

import shutil

from typing import Dict, List, Tuple



import numpy as np

import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler

from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange



from transformers import (

    WEIGHTS_NAME,

    AdamW,

    PreTrainedModel,

    PreTrainedTokenizer,

    RobertaConfig,

    RobertaForMaskedLM,

    RobertaTokenizer,

    get_linear_schedule_with_warmup,

)





try:

    from torch.utils.tensorboard import SummaryWriter

except ImportError:

    from tensorboardX import SummaryWriter





logger = logging.getLogger(__name__)





MODEL_CLASSES = {

    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),

}



class SortishSampler(Sampler):

    "Go through the text data by order of length with a bit of randomness."



    def __init__(self, data_source, key, bs:int):

        self.data_source,self.key,self.bs = data_source,key,bs



    def __len__(self) -> int: return len(self.data_source)



    def __iter__(self):

        idxs = np.random.permutation(len(self.data_source))

        sz = self.bs*50

        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]

        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])

        sz = self.bs

        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]

        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,

        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.

        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([],dtype=np.int)

        sort_idx = np.concatenate((ck_idx[0], sort_idx))

        return iter(sort_idx)



class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, args, lines, block_size=128):

        print(lines[0])

        self.examples = tokenizer.batch_encode_plus(lines, max_length=block_size)["input_ids"]

                

    def __len__(self):

        return len(self.examples)



    def __getitem__(self, i):

        return torch.tensor(self.examples[i])





def load_and_cache_examples(args, tokenizer, lines):

        return LineByLineTextDataset(tokenizer, args, lines=lines, block_size=args['block_size'])



def set_seed(args):

    random.seed(args['seed'])

    np.random.seed(args['seed'])

    torch.manual_seed(args['seed'])

    if args['n_gpu'] > 0:

        torch.cuda.manual_seed_all(args['seed'])



def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, args['mlm_probability'])

    special_tokens_mask = [

        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()

    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if tokenizer._pad_token is not None:

        padding_mask = labels.eq(tokenizer.pad_token_id)

        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)



    # 10% of the time, we replace masked input tokens with random word

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs = inputs.type(torch.LongTensor)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels





def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:

    """ Train the model """

    

    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])



    def collate(examples: List[torch.Tensor]):

        if tokenizer._pad_token is None:

            return pad_sequence(examples, batch_first=True)

        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    

    train_sampler = SortishSampler(train_dataset, key=lambda t : len(train_dataset[t]), bs=args['train_batch_size'])

    train_dataloader = DataLoader(

        train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'], collate_fn=collate

    )



    if args['max_steps'] > 0:

        t_total = args['max_steps']

        args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1

    else:

        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']



    # Prepare optimizer and schedule (linear warmup and decay)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [

        {

            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],

            "weight_decay": args['weight_decay'],

        },

        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},

    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])

    scheduler = get_linear_schedule_with_warmup(

        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total

    )



    # Check if saved optimizer or scheduler states exist

    if (

        args['model_name_or_path']

        and os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt"))

        and os.path.isfile(os.path.join(args['model_name_or_path'], "scheduler.pt"))

    ):

        # Load in optimizer and scheduler states

        optimizer.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "optimizer.pt")))

        scheduler.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "scheduler.pt")))



    # Train!

    logger.info("***** Running training *****")

    logger.info("  Num examples = %d", len(train_dataset))

    logger.info("  Num Epochs = %d", args['num_train_epochs'])

    logger.info("  Instantaneous batch size per GPU = %d", args['per_gpu_train_batch_size'])

    logger.info(

        "  Total train batch size (w. parallel, distributed & accumulation) = %d",

        args['train_batch_size']

        * args['gradient_accumulation_steps']

        * (torch.distributed.get_world_size() if args['local_rank'] != -1 else 1),

    )

    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])

    logger.info("  Total optimization steps = %d", t_total)



    global_step = 0

    epochs_trained = 0

    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint

    if args['model_name_or_path'] and os.path.exists(args['model_name_or_path']):

        try:

            # set global_step to gobal_step of last saved checkpoint from model path

            checkpoint_suffix = args['model_name_or_path'].split("-")[-1].split("/")[0]

            global_step = int(checkpoint_suffix)

            epochs_trained = global_step // (len(train_dataloader) // args['gradient_accumulation_steps'])

            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args['gradient_accumulation_steps'])



            logger.info("  Continuing training from checkpoint, will skip to saved global_step")

            logger.info("  Continuing training from epoch %d", epochs_trained)

            logger.info("  Continuing training from global step %d", global_step)

            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        except ValueError:

            logger.info("  Starting fine-tuning.")



    tr_loss, logging_loss = 0.0, 0.0



    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

    model_to_resize.resize_token_embeddings(len(tokenizer))



    model.zero_grad()

    train_iterator = trange(

        epochs_trained, int(args['num_train_epochs']), desc="Epoch", disable=args['local_rank'] not in [-1, 0]

    )

    set_seed(args)  # Added here for reproducibility

    epoch_count=0

    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        for step, batch in enumerate(epoch_iterator):



            # Skip past any already trained steps if resuming training

            if steps_trained_in_current_epoch > 0:

                steps_trained_in_current_epoch -= 1

                continue



            inputs, labels = mask_tokens(batch, tokenizer, args) if args['mlm'] else (batch, batch)

    

            inputs = inputs.to(args['device'])

            labels = labels.to(args['device'])

            model.train()

#             try:

            outputs = model(inputs.type(torch.LongTensor).to(args['device']), masked_lm_labels=labels.type(torch.LongTensor).to(args['device'])) if args['mlm'] else model(inputs, labels=labels)

#             except Exception as e:

#                 print(e)

#                 print(inputs, labels)

#                 print(inputs.shape, labels.shape)

#                 break

                

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)



            if args['n_gpu'] > 1:

                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args['gradient_accumulation_steps'] > 1:

                loss = loss / args['gradient_accumulation_steps']



            

            loss.backward()

#             print(loss)

            tr_loss += loss.item()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:

                

                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model.zero_grad()

                global_step += 1

#                 torch.cuda.empty_cache()

                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:

                    # Log metrics

                    if (

                        args['local_rank'] == -1 and args['evaluate_during_training']

                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(args, model, tokenizer)

                        for key, value in results.items():

                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

#                     tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

#                     tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args['logging_steps'], global_step)

                    logging_loss = tr_loss



            if args['max_steps'] > 0 and global_step > args['max_steps']:

                epoch_iterator.close()

                break

        print('training loss --->' , (tr_loss - logging_loss) / args['logging_steps'])

        epoch_count+=1

#         output_dir = os.path.join(args['output_dir'], "epoch_{}".format(epoch_count))

#         os.makedirs(output_dir, exist_ok=True)

#         model_to_save = (

#             model.module if hasattr(model, "module") else model

#         )  # Take care of distributed/parallel training

#         model_to_save.save_pretrained(output_dir)

#         tokenizer.save_pretrained(output_dir)



#         torch.save(args, os.path.join(output_dir, "training_args['bin']"))

#         logger.info("Saving model checkpoint to %s", output_dir)



#         torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

#         torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

#         logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args['max_steps'] > 0 and global_step > args['max_steps']:

            train_iterator.close()

            break



    return global_step, tr_loss / global_step





def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:

    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_output_dir = args['output_dir']



    



    if args['local_rank'] in [-1, 0]:

        os.makedirs(eval_output_dir, exist_ok=True)



    args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])

    # Note that DistributedSampler samples randomly



    def collate(examples: List[torch.Tensor]):

        if tokenizer._pad_token is None:

            return pad_sequence(examples, batch_first=True)

        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)



    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(

        eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'], collate_fn=collate

    )



    # multi-gpu evaluate

    if args['n_gpu'] > 1:

        model = torch.nn.DataParallel(model)



    # Eval!

    logger.info("***** Running evaluation {} *****".format(prefix))

    logger.info("  Num examples = %d", len(eval_dataset))

    logger.info("  Batch size = %d", args['eval_batch_size'])

    eval_loss = 0.0

    nb_eval_steps = 0

    model.eval()



    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        inputs, labels = mask_tokens(batch, tokenizer, args) if args['mlm'] else (batch, batch)

        inputs = inputs.to(args['device'])

        labels = labels.to(args['device'])

        with torch.no_grad():

            outputs = model(inputs.type(torch.LongTensor).to(args['device']), masked_lm_labels=labels.type(torch.LongTensor).to(args['device'])) if args['mlm'] else model(inputs, labels=labels)

            lm_loss = outputs[0]

            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1



    eval_loss = eval_loss / nb_eval_steps

    perplexity = torch.exp(torch.tensor(eval_loss))



    result = {"perplexity": perplexity, 'eval_loss':eval_loss}



    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

    with open(output_eval_file, "w") as writer:

        logger.info("***** Eval results {} *****".format(prefix))

        for key in sorted(result.keys()):

            logger.info("  %s = %s", key, str(result[key]))

            writer.write("%s = %s\n" % (key, str(result[key])))



    return result





def main():

    args = {}

    args['should_continue'] = False

    args['mlm_probability'] = 0.15

    args['config_name'] = None

    args['tokenizer_name'] = None

    args['cache_dir'] = None

    args['block_size'] = -1

    args['do_train']=True

    args['do_eval'] = True

    args['evaluate_during_training'] = False

    args['per_gpu_train_batch_size'] = 32

    args['per_gpu_eval_batch_size'] = 32

    args['gradient_accumulation_steps'] = 1

    args['learning_rate'] = 5e-5

    args['weight_decay'] = 0.1

    args['adam_epsilon'] = 1e-6

    args['max_grad_norm'] = 1.0

    args['max_steps'] = -1

    args['logging_steps'] = 500

    args['seed'] = seed

    args['custom_vocab_file'] = None

    args['local_rank'] = -1

    args['overwrite_output_dir'] = True

    

    args['output_dir'] = './roberta_finetuned/'

    args['model_type'] = 'roberta'

    args['model_name_or_path'] = 'roberta-base'

    args['mlm'] = True

    args['num_train_epochs'] = 4

    args['warmup_steps'] = 1000

    

    

    if args['model_type'] in ["bert", "roberta", "distilbert", "camembert"] and not args['mlm']:

        raise ValueError(

            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "

            "flag (masked language modeling)."

        )

    if (

        os.path.exists(args['output_dir'])

        and os.listdir(args['output_dir'])

        and args['do_train']

        and not args['overwrite_output_dir']

    ):

        raise ValueError(

            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(

                args['output_dir']

            )

        )



    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args['n_gpu'] = torch.cuda.device_count()

    

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]



    if args['config_name']:

        config = config_class.from_pretrained(args['config_name'], cache_dir=args['cache_dir'])

    elif args['model_name_or_path']:

        config = config_class.from_pretrained(args['model_name_or_path'], cache_dir=args['cache_dir'])

    else:

        config = config_class()



    if args['tokenizer_name']:

        tokenizer = tokenizer_class.from_pretrained(args['tokenizer_name'], cache_dir=args['cache_dir'])

    elif args['model_name_or_path']:

        tokenizer = tokenizer_class.from_pretrained(args['model_name_or_path'], cache_dir=args['cache_dir'])

    else:

        raise ValueError(

            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"

            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)

        )

    if args['block_size'] <= 0:

        args['block_size'] = tokenizer.max_len_single_sentence

        # Our input block size will be the max possible for the model

    else:

        args['block_size'] = min(args['block_size'], tokenizer.max_len_single_sentence)



    if args['model_name_or_path']:

        model = model_class.from_pretrained(

            args['model_name_or_path'],

            from_tf=bool(".ckpt" in args['model_name_or_path']),

            config=config,

            cache_dir=args['cache_dir'],

        )

        

    else:

        logger.info("Training new model from scratch")

        model = model_class(config=config)

    

    if args['tokenizer_name'] and 'bert' in args['tokenizer_name']:

        print('using custom tokenizer')

        with open('{}/vocab.txt'.format(args['tokenizer_name']),'r') as f:

            custom_vocab = [line for line in f.read().splitlines()]

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        bert_org_path = bert_tokenizer.save_vocabulary('.')[0]

        with open(bert_org_path,'r') as f:

            bert_vocab = [line for line in f.read().splitlines()]

        os.remove(bert_org_path)

        bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        bert_model.resize_token_embeddings(len(custom_vocab))

        model.resize_token_embeddings(len(custom_vocab))

        bert_weights_vocab = {i:j for i,j in  zip(bert_vocab ,bert_model.state_dict()['bert.embeddings.word_embeddings.weight'])}

        custom_weights_vocab = {i:j for i,j in zip(custom_vocab ,bert_model.state_dict()['bert.embeddings.word_embeddings.weight'])}

        for i in custom_vocab:

            try:

                weight = bert_weights_vocab[i]

                custom_weights_vocab[i] = weight

            except KeyError:

                pass

        model.state_dict()['bert.embeddings.word_embeddings.weight'] = custom_weights_vocab

        tokenizer = tokenizer_class.from_pretrained(args['tokenizer_name'], cache_dir=args['cache_dir'])

        config.vocab_size = len(tokenizer)

        del bert_model, bert_tokenizer

    

    if args['tokenizer_name'] and 'gpt-2' in args['tokenizer_name']:

        import json

        print('using custom tokenizer')

        custom_vocab = list(json.load(open('{}/vocab.json'.format(args['tokenizer_name']))).keys())

        bert_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        bert_org_path = bert_tokenizer.save_vocabulary('.')[0]

        bert_vocab = list(json.load(open(bert_org_path)).keys())

        os.remove(bert_org_path)

        bert_model = GPT2LMHeadModel.from_pretrained('gpt2')

        bert_model.resize_token_embeddings(len(custom_vocab))

        model.resize_token_embeddings(len(custom_vocab))

        bert_weights_vocab = {i:j for i,j in  zip(bert_vocab ,bert_model.state_dict()['transformer.wte.weight'])}

        custom_weights_vocab = {i:j for i,j in zip(custom_vocab ,bert_model.state_dict()['transformer.wte.weight'])}

        for i in custom_vocab:

            try:

                weight = bert_weights_vocab[i]

                custom_weights_vocab[i] = weight

            except KeyError:

                pass

        model.state_dict()['transformer.wte.weight'] = custom_weights_vocab

        tokenizer = tokenizer_class.from_pretrained(args['tokenizer_name'], cache_dir=args['cache_dir'])

        tokenizer.add_special_tokens({'pad_token':'<pad>', 'unk_token':'<unk>', 'bos_token':'<cls>', 'eos_token':'<sep>'})

        config.vocab_size = len(tokenizer)

        del bert_model, bert_tokenizer

    

    if args['custom_vocab_file']:

        logger.info("adding custom vocab from file : %s", args['custom_vocab_file'])

        import pickle

        custom_vocab = pickle.load(open(args['custom_vocab_file'],'rb'))

#         custom_vocab = ['Ġ'+i for i in custom_vocab]

        tokenizer.add_tokens(custom_vocab)

        if tokenizer._pad_token==None: tokenizer.add_special_tokens({'pad_token':'<pad>'})

        model.resize_token_embeddings(len(tokenizer))

        config.vocab_size = len(tokenizer)

        model.lm_head.bias = torch.nn.Parameter(torch.nn.functional.pad(

                model.lm_head.bias,

                (0, model.get_output_embeddings().weight.shape[0]-model.lm_head.bias.shape[0]),

                "constant",

                0,

            ))

    logger.info("Vocab Size : %s", len(tokenizer))

    

    

    init_layers = [9, 10, 11]

    dense_names = ["query", "key", "value", "dense"]

    layernorm_names = ["LayerNorm"]

    for name, module in model.named_parameters():

        if any(f".{i}." in name for i in init_layers):

            if any(n in name for n in dense_names):

                if "bias" in name:

                    module.data.zero_()

                elif "weight" in name:

                    module.data.normal_(mean=0.0, std=0.02)

            elif any(n in name for n in layernorm_names):

                if "bias" in name:

                    module.data.zero_()

                elif "weight" in name:

                    module.data.fill_(1.0)

    

    

    

    model.to(args['device'])



    logger.info("Training/evaluation parameters %s", args)

    

    df = pd.concat([pd.read_csv('../input/tweet-sentiment-extraction/train.csv'),pd.read_csv('../input/tweet-sentiment-extraction/test.csv')],ignore_index=True)

    df = df.dropna(subset=['text'])

    tr_df, ts_df = train_test_split(df, test_size=0.05, random_state = seed)

    tr_lines = tr_df.text.tolist()

    te_lines = ts_df.text.tolist()

    del tr_df, ts_df

    import gc

    gc.collect()

    eval_dataset = load_and_cache_examples(args, tokenizer, te_lines)

    # Training

    if args['do_train']:

        test_text = 'this is sentence to encode, and this is wrgn'

        print('original text: ',test_text)

        print('tokenized: ', tokenizer.tokenize(test_text))

        print('encoded: ', tokenizer.encode(test_text))

        

        

        train_dataset = load_and_cache_examples(args, tokenizer, tr_lines)



        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()

    if args['do_train']:

        # Create output directory if needed

        os.makedirs(args['output_dir'], exist_ok=True)



        logger.info("Saving model checkpoint to %s", args['output_dir'])

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.

        # They can then be reloaded using `from_pretrained()`

        model_to_save = (

            model.module if hasattr(model, "module") else model

        )  # Take care of distributed/parallel training

        model_to_save.save_pretrained(args['output_dir'])

        tokenizer.save_pretrained(args['output_dir'])



        # Good practice: save your training arguments together with the trained model

        torch.save(args, os.path.join(args['output_dir'], "training_args.bin"))



        # Load a trained model and vocabulary that you have fine-tuned

        model = model_class.from_pretrained(args['output_dir'])

        tokenizer = tokenizer_class.from_pretrained(args['output_dir'])

        model.to(args['device'])



    # Evaluation

    results = {}

    if args['do_eval']:

        checkpoints = [args['output_dir']]

        for checkpoint in checkpoints:

            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""

            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""



            model = model_class.from_pretrained(checkpoint)

            model.to(args['device'])

            

            result = evaluate(args, model, tokenizer,eval_dataset ,prefix=prefix)

            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())

            results.update(result)



    return results
main()
!ls