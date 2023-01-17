# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

from collections import defaultdict



# Абстрактный класс множества значений. Объект множества сам отвечает за выборку заданного (size) числа значений из него.

class HyperSampler:

    def sample(self, size):

        raise NotImplementedError



# Простая реализация, в которой функция выборки просто задается пользователем.

# Вариант использования - LambdaSampler(lambda size: np.random.randn(size))  - выборка из стандартного нормального распределения

class LambdaSampler(HyperSampler):

    def __init__(self, fn):

        super().__init__()

        self._fn = fn



    def sample(self, size):

        return self._fn(size)



# В данной реализации выборка не случайна. Задается массив значений, из которого берется первые size значений,

# по одному на итерацию поиска

# По этой причине размер массива должен быть не менее числа итераций поиска гиперпараметра.

class FixedArraySampler(HyperSampler):

    def __init__(self, array):

        super().__init__()

        self._arr = np.asarray(array)



    def sample(self, size):

        if len(self._arr) < size:

            raise ValueError("len(self._arr) < size")

        return self._arr[:size]



# Выборка из категориального равномерного распределения. 

# Вы задаёте множество значений, и из него случайным образом отбирается size элементов 

# (с заменой, т.е. один элемент может быть выбран > 1 раза)

class RandomArraySampler(HyperSampler):

    def __init__(self, array):

        super().__init__()

        self._arr = np.asarray(array)



    def sample(self, size):

        idx = np.random.choice(len(self._arr), size=size)

        return self._arr[idx]



# Обертка над LambdaSampler    

def h_lambda(fn):

    return LambdaSampler(fn)



# Обертка над RandomArraySampler, позволяющая писать так h_enum(32,64,128) вместо RandomArraySampler([32,64,128])

def h_enum(*values):

    return RandomArraySampler(values)



# Обертка над RandomArraySampler из существующей коллекции (lst = [1,2,3], h_enum(lst)), по сути для абстракции и сокращения имени

def h_set(values):

    return RandomArraySampler(values)



#Аналогичные обертки над FixedArraySampler

def h_fixed_enum(*values):

    return FixedArraySampler(values)



def h_fixed_set(values):

    return FixedArraySampler(values)



# Разбиение словаря на два непересекающихся в зависимости от значения критерия

def split_dictionary(d, criterion):

    a, b = {}, {}

    for k, v in d.items():

        if criterion(k, v):

            a[k] = v

        else:

            b[k] = v

    return a, b



def hyper_search(num_trials, parameters,

                            iteration_function,

                            initial_state,

                            progress_bar=None):

    """Функция поиска гиперпараметров

    

    Параметры:

    num_trials -- число итераций поиска

    parameters -- словарь, ключами которого служат имена параметров, 

        а значениями - либо фиксированные значения (тогда они будут передаваться на каждую итерацию одинаковыми),

        либо экземпляры класса HyperSampler, из которых на каждую новую итерацию выбирается новое значение

    iteration_function -- задаваемая пользователем функция итерации. 

        Её первым аргументом является текущее состояние поиска (любой объект),

        вторым - номер итерации,

        Также в неё распаковывается словарь с ключами из parameters,

        поэтому она либо должна перечислить каждый из этих ключей в своём списке параметров,

        либо принимать неограниченное количество именованных параметров (**kvargs)

        Функция возвращает новое состояние поиска.

        Таким образом поиск представляет собой reduce-алгоритм

    initial_state -- начальное состояние поиска. Может быть любым объектом, включая None.

    В частности, если вы несколько раз запускали hyper_search, вы можете скормить сюда результат предыдущего запуска.

    progress_bar -- Имеет три возможных значения - None (не отображать прогресс), 'tqdm' - текстовый progress bar, 'tqdm_notebook' - для jupyter

    

    Возвращаемое значение:

    Конечное состояние поиска (результат вызова iteration_function на последней итерации)

     """

    # Разделяем словарь по признаку необходимости выборки

    sampled, fixed = split_dictionary(parameters, lambda _, v: isinstance(v, HyperSampler))

    # Сразу, наперед, отбираем num_trials значений для каждого нефиксированного параметра

    random_queue = {}

    for name, population in sampled.items():

        random_queue[name] = population.sample(num_trials)



    if progress_bar == 'tqdm':

        from tqdm import tqdm

        tqdm_function = lambda iterable, **kws: tqdm(iterable,**kws)

    elif progress_bar == 'tqdm_notebook':

        from tqdm import tqdm_notebook

        tqdm_function = lambda iterable, **kws: tqdm_notebook(iterable, **kws)

    else:

        tqdm_function = lambda iterable, **kws: iterable



    # Превращаем numpy-скаляры в соотв. классы python, чтобы не сломать некоторые функции.

    def decay(x):

        if isinstance(x, np.generic):

            return x.item()

        return x



    current_state = initial_state



    for trial in tqdm_function(range(num_trials), desc='Trial #'):

        # Собираем значения гиперпараметров на текущую итерацию вместе

        current_trial_random = {k: decay(sample[trial]) for k,sample in random_queue.items()}

        iteration_setting = dict(current_trial_random, **fixed)

        iteration_setting.update(fixed)

        # Вызываем функцию итерации и вызываем 

        current_state = iteration_function(current_state, trial, **iteration_setting)



    return current_state
import torch

import torch.nn as nn
np.random.seed(5771)

torch.manual_seed(5661)
class PersistentModelWrapper:

    def __init__(self, path, initial_criterion):

        self.path = path

        self.criterion = initial_criterion



    def update(self, model, optimizer, criterion):

        self.criterion = criterion

        torch.save(

            {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'criterion': criterion},

            self.path)



    def load_model_data(self):

        return torch.load(self.path)



    def restore(self, model, optimizer):

        model_data = self.load_model_data()

        model.load_state_dict(model_data['model_state'])

        optimizer.load_state_dict(model_data['optimizer_state'])

!pip install pytorch-pretrained-bert
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text = "[CLS] Каким образом так получилось, что мы стоим на краю этой дороги [SEP]"

tokenized_text = tokenizer.tokenize(text)

masked_index = 5

tokenized_text[masked_index] = '[MASK]'

print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0 for _ in range(len(indexed_tokens))]

print(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])

segments_tensors = torch.tensor([segments_ids])
# # model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased').eval()

# model = model.cuda()

# with torch.no_grad():

#     predictions = model.bert(tokens_tensor.cuda(), segments_tensors.cuda())

# topk = torch.topk(predictions[0,masked_index],10)

# print(topk)
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
# class DataProcessor(object):

#     """Base class for data converters for sequence classification data sets."""



#     def get_train_examples(self, data_dir):

#         """Gets a collection of `InputExample`s for the train set."""

#         raise NotImplementedError()



#     def get_dev_examples(self, data_dir):

#         """Gets a collection of `InputExample`s for the dev set."""

#         raise NotImplementedError()



#     def get_labels(self):

#         """Gets the list of labels for this data set."""

#         raise NotImplementedError()



#     @classmethod

#     def _read_tsv(cls, input_file, quotechar=None):

#         """Reads a tab separated value file."""

#         with open(input_file, "r", encoding="utf-8") as f:

#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)

#             lines = []

#             for line in reader:

#                 if sys.version_info[0] == 2:

#                     line = list(unicode(cell, 'utf-8') for cell in line)

#                 lines.append(line)

#             return lines
def convert_examples_to_features(examples, label_list, max_seq_length,

                                 tokenizer, output_mode, total_examples=None):

    """Loads a data file into a list of `InputBatch`s."""



    label_map = {label : i for i, label in enumerate(label_list)}



    features = []

    for (ex_index, example) in enumerate(examples):

        if ex_index % 10000 == 0:

            logger.info("Writing example {} of {}".format(ex_index, total_examples))



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

        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # (b) For single sequences:

        #  tokens:   [CLS] the dog is hairy . [SEP]

        #  type_ids: 0   0   0   0  0     0 0

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

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        segment_ids = [0] * len(tokens)



        if tokens_b:

            tokens += tokens_b + ["[SEP]"]

            segment_ids += [1] * (len(tokens_b) + 1)



        input_ids = tokenizer.convert_tokens_to_ids(tokens)



        # The mask has 1 for real tokens and 0 for padding tokens. Only real

        # tokens are attended to.

        input_mask = [1] * len(input_ids)



        # Zero-pad up to the sequence length.

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding

        input_mask += padding

        segment_ids += padding



        assert len(input_ids) == max_seq_length

        assert len(input_mask) == max_seq_length

        assert len(segment_ids) == max_seq_length



        if output_mode == "classification":

            label_id = label_map[example.label]

        elif output_mode == "regression":

            label_id = float(example.label)

        else:

            raise KeyError(output_mode)



        if ex_index < 5:

            logger.info("*** Example ***")

            logger.info("guid: %s" % (example.guid))

            logger.info("tokens: %s" % " ".join(

                    [str(x) for x in tokens]))

            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

            logger.info(

                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            logger.info("label: %s (id = %d)" % (example.label, label_id))



        features.append(

                InputFeatures(input_ids=input_ids,

                              input_mask=input_mask,

                              segment_ids=segment_ids,

                              label_id=label_id))

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
import gc

gc.collect()
imdb_df = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv', encoding='latin-1')
imdb_df.sample(10)
dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]
test_df = imdb_df[(imdb_df.type == 'test')]
from sklearn import model_selection
train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)
for row in train_df.iterrows():

    print(row[1].type)

    break
def df_to_examples_imdb(df):

    for idx,row in df.iterrows():

        yield InputExample(idx,row.review,label=row.label)
train_features = convert_examples_to_features(df_to_examples_imdb(train_df),

                                              ['neg','pos'],

                                              max_seq_length=230,

                                              tokenizer=tokenizer,

                                              output_mode="classification")
val_features = convert_examples_to_features(df_to_examples_imdb(val_df),

                                              ['neg','pos'],

                                              max_seq_length=230,

                                              tokenizer=tokenizer,

                                              output_mode="classification")
def features_to_tensors(list_of_features):

    all_text_tensor = torch.tensor([f.input_ids for f in list_of_features], dtype=torch.long)

    all_mask_tensor = torch.tensor([f.input_mask for f in list_of_features], dtype=torch.long)

    all_segment_tensor = torch.tensor([f.segment_ids for f in list_of_features], dtype=torch.long)

    all_label_tensor = torch.tensor([f.label_id for f in list_of_features], dtype=torch.long)

    return all_text_tensor, all_mask_tensor, all_segment_tensor, all_label_tensor
from torch.utils.data import TensorDataset,DataLoader
train_text_tensor, train_mask_tensor, train_segment_tensor, train_label_tensor = features_to_tensors(train_features)

val_text_tensor, val_mask_tensor, val_segment_tensor, val_label_tensor = features_to_tensors(val_features)
train_dataset = TensorDataset(train_text_tensor, train_mask_tensor, train_segment_tensor, train_label_tensor)

val_dataset = TensorDataset(val_text_tensor, val_mask_tensor, val_segment_tensor, val_label_tensor)
print(train_label_tensor[:2])
from pytorch_pretrained_bert import BertAdam
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2).cuda()
!nvidia-smi
# Batch size: 16, 32

# • Learning rate (Adam): 5e-5, 3e-5, 2e-5

# • Number of epochs: 3, 4
from pytorch_pretrained_bert import BertConfig
import os
class BertPersistentWrapper:

    def __init__(self, prefix, initial_criterion, num_labels):

        self.prefix = prefix

        self.model_path = prefix + '_model.bin'

        self.config_path = prefix + '_config.bin'

#         self.vocab_path = prefix + '_vocab.bin'

        self.criterion = initial_criterion

        self.num_labels = num_labels



    def update(self, model, criterion):

        self.criterion = criterion

        model_to_save = model.module if hasattr(model, 'module') else model

        torch.save(model_to_save.state_dict(), self.model_path)

        model_to_save.config.to_json_file(self.config_path)

#         tokenizer.save_vocabulary(output_vocab_file)



#     def load_model_data(self):

#         return torch.load(self.path)



    def restore(self):

        config = BertConfig.from_json_file(self.config_path)

        model = BertForSequenceClassification(config, num_labels=self.num_labels)

        model.load_state_dict(torch.load(self.model_path))

        return model

    

    def destroy(self):

        os.remove(self.model_path)

        os.remove(self.config_path)
class SearchState:

    def __init__(self, best_model, parameter_stats):

        self.best_model = best_model

        self.parameter_stats = parameter_stats

        

SearchState = namedtuple('SearchState',['best_model', 'parameter_stats'])
def bert_wrapper_test():

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2)

    print(model.num_labels)

    bw = BertPersistentWrapper('wrapper_test',10, 2)

    bw.update(model,5)

    res_model = bw.restore()

    print(res_model)

    print(res_model.num_labels)

    bw.destroy()

    

bert_wrapper_test()
!ls
import gc

gc.collect()
from tqdm import tqdm_notebook
def train_bert(hyper_state, hyper_trial,

               n_epochs,

               gradient_accumulation_steps,

               batch_size,

               learning_rate,

               warmup_proportion,

               train_dataset,

               val_dataset,

               num_labels,

               device):

    print('Trial', hyper_trial)

    print('n_epochs = {}, effective_batch_size={}, lr={}, warmup={}'.format(n_epochs,

                                                                            batch_size * gradient_accumulation_steps,

                                                                            learning_rate,

                                                                            warmup_proportion))

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels).cuda()

    num_train_optimization_steps = n_epochs * int(len(train_dataset) / batch_size / gradient_accumulation_steps)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)



    optimizer = BertAdam(model.parameters(), lr=learning_rate, warmup=warmup_proportion,

                         t_total=num_train_optimization_steps)



    best_model = BertPersistentWrapper(f'model{hyper_trial}.md', 0.0, num_labels)



    for epoch in tqdm_notebook(range(n_epochs), desc='Epoch'):

        model.train()

        tr_loss = 0.0

        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm_notebook(train_loader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch



            # define a new function to compute loss values for both output_modes

            logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))



            if gradient_accumulation_steps > 1:

                loss = loss / gradient_accumulation_steps



            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)

            nb_tr_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0:

                optimizer.step()

                optimizer.zero_grad()



        tr_loss /= len(train_loader)

        print('Epoch {}, training_loss={}'.format(epoch, tr_loss))



        model.eval()

        with torch.no_grad():

            running_corrects = 0

            running_total = 0



            running_loss = 0.0

            for batch in tqdm_notebook(val_loader):

                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids = batch



                logits = model(input_ids, segment_ids, input_mask, labels=None)

                preds = logits.view(-1, num_labels).argmax(dim=1)



                running_total += input_ids.size(0)

                running_corrects += (preds == label_ids.view(-1)).sum().item()



                loss_fct = nn.CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                running_loss += loss.item()



        val_loss = running_loss / len(val_loader)

        val_accuracy = running_corrects / running_total

        print('Epoch {}, val_loss={}, val_accuracy={}'.format(epoch, val_loss, val_accuracy))



        if val_accuracy > best_model.criterion:

            best_model.update(model, val_accuracy)



    del model

    torch.cuda.empty_cache()

#     print('n_epochs = {}, effective_batch_size={}, lr={}, warmup={}'.format(n_epochs,

#                                                                             batch_size * gradient_accumulation_steps,

#                                                                             learning_rate,

#                                                                             warmup_proportion))

    param_stats = dict(

        n_epochs=n_epochs,

        gradient_batch=batch_size * gradient_accumulation_steps,

        lr=learning_rate,

        accuracy=val_accuracy

    )

    

    if not hyper_state:

        return SearchState(best_model, [param_stats])

    

    hyper_state.parameter_stats.append(param_stats)

    if best_model.criterion > hyper_state.best_model.criterion:

        hyper_state.best_model.destroy()

        hyper_state.best_model = best_model

    else:

        best_model.destroy()

    return hyper_state

                
settings_for_random_search = {

    'learning_rate': h_enum(5e-5, 3e-5, 2e-5),

    'warmup_proportion': 0.1,

    'gradient_accumulation_steps': h_enum(1,2),

    'batch_size': 16,

    'n_epochs': h_enum(3,4),

    'train_dataset': train_dataset,

    'val_dataset': val_dataset,

    'num_labels': 2,

    'device': torch.device('cuda')

}
result_state = hyper_search(6,settings_for_random_search,train_bert,None,'tqdm_notebook')

pd.DataFrame.from_records(result_state.parameter_stats)
bert_model = result_state.best_model.restore()
torch.cuda.empty_cache()
bert_model = bert_model.cuda()
import gc

gc.collect()
def predict_loader(bert_model ,loader, device='cuda'):

    bert_model.eval()

    predictions = []

    correct_predictions = []

    with torch.no_grad():

        for batch in tqdm_notebook(loader):

            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch

            logits = bert_model(input_ids, segment_ids, input_mask, labels=None)

            predictions.extend(logits.argmax(dim=1).tolist())

            correct_predictions.extend(label_ids.tolist())

#             break

    return predictions, correct_predictions
y_pred, y_test = predict_loader(bert_model, DataLoader(val_dataset, batch_size=16))
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
test_features = convert_examples_to_features(df_to_examples_imdb(test_df),

                                              ['neg','pos'],

                                              max_seq_length=230,

                                              tokenizer=tokenizer,

                                              output_mode="classification")
test_text_tensor, test_mask_tensor, test_segment_tensor, test_label_tensor = features_to_tensors(test_features)
test_dataset = TensorDataset(test_text_tensor, test_mask_tensor, test_segment_tensor, test_label_tensor)

test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)

y_pred, y_test = predict_loader(bert_model, test_loader)

print(metrics.classification_report(y_test, y_pred))

print(metrics.accuracy_score(y_test, y_pred))