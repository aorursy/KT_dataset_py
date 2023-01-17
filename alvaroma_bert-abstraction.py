import os

import json

import numpy as np



class Task():

    '''

    Helper class to read a Task as a sequence of tokens

    '''

    def __init__(self, path, name):

        filename = os.path.join(path, name)

        with open(filename, 'r') as fp:

            self.tasks = json.load(fp)



    def max_size(self):

        max_w, max_h = 0, 0

        for t in self.tasks['train']:

            w, h = self._size(t['input'])

            if w > max_w: max_w = w

            if h > max_h: max_h = h

            w, h = self._size(t['output'])

            if w > max_w: max_w = w

            if h > max_h: max_h = h

        for t in self.tasks['test']:

            w, h = self._size(t['input'])

            if w > max_w: max_w = w

            if h > max_h: max_h = h

            if 'output' in t:

                w, h = self._size(t['output'])

                if w > max_w: max_w = w

                if h > max_h: max_h = h

        return max_w, max_h, max(max_w, max_h)



    def _size(self, lst):

        h = len(lst)

        w = len(lst[0])

        return w, h



    def get_train(self):

        for t in self.tasks['train']:

            inp = Task.as_sequence(t['input'])

            out = Task.as_sequence(t['output'])

            yield inp, out



    def get_valid(self):

        for t in self.tasks['test']:

            inp = Task.as_sequence(t['input'])

            out = Task.as_sequence(t['output'])

            yield inp, out



    def get_tests(self):

        for t in self.tasks['test']:

            inp = Task.as_sequence(t['input'])

            yield inp



    @staticmethod

    def as_sequence(lst):

        sequ = []

        h = len(lst)

        w = len(lst[0])

        sequ.append('[CLS]')

        for y in range(h):

            for x in range(w):

                c = lst[y][x]

                sequ.append(str(c))

                #sequ.append(' ')

            sequ.append('[SEP]')

        return sequ



    @staticmethod

    def pred_size(sequ):

        sequ = sequ[1:]

        w = 0

        h = 0

        x = 0

        y = 0

        for n in range(len(sequ)):

            if sequ[n] == 0: break

            if sequ[n] == 3:

                w = max(x, w)

                x = 0

                y += 1

                continue

            x += 1

        h = y

        return w, h



    @staticmethod

    def as_array(sequ, w=16, h=16):

        sequ = sequ[1:]

        array = np.zeros((h, w), dtype='int')

        n = 0

        for y in range(h):

            for x in range(w):

                if n >= len(sequ): return array

                c = sequ[n] - 5

                array[y][x] = max(0, c)

                n += 1

            n += 1

        return array



    @staticmethod

    def as_list(arr):

        lst = []

        for row in arr:

            lst.append(list(row))

        return lst

    
import os

import collections

import numpy as np



from torch.utils.data import Dataset



from transformers import BertTokenizer



class TaskDataset(Dataset):

    def __init__(self, path, name, mode='train'):

        self.tokenizer = BertTokenizer('vocab.txt')

        self.items = self.__load(path, name, mode)



    def __load(self, path, name, mode):

        items = []

        task = Task(path, name)

        self.task_width, self.task_height, _ = task.max_size()

        max_length = 2 + (self.task_width + 1) * self.task_height

        if mode == 'train':

            for inp, out in task.get_train():

                inp = self.__tokenize(inp, max_length)

                out = self.__tokenize(out, max_length)

                items.append((name, inp, out))

        if mode == 'valid':

            for inp, out in task.get_valid():

                inp = self.__tokenize(inp, max_length)

                out = self.__tokenize(out, max_length)

                items.append((name, inp, out))

        if mode == 'tests':

            for inp in task.get_tests():

                inp = self.__tokenize(inp, max_length)

                items.append((name, inp, inp))

        return items



    def __tokenize(self, sequ, max_length):

        sequ_dict = self.tokenizer.encode_plus(sequ, add_special_tokens=False, max_length=max_length, pad_to_max_length=True)

        return np.array(sequ_dict['input_ids'], dtype=np.int64), np.array(sequ_dict['attention_mask'], dtype=np.int64), np.array(sequ_dict['token_type_ids'], dtype=np.int64), 



    def __len__(self):

        return len(self.items)



    def __getitem__(self, idx):

        return self.items[idx]

    
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as op



from transformers import BertConfig, BertModel, BertPreTrainedModel



INTERMEDIATE_SIZE = 1024

HIDDEN_SIZE = 1024

VOCAB_SIZE = 16



def create_model_config(w, h):

    return {

      "architectures": [

        "BertForMaskedLM"

      ],

      "attention_probs_dropout_prob": 0.5,

      "hidden_act": "gelu",

      "hidden_dropout_prob": 0.5,

      "hidden_size": HIDDEN_SIZE,

      "initializer_range": 0.02,

      "intermediate_size": INTERMEDIATE_SIZE,

      "max_position_embeddings": 2 + (w + 1) * h,

      "num_attention_heads": 8,

      "num_hidden_layers": 8,

      "type_vocab_size": 1,

      "vocab_size": VOCAB_SIZE,

      "num_labels": 1

    }



class Network(BertPreTrainedModel):

    def __init__(self, w, h):

        config = BertConfig.from_dict(create_model_config(w, h))

        super(Network, self).__init__(config)



        self.bert = BertModel(config)

        self.fc = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)



        self.init_weights()



    def forward(self, X):

        ids = X[0]

        msk = X[1]

        typ = X[2]



        output = self.bert(ids, attention_mask=msk, token_type_ids=typ)

        x = output[0]

        x = F.relu(self.fc(x))

        x = x.transpose(1, 2)



        return x

    
import numpy as np



class Summary():

    def __init__(self):

        self.epoch = -1

        self.loss = None

        self.accuracy = None

        self.values = None

        self.history = []



    def register(self, epoch, loss, accuracy, values={}):

        values['epoch'] = epoch

        values['loss'] = loss

        values['accuracy'] = accuracy

        self.history.append(values)

        if self.epoch < 0 or accuracy > self.accuracy:

            self.epoch = epoch

            self.loss = loss

            self.accuracy = accuracy

            self.values = values

            return True

        return False



    def hash(self):

        def r4(v): return "{:6.4f}".format(v)

        return F'{self.epoch:02}-{r4(self.accy).strip()}'



    def __str__(self):

        def r4(v): return "{:6.4f}".format(v)

        return F'{self.epoch}\t{r4(self.values["loss"])}\t{r4(self.values["accuracy"])}'

    

class BaseMetrics():

    def __init__(self):

        self.loss = None

        self.preds = None

        self.targs = None

        self.values = None



    def begin(self):

        self.loss = []

        self.preds = []

        self.targs = []

        self.values = None



    def update(self, loss, dz, dy):

        self.loss.append(self._get_loss(loss))

        self.preds.append(self._get_preds(dz))

        self.targs.append(self._get_targets(dy))



    def _get_loss(self, loss):

        return loss.item()



    def _get_preds(self, dz):

        return dz.detach().cpu().numpy()



    def _get_targets(self, dy):

        return dy.detach().cpu().numpy()



    def commit(self, epoch):

        loss = np.mean(self.loss)

        accuracy = 0.0

        self.values = {

                'loss': 0.0,

                'accuracy': 0.0

            }

        return accuracy



    def __str__(self):

        def r4(v): return "{:6.4f}".format(v)

        return F'{r4(self.values["loss"])}\t{r4(self.values["accuracy"])}'



class CrossEntropyMetrics(BaseMetrics):

    def _get_preds(self, dz):

        dz = dz[:,:,1:]

        return dz.detach().cpu().numpy()



    def _get_targets(self, dy):

        dy = dy[:,1:]

        return dy.detach().cpu().numpy()



    def commit(self, epoch):

        loss = np.mean(self.loss)

        preds = np.concatenate(self.preds, axis=0)

        targs = np.concatenate(self.targs, axis=0)



        preds = preds.argmax(axis=1)

        targs = targs

        preds = preds.reshape(preds.shape[0], -1)

        targs = targs.reshape(targs.shape[0], -1)

        accuracy = (sum(preds == targs) / len(preds)).mean()



        match = 0

        for pred, targ in zip(preds, targs):

            if sum(pred != targ) == 0:

                match += 1



        accuracy += match



        self.values = {

                'loss': loss,

                'accuracy': accuracy

            }

        return accuracy

    
import numpy as np



import matplotlib.pyplot as plt



class EasyGrid():

    def __init__(self, cmap=None, norm=None):

        self.images = []

        self.titles = []

        self.cmap = cmap

        self.norm = norm



    def append(self, image, title=''):

        self.images.append(image)

        self.titles.append(title)



    def show(self, rows, cols, title='', figsize=(4, 2), gridlines=False):

        f, axs = plt.subplots(rows, cols, figsize=figsize)

        f.suptitle(title, fontsize=16)

        inx = 0

        for y in range(rows):

            for x in range(cols):

                if inx == len(self.images): break

                if rows > 1 and cols > 1:

                    ax = axs[y, x]

                elif rows > 1:

                    ax = axs[y]

                else:

                    ax = axs[x]

                ax.imshow(self.images[inx], cmap=self.cmap, norm=self.norm)

                ax.set_title(self.titles[inx])

                if gridlines:

                    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)   

                    ax.set_yticks([x-0.5 for x in range(1+len(self.images[inx]))])

                    ax.set_xticks([x-0.5 for x in range(1+len(self.images[inx][0]))])     

                    ax.set_xticklabels([])

                    ax.set_yticklabels([])

                inx += 1

        plt.show()



from matplotlib import colors



CMAP = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

NORM = colors.Normalize(vmin=0, vmax=9)



def plot_task_array(arrays, cols=2, title=''):

    grid = EasyGrid(CMAP, NORM)

    for array in arrays:

        grid.append(array)

    l = len(arrays)

    grid.show(l // cols + l % cols, cols, title=title, gridlines=True)

    
import os

import random



# flattener

def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred



# Seed

def random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.optim as op



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



PATH = '../input/abstraction-and-reasoning-challenge'

PATH_TESTS = PATH + '/test'



EPOCHS = 100

BATCH = 4

LR = 0.0001



SEED = 1234

random_seed(SEED)



#   Tokens

TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[###]']

with open('vocab.txt', 'w') as fp:

    for t in TOKENS:

        fp.write(t + '\n')



def train(name, epochs=EPOCHS):

    # Data

    ds_train = TaskDataset(PATH_TESTS, name, mode='train')

    ld_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH, num_workers=4, shuffle=True)



    # Size

    w = ds_train.task_width

    h = ds_train.task_height



    # Model

    model = Network(w, h)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = op.Adam(model.parameters(), lr=LR)



    metrics = CrossEntropyMetrics()



    # Train

    summary_train = Summary()

    for epoch in range(epochs):

        model.train()

        metrics.begin()

        for step, (id, x, y) in enumerate(ld_train):

            optimizer.zero_grad()



            x0 = x[0].to(DEVICE)

            x1 = x[1].to(DEVICE)

            x2 = x[2].to(DEVICE)

            dy = y[0].to(DEVICE)



            dz = model((x0, x1, x2))



            loss = criterion(dz[:,:,1:], dy[:,1:])

            metrics.update(loss, dz, dy)



            loss.backward()

            optimizer.step()

        accuracy = metrics.commit(epoch)

        summary_train.register(epoch, loss.item(), accuracy, metrics.values)



    print(summary_train)

    return model





def predict(model, name):

    # Data

    ds_tests = TaskDataset(PATH_TESTS, name, mode='tests')

    ld_tests = torch.utils.data.DataLoader(ds_tests, batch_size=1, shuffle=False)



    # Size

    w = ds_tests.task_width

    h = ds_tests.task_height



    preds = []

    model.eval()

    with torch.no_grad():

        for step, (id, x, y) in enumerate(ld_tests):

            x0 = x[0].to(DEVICE)

            x1 = x[1].to(DEVICE)

            x2 = x[2].to(DEVICE)



            dz = model((x0, x1, x2))

            z = dz.detach().cpu().numpy()

            z = z[0]

            az = z.argmax(0)

            w, h = Task.pred_size(az)

            w = max(1, w)

            h = max(1, h)

            pred = Task.as_array(az, w, h)

            pred = Task.as_list(pred)

            flat = flattener(pred)

            preds.append(flat)



    return preds



submission = pd.read_csv(PATH + '/sample_submission.csv', index_col='output_id')



STEPS = -1



step = 0

hash = set()

for output_id in submission.index:

    task_id = output_id.split('_')[0]

    if task_id in hash:

        continue

    hash.add(task_id)

    name = str(task_id + '.json')



    model = train(name, EPOCHS)

    preds = predict(model, name)



    for n in range(len(preds)):

        pred = preds[n] + ' ' + preds[n] + ' ' + preds[n]

        submission.loc[F'{task_id}_{n}', 'output'] = pred

    step += 1

    if step == STEPS: break



submission.to_csv('submission.csv')

df = pd.read_csv('submission.csv')

df.head()



tokenizer = BertTokenizer('vocab.txt')

def tokenize(sequ, max_length):

    sequ_dict = tokenizer.encode_plus(sequ, add_special_tokens=False, max_length=max_length, pad_to_max_length=True)

    return np.array(sequ_dict['input_ids'], dtype=np.int64)



for values in df.values:

    name = values[0]

    name = name[:-2] + '.json'

    print('TRAIN', name)

    task = Task(PATH_TESTS, name)

    for inp, out in task.get_train():

        task_width, task_height, _ = task.max_size()

        max_length = 2 + (task_width + 1) * task_height

        x = tokenize(inp, max_length)

        y = tokenize(out, max_length)

        wx, hx = Task.pred_size(x)

        arr_x = Task.as_array(x, wx, hx)

        wy, hy = Task.pred_size(y)

        arr_y = Task.as_array(y, wy, hy)

        plot_task_array([arr_x, arr_y])

    ix = 0

    

    print('PREDS', name)

    for inp in task.get_tests():

        try:

            x = tokenize(inp, max_length)

            wx, hx = Task.pred_size(x)

            arr_x = Task.as_array(x, wx, hx)



            parts = values[1].split(' ')[0].split('|')

            parts = parts[1:-1]

            h = len(parts)

            w = len(parts[0])

            a = np.zeros((h, w), dtype='int')

            for y in range(h):

                for x in range(w):

                    a[y][x] = parts[y][x]        

            plot_task_array([arr_x, a])

        except Exception as ex:

            print(ex)

    print()