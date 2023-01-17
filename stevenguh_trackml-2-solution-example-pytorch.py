import numpy as np

import pandas as pd

import os

from math import ceil



import torch

import torch.nn as nn

from torch.nn import init

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Sampler



import line_profiler

%load_ext line_profiler



DATASET_PATH = '../input/trackml-model/'



from tqdm import tqdm_notebook

print(os.listdir("../input"))

print(os.listdir("../input/trackml/"))

print(os.listdir(DATASET_PATH))

prefix='../input/trackml-particle-identification/'
from contextlib import contextmanager

from timeit import default_timer



@contextmanager

def elapsed_timer():

    start = default_timer()

    elapser = lambda: default_timer() - start

    yield lambda: elapser()

    end = default_timer()

    elapser = lambda: end-start
def get_event(event, filter=None):

    hits = pd.read_csv(prefix+'train_1/%s-hits.csv'%event)

    cells = pd.read_csv(prefix+'train_1/%s-cells.csv'%event)

    truth = pd.read_csv(prefix+'train_1/%s-truth.csv'%event)

    particles = pd.read_csv(prefix+'train_1/%s-particles.csv'%event)

    return hits, cells, truth, particles



def create_model(fs = 10):

    return nn.Sequential(

        nn.Linear(fs, 800),

        nn.SELU(),

        nn.Linear(800, 400),

        nn.SELU(),

        nn.Linear(400, 400),

        nn.SELU(),

        nn.Linear(400, 400),

        nn.SELU(),

        nn.Linear(400, 200),

        nn.SELU(),

        nn.Linear(200, 1),

        nn.Sigmoid()

    )
USE_GPU = True



TRAIN_1 = False

TRAIN_2 = False

TRAIN = TRAIN_1 or TRAIN_2

REDUCE_ON_PLATEAU = True



LOADING_MODEL = True

LOADING_MODEL_H = True



PRE_PROCESS = False

PRE_PROCESS_H = False



SAVING = False

LOADING_PREFIX = DATASET_PATH

EVENT_SIZE_PATH = 'event_rows.csv'

EVENT_SIZE_H_PATH = 'event_rows-h.csv'



if USE_GPU and torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

def get_features(event_name):

    hits, cells, truth, particles = get_event(event_name)

    

    # Filter out un-used columns early

    hits = hits[['hit_id', 'x', 'y', 'z']]

    truth = truth[['particle_id', 'hit_id']]    

    



    # as_index=False so the group by retain the column name

    cell_by_hit_id = cells.groupby(['hit_id'], as_index=False)

    cell_count = cell_by_hit_id.value.count().rename(columns={'value':'cell_count'})

    charge_value = cell_by_hit_id.value.sum().rename(columns={'value':'charge_value'})

    

    # Scaling

    hits[['x', 'y', 'z']] /= 1000

    cell_count['cell_count'] /= 10

    

    truth = pd.merge(truth, cell_count, on='hit_id')

    truth = pd.merge(truth, charge_value, on='hit_id')

    truth = pd.merge(truth, hits, on='hit_id')

    # The columns of truth are as follow

    # ['particle_id', 'hit_id', 'x', 'y', 'z', 'cell_count', 'charge_value']

    return truth



def pre_process(event_name, print_size=True):

    features = get_features(event_name)

    

    columns_needed = ['x', 'y', 'z', 'cell_count', 'charge_value']

    columns_needed_all = [c + '_x' for c in columns_needed] + [c + '_y' for c in columns_needed] + ['label']



    # Get all the hits that's identified with a particle

    true_pairs = features[features.particle_id != 0]

    # Merge to create all hit pairs that's identified with the same particle

    true_pairs = pd.merge(true_pairs, true_pairs, on='particle_id')

    # Filter all the pairs that has the same hit_id

    true_pairs = true_pairs[true_pairs.hit_id_x != true_pairs.hit_id_y]

    # Add a new column to indicate this dataset is the true dataset

    true_pairs['label'] = 1

    # Filter the only columns needed

    true_pairs = true_pairs[columns_needed_all]

    

    FALSE_PAIR_RATIO = 3

    size = len(true_pairs) * FALSE_PAIR_RATIO

    p_id = features.particle_id.values

    # Generated random hit idx pairs

    i = np.random.randint(len(features), size=size)

    j = np.random.randint(len(features), size=size)

    # Get the hit idx pair that's either assoicated with particle id 0 or different particle id

    hit_idx = (p_id[i]==0) | (p_id[i]!=p_id[j])

    i, j = i[hit_idx], j[hit_idx]

    # Filter and create features with the correct order of the columns

    features = features[columns_needed]

    false_pairs = pd.DataFrame(

        np.hstack((features.values[i], features.values[j], np.zeros((len(i),1)))),

        columns=columns_needed_all)



    processed = pd.concat([true_pairs, false_pairs], axis=0)

    processed = processed.sample(frac=1).reset_index(drop=True)

    

    if print_size:

        # Create a DataFrame just to pretty-print ;)

        print(event_name)

        print(pd

              .DataFrame(data={

                  'True': ['{:,}'.format(len(true_pairs))],

                  'False': ['{:,}'.format(len(false_pairs))],

                  'Total': ['{:,}'.format(len(processed))]

              })

              .to_string(index=False))

    return processed
if PRE_PROCESS:

    event_rows = []

    for i in tqdm_notebook(range(10, 20)):

        event_name = 'event0000010%02d'%i

        file_name = '%s.feather' % event_name

        processed = pre_process(event_name)

        event_rows.append((file_name, len(processed.index)))

        processed.to_feather(file_name) # Save to disk

        print('saved %s' % file_name)



    pd.DataFrame(event_rows).to_csv(EVENT_SIZE_PATH, index=False)

    print('event rows saved')

    del processed

else:

    print('load event rows')

    event_rows = list(pd.read_csv(LOADING_PREFIX + EVENT_SIZE_PATH).itertuples(index=False, name=None))

    event_rows = [(LOADING_PREFIX + r[0], r[1]) for r in event_rows]
from datetime import datetime

from feather import read_dataframe as feather_read

from multiprocessing import current_process

from threading import current_thread

import bisect



class FeatherCache():

    @staticmethod

    def cumsum(processed_rows):

        r, s = [], 0

        for row in processed_rows:

            l = row[1]

            r.append(l + s)

            s += l

        return r

    

    def __init__(self, processed_rows, cache_size=2, print_proc=False):

        self.processed_rows = processed_rows

        self.cumulative_sizes = self.cumsum(processed_rows)

        self.cache_size = cache_size

        self.print_proc = print_proc

        

        # warm up the loading by having two processed events loaded

        self.cache = {}

        for file_name, size in processed_rows[0:cache_size]:

            self.cache[file_name] = feather_read(file_name)

        

        self.time_stamps = {}

        



    def __len__(self):

        return self.cumulative_sizes[-1]

    

    @property

    def LRU_filename(self):

        least = None

        for file_name, time_stamp in self.time_stamps.items():

            if least is None:

                least = (file_name, time_stamp)

            elif time_stamp < least[1]:

                least = (file_name, time_stamp)

        return least[0]

    

    #TODO prefetch in another process when the file is loaded

    # https://stackoverflow.com/questions/45394783/multiprocess-reading-from-file

    def get_file_dataframe(self, file_name):

        if file_name in self.cache:

            # If in the cache, just get it

            self.time_stamps[file_name] = datetime.now()

            return self.cache[file_name]

        else:

            if self.print_proc:

                process_name = current_process().name

                thread_name = current_thread().name

                print('reading %s from thread %s, and process %s' % (file_name, thread_name, process_name))

            if len(self.cache) > self.cache_size:

                key = self.LRU_filename

                if self.print_proc:

                    print('delete %s' % key)

                del self.cache[key]

                del self.time_stamps[key]



            self.cache[file_name] = feather_read(file_name)

            self.time_stamps[file_name] = datetime.now()

            return self.cache[file_name]

                

    def get_map(self, indcies):

        # Map the indices back to to file_name and its corrsponding indcies

        file_dict = {}

        # Optimize for single dataset

        if (len(indcies) >= 2):

            front_idx = bisect.bisect_right(self.cumulative_sizes, indcies[0])

            back_idx = bisect.bisect_right(self.cumulative_sizes, indcies[-1])

            if front_idx == back_idx:

                file_name = self.processed_rows[front_idx][0]

                if front_idx == 0:

                    return {file_name : indcies}

            

        #else:

        for idx in indcies:

            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

            if dataset_idx == 0:

                sample_idx = idx

            else:

                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

            file_name = self.processed_rows[dataset_idx][0]

            file_dict.setdefault(file_name, []).append(sample_idx)

        return file_dict

    

    def get_items(self, indcies):

        d = self.get_map(indcies)

        ds = None

        for file_name in d:

            sample_idxs = d[file_name]

            f_ds = self.get_file_dataframe(file_name).iloc[sample_idxs]

            if ds is None:

                ds = f_ds

            else:

                ds = ds.append(f_ds, ignore_index=True)

        return ds

            

class FeatherDataset(Dataset):

    def __init__(self, feather_cache, to_items_fn=None):

        self.cache = feather_cache

        self.to_items_fn = to_items_fn

        

    def __len__(self):

        return len(self.cache)



    def __getitem__(self, idx):

        return idx;

    

    def get_items(self, indcies):

        ds = self.cache.get_items(indcies)

        rows = torch.as_tensor(ds.values)

        return rows if self.to_items_fn is None else self.to_items_fn(rows)

    

    @property

    def collate_fn(self):

        return self.get_items



class DataframeDataset(Dataset):

    def __init__(self, dataframe, opti_seq=False, to_items_fn=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.dataframe = dataframe

        self.opti_seq = opti_seq

        self.to_items_fn = to_items_fn

        

    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self, idx):

        return idx

    

    def get_items(self, indcies):

        if self.opti_seq and len(indcies) >= 2:

            rows = torch.as_tensor(self.dataframe.values[indcies[0]:indcies[-1]+1])

        else:

            rows = torch.as_tensor(self.dataframe.iloc[indcies].values)

        return rows if self.to_items_fn is None else self.to_items_fn(rows)

    

    @property

    def collate_fn(self):

        return self.get_items
class SequentialRangeSampler(Sampler):

    def __init__(self, data_source, num_samples=None):

        self.data_source = data_source

        self.num_samples = range(len(self.data_source)) if num_samples is None else num_samples



    def __iter__(self):

        return iter(self.num_samples)



    def __len__(self):

        return len(self.data_source)
def create_loaders(dataset, batch_size, validation_split):

    dataset_size = len(dataset)

    num_val = int(validation_split * dataset_size)

    num_train = dataset_size - num_val



    loader_train = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True,

                              sampler=SequentialRangeSampler(range(num_train)),

                              collate_fn=dataset.collate_fn)

    loader_val = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True,

                            sampler=SequentialRangeSampler(range(num_train, dataset_size)),

                            collate_fn=dataset.collate_fn)

    return loader_train, loader_val
if TRAIN_1:

    batch_size = 8000

    validation_split = .05 # 5%

    cache = FeatherCache(event_rows)

    dataset = FeatherDataset(cache, lambda rows : (rows[:, :-1], rows[:, -1].view(-1, 1)))

    loader_train, loader_val = create_loaders(dataset, batch_size, validation_split)
def check_accuracy(loader, model, thr=0.5):

    num_correct = 0

    num_samples = 0

    model.eval()  # set model to evaluation mode

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU

            y = y.view(-1).to(device=device, dtype=torch.uint8)

            scores = model(x)

            scores = (scores > thr).view(-1)

            num_correct += (scores == y).sum()

            num_samples += scores.size(0)

        acc = float(num_correct) / num_samples

        return (num_correct, num_samples, acc)
def train_model(model, optimizer, criterion, loader_train, loader_val, epochs=1, reduce_on_plateau=False, epoch_callback=None):

    """

    Train a model using the PyTorch Module API.

    

    Inputs:

    - model: A PyTorch Module giving the model to train.

    - optimizer: An Optimizer object we will use to train the model

    - criterion: A loss function

    - epochs: (Optional) A Python integer giving the number of epochs to train for

    

    Returns: Nothing, but prints model accuracies during training.

    """

    with elapsed_timer() as elapser:

        model = model.to(device=device)  # move the model parameters to CPU/GPU

        total_second = 0

        if reduce_on_plateau:

            scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, threshold=1e-3, verbose=True)

        for e in tqdm_notebook(range(epochs)):

            begin_epoch = elapser()

            for t, (x, y) in enumerate(tqdm_notebook(loader_train, desc='Epoch %d' % e, leave=False)):

                model.train()  # put model to training mode

                x = x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU

                y = y.view(-1, 1).to(device=device, dtype=torch.float) # BCELoss only support float as y



                # Zero out all of the gradients for the variables which the optimizer

                # will update.

                optimizer.zero_grad()



                scores = model(x)

                loss = criterion(scores, y)



                # This is the backwards pass: compute the gradient of the loss with

                # respect to each  parameter of the model.

                loss.backward()



                # Actually update the parameters of the model using the gradients

                # computed by the backwards pass.

                optimizer.step()



            num_correct, num_samples, acc = check_accuracy(loader_val, model)

            end_epoch = elapser()

            print('%.2fs - Epoch %d, Iteration %d, loss = %.4f, %d / %d correct (%.2f %%)' % (end_epoch - begin_epoch, e, t, loss.item(), num_correct, num_samples, acc * 100))

            if epoch_callback is not None:

                epoch_callback(num_correct, num_samples, acc, loss)

            if reduce_on_plateau:

                scheduler.step(acc)

    print('Total time: %.2fs' % elapser())
model_torch = create_model().to(device)
if TRAIN_1 and REDUCE_ON_PLATEAU:

    lr = -3

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=50, reduce_on_plateau=True)
if TRAIN_1 and not REDUCE_ON_PLATEAU:

    lr = -5

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=1)
if TRAIN_1 and not REDUCE_ON_PLATEAU:

    lr = -4

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=20)
if TRAIN_1 and not REDUCE_ON_PLATEAU:

    lr = -5

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=3)
if TRAIN_1 and SAVING:

    print('saving model')

    torch.save(model_torch.state_dict(), './torch_model.pt')
def predict(dataframe, model, batch_size=8000, num_worker=0):

    rows = torch.as_tensor(dataframe.values)

    num_elements = len(rows)

    num_batches = -(-num_elements // batch_size) # Round up

    model.eval()  # set model to evaluation mode

    scores = torch.zeros(num_elements, dtype=torch.float)

    with torch.no_grad():

        for i in range(num_batches):

            start = i * batch_size

            end = num_elements if i == num_batches - 1 else start + batch_size

            x_batch = rows[start:end]

            x_batch = x_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU

            scores[start:end] = model(x_batch).view(-1)



    return scores



def predict_true(dataframe, model, batch_size=8000, thr=0.5):

    scores = predict(dataframe, model, batch_size)

    indices = (scores > thr).nonzero()[:, 0]

    return dataframe.iloc[indices]
def negative_mine(model, event_name, smaple_size=30000000, print_size=True):

    with elapsed_timer() as elapser: 

        hits, cells, truth, particles = get_event(event_name)



        # Filter out un-used columns early

        hits = hits[['hit_id', 'x', 'y', 'z']]

        truth = truth[['particle_id', 'hit_id']]    





        # as_index=False so the group by retain the column name

        cell_by_hit_id = cells.groupby(['hit_id'], as_index=False)

        cell_count = cell_by_hit_id.value.count().rename(columns={'value':'cell_count'})

        charge_value = cell_by_hit_id.value.sum().rename(columns={'value':'charge_value'})



        # Scaling

        hits[['x', 'y', 'z']] /= 1000

        cell_count['cell_count'] /= 10



        features = pd.merge(truth, cell_count, on='hit_id')

        features = pd.merge(features, charge_value, on='hit_id')

        features = pd.merge(features, hits, on='hit_id')

        # The columns of truth are as follow

        # ['particle_id', 'hit_id', 'x', 'y', 'z', 'cell_count', 'charge_value']



        columns_needed = ['x', 'y', 'z', 'cell_count', 'charge_value']

        columns_needed_all = [c + '_x' for c in columns_needed] + [c + '_y' for c in columns_needed]



        p_id = features.particle_id.values

        # Generated random hit idx pairs

        i = np.random.randint(len(features), size=smaple_size)

        j = np.random.randint(len(features), size=smaple_size)

        # Get the hit idx pair that's either assoicated with particle id 0 or different particle id

        hit_idx = (p_id[i]==0) | (p_id[i]!=p_id[j])

        i, j = i[hit_idx], j[hit_idx]

        # Filter and create features with the correct order of the columns

        features = features[columns_needed]

        false_pairs = pd.DataFrame(

            np.hstack((features.values[i], features.values[j])),

            columns=columns_needed_all)



        before_size = len(false_pairs)

        false_pairs = predict_true(false_pairs, model_torch).reset_index(drop=True)

        false_pairs['label'] = 0

        after_size = len(false_pairs)

        if print_size:

            print(event_name)

            print('%.2fs - Before: %s, After: %s, Percent Pass: %d%%' % (elapser(), '{:,}'.format(before_size), '{:,}'.format(after_size), after_size/before_size*100))

        return false_pairs
def preprocess_h():

    event_rows_h = []

    for idx, i in enumerate(tqdm_notebook(range(10,20))):

        event_name = 'event0000010%02d' % i

        file_name = '%s.feather' % event_name

        processed_negative = negative_mine(model_torch, event_name)

        with elapsed_timer() as elapser:

            processed = feather_read(event_rows[idx][0]) # read the path from event_rows loaded

            processed = processed.append(processed_negative, ignore_index=True)

            processed = processed.sample(frac=1).reset_index(drop=True)

            print('Read, append and re-sample: %.2fs' % elapser())

        event_rows_h.append((file_name, len(processed.index)))

        processed.to_feather(file_name) # Save to disk

        print('saved %s' % file_name)



    pd.DataFrame(event_rows_h).to_csv(EVENT_SIZE_H_PATH, index=False)

    print('event rows h saved')

    return event_rows_h



# if you skip step2, you still need to run step1 to get training data.

if LOADING_MODEL:

    print('load model')

    model_torch.load_state_dict(torch.load('../input/trackml-model/torch_model.pt'))



# Preprocess

if PRE_PROCESS_H:

    event_rows_h = preprocess_h()

else:

        print('load event rows hard')

        event_rows_h = list(pd.read_csv(LOADING_PREFIX + EVENT_SIZE_H_PATH).itertuples(index=False, name=None))

        event_rows_h = [(LOADING_PREFIX + r[0], r[1]) for r in event_rows_h]
if TRAIN_2:

    batch_size = 8000

    validation_split = .05 # 5%

    cache = FeatherCache(event_rows_h[::-1]) # invert to switch it up?

    dataset = FeatherDataset(cache, lambda rows : (rows[:, :-1], rows[:, -1].view(-1, 1)))

    loader_train, loader_val = create_loaders(dataset, batch_size, validation_split)
if TRAIN_2 and REDUCE_ON_PLATEAU:

    lr = -4

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=50, reduce_on_plateau=True)
if TRAIN_2 and not REDUCE_ON_PLATEAU:

    lr = -4

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=30)
if TRAIN_2 and not REDUCE_ON_PLATEAU:

    lr = -5

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=10)
if TRAIN_2 and not REDUCE_ON_PLATEAU:

    lr = -6

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=3)
if TRAIN_2 and not REDUCE_ON_PLATEAU:

    lr = -7

    optimizer = optim.Adam(model_torch.parameters(), lr=10**lr)

    criterion = nn.BCELoss()

    train_model(model_torch, optimizer, criterion, loader_train, loader_val, epochs=3)
if TRAIN_2 and SAVING:

    torch.save(model_torch.state_dict(), './torch_model_h.pt')

if TRAIN_2:

    del dataframe

    del loader_train

    del loader_val
if LOADING_MODEL_H:

    print('load model_h')

    model_torch.load_state_dict(torch.load('../input/trackml-model/torch_model_h.pt'))
batch_size = 8000

validation_split = .05 # 5%

cache = FeatherCache(event_rows_h[::-1]) # invert to switch it up?

dataset = FeatherDataset(cache, lambda rows : (rows[:, :-1], rows[:, -1].view(-1, 1)))

loader_train, loader_val = create_loaders(dataset, batch_size, validation_split)
event = 'event000001001'

features = get_features(event)

hits, cells, truth, particles = get_event(event)



# Count number of hits of each group 'volume_id','layer_id','module_id'

# The hit_id is also sorted by 

# Use the line below is see for yourself

# (hits.groupby(['volume_id','layer_id','module_id'])['hit_id'].head(len(hits)) == np.arange(1, len(hits)+1)).all()

# Also, the sum of counts is same as the number of all hits. np.sum(count) == len(hits)

count = hits.groupby(['volume_id','layer_id','module_id'])['hit_id'].count().values

# Map hits index to the individual identifible module

module_id = np.zeros(len(hits), dtype='int32')



# for each individual identifible module

for i in range(len(count)):

    si = np.sum(count[:i])

    module_id[si:si+count[i]] = i

    #print('%d:%d(%d + %d) = %d' % (si, si+count[i], si, count[i], i))
def predict_pairs_test(model, hit_idx, features, thr=0.7):

    num_hits = len(features)

    columns_needed = ['x', 'y', 'z', 'cell_count', 'charge_value']

    features = features[columns_needed]



    # Load the features of a single hit to fill the "features"

    # Effectively, we are preparing the features pairs where

    # the second pair of the feature set is the hit we are

    # looking at. The "features" looks as as follow:

    # [f(0), f(3)]

    # [f(1), f(3)]

    # [    ...   ]

    # [f(n), f(3)]

    target = features.iloc[[hit_idx]]

    columns = features.columns

    features = features.add_suffix('_x')

    for c in columns:

        # Broadcast to the whole column

        features[c + '_y'] = target[c][hit_idx]

    

    scores = predict(features, model, batch_size=num_hits)

    indices = (scores > thr).nonzero()[:, 0]

    # Re-eval the feature pairs that give scores higher than thr

    # by flipping the order of the pair

    swaped_features = features.iloc[indices]

    swaped_columns = [c + '_y' for c in columns_needed] + [c + '_x' for c in columns_needed]

    swaped_scores = predict(swaped_features[swaped_columns], model, batch_size=len(indices))

    

    # Average the scores of selected indices

    scores[indices] = (scores[indices] + swaped_scores)/2

    

    return scores



def get_path(hit, features, module_id, model, mask, thr):

    path = [hit]

    a = 0

    while True:

        # Use the last hit in the path to predict the next hit

        c = predict_pairs_test(model, path[-1], features, thr/2)

        c = c.numpy()

        # Update the mask to the hits that passed the thr and mask

        mask = (c > thr) * mask

        # Set the mask so we can't use the same hit as the hit next

        mask[path[-1]] = 0

        

        if 1: # ???

            # Get hit_index of all the predictions that passed thr

            cand = np.where(c > thr)[0]

            # If there's at least on hit that passed the thr

            if len(cand) > 0:

                # Mask any hit index that is in the module group of

                # any of hit in the current path

                mask[cand[np.isin(module_id[cand], module_id[path])]] = 0

        # a is the accumulated scores

        # len(a) should get smaller over iterations before mask will keep limiting

        # the possible next hit.

        # The a.max of the last iteration is eliminated by the mask, so that

        # next hit will be based on the sum of the scores of the previous scores

        # and the current scores that's base on the last hit given they are premitted

        # by the mask.

        a = (c + a) * mask

        if a.max() < thr * len(path):

            break

        path.append(a.argmax())

    return path

# select one hit to construct a track

for hit in range(3):

#     path = %lprun -m torch.utils.data get_path(hit, features, module_id, model_torch, np.ones(len(truth)), 0.95)

#     path = %lprun -f DataframeDataset.get_items get_path(hit, features, module_id, model_torch, np.ones(len(truth)), 0.95)

#     path = %lprun -f predict get_path(hit, features, module_id, model_torch, np.ones(len(truth)), 0.95)

    path = get_path(hit, features, module_id, model_torch, np.ones(len(truth)), 0.95)

    # Get all the hit index of particle_id that's assoicated with 'hit'

    gt = np.where(truth.particle_id == truth.particle_id[hit])[0]

    path.sort()

    print('hit_id = ', hit+1)

    print('reconstruct :', path)

    print('ground truth:', gt.tolist())
def predict_all_pairs(model, features):

    num_hits = len(features)

    columns_needed = ['x', 'y', 'z', 'cell_count', 'charge_value']

    features = features[columns_needed]

    

    preds = [0] * num_hits

    for i in tqdm_notebook(range(len(features)-1)):

        # Load the features of a single hit to fill the "features"

        # Effectively, we are preparing the features pairs where

        # the second pair of the feature set is the hit we are

        # looking at. The "features" looks as as follow:

        # [f(0), f(3)]

        # [f(1), f(3)]

        # [    ...   ]

        # [f(n), f(3)]

        target = features.iloc[[i]]

        features_predict = features.iloc[i+1:]

        columns = features_predict.columns

        features_predict = features_predict.add_suffix('_x')

        for c in columns:

            features_predict[c + '_y'] = target[c][i]



        scores = predict(features_predict, model, batch_size=num_hits)

        indices = (scores > 0.2).nonzero()[:, 0]



        if len(indices) > 0:

            # Re-eval the feature pairs that give scores higher than thr

            # by flipping the order of the pair

            swaped_features = features_predict.iloc[indices]

            swaped_columns = [c + '_y' for c in columns_needed] + [c + '_x' for c in columns_needed]

            swaped_scores = predict(swaped_features[swaped_columns], model, batch_size=len(indices))



            # Average the scores of selected indices

            scores[indices] = (scores[indices] + swaped_scores)/2

        

        indices = (scores > 0.5).nonzero()[:, 0]

        # Append the result of hit i as two pairs as numpy array of hit_index and scores

        preds[i] = [(indices+i+1).numpy(), scores[indices].numpy()]

        

    preds[-1] = [np.array([], dtype='int64'), np.array([], dtype='float32')]



    # rebuild to NxN

    # for ii in reversed(range(len(preds))):

    for i in range(len(preds)): # for each hit

        ii = len(preds)-i-1 # looping backward

        for j in range(len(preds[ii][0])): # for each prediction of a hit

            jj = preds[ii][0][j] # get the hit index of the prediction

            # Build symmetry between the prediction and the hit

            # like TTA above.

            preds[jj][0] = np.insert(preds[jj][0], 0 ,ii)

            preds[jj][1] = np.insert(preds[jj][1], 0 ,preds[ii][1][j])

    return preds
# Predict all pairs for reconstruct by all hits. (takes 2.5hr but can skip)

skip_predict = True



if skip_predict is False:

    preds = predict_all_pairs(model_torch, features)

    if SAVING:

        np.save('my_%s.npy'%event, preds)

else:

    print('load predicts')

    preds = np.load('../input/trackml/my_%s.npy'%event)
def get_path2(hit, mask, thr):

    path = [hit]

    a = 0

    while True:

        c = get_predict2(path[-1])

        mask = (c > thr)*mask

        mask[path[-1]] = 0

        

        if 1:

            cand = np.where(c>thr)[0]

            if len(cand)>0:

                mask[cand[np.isin(module_id[cand], module_id[path])]]=0

                

        a = (c + a)*mask

        if a.max() < thr*len(path):

            break

        path.append(a.argmax())

    return path



def get_predict2(p):

    c = np.zeros(len(preds))

    c[preds[p, 0]] = preds[p, 1]

    return c
# reconstruct by all hits. (takes 0.6hr but can skip)

skip_reconstruct = True



if skip_reconstruct == False:

    tracks_all = []

    thr = 0.85

    # This is probably the optimization to get the best path.

    x4 = True

    for hit in tqdm_notebook(range(len(preds))):

        m = np.ones(len(truth)) # all ones as mask

        path  = get_path2(hit, m, thr)

        if x4 and len(path) > 1:

            # ban the second hit from the frist run on the second run

            m[path[1]]=0

            path2  = get_path2(hit, m, thr)

            if len(path) < len(path2):

                # If the path from the second run is longer than the frist

                # run it again with the second hit from second run blocked

                path = path2

                m[path[1]]=0

                path2  = get_path2(hit, m, thr)

                if len(path) < len(path2):

                    # if The path from the thrid run is longer than

                    # the second run, use it

                    path = path2

            elif len(path2) > 1:

                # if the path from the second is small or equal to

                # the frist run, try rerunning with

                # second hit from the frist path re-enabled, and the 

                # second hit from the second path banned.

                m[path[1]] = 1

                m[path2[1]] = 0

                path2  = get_path2(hit, m, thr)

                if len(path) < len(path2):

                    # If the path from the thrid run is longer than the frist,

                    # use it.

                    path = path2

        tracks_all.append(path)

    #np.save('my_tracks_all', tracks_all)

else:

    print('load tracks')

    tracks_all = np.load('../input/trackml/my_tracks_all.npy')
# This is like to get the weight of each track

def get_track_score(tracks_all, n=4):

    scores = np.zeros(len(tracks_all))

    for i, path in enumerate(tracks_all):

        count = len(path) # number of hits in the path

        if count > 1:

            tp = 0 # true positive

            fp = 0 # false positive

            for p in path:

                # Check to see how many hits of varible path are 

                # in the track that originated from

                # each hit (p) in the varible path

                tp = tp + np.sum(np.isin(tracks_all[p], path, assume_unique=True))

                # Optimization?

                # s = np.sum(np.isin(tracks_all[p], path, assume_unique=True))

                # fp = fp + np.invert(s)

                fp = fp + np.sum(np.isin(tracks_all[p], path, assume_unique=True, invert=True))

            # ??

            # 1) Give extra weight to the the fp

            # 2) Subtrack the count is to subtract tp off of

            # first hit of the tracks that originated from

            # each hit of vraible path.

            # 3) Divide count which gives the the average score of track searched

            # 4) Divide count-1 which gives the average socre of the each prediction

            # after the hit hit of the path

            scores[i] = (tp-fp*n-count)/count/(count-1)

        else: # if path has less than 2 points, the scores is negative inf

            scores[i] = -np.inf

    return scores



# A faster scoring function

# https://www.kaggle.com/cpmpml/a-faster-python-scoring-function

def score_event_fast(truth, submission):

    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')

    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()

    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    

    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()

    truth = truth.merge(df1, how='left', on='particle_id')

    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()

    truth = truth.merge(df1, how='left', on='track_id')

    truth.count_both *= 2

    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()

    particles = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].particle_id.unique()



    return score, truth[truth.particle_id.isin(particles)].weight.sum(), 1-truth[truth.track_id>0].weight.sum()



def evaluate_tracks(tracks, truth):

    # use truth.hit_id, so the hit_ids is 1-index to len(hits)

    submission = pd.DataFrame({'hit_id': truth.hit_id, 'track_id': tracks})

    score = score_event_fast(truth, submission)[0]

    track_id = tracks.max() # number of tracks predicted

    print('Score:%.4f\nHits/Tracks:%2.2f\nTracks identified:%4d\nHits not id:%5d\nScore missed:%.4f\nTotal weight of unidenified tracks:%.4f' %(

        score,

        # number of hits that's identified with a tracks / number of tracks predicted

        # average hits / track identified

        np.sum(tracks>0)/track_id,

        track_id, # number of tracks predicted

        np.sum(tracks==0), # number of hits that's not assoicated with a track

        # Truth weights of the tracks we didn't identify

        1-score-np.sum(truth.weight.values[tracks==0]),

        # Sum of the weights of unidenified tracks

        np.sum(truth.weight.values[tracks==0])))



def extend_path(path, mask, thr, last = False):

    a = 0

    # To recreate the a value

    for p in path[:-1]: # for each hit in the path except the last one

        c = get_predict2(p)

        if last == False:

            mask = (c > thr)*mask

        mask[p] = 0

        cand = np.where(c>thr)[0]

        mask[cand[np.isin(module_id[cand], module_id[path])]]=0

        a = (c + a)*mask



    while True:

        c = get_predict2(path[-1])

        if last == False: # Only add one more hit in the path

            mask = (c > thr)*mask

        mask[path[-1]] = 0

        cand = np.where(c>thr)[0]

        mask[cand[np.isin(module_id[cand], module_id[path])]]=0

        a = (c + a)*mask

            

        # No more hit above the thr

        if a.max() < thr * len(path):

            break



        path.append(a.argmax())

        if last: break

    

    return path
# calculate track's confidence (about 2 mins)

scores = get_track_score(tracks_all, 8)
# merge tracks by confidence and get score

# get all index that would sort from large to small

idx = np.argsort(scores)[::-1]

tracks = np.zeros(len(hits))

track_id = 0



for hit in idx: # for each initial hit in the tracks that has the highest score

    path = np.array(tracks_all[hit]) # convert from list to np.array

    # Remove all the hits in the path that's been part of other tracks' hits

    path = path[np.where(tracks[path] == 0)[0]]



    if len(path) > 3: # If the path has at least 3 hits

        # prioritize the track number by its weight

        # and make sure the id starts with 1

        track_id = track_id + 1

        # set the hit index to the all the hits in the path

        tracks[path] = track_id



evaluate_tracks(tracks, truth)
# multistage



# Get the idx of the scores sorted by big to small,

# which is the hit index

idx = np.argsort(scores)[::-1]

# Remake the tracks

tracks = np.zeros(len(hits))

track_id = 0



for hit in idx: # for each hit index sorted by scores

    # get the path where it's originated from hit

    path = np.array(tracks_all[hit])

    # Remove all the hits in the path that's been part of other tracks' hits

    path = path[np.where(tracks[path]==0)[0]]



    if len(path) > 6:

        track_id = track_id + 1  

        tracks[path] = track_id



evaluate_tracks(tracks, truth)



# for each of the track_id that's identified,

# try to extend the existing path with the un-identified hits

for track_id in range(1, int(tracks.max())+1):

    # Get the path/hit_ids with track_id

    path = np.where(tracks == track_id)[0]

    # extend the path with the hits with no track identified

    path = extend_path(path.tolist(), 1*(tracks==0), 0.6)

    # Update the racks

    tracks[path] = track_id



evaluate_tracks(tracks, truth)



# Try to extened the paths that didn't pass the threshold before

for hit in idx: # for each hit index sorted by scores

    # get the path where it's originated from hit

    path = np.array(tracks_all[hit])

    # Remove all the hits in the path that's been part of other tracks' hits

    path = path[np.where(tracks[path]==0)[0]]



    if len(path) > 3:

        path = extend_path(path.tolist(), 1*(tracks==0), 0.6)

        track_id = track_id + 1  

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



# for each of the track_id that's identified,

# try to extend the existing path with the un-identified hits

# But this time with a smaller threshold

for track_id in range(1, int(tracks.max())+1):

    path = np.where(tracks == track_id)[0]

    path = extend_path(path.tolist(), 1*(tracks==0), 0.5)

    tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



# Try to extend the small path

for hit in idx:

    path = np.array(tracks_all[hit])

    path = path[np.where(tracks[path]==0)[0]]



    if len(path) > 1:

        # Try extend the path at least has 2 hits

        path = extend_path(path.tolist(), 1*(tracks==0), 0.5)

    if len(path) > 2:

        # if the extended path has at least 3 hits

        track_id = track_id + 1

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



# for each of the track_id that's identified,

# try to extend the existing path with the un-identified hits

# But this time with a smaller threshold

for track_id in range(1, int(tracks.max())+1):

    # Get the path/hit_ids with track_id

    path = np.where(tracks == track_id)[0]

    # if the number of hits is even

    if len(path) % 2 == 0:

        # try extend them

        path = extend_path(path.tolist(), 1*(tracks==0), 0.5, True)

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)