!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os

import time

import numpy as np 

import pandas as pd

from glob import glob

from collections import deque

import matplotlib.pyplot as plt

from PIL import Image



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler



import torchvision

import torchvision.transforms as T 

import torchvision.models as models



# imports the torch_xla package

import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
root = '../input/104-flowers-garden-of-eden'



train_df = pd.DataFrame()

folder = os.listdir(root)

for f in folder:

    classes = os.listdir(os.path.join(root,f,'train'))

    for c in classes:

        images = os.listdir(os.path.join(root,f,'train',c))

        tmp_df = pd.DataFrame(images,columns=['image_name'])

        tmp_df['class'] = c

        tmp_df['folder'] = f

        tmp_df['type'] = 'train'

        train_df = train_df.append(tmp_df, ignore_index=True)

print('train:',train_df.shape)   



val_df = pd.DataFrame()

folder = os.listdir(root)

for f in folder:

    classes = os.listdir(os.path.join(root,f,'val'))

    for c in classes:

        images = os.listdir(os.path.join(root,f,'val',c))

        tmp_df = pd.DataFrame(images,columns=['image_name'])

        tmp_df['class'] = c

        tmp_df['folder'] = f

        tmp_df['type'] = 'val'

        val_df = val_df.append(tmp_df, ignore_index=True)

print('val:',val_df.shape)     
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']  
print('num class:', len(CLASSES))

train_df['label'] = train_df['class'].apply(lambda x: CLASSES.index(x))

val_df['label'] = val_df['class'].apply(lambda x: CLASSES.index(x))  
test_df = pd.DataFrame()

f = 'jpeg-224x224'



images = os.listdir(os.path.join(root,f,'test'))

tmp_df = pd.DataFrame(images,columns=['image_name'])

tmp_df['class'] = 'unknown'

tmp_df['folder'] = f

tmp_df['type'] = 'test'

test_df = test_df.append(tmp_df, ignore_index=True)

print('test:',test_df.shape)    
class flowerDataset(Dataset):

    def __init__(self, df, root = '../input/104-flowers-garden-of-eden'):

        self.df = df

        self.root = root

        self.transforms = T.Compose([T.Resize((224,224)), T.ToTensor()])

        

    def __getitem__(self, idx):



        img_path = os.path.join(self.root, 

                                self.df.iloc[idx]['folder'], 

                                self.df.iloc[idx]['type'],

                                self.df.iloc[idx]['class'],

                                self.df.iloc[idx]['image_name'])

        img = Image.open(img_path)

        img_tensor = self.transforms(img)

        target_tensor = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)

        return img_tensor, target_tensor

    

    def __len__(self):

        return len(self.df)

    

    

class testDataset(Dataset):

    def __init__(self, df, root = '../input/104-flowers-garden-of-eden'):

        self.df = df

        self.root = root

        self.transforms = T.Compose([T.ToTensor()])

        

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.df.iloc[idx]['folder'], 

                                self.df.iloc[idx]['type'],

                                self.df.iloc[idx]['image_name'])

        img = Image.open(img_path)

        img_tensor = self.transforms(img)

        return img_tensor,  self.df.iloc[idx]['image_name'][:-5]

    

    def __len__(self):

        return len(self.df)
train_dataset = flowerDataset(train_df)

print(train_dataset.__len__())



train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = True)

train_iter = iter(train_loader)



images, labels = next(train_iter)

print(images.size())

print(labels.size())



plot_size = 32



fig = plt.figure(figsize=(25, 10))

for idx in np.arange(plot_size):

    ax = fig.add_subplot(4, plot_size/4, idx+1, xticks=[], yticks=[])

    ax.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title(classes[labels[idx].item()])
def train_net():

    torch.manual_seed(FLAGS['seed'])

    

    device = xm.xla_device()   

    world_size = xm.xrt_world_size()

    

    xm.master_print(f'device: {device}')

    xm.master_print(f'world_size: {world_size}')

    

    ### train loader

    train_dataset = flowerDataset(train_df)

    train_sampler = DistributedSampler(train_dataset,

                                       num_replicas = world_size,

                                       rank = xm.get_ordinal(),

                                       shuffle = True)

    train_loader = DataLoader(train_dataset,

                              batch_size = FLAGS['batch_size'],

                              sampler = train_sampler,

                              num_workers = FLAGS['num_workers'],

                              drop_last = True)

    

    ### val loader

    val_dataset = flowerDataset(val_df)

    val_sampler = DistributedSampler(val_dataset,

                                     num_replicas = world_size,

                                     rank = xm.get_ordinal(),

                                     shuffle = True)

    

    val_loader = DataLoader(val_dataset,

                            batch_size = FLAGS['batch_size'],

                            sampler = val_sampler,

                            num_workers = FLAGS['num_workers'],

                            drop_last = True)

    

    #### model

    model = models.resnet18()

    model.load_state_dict(torch.load('/kaggle/input/resnet18/resnet18.pth'))

    model.fc = nn.Linear(512, 104)

    model.to(device)

    

    ### Scale learning rate to num cores

    optimizer = optim.SGD(model.parameters(), 

                          lr = FLAGS['learning_rate'] * world_size,

                          momentum = FLAGS['momentum'], 

                          weight_decay=5e-4)

    

    loss_fn = torch.nn.CrossEntropyLoss()

    

    def train_loop_fn(loader):

        tracker = xm.RateTracker()

        model.train()

        loss_window = deque(maxlen = FLAGS['log_steps'])

        for x, (data, target) in enumerate(loader):

            optimizer.zero_grad()

            output = model(data)

            loss = loss_fn(output, target)

            loss_window.append(loss.item())

            loss.backward()

            xm.optimizer_step(optimizer)

            tracker.add(FLAGS['batch_size'])

            if (x+1) % FLAGS['log_steps'] == 0:

                print('[xla:{}]({}) Loss={:.5f} '.format(xm.get_ordinal(), x+1, np.mean(loss_window)), flush=True)

                

    def val_loop_fn(loader):

        total_samples, correct = 0, 0

        model.eval()

        for data, target in loader:

            with torch.no_grad():

                output = model(data)

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

            total_samples += data.size()[0]



        accuracy = 100.0 * correct / total_samples

        print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy), flush=True)

        return accuracy





    for epoch in range(1,FLAGS['num_epochs'] + 1):

        para_loader = pl.ParallelLoader(train_loader, [device])

        train_loop_fn(para_loader.per_device_loader(device))

        xm.master_print("Finished training epoch {}".format(epoch))

        

        para_loader = pl.ParallelLoader(val_loader, [device])

        accuracy = val_loop_fn(para_loader.per_device_loader(device))

        

        best_accuracy = 0.0

        if accuracy > best_accuracy:

            xm.save(model.state_dict(), 'trained_resnet18_model.pth')

            best_accuracy = accuracy

        

def _mp_fn(rank, flags):

    global FLAGS

    FLAGS = flags

    torch.set_default_tensor_type('torch.FloatTensor')

    train_start = time.time()

    train_net()

    elapsed_train_time = time.time() - train_start

    print("Process", rank, "finished training. Train time was:", elapsed_train_time)
# Define Parameters

FLAGS = {}

FLAGS['seed'] = 1

FLAGS['num_workers'] = 4

FLAGS['num_cores'] = 8

FLAGS['num_epochs'] = 10

FLAGS['log_steps'] = 50

FLAGS['batch_size'] = 16

FLAGS['learning_rate'] = 0.0001

FLAGS['momentum'] = 0.9



xmp.spawn(_mp_fn, args = (FLAGS,), nprocs = FLAGS['num_cores'],start_method='fork')
model = models.resnet18()

model.fc = nn.Linear(512, 104)

model.load_state_dict(torch.load('trained_resnet18_model.pth'))



device = xm.xla_device()

model.to(device)

model.eval()



print(device)
batch_size = 128

test_dataset = testDataset(test_df)

test_loader = DataLoader(test_dataset, batch_size = batch_size)

n = test_dataset.__len__()
label = []

id = []

for x, (images, names) in enumerate(test_loader):

    images = images.to(device)

    with torch.no_grad():

        output = model(images)

    preds = list(output.max(1)[1].cpu().numpy())

    label.extend(preds)

    id.extend(names)

    print('\rProcess {} %'.format(round(100*x*batch_size/n)),end="")

    

print('\rProcess 100 %')    



predictions = pd.DataFrame(data={'id':id,'label':label})
predictions.to_csv('submission.csv', index = False)