# !pip install torchsummary

# !pip install --upgrade efficientnet-pytorch
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import gc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



import torch

import torch.utils.data as data_utils

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torch.optim import lr_scheduler

from torch import nn



# from torchsummary import summary

from torchvision import transforms,models

import torch.nn.functional as F

from tqdm.auto import tqdm

from torch import Tensor



# from efficientnet_pytorch import EfficientNet

from collections import OrderedDict



from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
full_train_df = pd.read_feather('/kaggle/input/full-bengali-graphemes-normalized/full_train_df.feather')

target_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

X_train = full_train_df.drop(target_cols, axis=1)

Y_train = full_train_df[target_cols]
del full_train_df

gc.collect()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.002, random_state=666)

gc.collect()
IMG_SIZE = 64

CHANNELS = 1

W, H = IMG_SIZE, IMG_SIZE

BATCH_SIZE=512
# Convert to PyTorch tensors

X_train = torch.from_numpy(X_train.values.reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE))

X_val = torch.from_numpy(X_val.values.reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE))

Y_train = torch.from_numpy(Y_train.values)

Y_val = torch.from_numpy(Y_val.values)
print(f'Size of X_train: {X_train.shape}')

print(f'Size of Y_train: {Y_train.shape}')

print(f'Size of X_val: {X_val.shape}')

print(f'Size of Y_val: {Y_val.shape}')
# Visualize few samples of training dataset

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

count=0

for row in ax:

    for col in row:

        col.imshow(X_train[count].reshape(IMG_SIZE, IMG_SIZE).cpu().detach().numpy().astype(np.float64))

        col.set_title(str(Y_train[count].cpu().detach().numpy()))

        count += 1

plt.show()
class GraphemesDataset(Dataset):

    """

    Custom Graphemes dataset

    """

    def __init__(self, X, Y):

        self.X = X

        self.Y = Y

    

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, index):

        return self.X[index], self.Y[index]
train_dataset = GraphemesDataset(X_train, Y_train)

train_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,

                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):

        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *

                                           growth_rate, kernel_size=1, stride=1,

                                           bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),

        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,

                                           kernel_size=3, stride=1, padding=1,

                                           bias=False)),

        self.drop_rate = float(drop_rate)

        self.memory_efficient = memory_efficient



    def bn_function(self, inputs):

        # type: (List[Tensor]) -> Tensor

        concated_features = torch.cat(inputs, 1)

        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484

        return bottleneck_output



    # todo: rewrite when torchscript supports any

    def any_requires_grad(self, input):

        # type: (List[Tensor]) -> bool

        for tensor in input:

            if tensor.requires_grad:

                return True

        return False



    @torch.jit.unused  # noqa: T484

    def call_checkpoint_bottleneck(self, input):

        # type: (List[Tensor]) -> Tensor

        def closure(*inputs):

            return self.bn_function(*inputs)



        return cp.checkpoint(closure, input)



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (List[Tensor]) -> (Tensor)

        pass



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (Tensor) -> (Tensor)

        pass



    # torchscript does not yet support *args, so we overload method

    # allowing it to take either a List[Tensor] or single Tensor

    def forward(self, input):  # noqa: F811

        if isinstance(input, Tensor):

            prev_features = [input]

        else:

            prev_features = input



        if self.memory_efficient and self.any_requires_grad(prev_features):

            if torch.jit.is_scripting():

                raise Exception("Memory Efficient not supported in JIT")



            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)

        else:

            bottleneck_output = self.bn_function(prev_features)



        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:

            new_features = F.dropout(new_features, p=self.drop_rate,

                                     training=self.training)

        return new_features
class _DenseBlock(nn.ModuleDict):

    _version = 2



    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):

        super(_DenseBlock, self).__init__()

        for i in range(num_layers):

            layer = _DenseLayer(

                num_input_features + i * growth_rate,

                growth_rate=growth_rate,

                bn_size=bn_size,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient,

            )

            self.add_module('denselayer%d' % (i + 1), layer)



    def forward(self, init_features):

        features = [init_features]

        for name, layer in self.items():

            new_features = layer(features)

            features.append(new_features)

        return torch.cat(features, 1)
class DenseNet(nn.Module):

    r"""Densenet-BC model class, based on

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        growth_rate (int) - how many filters to add each layer (`k` in paper)

        block_config (list of 4 ints) - how many layers in each pooling block

        num_init_features (int) - the number of filters to learn in the first convolution layer

        bn_size (int) - multiplicative factor for number of bottle neck layers

          (i.e. bn_size * k features in the bottleneck layer)

        drop_rate (float) - dropout rate after each dense layer

        num_classes (int) - number of classification classes

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """



    __constants__ = ['features']



    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),

                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):



        super(DenseNet, self).__init__()



        # First convolution

        self.features = nn.Sequential(OrderedDict([

            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),

            ('norm0', nn.BatchNorm2d(num_init_features)),

            ('relu0', nn.ReLU(inplace=True)),

            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ]))



        # Each denseblock

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(

                num_layers=num_layers,

                num_input_features=num_features,

                bn_size=bn_size,

                growth_rate=growth_rate,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient

            )

            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:

                trans = _Transition(num_input_features=num_features,

                                    num_output_features=num_features // 2)

                self.features.add_module('transition%d' % (i + 1), trans)

                num_features = num_features // 2



        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))



        # Linear layer

        self.classifier = nn.Linear(num_features, num_classes)



        # Official init from torch repo.

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out
def _densenet(growth_rate, block_config, num_init_features, **kwargs):

    return DenseNet(growth_rate, block_config, num_init_features, **kwargs)
def densenet121(**kwargs):

    r"""Densenet-121 model from

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    """

    return _densenet(32, (6, 12, 24, 16), 64, **kwargs)
class DenseNet121Wrapper(nn.Module):

    def __init__(self):

        super(DenseNet121Wrapper, self).__init__()

        

        # Load imagenet pre-trained model 

        self.effNet = densenet121()

        

        # Appdend output layers based on our date

        self.fc_root = nn.Linear(in_features=1000, out_features=168)

        self.fc_vowel = nn.Linear(in_features=1000, out_features=11)

        self.fc_consonant = nn.Linear(in_features=1000, out_features=7)

        

    def forward(self, X):

        output = self.effNet(X)

        output_root = self.fc_root(output)

        output_vowel = self.fc_vowel(output)

        output_consonant = self.fc_consonant(output)

        

        return output_root, output_vowel, output_consonant
model = DenseNet121Wrapper().to(device)



# Print summary of our model

# summary(model, input_size=(CHANNELS, IMG_SIZE, IMG_SIZE))
torch.cuda.empty_cache()
LEARNING_RATE = 0.02

EPOCHS = 100

CUTMIX_ALPHA = 1
model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)

criterion = nn.CrossEntropyLoss()
def get_accuracy(root_preds, target_root, vowel_pred, target_vowel, consonant_pred, target_consonant):

    assert len(root_preds) == len(target_root) and len(vowel_pred) == len(target_vowel) and len(consonant_pred) == len(target_consonant)

    

    total = len(target_root) + len(target_vowel) + len(target_consonant)

    _, predicted_root = torch.max(root_preds.data, axis=1)

    _, predicted_vowel = torch.max(vowel_pred.data, axis=1)

    _, predicted_consonant = torch.max(consonant_pred.data, axis=1)

    

    del root_preds

    del vowel_pred

    del consonant_pred

    torch.cuda.empty_cache()



    correct = (predicted_root == target_root).sum().item() + (predicted_vowel == target_vowel).sum().item() + (predicted_consonant == target_consonant).sum().item()

    

    del target_root

    del target_vowel

    del target_consonant

    torch.cuda.empty_cache()

    return correct / total
def shuffle_minibatch(x, y):

    assert x.size(0)== y.size(0) # Size should be equal

    indices = torch.randperm(x.size(0))

    return x[indices], y[indices]
del X_train

del Y_train

gc.collect()
def clear_cache():

    gc.collect()

    torch.cuda.empty_cache()
X_val = X_val.to(device)

Y_val = Y_val.to(device)



# Split validation's Y into 3 separate targets

target_val_root, target_val_vowel, target_val_consonant = Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]

del Y_val

clear_cache()
def rand_bbox(size, lam):

    W = size[2]

    H = size[3]

    cut_rat = np.sqrt(1. - lam)

    cut_w = np.int(W * cut_rat)

    cut_h = np.int(H * cut_rat)



    # uniform

    cx = np.random.randint(W)

    cy = np.random.randint(H)



    bbx1 = np.clip(cx - cut_w // 2, 0, W)

    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    bby2 = np.clip(cy + cut_h // 2, 0, H)



    return bbx1, bby1, bbx2, bby2
def cutmix(data, targets1, targets2, targets3, alpha):

    indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]

    shuffled_targets1 = targets1[indices]

    shuffled_targets2 = targets2[indices]

    shuffled_targets3 = targets3[indices]



    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))



    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets
def mixup(data, targets1, targets2, targets3, alpha):

    indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]

    shuffled_targets1 = targets1[indices]

    shuffled_targets2 = targets2[indices]

    shuffled_targets3 = targets3[indices]



    lam = np.random.beta(alpha, alpha)

    data = data * lam + shuffled_data * (1 - lam)

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]



    return data, targets
def cutmix_criterion(preds1,preds2,preds3, targets):

    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]

    criterion = nn.CrossEntropyLoss(reduction='mean')

    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)
def mixup_criterion(preds1,preds2,preds3, targets):

    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]

    criterion = nn.CrossEntropyLoss(reduction='mean')

    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)
total_steps = len(train_loader)

val_acc_list = []

is_mixup, is_cutmix = True, True

for epoch in range(EPOCHS):

    for i, (x_train, y_train) in tqdm(enumerate(train_loader), total=total_steps):

        x_train = x_train.to(device)

        target_root = y_train[:, 0].to(device, dtype=torch.long)

        target_vowel = y_train[:, 1].to(device, dtype=torch.long)

        target_consonant = y_train[:, 2].to(device, dtype=torch.long)

        

        if np.random.rand()<0.6:

            images, targets = mixup(x_train, target_root, target_vowel, target_consonant, CUTMIX_ALPHA)

            if is_mixup:

                # Visualize few samples of training dataset

                fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

                count=0

                for row in ax:

                    for col in row:

                        col.imshow(images[count].reshape(IMG_SIZE, IMG_SIZE).cpu().detach().numpy().astype(np.float64))

                        count += 1

                plt.show()

                is_mixup = False

            # Forward pass

            root_preds, vowel_pred, consonant_pred = model(images)

            loss = mixup_criterion(root_preds, vowel_pred, consonant_pred, targets) 

        else:

            images, targets = cutmix(x_train, target_root, target_vowel, target_consonant, CUTMIX_ALPHA)

            if is_cutmix:

                # Visualize few samples of training dataset

                fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

                count=0

                for row in ax:

                    for col in row:

                        col.imshow(images[count].reshape(IMG_SIZE, IMG_SIZE).cpu().detach().numpy().astype(np.float64))

                        count += 1

                plt.show()

                is_cutmix = False

            # Forward pass

            root_preds, vowel_pred, consonant_pred = model(images)

            loss = cutmix_criterion(root_preds, vowel_pred, consonant_pred, targets)

        

        del x_train

        clear_cache()

        

        # Backpropagate

        optimizer.zero_grad()  # Reason: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        loss.backward()

        optimizer.step()

    

    lr_scheduler.step(loss.item())

    del root_preds

    del target_root

    del vowel_pred

    del target_vowel

    del consonant_pred

    del target_consonant

    clear_cache()



    # Calculate validation accuracy after each epoch

    # Predict on validation set

    root_val_preds, vowel_val_pred, consonant_val_pred = model(X_val)



    val_acc = get_accuracy(root_val_preds, target_val_root, vowel_val_pred, target_val_vowel, consonant_val_pred, target_val_consonant)

    val_acc_list.append(val_acc)



    del root_val_preds

    del vowel_val_pred

    del consonant_val_pred

    clear_cache()



    print('Epoch [{}/{}], Loss: {:.4f}, Validation accuracy: {:.2f}%'

          .format(epoch + 1, EPOCHS, loss.item(), val_acc * 100))
plt.style.use('ggplot')

plt.figure()

plt.plot(np.arange(0, EPOCHS), val_acc_list, label='val_accuracy')



plt.title('Accuracy')

plt.xlabel('# of epochs')

plt.ylabel('Accuracy')

plt.legend(loc='upper right')

plt.show()
torch.save(model.state_dict(), '100epochs_densenet121_lr.pth')