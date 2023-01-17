import os

import platform

import random

from argparse import Namespace



from tqdm.notebook import tqdm



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# import torchsummary
dtype = torch.float32

device = torch.device('cuda:0')

seed = 1234





def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything(seed=seed)
data_path = './data/' if 'windows' in platform.platform().lower() else '../input/lish-moa/'
train_features = pd.read_csv(data_path + 'train_features.csv')

test_features = pd.read_csv(data_path + 'test_features.csv')

train_targets_scored = pd.read_csv(data_path + 'train_targets_scored.csv')
train_features.info()
train_features.shape, train_targets_scored.shape, test_features.shape
train_features
class MoADataset(Dataset):

    def __init__(self, dtype, features, targets=None, feature_columns=None):

        self.dtype = dtype



        if isinstance(features, (pd.DataFrame, pd.Series)):

            if feature_columns is not None:

                features = features[feature_columns]

            features = features.values

        self.features = torch.tensor(features, dtype=self.dtype)

        self.feature_columns = feature_columns



        if targets is None:

            targets = -np.ones(self.features.shape[0])  # фиктивный таргет, если идет инференс модели

        elif isinstance(targets, (pd.DataFrame, pd.Series)):

            targets = targets.values

        self.targets = torch.tensor(targets, dtype=self.dtype)



    def __getitem__(self, i):

        return self.features[i], self.targets[i]

#         return {

#             'x': self.features[i],

#             'y': self.targets[i]

#         }



    def __len__(self):

        return self.features.shape[0]
def prepare_data(features, targets=None):

    # TODO: Здесь могут быть проблемы с тем, что тест и трейн обработаются по-разному!

    features_enc = pd.get_dummies(features, columns=['cp_type', 'cp_dose']).drop(columns=['sig_id'])

#     feature_columns = features_enc.drop(columns=['sig_id']).columns.values



    if targets is None:

        return features_enc  # , feature_columns



    targets_enc = targets.drop(columns=['sig_id'])  # .columns.values

    return features_enc, targets_enc  # , feature_columns, target_columns
train_features_enc, train_targets_scored_enc = prepare_data(train_features, train_targets_scored)

test_features_enc = prepare_data(test_features)
(

    train_features_tr, train_features_val,

    train_targets_scored_tr, train_targets_scored_val

) = train_test_split(train_features_enc, train_targets_scored_enc, test_size=0.2,

                     random_state=seed, shuffle=True)



train_dataset = MoADataset(dtype, train_features_tr, train_targets_scored_tr)

val_dataset = MoADataset(dtype, train_features_val, train_targets_scored_val)

test_dataset = MoADataset(dtype, test_features_enc)
batch_size = 2 ** 8  # 1024

num_workers = 0
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
class MoAModel(nn.Module):

    def __init__(self, device, dtype, num_in_features, num_hidden_features, num_out_features, dropout_rate=0.5):

        super().__init__()



        self.device = device

        self.dtype = dtype



        self.net = nn.Sequential(

            nn.BatchNorm1d(num_in_features),  # вместо нормирования входных данных



            nn.Linear(num_in_features, num_hidden_features),

            nn.BatchNorm1d(num_hidden_features),

            nn.ReLU(),

            nn.Dropout(dropout_rate),



            nn.Linear(num_hidden_features, num_hidden_features),

            nn.BatchNorm1d(num_hidden_features),

            nn.ReLU(),

            nn.Dropout(dropout_rate),



            nn.Linear(num_hidden_features, num_out_features)

        ).to(self.device)



    def forward(self, x):

        return self.net(x.to(self.device, self.dtype))
def print_results(cur_results, mode, cur_iter, print_every):

    if print_every == 'summary':

        print(

            f'Summary: epoch {cur_results.epoch + 1:3}, '

            f'mode {mode:6}, ',

            end=''

        )



        if mode != 'test':

            losses = cur_results.train_loss if mode == 'train' else cur_results.val_loss

            print(

                f'loss {np.mean(losses):12.5f}, '

            )

        else:

            print()

    elif cur_iter % print_every == 0:

        print(

            f'Epoch {cur_results.epoch + 1:3}, '

            f'mode {mode:6}, '

            f'iter {cur_iter:5}, ',

            end=''

        )



        if mode != 'test':

            losses = cur_results.train_loss if mode == 'train' else cur_results.val_loss

            print(

                f'loss {losses[-1]:12.5f}, '

            )

        else:

            print()
num_in_features = train_features_enc.shape[1]

num_out_features = train_targets_scored_enc.shape[1]



num_hidden_features = 1024





model = MoAModel(device, dtype, num_in_features, num_hidden_features, num_out_features)



# torchsummary.summary(model);







print_every = 5



max_epoch = 100

lr = 1e-3







criterion = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(model.parameters(), lr=lr)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)  # len(train_dataset) // batch_size)
# results = []



# for epoch in tqdm(range(max_epoch), desc='Epoch'):

#     # History

#     cur_results = Namespace(

#         epoch=epoch,

#         train_loss=[],

#         lr=[],

#         val_loss=[],

#         test_preds=[],

#     )



#     # Training

#     model.train()

#     for cur_iter, (x, y) in enumerate(train_dataloader):

#         optimizer.zero_grad()



#         scores = model(x)

#         loss = criterion(scores, y.to(device))



#         loss.backward()

#         optimizer.step()



#         cur_results.train_loss.append(loss.item())

#         cur_results.lr.append(lr_scheduler.get_last_lr()[0])



# #         print_results(cur_results, 'train', cur_iter, print_every)

#     print_results(cur_results, 'train', -1, print_every='summary')



#     lr_scheduler.step()



#     # Validation

#     model.eval()

#     for cur_iter, (x, y) in enumerate(val_dataloader):

#         with torch.no_grad():

#             scores = model(x)

#         loss = criterion(scores, y.to(device))

#         cur_results.val_loss.append(loss.item())



# #         print_results(cur_results, 'val', cur_iter, print_every)

#     print_results(cur_results, 'val', -1, print_every='summary')



#     # Test predictions

#     model.eval()

#     for cur_iter, (x, y) in enumerate(test_dataloader):

#         with torch.no_grad():

#             scores = model(x)

#         preds = torch.sigmoid(scores)

#         cur_results.test_preds.append(preds.cpu())



# #         print_results(cur_results, 'test', cur_iter, print_every)

#     print_results(cur_results, 'test', -1, print_every='summary')



#     results.append(cur_results)
(

    train_features_tr, train_features_val,

    train_targets_scored_tr, train_targets_scored_val

) = train_test_split(train_features_enc, train_targets_scored_enc, test_size=0.2,

                     random_state=seed, shuffle=True)



train_dataset = MoADataset(dtype, train_features_enc, train_targets_scored_enc)

# val_dataset = MoADataset(dtype, train_features_val, train_targets_scored_val)

test_dataset = MoADataset(dtype, test_features_enc)
batch_size = 2 ** 8  # 1024

num_workers = 0
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
results = []



for epoch in tqdm(range(max_epoch), desc='Epoch'):

    # History

    cur_results = Namespace(

        epoch=epoch,

        train_loss=[],

        lr=[],

        val_loss=[],

        test_preds=[],

    )



    # Training

    model.train()

    for cur_iter, (x, y) in enumerate(train_dataloader):

        optimizer.zero_grad()



        scores = model(x)

        loss = criterion(scores, y.to(device))



        loss.backward()

        optimizer.step()



        cur_results.train_loss.append(loss.item())

        cur_results.lr.append(lr_scheduler.get_last_lr()[0])



#         print_results(cur_results, 'train', cur_iter, print_every)

    print_results(cur_results, 'train', -1, print_every='summary')



    lr_scheduler.step()



#     # Validation

#     model.eval()

#     for cur_iter, (x, y) in enumerate(val_dataloader):

#         with torch.no_grad():

#             scores = model(x)

#         loss = criterion(scores, y.to(device))

#         cur_results.val_loss.append(loss.item())



# #         print_results(cur_results, 'val', cur_iter, print_every)

#     print_results(cur_results, 'val', -1, print_every='summary')



    # Test predictions

    model.eval()

    for cur_iter, (x, y) in enumerate(test_dataloader):

        with torch.no_grad():

            scores = model(x)

        preds = torch.sigmoid(scores)

        cur_results.test_preds.append(preds.cpu())



#         print_results(cur_results, 'test', cur_iter, print_every)

    print_results(cur_results, 'test', -1, print_every='summary')



    results.append(cur_results)
results[-1].val_loss
train_dataset.__len__()
# test_preds = [torch.cat(result.test_preds) for result in results[9::5]]

test_preds = [torch.cat(result.test_preds) for result in results[99:]]

ensemble_test_preds = torch.stack(test_preds, dim=0)

final_test_preds = ensemble_test_preds.mean(0)
final_test_preds
# best_idx = final_test_preds.argsort(dim=1)



# test_predictions = []



# keep_num_best = 3

# for final_test_pred, bad_idx in zip(final_test_preds, best_idx[:, :-keep_num_best]):

#     final_test_pred[bad_idx] = 0

#     test_predictions.append(final_test_pred)



# test_predictions = torch.stack(test_predictions)
answer = pd.read_csv(data_path + 'sample_submission.csv')
answer
answer.iloc[:, 1:] = final_test_preds  # test_predictions
answer
answer.to_csv('submission.csv', index=False)