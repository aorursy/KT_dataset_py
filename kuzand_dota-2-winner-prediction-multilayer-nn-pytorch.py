import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import copy

import datetime

import pytz

import time

import random



# PyTorch stuff

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torch.utils import data



# Sklearn stuff

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import roc_auc_score
SEED = 17



# Input data files are available in the "../input/" directory.

PATH_TO_DATA = '../input'

print(os.listdir(PATH_TO_DATA))
# Train dataset

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 

                                   index_col='match_id_hash')



# Test dataset

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df_train_features.head()
# Check if there is missing data

print(df_train_features.isnull().values.any())

print(df_test_features.isnull().values.any())
df_full_features = pd.concat([df_train_features, df_test_features])



# Index to split the training and test data sets

idx_split = df_train_features.shape[0]



# That is, 

# df_train_features == df_full_features[:idx_split]

# df_test_features == df_full_features[idx_split:]
df_full_features.drop(['game_time', 'game_mode', 'lobby_type', 'objectives_len', 'chat_len'],

                      inplace=True, axis=1)
def write_to_submission_file(predicted_labels):

    df_submission = pd.DataFrame({'radiant_win_prob': predicted_labels}, 

                                     index=df_test_features.index)



    submission_filename = 'submission_{}.csv'.format(

        datetime.datetime.now(tz=pytz.timezone('Europe/Athens')).strftime('%Y-%m-%d_%H-%M-%S'))

    

    df_submission.to_csv(submission_filename)

    

    print('Submission saved to {}'.format(submission_filename))
np.sort(np.unique(df_full_features['r1_hero_id'].values.flatten()))
np.all(df_train_features[[f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]].nunique(axis=1) == 10)
for t in ['r', 'd']:

    for i in range(1, 6):

        df_full_features = pd.get_dummies(df_full_features, columns = [f'{t}{i}_hero_id'])
df_full_features_scaled = df_full_features.copy()

df_full_features_scaled[df_full_features.columns.tolist()] = MinMaxScaler().fit_transform(df_full_features_scaled[df_full_features.columns.tolist()])  # alternatively use StandardScaler
df_full_features_scaled.head()
X_train = df_full_features_scaled[:idx_split]

X_test = df_full_features_scaled[idx_split:]



y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})
X_train.head()
mlp = nn.Sequential(nn.Linear(6, 4),

                    nn.ReLU(),

                    nn.Linear(4, 4),

                    nn.ReLU(),

                    nn.Linear(4, 1),

                    nn.Sigmoid()

                   )
class MLP(nn.Module):

    ''' Multi-layer perceptron with ReLu and Softmax.

    

    Parameters:

    -----------

        n_input (int): number of nodes in the input layer 

        n_hidden (int list): list of number of nodes n_hidden[i] in the i-th hidden layer 

        n_output (int):  number of nodes in the output layer 

        drop_p (float): drop-out probability [0, 1]

        random_state (int): seed for random number generator (use for reproducibility of result)

    '''

    def __init__(self, n_input, n_hidden, drop_p, random_state=SEED):

        super().__init__()   

        self.random_state = random_state

        set_random_seed(SEED)

        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(n_hidden[:-1], n_hidden[1:])])

        self.output_layer = nn.Linear(n_hidden[-1], 1)       

        self.dropout = nn.Dropout(p=drop_p)  # method to prevent overfitting

                

    def forward(self, X):

        ''' Forward propagation -- computes output from input X.

        '''

        for h in self.hidden_layers:

            X = F.relu(h(X))

            X = self.dropout(X)

        X = self.output_layer(X)

        return torch.sigmoid(X)

    

    def predict_proba(self, X_test):

        return self.forward(X_test).detach().squeeze(1).cpu().numpy()

    

    



def set_random_seed(rand_seed=SEED):

    ''' Helper function for setting random seed. Use for reproducibility of results'''

    if type(rand_seed) == int:

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True

        random.seed(rand_seed)

        np.random.seed(rand_seed)

        torch.manual_seed(rand_seed)

        torch.cuda.manual_seed(rand_seed)

        
def train(model, epochs, criterion, optimizer, scheduler, dataloaders, verbose=False):

    ''' 

    Train the given model...

    

    Parameters:

    -----------

        model: model (MLP) to train

        epochs (int): number of epochs

        criterion: loss function e.g. BCELoss

        optimizer: optimizer e.g SGD or Adam 

        scheduler: learning rate scheduler e.g. StepLR

        dataloaders: train and validation dataloaders

        verbose (boolean): print training details (elapsed time and losses)



    '''

    t0_tot = time.time()

    

    set_random_seed(model.random_state)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on {device}...')

    model.to(device)

    

    # Best model weights (deepcopy them because model.state_dict() changes during the training)

    best_model_wts = copy.deepcopy(model.state_dict()) 

    best_loss = np.inf

    losses = {'train': [], 'valid': []}

    roc_auc_values = []

    for epoch in range(epochs): 

        t0 = time.time()

        print(f'============== Epoch {epoch + 1}/{epochs} ==============')

        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                if verbose: print(f'lr: {scheduler.get_lr()}')

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0 

            for ii, (X_batch, y_batch) in enumerate(dataloaders[phase], start=1):                               

                # Move input and label tensors to the GPU

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)



                # Reset the gradients because they are accumulated

                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(X_batch).squeeze(1)  # forward prop

                    loss = criterion(outputs, y_batch)  # compute loss

                    if phase == 'train':

                        loss.backward()  # backward prop

                        optimizer.step()  # update the parameters

                        

                running_loss += loss.item() * X_batch.shape[0]

                

            ep_loss = running_loss/len(dataloaders[phase].dataset)  # average loss over an epoch

            losses[phase].append(ep_loss)

            if verbose: print(f' ({phase}) Loss: {ep_loss:.5f}')

                        

            # Best model by lowest validation loss

            if phase == 'valid' and ep_loss < best_loss:

                best_loss = ep_loss

                best_model_wts = copy.deepcopy(model.state_dict())      

                

        roc_auc = roc_auc_score(dataloaders['valid'].dataset.tensors[1].numpy(),

                                model.predict_proba(dataloaders['valid'].dataset.tensors[0].to(device)))

        roc_auc_values.append(roc_auc)

        if verbose: print('(valid) ROC AUC:', roc_auc)

        if verbose: print(f'\nElapsed time: {round(time.time() - t0, 3)} sec\n')

        

    print(f'\nTraining completed in {round(time.time() - t0_tot, 3)} sec')

    

    # Load the best model weights to the trained model

    model.load_state_dict(best_model_wts)

    model.losses = losses   

    model.roc_auc_values = roc_auc_values

    model.to('cpu')

    model.eval()

    return model





def plot_losses(train_losses, val_losses):

    y = [train_losses, val_losses]

    c = ['C7', 'C9']

    labels = ['Train loss', 'Validation loss']

    # Plot train_losses and val_losses wrt epoch

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = list(range(1, len(train_losses)+1))

    for i in range(2):

        ax.plot(x, y[i], lw=3, label=labels[i], color=c[i])

        ax.set_xlabel('Epoch', fontsize=16)

        ax.set_ylabel('Loss', fontsize=16)

        ax.set_xticks(range(0, x[-1]+1, 2))  

        ax.legend(loc='best')

    plt.tight_layout()

    plt.show()

        

        

def plot_roc_auc(roc_auc_values):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = list(range(1, len(roc_auc_values)+1))

    ax.plot(x, roc_auc_values, lw=3, color='C6')

    ax.set_xlabel('Epoch', fontsize=16)

    ax.set_ylabel('ROC AUC', fontsize=16)

    ax.set_xticks(range(0, x[-1]+1, 2))  

    plt.tight_layout()

    plt.show()
# Perform a train/validation split

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)



# Convert to pytorch tensors

X_train_tensor = torch.from_numpy(X_train_part.values).float()

X_valid_tensor = torch.from_numpy(X_valid.values).float()

y_train_tensor = torch.from_numpy(y_train_part.values).float()

y_valid_tensor = torch.from_numpy(y_valid.values).float()

X_test_tensor = torch.from_numpy(X_test.values).float()



# Create the train and validation dataloaders

train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)

valid_dataset = data.TensorDataset(X_valid_tensor, y_valid_tensor)



dataloaders = {'train': data.DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=2), 

               'valid': data.DataLoader(valid_dataset, batch_size=1000, shuffle=False, num_workers=2)}
mlp = MLP(n_input=X_train.shape[1], n_hidden=[200, 100], drop_p=0.4)



criterion = nn.BCELoss()  # Binary cross entropy

optimizer = optim.Adam(mlp.parameters(), lr=0.01, weight_decay=0.005)  # alternatevily torch.optim.SGD

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



epochs = 15

train(mlp, epochs, criterion, optimizer, scheduler, dataloaders, verbose=True)
plot_losses(mlp.losses['train'], mlp.losses['valid'])
plot_roc_auc(mlp.roc_auc_values)
print(f'Best ROC AUC: {roc_auc_score(y_valid.values, mlp.predict_proba(X_valid_tensor))}')
# Save

torch.save(mlp.state_dict(), 'mlp.pth')



# Load

mlp =  MLP(n_input=X_train.shape[1], n_hidden=[200, 100], drop_p=0.4)

mlp.load_state_dict(torch.load('mlp.pth'))

mlp.eval()
mlp_pred = mlp.predict_proba(X_test_tensor)



write_to_submission_file(mlp_pred)