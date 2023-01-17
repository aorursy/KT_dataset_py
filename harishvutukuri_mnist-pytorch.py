import time



import torch

from torch import nn, optim 

import torch.utils.data as data

from torch.autograd import Variable



import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')
# random seed everything

def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False



# Helper function for Execution time of the script

def execution_time(start):

    _ = time.time()

    hours, _ = divmod(_-start, 3600)

    minutes, seconds = divmod(_, 60)

    print("Execution Time:  {:0>2} hours: {:0>2} minutes: {:05.2f} seconds".format(int(hours),int(minutes),seconds))

    



start = time.time()

random_seed(42, False)
# Model Class



# output = (input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1))/stride + 1



class MnistModel(nn.Module):

    def __init__(self, classes):

        super(MnistModel, self).__init__()

        

        self.classes = classes

        

        # initialize the layers in the first (CONV => RELU) * 2 => POOL + DROP

        self.conv1A = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)  # (N,1,28,28) -> (N,16,24,24)

        self.act1A = nn.ReLU()

        self.conv1B = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # (N,16,24,24) -> (N,32,20,20)

        self.act1B = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2) # (N,32,20,20) -> (N,32,10,10)

        self.do1 = nn.Dropout(0.25)

        

        # initialize the layers in the second (CONV => RELU) * 2 => POOL + DROP

        self.conv2A = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) # (N,32,10,10) -> (N,64,8,8)

        self.act2A = nn.ReLU()

        self.conv2B = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # (N,64,8,8) -> (N,128,6,6)

        self.act2B = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2) # (N,128,6,6) -> (N,128,3,3)

        self.do2 = nn.Dropout(0.25)

        

        # initialize the layers in our fully-connected layer set

        self.dense3 = nn.Linear(128*3*3, 32) # (N,128,3,3) -> (N,32)

        self.act3 = nn.ReLU()

        self.do3 = nn.Dropout(0.25)

        

        # initialize the layers in the softmax classifier layer set

        self.dense4 = nn.Linear(32, self.classes) # (N, classes)

    

    def forward(self, x):

        

        # build the first (CONV => RELU) * 2 => POOL layer set

        x = self.conv1A(x)

        x = self.act1A(x)

        x = self.conv1B(x)

        x = self.act1B(x)

        x = self.pool1(x)

        x = self.do1(x)

        

        # build the second (CONV => RELU) * 2 => POOL layer set

        x = self.conv2A(x)

        x = self.act2A(x)

        x = self.conv2B(x)

        x = self.act2B(x)

        x = self.pool2(x)

        x = self.do2(x)

        

        # build our FC layer set

        x = x.view(x.size(0), -1)

        x = self.dense3(x)

        x = self.act3(x)

        x = self.do3(x)

        

        # build the softmax classifier

        x = nn.functional.log_softmax(self.dense4(x), dim=1)

        

        return x
# Dataset 

class MnistDataset(data.Dataset):

    def __init__(self, df, target, test=False):

        self.df = df

        self.test = test

        

        # if test=True skip this step

        if not self.test:        

            self.df_targets = target

        

    def __len__(self):

        # return length of the dataset

        return len(self.df)

    

    def __getitem__(self, idx):

        # if indexes are in tensor, convert to list

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        # if test=False return bunch of images, targets

        if not self.test:

            return torch.Tensor(self.df[idx].astype(float)), self.df_targets[idx]

        # if test=True return only images

        else:

            return torch.Tensor(self.df[idx].astype(float))
# Loss Function

def loss_fn(outputs, targets):

    return nn.NLLLoss()(outputs, targets)
# Train Model Loop

def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):

    

    # set model to train

    model.train()

    # iterate over data loader

    for bi, d in enumerate(data_loader):

        ids = d[0]

        targets = d[1]

        

        # sending to device (cpu/gpu)

        ids = ids.to(device, dtype=torch.float)

        targets = targets.to(device, dtype=torch.long)

        

        # Clear gradients w.r.t. parameters

        optimizer.zero_grad()

        # Forward pass to get output/logits

        outputs = model(x=ids)

        # Calculate Loss: softmax --> negative log likelihood loss

        loss = loss_fn(outputs, targets)

        # Getting gradients w.r.t. parameters

        loss.backward()

        # Updating parameters

        optimizer.step()

        if scheduler is not None:

            # Updating scheduler

            if type(scheduler).__name__ == 'ReduceLROnPlateau':

                scheduler.step(loss)

            else:

                scheduler.step()

        #if bi % 100 == 0:

            #print(f'Iter [{bi}], Loss: {loss}')

    print(f'Loss on Train Data: {loss}')
# Validation Model Loop

def eval_loop_fn(data_loader, model, device):

    

    # full list of targets, outputs

    fin_targets = []

    fin_outputs = []

    # set model to eveluate

    model.eval()  # as model is set to eval, there will be no optimizer and scheduler update

    

    # iterate over data loader

    for _, d in enumerate(data_loader):

        ids = d[0]

        targets = d[1]



        ids = ids.to(device, dtype=torch.float)

        targets = targets.to(device, dtype=torch.long)



        outputs = model(x=ids)

        loss = loss_fn(outputs, targets)

        loss.backward()

        

        # Get predictions from the maximum value

        _, outputs = torch.max(outputs.data, 1)

        

        # appending the values to final lists 

        fin_targets.append(targets.cpu().detach().numpy())

        fin_outputs.append(outputs.cpu().detach().numpy())

    

    return np.vstack(fin_outputs), np.vstack(fin_targets)
# Predicition for Test Data

def test_loop_fn(test, model, device):

    

    model.eval()

    # convert test data to FloatTensor

    test = Variable(torch.Tensor(test))

    test = test.to(device, dtype=torch.float)

    

    # Get predictions

    pred = model(test)

    # Get predictions from the maximum value

    _, predlabel = torch.max(pred.data, 1)

    # converting to list

    predlabel = predlabel.tolist()

    

    # Plotting the predicted results

    L = 5

    W = 5

    _, axes = plt.subplots(L, W, figsize = (12,12))

    axes = axes.ravel() # 



    for i in np.arange(0, L * W):  

        axes[i].imshow(test[i].reshape(28,28))

        axes[i].set_title("Prediction Class = {:0.1f}".format(predlabel[i]))

        axes[i].axis('off')

    

    plt.suptitle('Predictions on Test Data')

    plt.subplots_adjust(wspace=0.5)

    

    return predlabel
# Running Model

def run(path):

    # reading train an test data

    dfx = pd.read_csv(path+'train.csv')

    df_test = pd.read_csv(path+'test.csv')

    

    # variables for training model

    RANDOM_STATE = 42

    BATCH_SIZE = 100

    N_ITERS = 8500

    NUM_EPOCHS = int(N_ITERS / (len(dfx) / BATCH_SIZE))

    # target variable

    target = 'label' 

    classes = dfx[target].nunique()

    

    # spliting train data to train, validate

    df_train, df_valid = train_test_split(dfx, random_state= RANDOM_STATE, test_size = 0.2)

    df_train = df_train.reset_index(drop=True)

    df_valid = df_valid.reset_index(drop=True)

    

    # target labels

    train_targets = df_train[target].values

    valid_targets = df_valid[target].values

    

    # reshaping data to 28 x 28 images

    df_train = df_train.drop(target, axis=1).values.reshape(len(df_train), 1, 28, 28)

    df_valid = df_valid.drop(target, axis=1).values.reshape(len(df_valid), 1, 28, 28)

    df_test = df_test.values.reshape(len(df_test), 1, 28, 28)

    

    # Creating PyTorch Custom Datasets

    train_dataset = MnistDataset(df=df_train, target=train_targets)

    valid_dataset = MnistDataset(df=df_valid, target=valid_targets)

    

    # Creating PyTorch DataLoaders

    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_data_loader = data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    

    # device (cpu/gpu)

    device = "cpu"

    # learning rate

    lr = 1e-02

    # instatiate model and sending it to device

    model = MnistModel(classes=classes).to(device)

    # instantiate optimizer

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # instantiate scheduler

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 

                                                     patience=3, 

                                                     verbose=0, 

                                                     factor=0.5, 

                                                     min_lr=0.00001)

    

    # loop through epochs

    for epoch in range(NUM_EPOCHS):

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')

        # train on train data

        train_loop_fn(train_data_loader, model, optimizer, device, scheduler=None)

        # evaluate on validation data

        o, t = eval_loop_fn(valid_data_loader, model, device)

        print(f'Accuracy on Valid Data : {(o == t).mean() * 100} %')

    

    # Predict on test data

    preds = test_loop_fn(df_test, model, device)

    return preds
# Submission function

def submission(path, preds):

    sub = pd.read_csv(path+'sample_submission.csv')

    sub['Label'] = preds

    sub.to_csv('submission.csv', index=False)
path = '../input/digit-recognizer/'

preds = run(path)



submission(path, preds)



execution_time(start)