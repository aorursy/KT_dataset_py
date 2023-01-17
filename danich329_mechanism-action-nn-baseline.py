# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

print(train_df.shape)

train_df.head()
train_df.cp_type.value_counts()
train_df.cp_dose.value_counts()
train_labels_df = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

print(train_labels_df.shape)

train_labels_df.head()
val_df = train_df.sample(5000, random_state=100) 

print(val_df.shape)

val_df.head()
test_df = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

test_df.head()
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as opt

import torch.utils.data as data



import tqdm

import itertools
class FeedForward_Network(nn.Module):

    def __init__(self, input_size=100, output_size=100, dropout=[0.25, 0.25, 0.25, 0.25]):

        super(FeedForward_Network, self).__init__()

        

        self.layer1 = nn.Linear(input_size, 700)

        self.layer2 = nn.Linear(700, 500)

        self.layer3 = nn.Linear(500, 300)

        self.layer4 = nn.Linear(300, 250)

        self.layer5 = nn.Linear(250, output_size)

        

        self.dropout = dropout

        

    def forward(self, x):

        output = self.layer1(x)

        output = F.relu(output)

        output = F.dropout(output, p=self.dropout[0])

        output = self.layer2(output)

        output = F.relu(output)

        output = F.dropout(output, p=self.dropout[1])

        output = self.layer3(output)

        output = F.relu(output)

        output = F.dropout(output, p=self.dropout[2])

        output = self.layer4(output)

        output = F.relu(output)

        output = F.dropout(output, p=self.dropout[3])

        output = self.layer5(output)

        return torch.sigmoid(output)

data_map = {

    'train':(

        train_df.query(f"sig_id not in {val_df.sig_id.tolist()}"), 

        train_labels_df.query(f"sig_id not in {val_df.sig_id.tolist()}")

    ),

    'validation':(

        val_df, 

        train_labels_df.query(f"sig_id in {val_df.sig_id.tolist()}")

    )

}
#Create the dataloader

column_drops = ["sig_id", "cp_type", "cp_time", "cp_dose"]

dataloader_map = {

    label:data.DataLoader(

        data.TensorDataset(

            torch.FloatTensor(data_map[label][0].drop(column_drops, axis=1).values), 

            torch.FloatTensor(data_map[label][1].drop("sig_id", axis=1).values)

        ), 

        batch_size=256,

        shuffle=(label == 'train')

    )

    for label in data_map

}
model = FeedForward_Network(

    input_size=dataloader_map['train'].dataset[0][0].shape[0],

    output_size=dataloader_map['train'].dataset[0][1].shape[0]

)

loss_fn = nn.BCELoss()
#epochs = [50,100, 200]

#lr = [1e-3, 1e-5, 1e-7]

epochs = [50]

lr = [1e-3]

grid = itertools.product(epochs, lr)
for epoch_grid, lr_param in grid:

    optimizer = opt.Adam(model.parameters(), lr=lr_param)

    

    # Run the epochs

    for epoch in range(epoch_grid+1):

        

        if epoch > 0:

            print(epoch)

            train_loss = []

            

            for batch in dataloader_map['train']:

                # reset optimizer

                optimizer.zero_grad()

                

                # pass data through the model

                prediction = model(batch[0])

                loss = loss_fn(prediction, batch[1])

                

                # Track the model loss

                train_loss.append(loss.item())

                loss.backward()

                optimizer.step()



            print(f"train loss: {np.mean(train_loss)}")

                

                

        # Set model to evaluation

        model.eval()

        val_loss = []

        

        for batch in dataloader_map['validation']:

            # Pass data through model

            prediction = model(batch[0])

            

            # Get the loss

            loss = loss_fn(prediction, batch[1])

            

            # Track the loss

            val_loss.append(loss.item())

        

        print(f"val loss: {np.mean(val_loss)}")

feat_cols = (

    train_labels_df

    .drop("sig_id", axis=1)

    .columns

)



(

    pd.DataFrame(

        model(

            torch.FloatTensor(

                test_df

                .drop(column_drops, axis=1)

                .values

            )

        )

        .detach()

        .numpy(),

        columns=feat_cols

    )

    .assign(sig_id=test_df.sig_id.tolist())

    .to_csv("model_submission.csv", index=False)

)