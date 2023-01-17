%matplotlib inline

import pandas as pd

import numpy as np

from dateutil.relativedelta import relativedelta



import torch

from torch import nn, optim

from torch.utils.data import TensorDataset, DataLoader



from livelossplot import PlotLosses
def add_datetime_features(df, drop_first=True,

                          one_hot_features=["Month", "Dayofweek", "Hour"],

                          circular_encoding=[]):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                # "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",

                "Hour", "Minute",]



    datetime = pd.to_datetime(df.Date * (10 ** 9))

    df['Datetime'] = datetime 

    

    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature, drop_first=drop_first)], axis=1)

        elif feature in circular_encoding:

            diff = len(new_column.value_counts())

            df[feature + "_sin"] = np.sin(2 * np.pi * new_column / diff)

            df[feature + "_cos"] = np.cos(2 * np.pi * new_column / diff)

        else:

            df[feature] = new_column

    return df
df = pd.read_csv("../input/train_electricity.csv")

df = add_datetime_features(df,

                           one_hot_features=[],

                           circular_encoding=["Month", "Dayofweek", "Hour"],

                           drop_first=False)



print(df.columns)



eval_from = df['Datetime'].max() + relativedelta(months=-6)

train_df = df[df['Datetime'] < eval_from]

valid_df = df[df['Datetime'] >= eval_from]





label_col = "Consumption_MW"  # The target values are in this column

to_drop = [label_col, "Date", "Datetime"] 



train_df_X = train_df.drop(to_drop, axis=1)

valid_df_X = valid_df.drop(to_drop, axis=1)



means = train_df_X.mean()

stds = train_df_X.std()



def normalize(dfx, means=means, stds=stds):

    return (dfx - means) / stds



train_df_X = normalize(train_df_X)

valid_df_X = normalize(valid_df_X)



input_channels = len(train_df_X.columns)

input_channels
# let's use differences as the baseline

(train_df.Consumption_MW - train_df.Production_MW).describe()
batch_size = 1024



train_X = torch.from_numpy(train_df_X.values.astype(np.float32))

train_Y = torch.from_numpy((train_df.Consumption_MW - train_df.Production_MW).values.astype(np.float32)).unsqueeze(1)



valid_X = torch.from_numpy(valid_df_X.values.astype(np.float32))

valid_Y = torch.from_numpy((valid_df.Consumption_MW - valid_df.Production_MW).values.astype(np.float32)).unsqueeze(1)



dataloaders = {

    'train': DataLoader(

        TensorDataset(train_X, train_Y),

        batch_size=batch_size, shuffle=True, num_workers=4),

    'validation': DataLoader(

        TensorDataset(valid_X, valid_Y),

        batch_size=batch_size, shuffle=False, num_workers=4)

}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train_model(model, optimizer, num_epochs=10):

    liveloss = PlotLosses()

    model = model.to(device)

    criterion = nn.MSELoss()

    

    for epoch in range(num_epochs):

        logs = {}

        for phase in ['train', 'validation']:

            if phase == 'train':

                model.train()

            else:

                model.eval()



            running_loss = 0.0



            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                outputs = model(inputs)

                loss = criterion(outputs, labels)



                if phase == 'train':

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()



                running_loss += loss.item() * inputs.size(0)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            

            prefix = ''

            if phase == 'validation':

                prefix = 'val_'



            logs[prefix + 'rmse'] = np.sqrt(epoch_loss)

        

        liveloss.update(logs)

        liveloss.draw()
class Nonlinear2(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        

        self.fc = nn.Sequential(

            nn.Linear(input_channels, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, 1)

        )

        

    def forward(self, x):

        return -430.489065 + 505.123157 * self.fc(x)
model = Nonlinear2(8)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_model(model, optimizer, num_epochs=100)