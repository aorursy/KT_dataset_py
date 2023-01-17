# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

calender = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sell_prce = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
def get_train_label():

    val = sales_train.values[:,6:]

    i = len(val[0]) - 28

    train = val[:,i:i+28].reshape(1, 28, 30490)

    return train.astype(np.float32)
class LSTM(nn.Module):

    def __init__(self, input_size=30490, hidden_layer_size=1000, output_size=30490):

        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)



    def forward(self, input_seq):

        lstm_out, _ = self.lstm(input_seq)

        predictions = self.linear(lstm_out)

        return predictions[:,-1,:]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = LSTM()

model = model.to(device)
train = get_train_label()
# infer

model.eval()

model_path = '/kaggle/input/m5-lstm-weight/model.pth'

model.load_state_dict(torch.load(model_path, map_location='cpu'))

inputs = train[-1].reshape(1, 28 , 30490)

preds = np.empty((30490, 56))

preds_res = np.empty((60980, 28))

for i in tqdm(range(56)):

    inputs = torch.from_numpy(inputs).to(device)

    pred = model(inputs).cpu().detach().numpy()

    preds[:, i] = pred

    inputs = inputs.cpu().detach().numpy()

    inputs[:, 0:27,:] = inputs[:, 1:,:]

    inputs[:,-1,:] = pred

preds = np.where(preds < 0, 0, preds)

preds_val = preds[:, :28]

preds_eval = preds[:, 28:]

preds_res[:30490,:] = preds_val

preds_res[30490:,:] = preds_eval

for i in range(1, 28):

    submission[f'F{i}'] = preds_res[:,i]

submission.to_csv('submission.csv', index=False)
print(preds.min())

print(preds.max())