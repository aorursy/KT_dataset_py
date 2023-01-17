import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.sparse import load_npz

import gc

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
torch.cuda.set_device(0)

torch.cuda.manual_seed_all(42)
data_folder = '/kaggle/input/sparse-data-v3/'

X_train = load_npz(data_folder+'/train.npz')

X_test = load_npz(data_folder+'/test.npz')

y_train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', usecols=['isFraud', 'TransactionDT']).sort_values('TransactionDT')['isFraud']

print(X_train.shape, X_test.shape, y_train.shape)
def from_csr_to_tensor(X, y):

    x_batch = X.tocoo()

    x_batch = torch.sparse_coo_tensor(indices = torch.cuda.LongTensor([x_batch.row.tolist(), x_batch.col.tolist()]),

                            values = torch.cuda.FloatTensor(x_batch.data), size=[X.shape[0], X_train.shape[1]])

    y_batch = torch.from_numpy(y.values)

    x_batch = x_batch.type(torch.cuda.FloatTensor)

    y_batch = y_batch.type(torch.cuda.FloatTensor)

    

    return x_batch, y_batch
rows = X_train.shape[0]

rows_split = int(rows*0.8)

print(rows, rows_split)
X_valid = X_train[rows_split:]

y_valid = y_train[rows_split:]

X_fit = X_train[:rows_split]

y_fit = y_train[:rows_split]
x_batch_valid, y_batch_valid = from_csr_to_tensor(X_valid, y_valid)
class MyClassifier(nn.Module):

    def __init__(self):

        super(MyClassifier,self).__init__()

        self.fc1 = nn.Linear(X_train.shape[1], 1024)

        self.relu1 = nn.ReLU()

        self.dout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1024, 2048)

        self.relu2 = nn.ReLU()

        self.dout2 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(2048, 512)

        self.relu3 = nn.ReLU()

        self.dout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 64)

        self.prelu = nn.PReLU(1)

        self.out = nn.Linear(64, 1)

        self.out_act = nn.Sigmoid()

        

    def forward(self, input_):

        a1 = self.fc1(input_)

        h1 = self.relu1(a1)

        dout1 = self.dout1(h1)

        a2 = self.fc2(dout1)

        h2 = self.relu2(a2)

        dout2 = self.dout2(h2)

        a3 = self.fc3(dout2)

        h3 = self.relu3(a3)

        dout3 = self.dout3(h3)

        a4 = self.fc4(dout3)

        h4 = self.prelu(a4)

        a5 = self.out(h4)

        y = self.out_act(a5)

        return y

              

    def predict(self, x):

        #Apply softmax to output. 

        pred = F.softmax(self.forward(x))

        return torch.tensor(pred)
#Initialize the model        

model = MyClassifier()

model.cuda()

#Define loss criterion

criterion = nn.BCELoss()

#Define the optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
#Number of epochs

epochs = 50

#Batch size

batch_size = 16368

#Losses

losses = []

#Early-Stopping

best_epoch = 0

last_score = 0

es_rounds = 2

#Range

Rango = range(0, X_fit.shape[0], batch_size)

len_Rango = len(list(Rango))



for i in range(1, epochs+1):

    print('Training epoch {}'.format(i))

    loss_epoch = 0

    loss_valid = 0

    loss_aps = 0

    loss_auc = 0

    loss_auc_valid = 0

    loss_aps_valid = 0

    model.train()

    for j in Rango:

        x_batch, y_batch = from_csr_to_tensor(X_fit[j:j+batch_size], y_fit[j:j+batch_size])

        #Clear the previous gradients

        optimizer.zero_grad()

        #Precit the output for Given input

        y_pred = model(x_batch)

        #Compute Cross entropy loss

        loss = criterion(y_pred, y_batch)

        del x_batch

        del y_batch

        torch.cuda.empty_cache()

        gc.collect()

        #Compute gradients

        loss.backward()

        #Adjust weights

        optimizer.step()

        

    #Validate data with model.eval()    

    model.eval()

    y_pred_valid = model(x_batch_valid)

    loss_val = criterion(y_pred_valid, y_batch_valid)

    loss_valid += loss_val.item()

    loss_auc_valid += roc_auc_score(y_batch_valid.data.cpu().numpy(), y_pred_valid.data.cpu().numpy())

    loss_aps_valid += average_precision_score(y_batch_valid.data.cpu().numpy(), y_pred_valid.data.cpu().numpy())

        

    losses.append([loss_epoch, loss_valid, loss_auc, loss_auc_valid, loss_aps, loss_aps_valid])

    print(f'trains loss: {loss_epoch}, trains AUC: {loss_auc}, trains APS: {loss_aps}')

    print(f'tests loss: {loss_valid}, tests AUC: {loss_auc_valid}, tests APS: {loss_aps_valid}')

        

    if last_score <= loss_auc_valid:

        last_score = loss_auc_valid

        best_epoch = i

        es_rounds = 2

    else:

        if es_rounds > 0:

            es_rounds -=1

        else:

            print('EARLY-STOPPING !')

            print('Best epoch found: nº {}'.format(best_epoch))

            print('Exiting. . .')

            break





plt.figure(figsize=(16,12))

plt.subplot(3, 1, 1)

plt.title('Score per epoch')

plt.ylabel('Binary Cross-entropy')

# plt.plot(list(range(len(losses))), [x[0] for x in losses], label=['Trains BCE loss'])

plt.plot(list(range(len(losses))), [x[1] for x in losses], label=['Valids BCE loss'])

plt.legend()

plt.subplot(3, 1, 2)

plt.ylabel('ROC-AUC Score')

# plt.plot(list(range(len(losses))), [x[2] for x in losses], label=['Trains ROC_AUC'])

plt.plot(list(range(len(losses))), [x[3] for x in losses], label=['Valids ROC_AUC'])

plt.legend()

plt.subplot(3, 1, 3)

plt.xlabel('Epoch')

plt.ylabel('PR-AUC Score')

# plt.plot(list(range(len(losses))), [x[4] for x in losses], label=['Trains PR_AUC'])

plt.plot(list(range(len(losses))), [x[5] for x in losses], label=['Valids PR_AUC'])

plt.legend()

plt.show()
model = MyClassifier()

model.cuda()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
model.train()

for i in range(1, best_epoch+1):

    print('Training epoch {}'.format(i))

    loss_epoch = 0

    loss_aps = 0

    loss_auc = 0

    for j in Rango:

        x_batch, y_batch = from_csr_to_tensor(X_train[j:j+batch_size], y_train[j:j+batch_size])

        #Clear the previous gradients

        optimizer.zero_grad()

        #Precit the output for Given input

        y_pred = model(x_batch)

        #Compute Cross entropy loss

        loss = criterion(y_pred, y_batch)

        #Compute gradients

        loss.backward()

        #Adjust weights

        optimizer.step()

        

    del x_batch

    del y_batch

    torch.cuda.empty_cache()

    gc.collect()
del x_batch_valid

del y_batch_valid

del X_fit

del y_fit

del X_valid

del y_valid

torch.cuda.empty_cache()

gc.collect()
batch_size = 16368

y_dummy = pd.Series(np.array([1,1]))

Rango = range(0, X_test.shape[0], batch_size)

len_Rango = len(list(Rango))



model.eval()



final_preds = np.array([])



for j in Rango:

    x_batch_valid, _ = from_csr_to_tensor(X_test[j:j+batch_size], y_dummy)

    y_pred_valid = model(x_batch_valid)

    y_pred_valid = y_pred_valid.data.cpu().numpy()

    

    final_preds = np.concatenate((final_preds, y_pred_valid), axis=None)



    del x_batch_valid

    del y_pred_valid

    torch.cuda.empty_cache()

    gc.collect()

submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

submission['isFraud'] = final_preds
submission.head(10)
submission.to_csv('submission.csv', sep=',', header=True, index=None)