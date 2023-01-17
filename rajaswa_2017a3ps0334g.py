#IMPORTS

import time

import pandas as pd

import numpy as np



import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader, TensorDataset



from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.model_selection import train_test_split



epochs = 250
#SEED

seed_val = 0

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False



device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

print("using ", device)
#DATA IMPORT

train_df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

test_df = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')

sub_df = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')



del train_df['ID']

del test_df['ID']



y_train = np.array(train_df['Class'])

train_df.drop(['Class'], axis=1, inplace=True)

y_test = np.zeros(len(test_df))



columns = train_df.columns

categorical_cols = ['Number of Quantities', 'Number of Insignificant Quantities', 'Size']
#CATEGORICAL COLS: Number of Quantities, Number of Insignificant Quantities, Size

def get_categorical(col, train_df, test_df):

    dummies = pd.get_dummies(train_df[col], prefix=col, dummy_na=False)

    train_df = pd.concat([train_df, dummies], axis=1)

    train_df.drop([col], axis=1, inplace=True)

    

    dummies = pd.get_dummies(test_df[col], prefix=col, dummy_na=False)

    test_df = pd.concat([test_df, dummies], axis=1)

    test_df.drop([col], axis=1, inplace=True)

    

    try:

        del train_df[col+'_?']

    except:

        pass

    

    return train_df, test_df



for col in categorical_cols:

    train_df, test_df = get_categorical(col, train_df, test_df)
for df in [train_df, test_df]:

    for col in df.columns:

        df[col].replace(to_replace=['?'], value=0, inplace=True)

        df[col] = df[col].astype('float64')
train_df.head()
test_df.head()
#GETTING RAW VALUES FROM DATAFRAMES

x_train = train_df.to_numpy()

x_test = test_df.to_numpy()



print(x_train.shape, x_test.shape)
#DATASETS & DATALOADERS

train_dset = TensorDataset(torch.tensor(x_train), torch.IntTensor(y_train))

test_dset = TensorDataset(torch.tensor(x_test), torch.IntTensor(y_test))



train_loader = DataLoader(train_dset, batch_size=16, shuffle=True, num_workers=0)

test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
#MODEL

class Net(nn.Module):

    def __init__(self, neuron):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(15, neuron[0])

        self.fc2 = nn.Linear(neuron[0], neuron[1])

        self.fc3 = nn.Linear(neuron[1], 6)

        

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.01)



    def forward(self, x):

        out = self.dropout(self.relu(self.fc1(x)))

        out = self.dropout(self.relu(self.fc2(out)))

        out = self.relu(self.fc3(out))

        

        return out
#CRITERION

class_counts = [111, 22, 80, 33, 44, 81]

weights = torch.tensor([1, 1.2, 1.1, 1.2, 1.2, 1.1]).to(device)

criterion = nn.CrossEntropyLoss(weights)
model1 = Net([24, 16])

model2 = Net([26, 16])

model3 = Net([28, 16])



model_list = [model1, model2, model3]
#TRAINING

acc_list = []

model_no = 0

start_time = time.time()

for model in model_list:

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=False)

    model_no += 1

    model.train()

    model = model.to(device)

    

    for epoch in range(epochs):

        epoch_loss = 0.0

        count = 0

        for i, data in enumerate(train_loader):

            x, target = data

            x, target = x.to(device), target.to(device)

        

            optimizer.zero_grad()



            outputs = model(x.float())

            loss = criterion(outputs, target.long())



            loss.backward()

            optimizer.step()

        

            epoch_loss += loss.item()

            count += 1



    model.eval()

    preds = []

    ground_truth = []

    with torch.no_grad():

        for i, data in enumerate(train_loader):

            x, target = data

            x, target = x.to(device), target.to(device)

            

            outputs = model(x.float())

            

            for label in target:

                ground_truth.append(int(label.item()))

            _, predicted = torch.max(outputs.data, 1)

            for pred in predicted:

                preds.append(pred.cpu().numpy())



    preds = np.array(preds)

    ground_truth = np.array(ground_truth)

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, preds, average='weighted')

    accuracy = accuracy_score(ground_truth, preds)

    acc_list.append(accuracy)

    print('MODEL %2d ACC: %.3f F1: %.3f PREC: %.3f REC: %.3f TIME: %.3f' %

                  (model_no , accuracy, f1, prec, recall, time.time()-start_time))

    model = model.to('cpu')
final_preds_matrix = torch.FloatTensor([np.zeros(6)]*len(test_df)).to(device)

softmax = nn.Softmax()

preds = []

for model in model_list: 

    model.eval()

    model=model.to(device)



    with torch.no_grad():

        for i, data in enumerate(test_loader):

            x, target = data

            x, target = x.to(device), target.to(device)

            

            outputs = model(x.float())

            final_preds_matrix[i] += softmax(outputs.reshape(-1))

            

final_preds_matrix /= len(model_list)

for pred in final_preds_matrix:

    _, predicted = torch.max(pred.data, dim=0)

    preds.append(int(predicted.item()))
sub_df['Class'] = preds

print(sub_df['Class'].value_counts())
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub_df)