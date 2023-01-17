import os

import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import copy

%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler, RobustScaler
def smoothened(values, weight):

    last = values[0]

    smooth_values = []

    for value in values:

        temp_smoothed_value = last * weight + (1-weight)*value

        smooth_values.append(temp_smoothed_value)

        last = temp_smoothed_value

    return smooth_values
#Set Seed

torch.manual_seed(2020)
train = pd.read_csv("../input/bitsf312-lab1/train.csv")

test = pd.read_csv("../input/bitsf312-lab1/test.csv")



train['Size']=train['Size'].map({'Medium':0, 'Big':1, 'Small':-1, '?':'?'})

test['Size']=test['Size'].map({'Medium':0, 'Big':1, 'Small':-1, '?':'?'})

train.drop('ID', axis=1, inplace=True)



to_drop = []

for i in train.columns:

    temp = train.index[train[i] == '?'].tolist()

    if temp != []:

        to_drop.extend(temp)



for drop in reversed(np.sort(to_drop)):

    train.drop(drop, axis=0, inplace=True)
train.replace('?',np.NaN,inplace = True)

train[train.columns] = train[train.columns].astype(float)

# train[['Difficulty','Score']] = train[['Difficulty','Score']].fillna(train[['Difficulty','Score']].mean())

# mode_cols = train.columns

# mode_cols = mode_cols[:-3]

# train[mode_cols] = train[mode_cols].fillna(train[mode_cols].mode().iloc[0])

# train.isnull().any()

train.dropna(inplace=True)
from torch.utils.data import Dataset, DataLoader



class TabularDataset(Dataset):

  def __init__(self, data, cat_cols=None, output_col=None):

    """

    Characterizes a Dataset for PyTorch



    Parameters

    ----------



    data: pandas data frame

      The data frame object for the input data. It must

      contain all the continuous, categorical and the

      output columns to be used.



    cat_cols: List of strings

      The names of the categorical columns in the data.

      These columns will be passed through the embedding

      layers in the model. These columns must be

      label encoded beforehand. 



    output_col: string

      The name of the output variable column in the data

      provided.

    """



    self.n = data.shape[0]



    if output_col:

      self.y = data[output_col].astype(np.float32).values

    else:

      self.y =  np.zeros((self.n, 1))



    self.cat_cols = cat_cols if cat_cols else []

    # self.cont_cols = [col for col in data.columns

    #                   if col not in self.cat_cols + [output_col] + ['index']]

    self.cont_cols = ['Difficulty','Score']

    if self.cont_cols:

      self.cont_X = data[self.cont_cols].astype(np.float32).values

    else:

      self.cont_X = np.zeros((self.n, 1))



    if self.cat_cols:

      self.cat_X = data[cat_cols].astype(np.int64).values

    else:

      self.cat_X =  np.zeros((self.n, 1))



  def __len__(self):

    """

    Denotes the total number of samples.

    """

    return self.n



  def __getitem__(self, idx):

    """

    Generates one sample of data.

    """

    return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]





    from torch.utils.data import Dataset, DataLoader





class NormalDataset(Dataset):

  def __init__(self, data, cols=None, output_col=None):



    self.n = data.shape[0]



    if output_col:

      self.y = data[output_col].astype(np.float32).values

    else:

      self.y =  np.zeros((self.n, 1))



    self.cols = cols

    self.X = data[self.cols].astype(np.float32).values



  def __len__(self):

    """

    Denotes the total number of samples.

    """

    return self.n



  def __getitem__(self, idx):

    """

    Generates one sample of data.

    """

    return [self.y[idx], self.X[idx]]





class ElmoDataset(Dataset):

  def __init__(self, data, elmo, batch_to_ids, cat_cols, cols=None, output_col=None):



    self.n = data.shape[0]

    self.elmo = elmo

    self.batch_to_ids = batch_to_ids

    self.Xcat = data[cat_cols].astype(str).values

    self.vals = np.unique(np.ravel(train[categorical_features].astype(str).values)).tolist()

    char_ids = batch_to_ids(self.vals)

    elmo_rep = elmo(char_ids)

    # model.fit(train[msk][cols], train[msk]['Class'])#, sample_weight=weights)

    self.pairs = dict(zip(self.vals, elmo_rep['elmo_representations'][0].mean(dim=1).detach().numpy().tolist()))

    # print(self.Xcat)

    # print(self.Xcat.shape)



    self.cont_cols = ['Difficulty','Score']

    self.cont_X = data[self.cont_cols].astype(np.float32).values



    if output_col:

      self.y = data[output_col].astype(np.float32).values

    else:

      self.y =  np.zeros((self.n, 1))



    # self.cols = cols

    # self.X = data[self.cols].astype(np.float32).values



  def __len__(self):

    """

    Denotes the total number of samples.

    """

    return self.n



  def __getitem__(self, idx):

    """

    Generates one sample of data.

    """



    keys = self.Xcat[idx].tolist()

    catX = np.array([self.pairs[key] for key in keys])

    sample = np.mean(catX, axis=0).tolist()

    # print(len(sample))

    sample.extend(self.cont_X[idx].tolist())

    sample = torch.from_numpy(np.array(sample))

    # character_ids = self.batch_to_ids(self.Xcat[idx])

    # elmo_rep = elmo(character_ids)

    # elmo_rep = elmo_rep['elmo_representations'][0].mean(dim=1).mean(dim=1)

    # print(torch.from_numpy(self.cont_X[idx]).shape, elmo_rep.shape)

    # sample = self.cont_X[idx].tolist()

    # sample.extend(elmo_rep.detach().numpy().tolist())

    # sample = torch.from_numpy(np.array(sample))

    # sample = torch.cat([torch.from_numpy(self.cont_X[idx]),elmo_rep])#,dim=1)

    return [self.y[idx], sample]





class NormalDataset(Dataset):

  def __init__(self, data, cols=None, output_col=None):



    self.n = data.shape[0]



    if output_col:

      self.y = data[output_col].astype(np.float32).values

    else:

      self.y =  np.zeros((self.n, 1))



    self.cols = cols

    self.X = data[self.cols].astype(np.float32).values



  def __len__(self):

    """

    Denotes the total number of samples.

    """

    return self.n



  def __getitem__(self, idx):

    """

    Generates one sample of data.

    """

    return [self.y[idx], self.X[idx]]



class OneHotDataset(Dataset):

  def __init__(self, data, y):



    self.n = data.shape[0]

    self.X = data



    self.y = y

    # self.X = data[self.cols].astype(np.float32).values



  def __len__(self):

    """

    Denotes the total number of samples.

    """

    return self.n



  def __getitem__(self, idx):

    """

    Generates one sample of data.

    """

    return [self.y[idx], self.X[idx]]
msk = np.random.rand(train.shape[0])<0.8

X_train = train[msk]

X_val = train[~msk]



X_train.reset_index(inplace=True)

X_val.reset_index(inplace=True)



categorical_features = ['Size']



hot = OneHotEncoder().fit(train[categorical_features].astype(str).values)

temptt = hot.transform(X_train[categorical_features].astype(str)).toarray()

temptv = hot.transform(X_val[categorical_features].astype(str)).toarray()



cont_cols = ['Total Number of Words', 'Number of Special Characters', 'Number of Sentences',

             'Difficulty', 'Score']



train[cont_cols] = train[cont_cols].astype(float)

# X_tr = sklearn.preprocessing.normalize(X_train[cont_cols].values)

# X_va = sklearn.preprocessing.normalize(X_val[cont_cols].values)

X_tr = np.concatenate([temptt, sklearn.preprocessing.normalize(X_train[cont_cols].astype(float).values)], axis=1)

X_va = np.concatenate([temptv, sklearn.preprocessing.normalize(X_val[cont_cols].astype(float).values)], axis=1)

X_train = OneHotDataset(X_tr, X_train['Class'].values)

X_val = OneHotDataset(X_va, X_val['Class'].values)

batchsize = 64

trainloader = DataLoader(X_train, batchsize, shuffle=True, num_workers=1)

valloader = DataLoader(X_val, batchsize, shuffle=True, num_workers=1)
class Model2(nn.Module):

    def __init__(self, in_features, l1, l2, l3):

        super(Model2, self).__init__()

        self.layer1 = nn.Linear(in_features, l1)

        nn.init.xavier_normal_(self.layer1.weight)

        self.bn1 = nn.BatchNorm1d(l1)

        self.dp1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(l1, l2)

        nn.init.xavier_normal_(self.layer2.weight)

        self.bn2 = nn.BatchNorm1d(l2)

        self.dp2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(l2, l3)

        nn.init.xavier_normal_(self.layer3.weight)

        self.bn3 = nn.BatchNorm1d(l3)

        self.dp3 = nn.Dropout(0.5)

        self.layer4 = nn.Linear(l3, 6)

        nn.init.xavier_normal_(self.layer4.weight)



    def forward(self, x):

        x = self.dp1(self.bn1(F.relu(self.layer1(x))))

        x = self.dp2(self.bn2(F.relu(self.layer2(x))))

        x = self.dp3(self.bn3(F.relu(self.layer3(x))))

        return F.softmax(self.layer4(x))



model = Model2(8, 100, 250, 50)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

weights = max(np.unique(train['Class'].values, return_counts=True)[1])/np.unique(train['Class'].values, return_counts=True)[1]

criterion = nn.CrossEntropyLoss(torch.from_numpy(weights).float())

EPOCHS = 1000

losses, t_acc, v_acc = [], [], []
is_training = True

path_to_model = '../input/modelnnfllab1/model'

if os.path.exists(path_to_model):

    is_training = False

is_training = True

if is_training == False:

    print("Loading pre-trained model")

    model.load_state_dict(torch.load(path_to_model))
if is_training :

    EPOCHS = 500

    for epoch in range(EPOCHS):

        # scheduler.step()

        running_loss, train_correct, train_total, val_correct, val_total = 0.0, 0, 0, 0, 0

        for (input, label) in trainloader:

            # print(label.shape)

            # break

            model.train()

            input, label = label, input

            input, label = Variable(input.float()), Variable(label)

            optimizer.zero_grad()

            output = model(input)



            loss = criterion(output, label.long())

            loss.backward()

            optimizer.step()



            running_loss += loss.item()



            train_total += output.shape[0]

            train_correct += sum(torch.max(output,1)[1]==label)

        train_accuracy = 100*train_correct.item()/train_total



        for (input, label) in valloader:

            input, label = label, input

            model.eval()



            output = model(input.float())



            val_total += output.shape[0]

            val_correct += sum(torch.max(output,1)[1]==label)



        val_accuracy = 100*val_correct.item()/val_total

        v_acc.append(val_accuracy)

        t_acc.append(train_accuracy)



        if epoch==0:

            best = val_accuracy

            best_train = train_accuracy

        if val_accuracy > best:

            best = val_accuracy

            if train_accuracy > best_train:

                best_train = train_accuracy

                best_model = model.state_dict()

                best_model_fn = copy.deepcopy(model)

                torch.save(best_model, 'best_model')



        if epoch%10==0:

            print("========================================")

            print("EPOCH {} - Loss    : {}".format(epoch, loss.item()))

            print("Training Accuracy  : {}".format(train_accuracy))

            print("Validation Accuracy: {}".format(val_accuracy))

            print("========================================")



    fig, ax = plt.subplots(figsize=(7,7))



    ax.plot(np.arange(1, len(t_acc)+1), smoothened(t_acc,0.9), label="Train Accuracy")

    ax.plot(np.arange(1, len(v_acc)+1), smoothened(v_acc,0.9), label="Validation Accuracy")



    ax.legend()
categorical_features = ['Size']

tempttt = hot.transform(test[categorical_features].astype(float).astype(str)).toarray()

cont_cols = ['Total Number of Words', 'Number of Special Characters', 'Number of Sentences', 'Difficulty', 'Score']



test[cont_cols] = test[cont_cols].astype(float)

testtt = np.concatenate([tempttt, sklearn.preprocessing.normalize(test[cont_cols].astype(float).values)], axis=1)

# tempttt = sklearn.preprocessing.normalize(test[cont_cols])
output = model(torch.from_numpy(testtt).float())

pred = torch.max(output, 1)[1]

to_submit = pd.DataFrame(pd.concat([test['ID'],pd.Series(pred.numpy().tolist())], axis=1))

to_submit.columns = ['ID', 'Class']

to_submit.to_csv("submission.csv", sep=",", header=True, index=False)



from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(to_submit)