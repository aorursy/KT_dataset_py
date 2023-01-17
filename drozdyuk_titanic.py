import numpy as np

import torch

from torch import nn

import pandas as pd

from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device
class TitanicDataset(Dataset):

    def __init__(self, x, y=None):

        self.x = x

        self.y = y

        

    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, idx):

        data = self.x[idx, :]

        data = np.asarray(data).astype(np.float32)

        

        if self.y is not None:

            return data, np.array([self.y[idx]], dtype=np.float32)

        else:

            return data
df = pd.read_csv("/kaggle/input/titanic/train.csv")

def df_from(csv_file, label=None):

    df = pd.read_csv(csv_file)

    df = df.set_index('PassengerId')



    df['SibSp'] = df['SibSp']/8.0

    df['Parch'] = df['Parch']/9.0

    df['Fare'] = df['Fare']/512.0

    df['Age'] = df['Age']/80.0



    df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)

    df['Pclass'] = df['Pclass'].replace(1, 0).replace(2, 1).replace(3, 2)



    df['Embarked'] = df['Embarked'].replace('C', 0).replace('Q', 1).replace('S', 2).fillna(3)

    

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', "Parch", "Embarked", ]



    x = df[features]

    x = x.fillna(0.0)

    

    if label is not None:

        y = df[label]

        y = y.fillna(0.0)

        return x, y

    else:

        return x
df_train_x, df_train_y = df_from("/kaggle/input/titanic/train.csv", label='Survived')

NUM_FEATURES = len(df_train_x.columns)

df_train_x
# VALID_SIZE = 1

train_dataset = TitanicDataset(df_train_x.values, df_train_y.values)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)



# train_data, valid_data = torch.utils.data.random_split(train_dataset, lengths=[891 - VALID_SIZE, VALID_SIZE])



# len(train_data), len(valid_data)



# train_loader = DataLoader(train_data, batch_size=512, shuffle=True)

# valid_loader = DataLoader(valid_data, batch_size=1028, shuffle=True)
HIDDEN = 64

DROPOUT = 0.1

class Net(nn.Module):

    def __init__(self, num_inputs):

        super().__init__()

        

        self.embarked_emb = nn.Embedding(4, 3)

        self.embarked_linear = nn.Linear(3, out_features=1)

        

        self.p_class_emb = nn.Embedding(3, 3)

        self.p_class_linear = nn.Linear(3, out_features=1)



        self.sex_emb = nn.Embedding(2, 3)

        self.sex_linear = nn.Linear(3, out_features=1)

        

        self.model = nn.Sequential(

          nn.Linear(in_features=num_inputs, out_features=HIDDEN),

          nn.ReLU(),

          nn.Dropout(p=DROPOUT),

          nn.Linear(in_features=HIDDEN, out_features=HIDDEN),

          nn.ReLU(),

          nn.Dropout(p=DROPOUT),

          nn.Linear(in_features=HIDDEN, out_features=HIDDEN),

          nn.ReLU(),

          nn.Dropout(p=DROPOUT),

          nn.Linear(in_features=HIDDEN, out_features=1),

        )

        

    def forward(self, inputs):  

        p_class_idx = 0

        p_class = inputs[:, p_class_idx].long()

        p_class_emb = self.p_class_emb(p_class)

        p_class_linear = self.p_class_linear(p_class_emb)

        inputs[:, p_class_idx] = p_class_linear.squeeze()

        

        sex_idx = 1

        sex = inputs[:, sex_idx].long()

        sex_emb = self.sex_emb(sex)

        sex_linear = self.sex_linear(sex_emb)

        inputs[:, sex_idx] = sex_linear.squeeze()

        

        embarked_idx = 6

        embarked = inputs[:, embarked_idx].long()

        embarked_emb = self.embarked_emb(embarked)

        embarked_linear = self.embarked_linear(embarked_emb)

        inputs[:, embarked_idx] = embarked_linear.squeeze()

    

        return self.model(inputs)

        

n = Net(num_inputs=NUM_FEATURES).to(device)



loss_fn = nn.BCEWithLogitsLoss().to(device)

optim = torch.optim.Adam(n.parameters())
def train_step():    

    n.train()

    train_loss = 0.0

    c = 0

    for x, y in train_loader:

        x = x.to(device)

        y = y.to(device)   

     

        output = n(x)

        loss = loss_fn(output, y)

        loss.backward()

        train_loss += loss

        c += 1

        

        optim.step()

        optim.zero_grad()

        

    n.eval()

    

#     with torch.no_grad():

#         test_loss = sum([loss_fn(n(x.to(device)).to(device), y.to(device)) for x, y in valid_loader])

        

    return train_loss / c #, test_loss
EPOCHS = 500



for i in range(EPOCHS):

    train_loss = train_step()

    print(f'Train loss: {train_loss}')

#     print(f'Valid loss: {new_loss}')

#     if new_loss < best_loss:

#         best_loss = new_loss

#     else:

#         if new_loss > (best_loss + 0.005):

#             break

        
df_test = df_from("/kaggle/input/titanic/test.csv")

df_test
predictions = []

n.eval()

with torch.no_grad():

    for index, row in df_test.iterrows():

        x = torch.tensor([row.to_numpy()]).float()

        

        x = x.to(device)

        s = torch.sigmoid(n(x))

        outputs = (s > 0.5) * 1.0

        outputs = outputs.cpu().numpy()[0].astype(np.int32)

        predictions.append((index, outputs[0]))



with open('predictions.csv', 'w') as f:

    f.write('PassengerId,Survived\n')

    for p, pred in predictions:

        f.write(f'{p},{pred}\n')