import torch

from torch import nn, optim

import torch.nn.functional as F

import pandas as pd

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

import numpy as np

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/battles.csv')

df.head(5)
# no los interesa el número de la batalla así que a la basura!

data = df.drop(columns=['battle_number']).values

data.shape, data[:,0].min(), data[:,0].max()
class BattleDataset(Dataset):

    def __init__(self, battle_list):

        self.battle_list = battle_list

        

    def __getitem__(self, index):

        first, second, result = self.battle_list[index]

        return torch.tensor(first-1).long(), torch.tensor(second-1).long(), torch.tensor(result).float()



    def __len__(self):

        return self.battle_list.shape[0]
# valid_split_idx = int(data.shape[0] * 0.9)

# train_data = data[:valid_split_idx]

# valid_data = data[valid_split_idx:]



train_data, valid_data = train_test_split(data, test_size=0.1, random_state=99)



train_dataset = BattleDataset(train_data)

valid_dataset = BattleDataset(valid_data)



train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True) 

valid_loader = DataLoader(valid_dataset, batch_size=8, num_workers=2) 
# probemos que los loaders devuelvan los datos que esperamos...

first, second, result = next(iter(train_loader))

first, second, result
class EmbeddingModel(nn.Module):

    def __init__(self, embedding_size, dropout):

        super(EmbeddingModel, self).__init__()

        

        self.embedding = nn.Embedding(800, embedding_size) #800 es el numero de pokemon...

        

        self.out = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(embedding_size*2, 1),

            nn.Sigmoid()

        )

                

    def forward(self, first, second):

        first_embedding = self.embedding(first)

        second_embedding = self.embedding(second)

        combined = torch.cat((first_embedding, second_embedding), dim=-1)

        return self.out(combined)
# usamos un embedding size de 512 y un dropout de 0.1 porque me da a mi la gana, y punto

model = EmbeddingModel(embedding_size=512, dropout=0.1)
pred = model.forward(first, second)

pred.shape
best_loss = 9999999 # usaremos esto para solo ir guardando los mejores modelos
optimizer = optim.Adam(model.parameters(), lr=0.001) 

criterion = nn.MSELoss() # lo lógico sería usar BCELoss pero MSELoss aprende más rapido yoquese :/ ...

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epochs = 20



model.cuda()

for e in range(1, epochs+1):

    batch = 0

    total_loss = 0

    model.train()

    scheduler.step()

    for x1, x2, y in train_loader:

        batch += 1

        y = y.unsqueeze(-1).cuda()

        x1, x2 = x1.cuda(), x2.cuda()

        optimizer.zero_grad()

        

        pred = model.forward(x1, x2)    

        loss = criterion(pred, y)

                

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        

        if batch % 1000 == 0:

            print(f"Epoch {e} ({batch}/{len(train_loader)}): loss {total_loss / batch}")

            

    model.eval()

    valid_loss = 0

    valid_accuracy = 0

    with torch.no_grad():

        for x1, x2, y in valid_loader:

            y = y.unsqueeze(-1).cuda()

            x1, x2 = x1.cuda(), x2.cuda()

            

            pred = model.forward(x1, x2)

            loss = criterion(pred, y)

            

            valid_loss += loss.item()

            

            correct = (pred >= 0.5).long() == y.long()

            accuracy = correct.float().mean()

        

            valid_accuracy += accuracy           

            

    print(f"Epoch {e} ({len(train_loader)}/{len(train_loader)}): loss {total_loss / len(train_loader)} - valid_loss {valid_loss / len(valid_loader)} - valid_acc {valid_accuracy / len(valid_loader)}")

    

    if valid_loss < best_loss:

        best_loss = valid_loss

        torch.save(model.state_dict(), 'model.pt')

        print("model SAVED")

# nos aseguramos de cargar el modelo que ha obtenido menor valid_loss

model.load_state_dict(torch.load('model.pt'))
test_df = pd.read_csv('../input/test.csv')

test_data = test_df.values



out = []

model.eval()

with torch.no_grad():

    for battle, first, second in test_data:

        x1 = torch.tensor(first-1).long().unsqueeze(0).cuda()

        x2 = torch.tensor(second-1).long().unsqueeze(0).cuda()

        pred = model.forward(x1, x2).item()

        

        out.append([battle, int(pred >= 0.5)])

    

out = np.array(out)



out[:10]



submission = pd.DataFrame({'battle_number':out[:,0],'Winner':out[:,1]})

submission.to_csv('submission.csv', columns=['battle_number', 'Winner'], index=False)