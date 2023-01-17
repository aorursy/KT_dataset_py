import torch
train_csv = pd.read_csv('moral_TFT_train.csv')
train_D = train_csv.drop('Ranked',axis = 1)
train_L = train_csv.Ranked
test_D = pd.read_csv('moral_TFT_test.csv')
train_D= torch.FloatTensor(np.array(train_D))
train_L = torch.LongTensor(np.array(train_L))
test_D = torch.FloatTensor(np.array(test_D))
data_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_D, train_L),
                                          batch_size=100,
                                          shuffle=True,
                                          drop_last=True)
linear1= torch.nn.Linear(train_D.shape[1], 256, bias = True)
linear2 = torch.nn.Linear(256,256,bias = True)
linear3 = torch.nn.Linear(256,256,bias = True)
linear4 = torch.nn.Linear(256,256,bias = True)
linear5 = torch.nn.Linear(256,2, bias=  True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p = 0.3)
torch.nn.init.xavier_normal(linear1.weight)
torch.nn.init.xavier_normal(linear2.weight)
torch.nn.init.xavier_normal(linear3.weight)
torch.nn.init.xavier_normal(linear4.weight)
torch.nn.init.xavier_normal(linear5.weight)
model = torch.nn.Sequential(linear1, relu, dropout, linear2, relu, dropout, linear3, relu, dropout, linear4, relu, dropout, linear5)
loss =torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
import torch.nn.functional as F
total_batch = len(data_loader)
model.train()
for e in range(15):
  avg_cost= 0
  for x, y in data_loader:
    x = x.to(device)
    y=  y.to(device)
    #print(x.shape)
    optimizer.zero_grad()
    h_x = model(x)
    cost = loss(h_x, y)
    cost.backward()
    optimizer.step()
    avg_cost += cost / total_batch
  print('Epoch {}'.format(e), 'cost {}'.format(avg_cost))
test_D = torch.FloatTensor(np.array(test_D))
with torch.no_grad():
  model.eval()
  pred=  model(test_D.to(device))
result = pd.read_csv('sample.csv')
result['id'] = list(i for i in range(3300))
result['result'] = torch.argmax(pred, 1).cpu()
result.to_csv('submission.csv', index = False)