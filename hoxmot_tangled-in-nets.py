import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
if torch.cuda.is_available():
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    print("CUDA")
else:
    device = torch.device("cpu")
    kwargs = {}
    print("CPU")
train_file = '../input/cooking_train.json'
test_file = '../input/cooking_test.json'
train_batch = 128
test_batch = 500
log_interval = 10
dataset = pd.read_json(train_file)
testset = pd.read_json(test_file)
def preproc(line):
    return ' '.join(line).lower()
vect_x = CountVectorizer(preprocessor=preproc)
vect_x.fit(dataset.ingredients)
data_x = vect_x.transform(dataset.ingredients).todense()
data_x.shape
def elem_wrap(sthg):
    return sthg.tolist()[0].index(1)
vect_y = CountVectorizer()
data_y = vect_y.fit_transform(dataset.cuisine.values).todense()
data_y = np.apply_along_axis(elem_wrap, 1, data_y)

cuisine = sorted(set(dataset.cuisine))
len(cuisine)
def smart_elem_wrap(sthg):
    return cuisine.index(sthg)

vectorize_elem_wrap = np.vectorize(smart_elem_wrap)
data_y2 = vectorize_elem_wrap(dataset.cuisine.values)
data_y2 == data_y
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
class Net1_1(nn.Module):
    def __init__(self):
        super(Net1_1, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(2866, 5932)
        self.fc2 = nn.Linear(5932, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 1000)
        self.fc4 = nn.Linear(1000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(2866, 5732)
        self.fc2 = nn.Linear(5732, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(2866, 6000)
        self.fc2 = nn.Linear(6000, 5000)
        self.fc3 = nn.Linear(5000, 4000)
        self.fc4 = nn.Linear(4000, 2000)
        self.fc5 = nn.Linear(2000, 500)
        self.fc6 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)
def train(log_interval, model, device, train_loader, optimizer, epoch, verbose=False):
    model.train()
    dataset_len = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and verbose:
            print('Train Epoch:', epoch)
            print('\t{}/{}  -  {:.0f}%'.format(batch_idx * len(data), dataset_len,
                                               100 * batch_idx / len(train_loader)))
            print('\tLoss: {:.4f}'.format(loss.item()))
def test(model, device, test_loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    dataset_len = len(test_loader.dataset)
    test_loss /= dataset_len
    if verbose:
        print("Testing:")
        print("\tLoss: {:.4f}".format(test_loss))
        print("\tAccuracy: {}/{}  -  {:.3f}%".format(correct, dataset_len,
                                               100 * correct / dataset_len))
    return test_loss, correct
def conduct_experiment(model, device, epochs, log_interval, train_loader, optimizer, verification_loader):
    train_loss = []
    verify_loss = []
    train_scored = []
    verify_scored = []

    for ep in range(epochs):
        train(log_interval, model, device, train_loader, optimizer, ep + 1)
        ts, t = test(model, device, train_loader)
        vl, c = test(model, device, verification_loader)
        train_loss.append(ts)
        verify_loss.append(vl)
        train_scored.append(t)
        verify_scored.append(c)

    print("Trainset:")
    test(model, device, train_loader, True)
    print("Verifyset:")
    test(model, device, verification_loader, True)
    plt.plot(train_loss)
    plt.title('Train loss')
    plt.show()
    plt.plot(verify_loss)
    plt.title('Verify loss')
    plt.show()
    plt.plot(train_scored)
    plt.title('Train scored')
    plt.show()
    plt.plot(verify_scored)
    plt.title('Verify scored')
    plt.show()
train_x, verify_x, train_y, verify_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
test_x = vect_x.transform(testset.ingredients).todense()
ids = testset.id.values
class FoodTrain(torchdata.Dataset):
    def __init__(self, ingredients, cuisine):
        self.ingredients = torch.tensor(ingredients, dtype=torch.float32)
        self.cuisine = torch.tensor(cuisine, dtype=torch.int64)

    def __len__(self):
        return len(self.ingredients)

    def __getitem__(self, index):
        return self.ingredients[index], self.cuisine[index]
class FoodTest(torchdata.Dataset):
    def __init__(self, dataset, idx):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.index = torch.tensor(idx, dtype=torch.int32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.index[index]
train_loader = torchdata.DataLoader(FoodTrain(train_x, train_y), batch_size=train_batch, shuffle=True, **kwargs)
verification_loader = torchdata.DataLoader(FoodTrain(verify_x, verify_y), batch_size=test_batch, shuffle=True, **kwargs)
test_loader = torchdata.DataLoader(FoodTest(test_x, ids), batch_size=test_batch, shuffle=False, **kwargs)
torch.manual_seed(2137)
model1 = Net1().to(device)
optimizer = optim.Adagrad(model1.parameters(), lr=0.01)
# 10 epochs in final submission
conduct_experiment(model1, device, 10, log_interval, train_loader, optimizer, verification_loader)
torch.manual_seed(42)
model1_1 = Net1_1().to(device)
optimizer = optim.Adagrad(model1_1.parameters(), lr=0.01)
# 7 epochs in final submission
conduct_experiment(model1_1, device, 10, log_interval, train_loader, optimizer, verification_loader)
model2 = Net2().to(device)
optimizer = optim.Adagrad(model2.parameters(), lr=0.01)
# 4 epochs in final submission
conduct_experiment(model2, device, 10, log_interval, train_loader, optimizer, verification_loader)
model4 = Net4().to(device)
optimizer = optim.Adagrad(model4.parameters(), lr=0.01)
# 8 epochs in final submission
conduct_experiment(model4, device, 10, log_interval, train_loader, optimizer, verification_loader)
model5 = Net5().to(device)
optimizer = optim.Adagrad(model5.parameters(), lr=0.01)
# 5 epochs in final submission
conduct_experiment(model5, device, 10, log_interval, train_loader, optimizer, verification_loader)
model3 = Net3().to(device)
optimizer = optim.Adagrad(model3.parameters(), lr=0.01)
conduct_experiment(model3, device, 10, log_interval, train_loader, optimizer, verification_loader)
model1.eval()
model1_1.eval()
model2.eval()
model4.eval()
model5.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data_batch, target in verification_loader:
        data_batch, target = data_batch.to(device), target.to(device)
        output1 = model1(data_batch)
        output1_1 = model1_1(data_batch)
        output2 = model2(data_batch)
        output4 = model4(data_batch)
        output5 = model5(data_batch)
        outputs = torch.stack((output1, output1_1, output2, output4, output5))
        output = torch.mean(outputs, 0)
        test_loss += F.nll_loss(output, target, reduction='sum')
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

dataset_len = len(verification_loader.dataset)
test_loss /= dataset_len
print("Average loss: {:.4f}".format(test_loss))
print("Accuracy: {}/{}  -  {:.3f}%".format(correct, dataset_len, 100 * correct / dataset_len))
result = pd.DataFrame({'Id': [], 'cuisine': []}, dtype=np.int32)
with torch.no_grad():
    for data_batch, idx in test_loader:
        data_batch = data_batch.to(device)
        output1 = model1(data_batch)
        output1_1 = model1_1(data_batch)
        output2 = model2(data_batch)
        output4 = model4(data_batch)
        output5 = model5(data_batch)
        outputs = torch.stack((output1, output1_1, output2, output4, output5))
        output = torch.mean(outputs, 0)
        pred = output.argmax(dim=1)
        result = pd.concat([result, pd.DataFrame({'Id': idx.cpu(), 'cuisine': pred.cpu()}, dtype=np.int32)])
def choose(sthg):
    return cuisine[sthg]
result['cuisine'] = result['cuisine'].apply(choose, 0)