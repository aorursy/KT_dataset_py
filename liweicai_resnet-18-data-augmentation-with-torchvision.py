import numpy as np



train = np.loadtxt('../input/digit-recognizer/train.csv', delimiter=',', skiprows=1, dtype=np.uint8)

train_X = train[:, 1:].reshape(-1, 28, 28, 1)

train_y = train[:, 0].astype(np.int)



is_dev = np.arange(train_X.shape[0]) % 10 == 0

dev_X = train_X[is_dev]

dev_y = train_y[is_dev]

train_X = train_X[~is_dev]

train_y = train_y[~is_dev]



X_mean = train_X.astype(np.float32).mean() / 255

X_std = train_X.astype(np.float32).std() / 255



test_X = np.loadtxt('../input/digit-recognizer/test.csv', delimiter=',', skiprows=1, dtype=np.uint8)

test_X = test_X.reshape(-1, 28, 28, 1)



print(X_mean, X_std)
import torch

from torch.utils.data import TensorDataset, Dataset, DataLoader

from torchvision.transforms import Compose, ToPILImage, Pad, RandomAffine, RandomErasing, ToTensor, Normalize



class TransformDataset(Dataset):

    def __init__(self, transform, X, y=None):

        self.transform = transform

        self.X = X

        self.y = y

    

    def __len__(self):

        return self.X.shape[0]

    

    def __getitem__(self, index):

        X_item = self.transform(self.X[index])

        if self.y is None:

            return X_item,

        else:

            return X_item, self.y[index]



train_transform = Compose([

    ToPILImage(),

    Pad(2),

    RandomAffine(degrees=30, translate=(0.2, 0.2), shear=0.2),

    ToTensor(),

    Normalize((X_mean,), (X_std,)),

    RandomErasing()

])



test_transform = Compose([

    ToPILImage(),

    Pad(2),

    ToTensor(),

    Normalize((X_mean,), (X_std,))

])



train_loader = DataLoader(TransformDataset(train_transform, train_X, train_y), batch_size=100, shuffle=True)

dev_loader = DataLoader(TransformDataset(test_transform, dev_X, dev_y), batch_size=400, shuffle=False)

test_loader = DataLoader(TransformDataset(test_transform, test_X), batch_size=400, shuffle=False)
import torch.nn as nn

from torchvision.models.resnet import resnet18

        

model = resnet18(num_classes=10)

model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)

nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

model.maxpool = nn.Identity()



criterion = nn.CrossEntropyLoss()
from tqdm.autonotebook import tqdm



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train_epoch(model, criterion, optimizer, loader):

    count = 0

    total_loss = 0

    model.to(device)

    model.train()

    pbar = tqdm(loader, desc='Training', leave=False)

    for X, y in pbar:

        batch_size = X.shape[0]

        y_pred = model(X.to(device))

        loss = criterion(y_pred, y.to(device))

        pbar.set_postfix_str('Loss=%.4f' % loss.item())

        count += batch_size

        total_loss += loss.item() * batch_size

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return total_loss / count



def classification(model, loader):

    y_pred = []

    y_true = []

    evaluate = False

    model.to(device)

    model.eval()

    with torch.no_grad():

        for tensors in tqdm(loader, desc='Classification', leave=False):

            X = tensors[0]

            y_max = model(X.to(device)).argmax(dim=1).cpu()

            y_pred.append(y_max)

            if len(tensors) > 1:

                evaluate = True

                y_true.append(tensors[1])

    y_pred = torch.cat(y_pred)

    if evaluate:

        y_true = torch.cat(y_true)

        return (y_pred == y_true).type(torch.float32).mean()

    else:

        return y_pred.numpy()

max_dev_acc = 0

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=1, factor=0.1**0.5)



for epoch in tqdm(range(20), desc='Epochs'):

    train_loss = train_epoch(model, criterion, optimizer, train_loader)

    acc = classification(model, dev_loader)

    scheduler.step(acc)

    print('trainLoss=%.4f' % train_loss, 'devAcc=%.4f' % acc)

    if acc > max_dev_acc:

        max_dev_acc = acc

        torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))

y_pred = classification(model, test_loader)

with open('submission.csv', 'w') as f:

    print('ImageId,Label', file=f)

    for i, y in enumerate(y_pred):

        print(i + 1, y, sep=',', file=f)