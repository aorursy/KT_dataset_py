import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import random_split, TensorDataset, DataLoader
df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
labels_array = np.array(df.label)
features_array = df.loc[:, df.columns != 'label'].values / 255

X_train = torch.from_numpy(features_array).view(-1, 1, 28, 28).float()
targets = torch.from_numpy(labels_array).type(torch.LongTensor)
print(X_train.shape)

dataset = TensorDataset(X_train, targets)

train_size = 32000
val_size = 10000
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
for imgs, _ in train_loader:
    print('Original image shape', imgs.shape)
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    plt.imshow(make_grid(imgs, nrow=16).permute(1, 2, 0))
    break
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class Device():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
device = get_device()

train_loader = Device(train_loader, device)
val_loader = Device(val_loader, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class DigitRecognizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 10, kernel_size=3, stride=1, padding=1),
            
            nn.Flatten(),
            nn.Sigmoid()
        )
        
    
    def forward(self, xb):
        out = self.cnn(xb)
        xb = xb.view(xb.size(0), -1)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def evaluate(model, validation_loader):
    outputs = [model.validation_step(batch) for batch in validation_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, learning_rate, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), learning_rate)
        for epoch in range(epochs):
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)
        return history
model = DigitRecognizerModel()
to_device(model, device)
for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss: ', loss.item())
    break
history = [evaluate(model, val_loader)]
history
fit(100, model, train_loader, val_loader, learning_rate=0.1)
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
s_subm = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
s_subm.head()
test_features = test_df.loc[:, ].values / 255

X_test = torch.from_numpy(test_features).view(-1, 1, 28, 28).float()

X_loader = DataLoader(X_test, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

X_test = Device(X_loader, device)
def predict(test_data):
    model.eval()
    
    test_prediction = torch.LongTensor()
    
    for i, imgs in enumerate(test_data):
        output = model(imgs)
    
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_prediction = torch.cat((test_prediction, pred), dim=0)
    
    return test_prediction
preds = predict(X_test)
image_ids = np.arange(1, len(s_subm)+1)
submission_df = pd.DataFrame({
    'ImageId': image_ids,
    'Label': preds.numpy().reshape(preds.numpy().shape[0])
})
submission_df.to_csv('submission.csv', index=False)
