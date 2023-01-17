import copy
import cv2
import numpy as np 
import os
import pandas as pd 
import traceback
import zipfile


from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
print(os.listdir('data'))
print(len(os.listdir('/kaggle/working/data/train')))
print(os.listdir('/kaggle/working/data/train')[:10])
class DataPreparation():
    def __init__(self, img_size=64, data_path='/data'):
        self.img_size=img_size
        self.data_path = data_path
        self.labels = {'cat': 0, 'dog': 1}
        self.training_data =[]
        self.cat_cnt = 0
        self.dog_cnt = 0
        
    def prepare_train_data(self):
        for f in tqdm(os.listdir(self.data_path)):
            try:
                file_path = os.path.join(self.data_path, f)
                label = self.labels[f[:f.index('.')]]
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.training_data.append([np.array(img), label])
                
                if label == 0:
                    self.cat_cnt += 1
                elif label == 1:
                    self.dog_cnt += 1
                    
            except Exception as e:
                pass
                # print(e)
                
            
        np.random.shuffle(self.training_data)
        print('Cats: {}'.format(self.cat_cnt))
        print('Dogs: {}'.format(self.dog_cnt))
            
            
cats_and_dogs = DataPreparation(img_size=128, data_path='/kaggle/working/data/train')
cats_and_dogs.prepare_train_data()
X = torch.Tensor([i[0] for i in cats_and_dogs.training_data]).view(-1, 1, 128, 128)

X_max = X.max() # to re-use with new data
X /= X_max
y = torch.Tensor([i[1] for i in cats_and_dogs.training_data])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y)

del cats_and_dogs
import matplotlib.pyplot as plt

plt.imshow(X[12664].view(128, 128), cmap="gray")
print(y[12664])
class nnDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index].long()
        
    def __len__ (self):
        return len(self.X_data)
    

train = nnDataset(X_train, y_train)
test = nnDataset(X_test, y_test)

train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=1)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 слой
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=5,
                                     padding=2)
        
        # 2 слой
        self.conv2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=5,
                                     padding=2)
        
        # 3 слой
        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=5,
                                     padding=2)
        
        # 4 слой
        self.conv4 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1)
        
        self.fc1 = torch.nn.Linear(8*8*128, 512)
        self.fc2 = torch.nn.Linear(512, 128)        
        self.fc3 = torch.nn.Linear(128, 32)        
        self.fc4 = torch.nn.Linear(32, 2)
        
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)
       
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2,
                                        stride=2)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(512)
        self.batch_norm2 = torch.nn.BatchNorm1d(128)
        
        self.dropout50 =  torch.nn.Dropout(p=0.5)
        self.dropout25 =  torch.nn.Dropout(p=0.25)
        
        self.act = torch.nn.ReLU()

        
        

    def forward(self, x):
        x = self.conv1(x) # 128 x 128
        x = self.max_pool(x) # 64 x 64
        x = self.dropout25(x)
        x = self.act(x)
        
        
        x = self.conv2(x)
        x = self.max_pool(x) # 32 x 32
        x = self.dropout25(x)
        x = self.act(x)
        
         
        x = self.conv3(x)
        x = self.max_pool(x) # 16 x 16
        x = self.dropout25(x)
        x = self.act(x)
        
        x = self.conv4(x)
        x = self.max_pool(x) # 8 x 8
        x = self.dropout25(x)
        x = self.act(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x) # 8 x 8 x 128
        x = self.batch_norm1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.dropout25(x)
        x = self.act(x)
        x = self.fc4(x)
        
        return F.softmax(x, dim=1) # torch.sigmoid(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = binaryClassification(n_features=64*64, n_hidden_neurons=64, dropout_ratio=0.1)
model = Net()
model.to(device)

criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train_eval_nn(model,
                  train_dataloader, test_dataloader,
                  loss_function, optimizer,
                  n_epochs, early_stopping_patience,
                  lr=1e-3, l2_reg_alpha=0,
                  scheduler=None, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)
    
    if scheduler:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
        lr_scheduler = scheduler(optimizer)
    else:
        lr_scheduler = None

    
    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(n_epochs):
        try:
            model.train()
            mean_train_loss = 0
            train_batches_n = 0


            for idx, (batch_x, batch_y) in enumerate(train_dataloader):
                if idx > 10000:
                    break
                optimizer.zero_grad()

                x_batch = batch_x.to(device)
                y_batch = batch_y.to(device)
                preds = model(x_batch)
                loss_val = loss_function(preds, y_batch)
                # optimizer.zero_grad()

                loss_val.backward()
                optimizer.step()

                mean_train_loss += float(loss_val)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Среднее значение функции потерь на обучении', mean_train_loss)


            model.eval()               
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for idx, (batch_x, batch_y) in enumerate(test_dataloader):
                    if idx > 1000:
                        break

                    x_batch = batch_x.to(device)
                    y_batch = batch_y.to(device)

                    preds = model.forward(x_batch)
                    loss_val = loss_function(preds, y_batch)

                    mean_val_loss += float(loss_val)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            
            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
                
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                early_stopping_patience))
                break
            
            if lr_scheduler:
                lr_scheduler.step(mean_val_loss)
                
            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
                
        except Exception as ex:
            print('Ошибка при обучении {}\n{}'.format(ex, traceback.format_exc()))
            break
            
    return best_val_loss, best_model



best_val_loss, best_model = train_eval_nn(model=model, device=device,
              train_dataloader=train_loader,
              test_dataloader=test_loader,
              loss_function=criterion, optimizer=optimizer,
              n_epochs=500, early_stopping_patience=20,
               lr=1e-3, l2_reg_alpha=0,
              scheduler=lambda optim: torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1))
def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = batch_x.to(device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)
    
    
test_pred = predict_with_model(best_model, test, return_labels=True)


print('Accuracy:', accuracy_score(y_test, np.argmax(test_pred[0], axis=1)))
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
img = cv2.imread('/kaggle/input/mars-the-doggie/photo_2020-05-02_20-06-38.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (self.img_size, self.img_size))

plt.imshow(img, cmap="gray");
img = cv2.resize(img, (128, 128))

mars = torch.Tensor(img).view(-1, 1, 128, 128)
mars /= X_max

pred = best_model.forward(mars.to(device)).detach().cpu().numpy()
pred_label = np.argmax(pred, axis=1)
pred_label = 'Dog' if pred_label==1 else 'Cat'

print('Mars is {}'.format(pred_label))
img = cv2.imread('/kaggle/input/ivanych/photo_2020-05-02_20-46-12.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (self.img_size, self.img_size))

plt.imshow(img, cmap="gray");
img = cv2.resize(img, (128, 128))

dzhoy = torch.Tensor(img).view(-1, 1, 128, 128)
dzhoy /= X_max


pred = best_model.forward(dzhoy.to(device)).detach().cpu().numpy()
pred_label = np.argmax(pred, axis=1)
pred_label = 'Dog' if pred_label==1 else 'Cat'

print('Dzhoy is {}'.format(pred_label))
img = cv2.imread('/kaggle/input/marik/photo_2020-05-02_20-53-04.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (self.img_size, self.img_size))

plt.imshow(img, cmap="gray");
img = cv2.resize(img, (128, 128))

marik = torch.Tensor(img).view(-1, 1, 128, 128)
marik /= X_max


pred = best_model.forward(marik.to(device)).detach().cpu().numpy()
pred_label = np.argmax(pred, axis=1)
pred_label = 'Dog' if pred_label==1 else 'Cat'

print('Marik is {}'.format(pred_label))
img = cv2.imread('/kaggle/input/bulldog/photo_2020-05-02_21-08-43.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (self.img_size, self.img_size))

plt.imshow(img, cmap="gray");
img = cv2.resize(img, (128, 128))

bulldog = torch.Tensor(img).view(-1, 1, 128, 128)
bulldog /= X_max

pred = best_model.forward(bulldog.to(device)).detach().cpu().numpy()
pred_label = np.argmax(pred, axis=1)
pred_label = 'Dog' if pred_label==1 else 'Cat'

print('This is {}'.format(pred_label))
img = cv2.imread('/kaggle/input/big-cat/photo_2020-05-02_21-34-45.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (self.img_size, self.img_size))

plt.imshow(img, cmap="gray");
img = cv2.resize(img, (128, 128))

cat = torch.Tensor(img).view(-1, 1, 128, 128)
cat /= X_max

pred = best_model.forward(cat.to(device)).detach().cpu().numpy()
pred_label = np.argmax(pred, axis=1)
pred_label = 'Dog' if pred_label==1 else 'Cat'

print('This is {}'.format(pred_label))
