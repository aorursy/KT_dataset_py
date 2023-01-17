import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix

import numpy as np
import re

import matplotlib.pyplot as plt
%matplotlib inline

import copy
import collections

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(13)
np.random.seed(13)
# подгрузим данные

train_source = fetch_20newsgroups(subset='train')
test_source = fetch_20newsgroups(subset='test')


# векторизация
TOKEN_RE = re.compile(r'[\w\d]+')

def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, tokenizer=tokenize_text_simple_regex, max_df=0.8, min_df=5)
x_train = vectorizer.fit_transform(train_source['data']) 
x_test = vectorizer.transform(test_source['data'])
class SparseFeaturesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label


    
class simpleNet(torch.nn.Module):
    def __init__(self, vocabulary_size, n_hidden_neurons, n_classes):
        super(simpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(vocabulary_size, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, n_classes)
        self.sm = torch.nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    
    def inference(self,x):
        x = self.forward(x)
        x = self.sm(x)
        return x

def train_eval_nn(model, device,
                  train_dataloader, test_dataloader,
                  loss_function, optimizer,
                  n_epochs, early_stopping_patience):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)
    
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
                preds = model.forward(x_batch)

                loss_val = loss(preds, y_batch)
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
                    loss_val = loss(preds, y_batch)

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
                
            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
                
        except Exception as ex:
            print('Ошибка при обучении {}\n{}'.format(ex, traceback.format_exc()))
            break
            
    return best_val_loss, best_model
    
if __name__ == '__main__':  
    train_dataset = SparseFeaturesDataset(x_train, train_source['target'])
    test_dataset = SparseFeaturesDataset(x_test, test_source['target'])    


    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # видеокарта, если доступна
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # создаем экземпляр сети
    simple_net = simpleNet(vocabulary_size=len(vectorizer.vocabulary_),
                           n_hidden_neurons=200,
                           n_classes=20
                          )

    # задаем функцию потерь и оптимизатор
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_net.parameters(), lr=1e-3)


    best_val_loss, best_model = train_eval_nn(model=simple_net, device=device,
                                              train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                              loss_function=loss, optimizer=optimizer,
                                              n_epochs=500, early_stopping_patience=10)
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
    
    
test_pred = predict_with_model(best_model, test_dataset)
print('Доля верных ответов', accuracy_score(test_source['target'], test_pred.argmax(-1)))