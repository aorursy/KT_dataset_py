# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch
def grad_example(x, i):

    x.requires_grad_(True)

    ex = torch.exp(x)

    sm = ex[i] / ex.sum()

    res = sm.log()

    res.backward()

    print(x.grad)

    
grad_example(torch.tensor([1.,2.,3.]), 1)
def foo():

    x = torch.tensor([4., 17.], requires_grad=True) # Начальные значения x

    lr = 0.05 # скорость оптимизации

    for iteration in range(100):

        with torch.no_grad(): #pytorch запрещает изменять тензоры, для которых подсчитывается градиент, 

            # torch.no_grad() позволяет исполнить код, не отслеживая градиент, в т.ч. изменить значения в тензоре

            if x.grad is not None: # На первой итерации x.grad == None

                x.grad.zero_() # устанавливаем градиент в нуль (в противном случае результат следующего вычисления градиента прибавится к предыдущему)

                

        f = torch.sum(x ** 2) # Вычисляем минимизируемую функцию

        # Выводим (x.data - тензор, разделяющий память с x, но без отслеживания градиента, используем его, чтобы выводились только данные_

        # Если в тензоре только один элемент, item позволяет получить его как число

        print(x.data, f.item()) 

        # Вычисляем градиент

        f.backward()

        with torch.no_grad():

            # Делаем шаг (сдвигаем параметры в направлении, противоположном градиенту, на величину, пропорциональную частным производным и скорости обучения)

            x -= lr * x.grad

    print(x.data, f.item())
foo()
def task_2():

    X = torch.tensor([[-1 ,-1],[3, 5.1], [0, 4], [-1, 5], [3, -2], [4, 5]])

    w = torch.tensor([0.1, -0.1])

    b = torch.tensor(0.)

    y = torch.tensor([0.436, 14.0182, 7.278, 6.003, 7.478, 15.833])

    lr = 0.02

    w.requires_grad_(True)

    b.requires_grad_(True)

    for iterations in range(100):

        with torch.no_grad():

            if w.grad is not None:

                w.grad.zero_()

                b.grad.zero_()

        f = ((X.mv(w) + b - y)**2).sum()

        f.backward()

        with torch.no_grad():

            w -= lr*w.grad

            b -= lr*b.grad

        print(w, b)

    print(((X.mv(w) + b - y)**2).sum())

task_2()
from sklearn.datasets import fetch_openml
covertype = fetch_openml(data_id=180)
print(type(covertype), covertype)
cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)
print(covertype.target)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(covertype.target)
print(label_encoder.classes_) 
cover_target = label_encoder.transform(covertype.target)
print(cover_target)
print(cover_df.shape)
from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)
to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



print(columns_to_normalize)

print(idx_to_normalize)
cover_df[columns_to_normalize].sample(4)
from torch.utils.data import TensorDataset,DataLoader
tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)
print(tensor_train[:3]) #Первые три экземпляра данных
tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)
train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)
print(train_mean, train_std)
tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std
print(tensor_train[:3])
y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)
train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)
print(train_ds[400])
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)
for xx, yy in train_loader:

    print(xx[0])

    print(yy[0])

    break
from sklearn.metrics import classification_report



class Net():

    def __init__(self, t1, t2, t3):

        self.w = [torch.randn((t1, t2), device='cuda'), torch.randn((t2,t3), device='cuda')]

        self.b = [torch.randn(t2, device='cuda'), torch.randn(t3, device='cuda')]

        #self.w = [torch.randn(t1, t2), torch.randn(t2,t3)]

        #self.b = [torch.randn(t2), torch.randn(t3)]

        self.best_w = self.w.copy()

        self.best_b = self.b.copy()

        self.best_loss = None

        self.result = None

        

        

    def fit(self, train_loader, epoches, lr):

        for i in range(len(self.w)):

            self.w[i].requires_grad_(True)

            self.b[i].requires_grad_(True)        



        for epoche in range(epoches):

            batch_loss = 0

            for xx, yy in train_loader:

                xx.to(torch.device('cuda'))

                if self.w[0].grad is not None:

                    for i in range(len(self.w)):

                        self.w[i].grad.zero_()

                        self.b[i].grad.zero_()

                outh = (xx.cuda().mm(self.w[0]) + self.b[0]).relu()

                outy = (outh.cuda().mm(self.w[1]) + self.b[1]).log_softmax(1)

                loss = -outy[torch.arange(len(xx)),yy].sum()/len(xx)

                

                loss.backward()

                    

                with torch.no_grad():

                    batch_loss += loss.item()

                    for i in range(len(self.w)):

                        self.w[i] -= lr * self.w[i].grad

                        self.b[i] -= lr * self.b[i].grad

            with torch.no_grad():

                batch_loss /= len(xx)

                if self.best_loss is None or batch_loss < self.best_loss:

                    self.best_w = self.w.copy()

                    self.best_b = self.b.copy()

                    self.best_loss = batch_loss

            print("Batch loss at {} = {}".format(epoche+1, batch_loss))

        print(self.best_loss)



        

    def predict(self, test_loader):

        res = None

        with torch.no_grad():

            for xx, yy in test_loader:

                outh = (xx.cuda().mm(self.best_w[0]) + self.best_b[0]).relu()

                outy = (outh.cuda().mm(self.best_w[1]) + self.best_b[1]).log_softmax(1)

                if res is None:

                    res = outy.argmax(1)

                else:

                    res = torch.cat((res, outy.argmax(1)))

        print(classification_report(res.tolist(), y_test_tensor.tolist()))

        return res
n = Net(54, 72, 7)

lr = 0.02

epoches = 20

n.fit(train_loader, epoches, lr)
n.predict(test_loader)
wh.requires_grad_(True)

wy.requires_grad_(True)

bh.requires_grad_(True)

by.requires_grad_(True)

if best_loss is not None:

    wh = best_wh

    wy = best_wy

    bh = best_bh

    by = best_by

for i in range(20):    

    count = 1

    for xx, yy in train_loader:

        if wh.grad is not None:

            wh.grad.zero_()

            wy.grad.zero_()

            bh.grad.zero_()

            by.grad.zero_()

        

        outh = (xx.mm(wh) + bh).relu()

        outy = (outh.mm(wy) + by).log_softmax(1)

        j = -outy[torch.arange(len(xx)),yy].sum()/len(xx)

        

        j.backward()

        with torch.no_grad():

            if best_loss is None or j.item() < best_loss:

                best_wh = wh

                best_wy = wy

                best_bh = bh

                best_by = by

                best_loss = j

            #lr /= 10

            wh -= lr*wh.grad

            wy -= lr*wy.grad

            bh -= lr*bh.grad

            by -= lr*by.grad

        if(count%300 == 0):

            s = "Loss({}, {}): {}".format(i+1, count, j.item())

            print(s)

        count += 1

print(best_loss)
from sklearn.metrics import classification_report

res = None

with torch.no_grad():

    for xx, yy in test_loader:

        outh = (xx.mm(best_wh) + best_bh).relu()

        outy = (outh.mm(best_wy) + best_by).log_softmax(1)

        if res is None:

            res = outy.argmax(1)

        else:

            res = torch.cat((res, outy.argmax(1)))

    print(classification_report(res.tolist(), y_test_tensor.tolist()))
import matplotlib.pyplot as plt

import matplotlib.colors
phi = np.linspace(0, 20, 300)

r = 1 + phi



x = r * np.cos(phi)

y = r * np.sin(phi)

classes = np.zeros(300)

classes[:100] = 0

classes[100:200] = 1

classes[200:] = 2

points = np.column_stack((x + np.random.normal(scale=0.3,size = x.size), y + np.random.normal(scale=0.3,size=y.size)))

plt.plot(x,y)

plt.scatter(points[:,0], points[:,1], c=classes, cmap=matplotlib.colors.ListedColormap(['r','g','b']))
points_train, points_test, classes_train, classes_test = train_test_split(points,classes, test_size=0.2)
points_train, points_test = [torch.from_numpy(points_train).type(torch.FloatTensor), torch.from_numpy(points_test).type(torch.FloatTensor)]