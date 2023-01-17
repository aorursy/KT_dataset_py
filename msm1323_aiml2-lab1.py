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
def grad_fun_7(x, y):

    x.requires_grad_(True)

    y.requires_grad_(True)

    scal_sum = (x.dot(y)).sum()

    res = scal_sum/(x.norm()*y.norm())

    res.backward()

    print(x.grad)

    print(y.grad)

    
grad_fun_7(torch.tensor([1.,2.,3.]), torch.tensor([4.,5.,6.]))
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

    y_hat = torch.matmul(X,w) + b # или torch.sum(X*w,dim=1) + b

    print(y_hat)
task_2()
P = torch.tensor([[0.1955, 0.0311, 0.7734],

 [0.5093, 0.174, 0.3167],

 [0.1309, 0.061, 0.8081],

 [0.3826, 0.5054, 0.112]])



c = torch.tensor([2,0,1,0], dtype=torch.long) #С учетом индексов, начинающихся с 0

# Вектор от 0 до заданного числа

print('torch.arange(4): ', torch.arange(4))

# Мы можем индексировать тензоры списками или тензорами из целых чисел. Например tensor([5,9,1,2])[[0,3,0]] == tensor([5,2,5])

# В случае с матрицами, мы можем индексировать их двумя списками индексов одинакового размера

print('P[torch.arange(4), c]: ', P[torch.arange(4), c])

L = -torch.mean(torch.log(P[torch.arange(4), c]))

print(L.item())
from sklearn.datasets import fetch_openml
covertype = fetch_openml(data_id=180)
print(type(covertype), covertype)
cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)
cover_df.sample(10)
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
train_loader = DataLoader(train_ds,batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)
for xx, yy in train_loader:

    print(xx)

    print(yy)

    break
class NN:

    

    def __init__(self, dim_xIn, dim_h, num, dim_yOut):

        self.dim_h = dim_h

        self.hid_lrs = []

        self.hid_bs = []

        self.hid_lrs.append(torch.randn(dim_xIn, dim_h))

        self.hid_bs.append(torch.randn(dim_h))

        self.out_lr = torch.randn(dim_h, dim_yOut)

        self.out_b = torch.randn(dim_yOut)

        

    def forward_pass(self, cur):

        for i in range(len(self.hid_lrs)):

            cur = cur.matmul(self.hid_lrs[i]) + self.hid_bs[i]  

            cur = cur.relu()

        return(cur)

        

    def pre_fit(self, train_loader):

        self.hid_lrs[0].requires_grad_(True)

        self.hid_bs[0].requires_grad_(True)

        for i in range(len(self.hid_lrs)-1):

            hid_lr = torch.randn(self.dim_h, self.dim_h).requires_grad_(True)

            hid_bs = torch.randn(self.dim_h).requires_grad_(True)

            self.hid_lrs.append(hid_lr)

            self.hid_bs.append(hid_bs)

        self.out_lr.requires_grad_(True)

        self.out_b.requires_grad_(True)

        

    def fit(self, num, train_loader):

        self.h = 0.01

        self.era = num

        for e in range(self.era):

            for xx, yy in train_loader:

                if self.hid_lrs[0].grad is not None:

                    for i in range(len(self.hid_lrs)):

                        self.hid_lrs[i].grad.zero_()

                if self.hid_bs[0].grad is not None:

                    for i in range(len(self.hid_bs)):

                        self.hid_bs[i].grad.zero_()      

                if self.out_lr.grad is not None:

                    self.out_lr.grad.zero_()

                if self.out_b.grad is not None:

                    self.out_b.grad.zero_()

                hid_out = self.forward_pass(xx)

                out_lr = hid_out.matmul(self.out_lr) + self.out_b 

                out_lr = out_lr.log_softmax(dim = 1)

                loss_f = -(1/len(xx))*out_lr[torch.arange(len(xx)),yy].sum()

                loss_f.backward()

                with torch.no_grad():

                    for i in range(len(self.hid_lrs)):

                        self.hid_lrs[i] -= self.h*self.hid_lrs[i].grad

                        self.hid_lrs[i] -= self.h*self.hid_lrs[i].grad

                    self.out_lr -= self.h*self.out_lr.grad

                    self.out_b -= self.h*self.out_b.grad

            print(loss_f)

            

    def predict(self, xx, yy):

        hid_out = self.forward_pass(xx)

        out_lr = hid_out.matmul(self.out_lr) + self.out_b

        out_lr = out_lr.log_softmax(dim = 1)

        from sklearn.metrics import classification_report

        r = out_lr.argmax(1)

        print(classification_report(r, yy))
nn = NN(54, 10, 10, 7)

nn.pre_fit(train_loader)

nn.fit(800, train_loader)
nn.predict(tensor_test, y_test_tensor)
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