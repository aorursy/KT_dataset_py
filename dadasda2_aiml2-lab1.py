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
def grad(x,y,z):

    y = y.cuda()

    z = z.cuda()

    y.requires_grad_(True)

    z.requires_grad_(True)

    f = 2 * x + (3 - torch.sin(torch.dot(y,z)))**x

    f.backward()

    print(y.grad)
grad(1,torch.tensor([1.,2.,3.]),torch.tensor([1.,2.,3.]))
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

    X = torch.tensor([[-1 ,-1],[3, 5.1], [0, 4], [-1, 5], [3, -2], [4, 5]],requires_grad = True)

    w = torch.tensor([0.1, -0.1])

    b = torch.tensor(0.)

    y = torch.tensor([0.436, 14.0182, 7.278, 6.003, 7.478, 15.833],requires_grad = True)

    lr = 0.05 



    for iteration in range(100):

        with torch.no_grad():

            if X.grad is not None: 

                X.grad.zero_()   

            if y.grad is not None: 

                y.grad.zero_()     

                

        y_hat = torch.matmul(X,w) + b        

        f = torch.sum((y_hat - y)**2)

        print(X.data, y.data, f.item()) 

        f.backward()

        with torch.no_grad():

            X -= lr * X.grad

            y -= lr * y.grad

        print('grad',X.grad,y.grad)

    print(X.data, f.item())
task_2()
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

    print(xx.size())

    print(xx)

    print(yy)

    break
from sklearn.metrics import classification_report
class net:

    

    def __init__(self,x,hid,out):

        self.hid_layer  = torch.randn(x,hid)

        self.hid_layer2 = torch.randn(hid,hid)

        self.out_layer  = torch.randn(hid,out) 

        self.b_hid      = torch.randn(hid)

        self.b_out      = torch.randn(out)



    def train(self, epoch, train_loader):

        h = 0.01

        ep = epoch

        loss_pred = 1000

        

        self.hid_layer = self.hid_layer.cuda()

        self.out_layer = self.out_layer.cuda()

        self.b_hid = self.b_hid.cuda()

        self.b_out = self.b_out.cuda()

        self.hid_layer2 = self.hid_layer2.cuda()

        

        self.hid_layer.requires_grad_(True)

        self.out_layer.requires_grad_(True)

        self.b_hid.requires_grad_(True)

        self.b_out.requires_grad_(True)

        self.hid_layer2.requires_grad_(True)

        

        for i in range(ep):

            for xx,yy in train_loader:

                xx = xx.cuda()

                yy = yy.cuda()

                

                if self.hid_layer.grad is not None:

                    self.hid_layer.grad.zero_()

                if self.out_layer.grad is not None:

                    self.out_layer.grad.zero_()

                if self.b_hid.grad is not None:

                    self.b_hid.grad.zero_()

                if self.b_out.grad is not None:

                    self.b_out.grad.zero_()

                if self.hid_layer2.grad is not None:

                    self.hid_layer2.grad.zero_()

                

                hid_out = xx.matmul(self.hid_layer) + self.b_hid

                hid_out = hid_out.relu()

                hid_out = hid_out.matmul(self.hid_layer2) + self.b_hid

                hid_out = hid_out.relu()

                out = hid_out.matmul(self.out_layer) + self.b_out

                out = out.log_softmax(dim = 1)

                

                loss = -(1/len(xx))*out[torch.arange(len(xx)),yy].sum()

                loss.backward()

                

                with torch.no_grad():

                    self.hid_layer  -= h * self.hid_layer.grad

                    self.hid_layer2 -= h * self.hid_layer2.grad

                    self.out_layer  -= h * self.out_layer.grad

                    self.b_hid      -= h * self.b_hid.grad

                    self.b_out      -= h * self.b_out.grad

            

            print(loss)

           

        

    def pred(self,x,y):

        x = x.cuda()

        y = y.cuda()

        

        hid_out = x.matmul(self.hid_layer) + self.b_hid

        hid_out = hid_out.relu()

        hid_out = hid_out.matmul(self.hid_layer2) + self.b_hid

        hid_out = hid_out.relu()

        out = hid_out.matmul(self.out_layer) + self.b_out

        out = out.log_softmax(dim = 1)

        

        pred = out.argmax(1)

        pred = pred.cpu()

        return pred
n = net(54, 1000, 7)

n.train(15, train_loader)
prediction = n.pred(tensor_test, y_test_tensor)

print(classification_report(prediction,y_test_tensor))