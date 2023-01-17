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
def grad_example(u):

    u.requires_grad_(True)

    ex = torch.exp(u)

    p = ex / ex.sum()

    log = p.log()

    plog = p * log

    res = -plog.sum()

    res.backward()

    print(u.grad)

    
grad_example(torch.tensor([1.,2.,3.]))
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
def relu(m):

    return torch.max(torch.zeros(m.shape),m)

def softmax(m):

    e = torch.exp(m/1000)

    se = torch.sum(e,1)

    divv = torch.div(e.t(),se).t()

    #print('div',divv)

    return divv
#print(label_encoder.classes_.size) #7

#print(cover_df.shape[1]) #54

dh = 500

dx = cover_df.shape[1]

nc = label_encoder.classes_.size

trains = 2000



wh = torch.randn(dh, dx)

bh = torch.randn(dh)

wh1 = torch.randn(dh, dh)

bh1 = torch.randn(dh)

wp = torch.randn(nc, dh)

bp = torch.randn(nc)



#print(wh,bh,wp,bp)



#print(torch.eq(p.max(1)[1],yy).sum())



wh.requires_grad_(True)

bh.requires_grad_(True)

wh1.requires_grad_(True)

bh1.requires_grad_(True)

wp.requires_grad_(True)

bp.requires_grad_(True)



t = 10

prev = -1



for ii in torch.arange(trains):

    

    for xx, yy in train_loader:

        #print(xx)

        #print(yy)

        break



    h = relu(torch.matmul(xx, wh.t())+bh)

    h1 = relu(torch.matmul(h, wh1.t())+bh1)

    p = softmax(torch.matmul(h1, wp.t())+bp)

    

    loss = (1/len(yy))*torch.pow((p-1)[torch.arange(len(yy)),yy],2).sum()

    if round(ii.item()*100/trains) % 10 == 0 and round(ii.item()*100/trains) > prev: 

        print(round(ii.item()*100/trains),"%")

        prev = round(ii.item()*100/trains)

        print(loss)

    loss.backward()



    



    with torch.no_grad():

        wh -= t * wh.grad

        bh -= t * bh.grad

        wh1 -= t * wh1.grad

        bh1 -= t * bh1.grad

        wp -= t * wp.grad

        bp -= t * bp.grad

        #h -= wh.grad

        #p -= p.grad

    wh.grad.zero_()

    bh.grad.zero_()

    wh1.grad.zero_()

    bh1.grad.zero_()

    wp.grad.zero_()

    bp.grad.zero_()

print(torch.eq(p.max(1)[1],yy).sum().item()/len(yy))

from sklearn.metrics import classification_report

for xx, yy in test_loader:

    break

h = relu(torch.matmul(xx, wh.t())+bh)

h1 = relu(torch.matmul(h, wh1.t())+bh1)

p = softmax(torch.matmul(h1, wp.t())+bp)

pred = p.max(1)[1]

print(classification_report(yy, pred, labels=torch.arange(7)))
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