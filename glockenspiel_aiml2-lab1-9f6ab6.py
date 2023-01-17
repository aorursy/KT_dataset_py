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
def grad_example(w, x):

    w.requires_grad_(True)

    res = 1 / ( 1 + torch.exp(-w * x) )

    res.backward(torch.tensor([1.,1.,1.]))

    print(w.grad)

    x.requires_grad_(True)

    res = 1 / ( 1 + torch.exp(-w * x) )

    res.backward(torch.tensor([1.,1.,1.]))

    print(x.grad)

    
grad_example(torch.tensor([1.,-1.,-1.]), torch.tensor([1.,2.,3.]))
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
import torch 

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
import pandas as pd

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

#https://www.kaggle.com/nilanml/detecting-sarcasm-using-different-embeddings

print(train_ds[400])
train_loader = DataLoader(train_ds,batch_size=2000, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=2000)

from sklearn.model_selection import train_test_split

import numpy as np

import random as rd 

class Net: # V3

    def __init__(self, layers=None, x=None,y=None,nonlin = None):

        # Выбор функции активации

        if nonlin == 'tanh':

            self.nonlin = self.tanh

        elif nonlin == 'sigmoid':

            self.nonlin = self.sigmoid

        x1,x2,y1,y2 = train_test_split(x.numpy(),y.numpy(),test_size=0.33,random_state=rd.randint(0,100))

        self.X_train = x1

        self.X_test = x2

        self.y_train = y1

        self.y_test = y2

        # В сорцах лежат все данные о сети

        self.output_1 = np.array([[0 for u in range(7)] for ou in range(len(self.y_train))])

        self.output_2 = np.array([[0 for u in range(7)] for ou in range(len(self.y_test))])

        

        for n in range(len(self.y_train)):

            self.output_1[n][self.y_train[n]] = 1

        for n in range(len(self.y_test)):

            self.output_2[n][self.y_test[n]] = 1                                                 

        

        self.Source = {'synapses' : [],

                       'nonlin' : nonlin,

                       'width' : np.random.choice([0.5,1,1.25,1.5,1.75,2]), # Наблюдение, лучшая сходимость на 1

                       'speed' : [],

                       'momentum' : [],

                       'learn_samples' : {'IN' : self.X_train, 'OUT' : self.output_1},

                       'test_samples' : {'IN' : self.X_test, 'OUT' : self.output_2}}



        # Инициализация весов

        if type(layers) != type(None):

            for i in range(1, len(layers)):

                self.Source['synapses'].append(self.Source['width']*2*np.random.random((layers[i-1],layers[i])) - 1*self.Source['width'])

                self.Source['speed'].append(np.random.random((layers[i-1],layers[i])) * 0.001)

                self.Source['momentum'].append(np.random.random((layers[i-1],layers[i]))*3)

    

    def load_setting(self, filename):

        try:

            with open(filename, 'rb') as f:

                data = dill.loads(f.read())

                self.Source = data

                # Подгрузка акт функции

                nonlin = self.Source['nonlin']

                if nonlin == 'tanh':

                    self.nonlin = self.tanh

                elif nonlin == 'sigmoid':

                    self.nonlin = self.sigmoid

        except Exception as e:

            print('Данные сети не были загружены', e)



    def save_setting(self, filename):

        with open(filename, 'wb') as f:

            f.write(dill.dumps(self.Source))

        print('Настройки сети сохранены')





    def run(self, IN, rnd=1):

        L = [IN]

        for i in range(len(self.Source['synapses'])):#-1

            L.append(self.nonlin(np.dot(L[i],self.Source['synapses'][i])))

        if rnd == 0:

            return L[-1]

        else: # вывод с округлением

            return np.round(L[-1], rnd)

    

    def show_error_percent2(self):

        result_learn = self.run(self.Source['learn_samples']['IN'], rnd=0)

        result_test = self.run(self.Source['test_samples']['IN'], rnd=0)

        learn_OUT = self.Source['learn_samples']['OUT']

        test_OUT = self.Source['test_samples']['OUT']

        error_learn = np.mean(np.abs(result_learn - learn_OUT))

        error_test = np.mean(np.abs(result_test - test_OUT))

        return [error_learn, error_test]



    def show_error_percent(self):

        '''Return percent learn, test, updateIN, updateOUT'''

        result_learn = self.run(self.Source['learn_samples']['IN'], rnd=0)

        result_test = self.run(self.Source['test_samples']['IN'], rnd=0)

        len_OUT_learn = len(self.Source['learn_samples']['OUT'])

        len_OUT_test = len(self.Source['test_samples']['OUT'])

        learn_OUT = self.Source['learn_samples']['OUT']

        learn_IN = self.Source['learn_samples']['IN']

        test_OUT = self.Source['test_samples']['OUT']

        test_IN = self.Source['test_samples']['IN']

        count_good_learn = 0

        count_good_test = 0



        # Урезанная выборка (медот обучения на неправильных примерах (ускорение))

        updateIN = []

        updateOUT = []       

        # Считаем попадания на учебке

        #print(learn_OUT[0],result_learn[0])

        for i in range(len_OUT_learn):

            if (learn_OUT[i].argmax() == result_learn[i].argmax()):

                count_good_learn += 1

            



        

        # Считаем попадания на тесте

        for i in range(len_OUT_test):

            if (test_OUT[i].argmax() == result_test[i].argmax()):

                count_good_test += 1

            #if (test_OUT[i] > 0 and result_test[i] > 0.5) or (test_OUT[i] < 0 and result_test[i] < -0.5):

                #count_good_test += 1

        





        return [round(100*count_good_learn/len_OUT_learn), round(100*count_good_test/len_OUT_test)]



    def train(self, steps, show_print=0):

        speed = self.Source['speed']

        momentum = self.Source['momentum']

        learn_OUT = self.Source['learn_samples']['OUT']

        learn_IN = self.Source['learn_samples']['IN']

        last_update = self.Source['synapses'].copy()

        

        

        



        for i in range(len(last_update)):

            last_update[i] = last_update[i] * 0



        for j in range(steps):



            # Прямой проход

            L = [learn_IN]

            for i in range(len(self.Source['synapses'])):#-1

                L.append(self.nonlin(np.dot(L[i],self.Source['synapses'][i]), deriv=False))

            # Расчет ошибок

            error = learn_OUT - L[-1]

            Errors_delta = [0 for x in range(len(L)-1)]

            for i in reversed(range(len(self.Source['synapses']))):

                if i == len(self.Source['synapses'])-1:

                    error = learn_OUT - L[-1]

                    delta = error*self.nonlin(L[-1], deriv=True)

                    Errors_delta[-1] = delta



                else:

                    error = Errors_delta[i+1].dot(self.Source['synapses'][i+1].T)

                    delta = error*self.nonlin(L[i+1], deriv=True)

                    Errors_delta[i] = delta



            # Обновление весов

            for i in range(len(self.Source['synapses'])):

                add = L[i].T.dot(Errors_delta[i]) * speed[i] + last_update[i] * momentum[i]

                self.Source['synapses'][i] += add

                last_update[i] = L[i].T.dot(Errors_delta[i]) * speed[i]

            '''Конец основной части'''





             # Проверка и апдейт примеров (урезание)



            if show_print == 1 and (j%100) == 0:

                error_train_test = self.show_error_percent()

                print('Эпоха - {}, Правильный процент ответов на учебном наборе - {}, на тестовом - {}'.format(j,error_train_test[0],error_train_test[1]))

                

                      

    def nonlin(self, x, deriv=False):

        if(deriv==True):

            f = 1.0 / (1.0 + np.exp(-x))

            return f * (1.0 - f)

        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid(self, x, deriv=False):

        if(deriv==True):

            f = 1.0 / (1.0 + np.exp(-x))

            return f * (1.0 - f)

        return 1.0 / (1.0 + np.exp(-x))



    def tanh(self, x, deriv=False):

        if(deriv==True):

            return (1+x)*(1-x)

        return np.tanh(x)

    






for xx, yy in train_loader:

    net = Net([54,50,50,50,7],xx,yy)

    net.train(2000,1)

    break

    



    

    
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