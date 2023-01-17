import torch

from torchvision import datasets, transforms
batch_size = 32

learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(

    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),

    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(

    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),

    batch_size=batch_size)
for data, target in train_loader:

    print(data.shape, target)

    break
x = data.reshape(batch_size, 28*28)

print(x.shape)

    

w = torch.randn(28*28, 10, requires_grad=True) # теперь не 10х784, а 784х10

b = torch.randn(10, requires_grad=True)



# w2 = torch.randn(10, 10, requires_grad=True) # теперь не 10х784, а 784х10

# b2 = 



print(w.shape)

print(b.shape)
b
a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

a_.shape
y_ = 1 / (1 + torch.exp(- (a_)))
y_.argmax()

y_.shape
def oneHot(a, oneHot_batch_size):



    out = []

    for i in a:



        b = torch.zeros(10)

        b[i] = 1.

        out.append(b)

    out = torch.cat(out)

        

    out = out.reshape(oneHot_batch_size, 10)



    return out
y = oneHot(target, batch_size)

y.shape
BCE = torch.nn.BCELoss()

loss = BCE(y_, y)



print(loss, type(loss), loss.requires_grad)
%%time





optimizer = torch.optim.Adam([w, b], lr=learning_rate)



print(optimizer, type(optimizer))
%%time



# Learning Loop 

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

        y_ = 1 / (1 + torch.exp(- (a_)))

        

        loss = BCE(y_, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')

        

cnt = 0



for data, target in test_loader:

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    x = data.reshape(local_batch_size, 28*28)

    y = oneHot(target, local_batch_size)

    a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

    y_ = 1 / (1 + torch.exp(- (a_)))

    for i in range(local_batch_size):

        if y_[i].argmax() == target[i]:

            cnt+=1



print("accuracy", cnt/10000)

train_loader = torch.utils.data.DataLoader(

    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),

    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(

    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),

    batch_size=batch_size)





#Переопределяем веса, указываем количество нейронов

w1 = torch.randn(28*28, 32, requires_grad=True) # теперь не 784х10, а связи к промежуточному слою 784х32 

b1 = torch.randn(32, requires_grad=True)



w2 = torch.randn(32, 10, requires_grad=True) # связи от промежуточного слоя (32 нейрона) к финальным 10 нейронам  32х10

b2 = torch.randn(10, requires_grad=True)

optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=learning_rate)

%%time



# Learning Loop 784х32х10

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

        y_1 = 1 / (1 + torch.exp(- (a_1)))

#         print(x.shape)

        a_2 = torch.mm(y_1, w2/10) + b2/10

        y_2 = 1 / (1 + torch.exp(- (a_2)))

        

        loss = BCE(y_2, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')

        

cnt = 0



for data, target in test_loader:

    

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    

    x = data.reshape(local_batch_size, 28*28)



    a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

    y_1 = 1 / (1 + torch.exp(- (a_1)))

    #         print(x.shape)

    a_2 = torch.mm(y_1, w2/10) + b2/10

    y_2 = 1 / (1 + torch.exp(- (a_2)))

    

    for i in range(local_batch_size):

        if y_2[i].argmax() == target[i]:

            cnt+=1

            

print("accuracy", cnt/10000)
%%time

#Переопределяем веса, указываем количество нейронов (784х10)

w = torch.randn(28*28, 10, requires_grad=True) # теперь не 10х784, а 784х10

b = torch.randn(10, requires_grad=True)

optimizer = torch.optim.Adam([w, b], lr=learning_rate)





# Learning Loop 784х32х10





# Learning Loop (784х10)

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

        y_ = 1 / (1 + torch.exp(- (a_)))

        

        loss = BCE(y_, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')



#Проверка точности

cnt = 0



for data, target in test_loader:

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    x = data.reshape(local_batch_size, 28*28)

    y = oneHot(target, local_batch_size)

    a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

    y_ = 1 / (1 + torch.exp(- (a_)))

    for i in range(local_batch_size):

        if y_[i].argmax() == target[i]:

            cnt+=1





            

print("accuracy", cnt/10000)
train_loader = torch.utils.data.DataLoader(

    datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()),

    batch_size=1, shuffle=True)



test_loader = torch.utils.data.DataLoader(

    datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),

    batch_size=1)
import matplotlib.pyplot as plt
iii = 0

for data, target in train_loader:

#     print(data.shape)

    plt.imshow(data.reshape(28,28))

    plt.show() # plot first 10 digits examples

    iii += 1

    if iii == 2:

        break

    

train_loader = torch.utils.data.DataLoader(

    datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()),

    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(

    datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),

    batch_size=batch_size)



for data, target in train_loader:

    print(data.shape, target)

    break

x = data.reshape(batch_size, 28*28)

print(x.shape)
#Переопределяем веса, указываем количество нейронов 784 - 64 - 32 - 10

w1 = torch.randn(28*28, 64, requires_grad=True) # теперь не 784х10, а связи к первому промежуточному слою 784х64

b1 = torch.randn(64, requires_grad=True)



w2 = torch.randn(64,32, requires_grad=True) # связи первого промежуточного слоя (64 нейрона) и второго промежуточного слоя (32 нейрона)

b2 = torch.randn(32, requires_grad=True)



w3 = torch.randn(32, 10, requires_grad=True) # связи от второго промежуточного слоя (32 нейрона) к финальным 10 нейронам  32х10

b3 = torch.randn(10, requires_grad=True)



optimizer = torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr=learning_rate)

# Learning Loop 784 - 64 - 32 - 10

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

        y_1 = 1 / (1 + torch.exp(- (a_1)))

#         print(x.shape)

        a_2 = torch.mm(y_1, w2/10) + b2/10

        y_2 = 1 / (1 + torch.exp(- (a_2)))

        

        a_3 = torch.mm(y_2, w3/10) + b3/10

        y_3 = 1 / (1 + torch.exp(- (a_3)))

        

        loss = BCE(y_3, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')
cnt = 0



for data, target in test_loader:

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    

    

    x = data.reshape(local_batch_size, 28*28)

#     y = oneHot(target, local_batch_size)



    a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

    y_1 = 1 / (1 + torch.exp(- (a_1)))

    #         print(x.shape)

    a_2 = torch.mm(y_1, w2/10) + b2/10

    y_2 = 1 / (1 + torch.exp(- (a_2)))



    a_3 = torch.mm(y_2, w3/10) + b3/10

    y_3 = 1 / (1 + torch.exp(- (a_3)))



    

    for i in range(local_batch_size):

        if y_3[i].argmax() == target[i]:

            cnt+=1



            

cnt/10000
#Переопределяем веса, указываем количество нейронов (784х32х10)

w1 = torch.randn(28*28, 32, requires_grad=True) # теперь не 784х10, а связи к промежуточному слою 784х32 

b1 = torch.randn(32, requires_grad=True)



w2 = torch.randn(32, 10, requires_grad=True) # связи от промежуточного слоя (32 нейрона) к финальным 10 нейронам  32х10

b2 = torch.randn(10, requires_grad=True)



optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=learning_rate)



# Learning Loop 784х32х10

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

        y_1 = 1 / (1 + torch.exp(- (a_1)))

#         print(x.shape)

        a_2 = torch.mm(y_1, w2/10) + b2/10

        y_2 = 1 / (1 + torch.exp(- (a_2)))

        

        loss = BCE(y_2, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')



#Проверка точности

cnt = 0



for data, target in test_loader:

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    

    

    x = data.reshape(local_batch_size, 28*28)

#     y = oneHot(target, local_batch_size)



    a_1 = torch.mm(x, w1/10) + b1/10 # воспользуемся функцией torch.mm

    y_1 = 1 / (1 + torch.exp(- (a_1)))

    #         print(x.shape)

    a_2 = torch.mm(y_1, w2/10) + b2/10

    y_2 = 1 / (1 + torch.exp(- (a_2)))

    

    for i in range(local_batch_size):

        if y_2[i].argmax() == target[i]:

            cnt+=1



            

print("accuracy", cnt/10000)
%%time

#Переопределяем веса, указываем количество нейронов (784х10)

w = torch.randn(28*28, 10, requires_grad=True) # теперь не 10х784, а 784х10

b = torch.randn(10, requires_grad=True)



optimizer = torch.optim.Adam([w, b], lr=learning_rate)



# Learning Loop 784х32х10





# Learning Loop (784х10)

for _ in range(20):

    tot_loss = 0

    for data, target in train_loader:

        x = data.reshape(batch_size, 28*28)

        y = oneHot(target, batch_size)

#         print(y)

        

        a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

        y_ = 1 / (1 + torch.exp(- (a_)))

        

        loss = BCE(y_, y)

#         print(loss)

        

        optimizer.zero_grad() # clean all grad    

        loss.backward() # ~backprob~ step

        optimizer.step()

        tot_loss+=loss

#         print(w, b)

        

    if _ % 1 == 0:

        print(f'step_loss {_} = ', "%.10f" % float(tot_loss/60000.))

        print('____________')



#Проверка точности

cnt = 0



for data, target in test_loader:

    if data.shape[0] == batch_size:

        local_batch_size = batch_size

    else:

        local_batch_size = data.shape[0]

    x = data.reshape(local_batch_size, 28*28)

    y = oneHot(target, local_batch_size)

    a_ = torch.mm(x, w/10) + b/10 # воспользуемся функцией torch.mm

    y_ = 1 / (1 + torch.exp(- (a_)))

    for i in range(local_batch_size):

        if y_[i].argmax() == target[i]:

            cnt+=1





            

print("accuracy", cnt/10000)
train_loader = torch.utils.data.DataLoader(

    datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()),

    batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(

    datasets.CIFAR10('data', train=False, transform=transforms.ToTensor()),

    batch_size=1)
# iii = 0

for data, target in train_loader:

    print(data.shape)

#     plt.imshow(data.reshape(28,28))

#     plt.show() # plot first 10 digits examples

#     iii += 1

#     if iii == 2:

    break
# iii = 0

# for data, target in train_loader:

# #     print(data.shape)

#     plt.imshow(data.reshape(28,28))

#     plt.show() # plot first 10 digits examples

#     iii += 1

#     if iii == 2:

#         break