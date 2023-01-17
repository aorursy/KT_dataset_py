import torch as th

import syft as sy
x = th.tensor([1,2,3,4,5])

x
y = x + x
print(y)
hook = sy.TorchHook(th)
th.tensor([1,2,3,4,5])
bob = sy.VirtualWorker(hook, id="bob")
bob._objects
x = th.tensor([1,2,3,4,5])
x = x.send(bob)
bob._objects
x.location
x.id_at_location
x.id
x.owner
hook.local_worker
x
x = x.get()

x
bob._objects



# try this project here!

bob = sy.VirtualWorker(hook, 'bob')

ada = sy.VirtualWorker(hook, 'ada')



mytensor = th.Tensor([2,3,5,7])
tensor_pointer = mytensor.send(bob, ada)
print(bob._objects)

print(ada._objects)

tensor_pointer
mytensor = tensor_pointer.get()

mytensor
x = th.tensor([1,2,3,4,5]).send(bob)

y = th.tensor([1,1,1,1,1]).send(bob)
x
y
z = x + y
z
z = z.get()

z
z = th.add(x,y)

z
z = z.get()

z
x = th.tensor([1.,2,3,4,5], requires_grad=True).send(bob)

y = th.tensor([1.,1,1,1,1], requires_grad=True).send(bob)
z = (x + y).sum()
z.backward()
x = x.get()
x
x.grad
# try this project here!

myinput = th.tensor([[0.,0], [1,0], [0,1], [1,1]], requires_grad=True)

input_ptr = myinput.send(ada)

target = th.tensor([[0.], [1], [0], [1]], requires_grad=True).send(ada)

weights = th.tensor([[0.], [0.]], requires_grad=True).send(ada)
for i in range(10):

    prediction = input_ptr.mm(weights)

    loss = ((prediction - target)**2).sum()

    loss.backward()

    weights.data.sub_(weights.grad * 0.15)

    weights.grad *= 0



    print(loss.get().data)
ada._objects


del input_ptr

ada.clear_objects()

ada._objects
bob = bob.clear_objects()
bob._objects
x = th.tensor([1,2,3,4,5]).send(bob)
bob._objects
del x
bob._objects
x = th.tensor([1,2,3,4,5]).send(bob)
bob._objects
x = "asdf"
bob._objects
x = th.tensor([1,2,3,4,5]).send(bob)
x
bob._objects
x = "asdf"
bob._objects
del x
bob._objects
bob = bob.clear_objects()

bob._objects
for i in range(1000):

    x = th.tensor([1,2,3,4,5]).send(bob)
bob._objects
x = th.tensor([1,2,3,4,5]).send(bob)

y = th.tensor([1,1,1,1,1]).send(bob)
z = x + y

z
from torch import nn, optim
# A Toy Dataset

data = th.tensor([[1.,1],[0,1],[1,0],[0,0]], requires_grad=True)

target = th.tensor([[1.],[1], [0], [0]], requires_grad=True)



bob = sy.VirtualWorker(hook, 'bob')

ada = sy.VirtualWorker(hook, 'ada')
# A Toy Model

model = nn.Linear(2,1)
opt = optim.SGD(params=model.parameters(), lr=0.1)
def train(iterations=20):

    for iter in range(iterations):

        opt.zero_grad()



        pred = model(data)



        loss = ((pred - target)**2).sum()



        loss.backward()



        opt.step()



        print(loss.data)

        

train()
data_bob = data[0:2].send(bob)

target_bob = target[0:2].send(bob)
data_ada = data[2:4].send(ada)

target_ada = target[2:4].send(ada)
datasets = [(data_bob, target_bob), (data_ada, target_ada)]
def train(iterations=20):



    model = nn.Linear(2,1)

    opt = optim.SGD(params=model.parameters(), lr=0.1)

    

    for iter in range(iterations):



        for _data, _target in datasets:



            # send model to the data

            model = model.send(_data.location)



            # do normal training

            opt.zero_grad()

            pred = model(_data)

            loss = ((pred - _target)**2).sum()

            loss.backward()

            opt.step()



            # get smarter model back

            model = model.get()



            print(loss.get())
train()
bob.clear_objects()

ada.clear_objects()
x = th.tensor([1,2,3,4,5]).send(bob)
x = x.send(ada)
bob._objects
ada._objects
y = x + x
y
bob._objects
ada._objects
jon = sy.VirtualWorker(hook, id="jon")
bob.clear_objects()

ada.clear_objects()



x = th.tensor([1,2,3,4,5]).send(bob).send(ada)
bob._objects
ada._objects
x = x.get()

x
bob._objects
ada._objects
x = x.get()

x
bob._objects
bob.clear_objects()

ada.clear_objects()



x = th.tensor([1,2,3,4,5]).send(bob).send(ada)
bob._objects
ada._objects
del x
bob._objects
ada._objects
bob.clear_objects()

ada.clear_objects()
x = th.tensor([1,2,3,4,5]).send(bob)
bob._objects
ada._objects
x.move(ada)
bob._objects
ada._objects
x = th.tensor([1,2,3,4,5]).send(bob).send(ada)
bob._objects
ada._objects
x.remote_get()
bob._objects
ada._objects
x.move(bob)
x
bob._objects
ada._objects