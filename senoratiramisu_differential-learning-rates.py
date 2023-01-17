import numpy as np 

import torch

from torch import nn

from torch import optim

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import seaborn as sns

import torchvision.models as models     

from collections import defaultdict, namedtuple
#set device and seed

device = 'cuda'

torch.manual_seed(1)

torch.cuda.manual_seed(1)

np.random.seed(1)

torch.backends.cudnn.deterministic = True
transform = transforms.Compose(

    [transforms.Resize(224),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,

                                          shuffle=True, num_workers=0)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transform)



testloader = torch.utils.data.DataLoader(testset, batch_size=256,

                                         shuffle=False, num_workers=0)
def getmodel():

    model = models.resnet34(pretrained=False)

    model.fc.out_features = 10

    return model.to(device)
class stat:

    '''

    Store a single-valued descriptor of the gradients, as mean or std,

    after every minibatch

    '''

    def __init__(self,fn):

        self.values = defaultdict(list)

        self.fn = fn

        

    def update(self,grads):

        for k,v in grads.items():

            self.values[k].append(self.fn(v))

            

    def plot(self,labels):

        fig, ax = plt.subplots(1,1)

        x = [i for i in range(len(self.values[0]))]

        for k,v in self.values.items():

            ax.plot(x,v,label = labels[k])

        ax.legend()

        plt.show()
class histogram:

    '''

    Store a histogram of the gradients after every minibatch

    '''

    def __init__(self,bins = None, bound = 5e-4):

        self.values = defaultdict(list)

        self.bins = bins

        self.bound = bound

        

    def update(self,grads):

        if not self.bins:

            self.bins = {k: int(1 + np.log(len(v))/np.log(2)) for k,v in grads.items()}

        for k,v in grads.items():

            h = v.histc(bins=self.bins[k],min=-self.bound,max=self.bound).detach().tolist()

            self.values[k].append(h)

            

    def plot(self,labels,**kwargs):

        rows, cols = len(self.values)//2, 2

        fig, axs = plt.subplots(rows,cols,figsize = (15,5))

        axs = axs.flatten()

        fig.patch.set_facecolor('w')

        for (k,v),ax in zip(self.values.items(),axs):

            ax.imshow(np.array(self.values[k]).T,extent = (0,200,-15,15),**kwargs)

            ax.set_title(labels[k])

            ax.set_yticklabels('')

        fig.suptitle('Gradients Histograms over one epoch')

        plt.show()
class modelgrads:

    '''

    Main module. Get the gradients, update the stats, and it had some 

    other utility functions

    

    model: a torch resnet

    stats: a list of objects with an update method implemented 

    '''

    def __init__(self,model,stats=None):

        self.model = model

        self.stats = stats

        self._set_groups()

        

    def layers(self): 

        for m in self.model.modules():

            if hasattr(m,'weight'):

                yield m

        

    def _set_groups(self):

        layer_type = defaultdict(int,{nn.Linear: 1, nn.Conv2d: 2}) 

        for m in self.layers():

            setattr(m,'group',layer_type[type(m)])  

            if isinstance(m,nn.Conv2d) and m.in_channels != m.out_channels and m.kernel_size == (1,1):

                layer_type[nn.Conv2d] += 1

        self._nconv = layer_type[nn.Conv2d]

    

    def layer_groups(self):

        layer_groups = defaultdict(list)

        for m in self.layers(): layer_groups[m.group].append(m)

        return layer_groups

    

    def labels(self):

        non_conv = {0: 'BatchNorm Layers', 1: 'Linear Layers'}

        conv = {k: f'Conv Layer {k-1}' for k in range(2,self._nconv+1)}

        return {**non_conv,**conv}

        

    def _get_grads(self):

        grads = defaultdict(torch.cuda.FloatTensor)

        for m in self.layers(): 

            grads[m.group] = torch.cat((grads[m.group], m.weight.grad.flatten()))

        return grads



    def update_stats(self):

        grads = self._get_grads()

        for stat in self.stats:

            stat.update(grads)
def train_loop(model, optimizer, loss_fn, epochs, grads=None):

    

    for _ in range(epochs):

        model.train()

        current = 0

        for img, lab in trainloader:

            optimizer.zero_grad()

            out = model(img.float().to(device))

            loss = loss_fn(out, lab.cuda().to(device))

            loss.backward()

            optimizer.step()

            current += loss.item()

            if grads:

                grads.update_stats()

                

        train_loss = current / len(trainloader)

                

        with torch.no_grad():

            current, acc = 0, 0

            model.eval()

            for img, lab in testloader:

                out  = model(img.float().to(device))

                loss = loss_fn(out, lab.to(device))

                current += loss.item() 

                _, pred = nn.Softmax(-1)(out).max(-1)

                acc += (pred == lab.cuda()).sum().item()



            valid_loss = current / len(testloader)

            accuracy   = 100 * acc / len(testset)



        print(f'Train loss: {train_loss:.2f}, Validation loss: {valid_loss:.2f}, Accuracy: {accuracy}')
resnet = getmodel()
absmean = stat(fn = lambda v: v.abs().mean().detach().item())

mean = stat(fn = lambda v: v.mean().detach().item())

std = stat(fn = lambda v: v.std().detach().item())

hist = histogram()
resnet_grads = modelgrads(resnet, stats = [absmean, mean, std, hist])
optimizer = optim.Adam(resnet.parameters(), lr =1e-3)

loss_fn = nn.CrossEntropyLoss()
train_loop(model = resnet, optimizer = optimizer, loss_fn = loss_fn, epochs = 1, grads = resnet_grads)
labels = resnet_grads.labels()
hist.plot(labels, cmap = 'rainbow')
absmean.plot(labels)
mean.plot(labels)
std.plot(labels)
def optim_diff(layer_groups, lr, cr):

    opt = []

    for k,group in groups.items():

        if k == 1:

            opt.append({'params': [par for l in group for par in l.parameters()], 

                        'lr': 1e-2})

        else:

            opt.append({'params': [par for l in group for par in l.parameters()], 

                        'lr': lr*(1+cr)**(k-1)})  

    return optim.Adam(opt, lr = lr)
resnet = getmodel()

groups = modelgrads(resnet).layer_groups()

optimizer = optim_diff(groups, 1e-3,.1)

train_loop(model = resnet, optimizer = optimizer, loss_fn = loss_fn, epochs = 5)
resnet = getmodel()

groups = modelgrads(resnet).layer_groups()

optimizer = optim.Adam(resnet.parameters(), lr =1e-3)

train_loop(model = resnet, optimizer = optimizer, loss_fn = loss_fn, epochs = 5)