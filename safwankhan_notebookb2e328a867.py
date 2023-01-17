import os

from torchvision import datasets

from torchvision import transforms

import torch

import torch.nn as nn

from tqdm import tqdm

import numpy as np

from torch import optim

import pandas as pd

from sklearn.datasets import load_boston

from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
def get_boston_data(batch_size=64):

    """

    Args:

        train_batch_size(int): training batch size 

        test_batch_size(int): test batch size

    Returns:

        (torch.utils.data.DataLoader): train loader 

        (torch.utils.data.DataLoader): test loader

    """

    #Dataset Loading

    boston_dataset = load_boston()

    x=boston_dataset.data

    y=boston_dataset.target

    #Convert to Tensors

    inputs = torch.Tensor(x)

    targets = torch.Tensor(y)

    #Create Dataset

    boston_ds = TensorDataset(inputs, targets)

    batch_size = batch_size

    test_split = .2

    shuffle_dataset = True

    random_seed= 42



    # Creating data indices for training and validation splits:

    dataset_size = len(boston_ds)

    indices = list(range(dataset_size))

    split = int(np.floor(test_split * dataset_size))

    if shuffle_dataset :

        np.random.seed(random_seed)

        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]



    # Creating PT data samplers and loaders:

    train_sampler = SubsetRandomSampler(train_indices)

    test_sampler = SubsetRandomSampler(test_indices)

    kwargs = {'num_workers': 4, 'pin_memory': True}



    train_loader = torch.utils.data.DataLoader(boston_ds, batch_size=batch_size, 

                                               sampler=train_sampler,shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(boston_ds, batch_size=batch_size,

                                                    sampler=test_sampler,shuffle=False, **kwargs)



    return train_loader, test_loader
class AverageMeter(object):

    """Basic meter"""

    def __init__(self):

        self.reset()



    def reset(self):

        """ reset meter

        """

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        """ incremental meter

        """

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

def hsic_objective(hidden, h_target, h_data, sigma):





    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma)

    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)



    return hsic_hx_val, hsic_hy_val
def hsic_train(cepoch, model, data_loader, config_dict):



    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    prec1 = total_loss = hx_l = hy_l = -1



    batch_acc    = AverageMeter()

    batch_loss   = AverageMeter()

    batch_hischx = AverageMeter()

    batch_hischy = AverageMeter()



    batch_log = {}

    batch_log['batch_acc'] = []

    batch_log['batch_loss'] = []

    batch_log['batch_hsic_hx'] = []

    batch_log['batch_hsic_hy'] = []



    model = model.to('cpu')



    n_data = config_dict['batch_size'] * len(data_loader)

    

    



    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=120)

    for batch_idx, (data, target) in pbar:

        

                

        data   = data.to('cpu')

        target = target.to('cpu')

        output, hiddens = model(data)



        h_target = target.view(-1,1)

        #h_target = to_categorical(h_target, num_classes=10).float()

        h_data = data.view(-1, np.prod(data.size()[1:]))

        



     

        idx_range = []

        it = 0



        # So the batchnorm is not learnable, making only @,b at layer

        for i in range(len(hiddens)):

            idx_range.append(np.arange(it, it+2).tolist())

            it += 2

    

        for i in range(len(hiddens)):

            

            output, hiddens = model(data)

            params, param_names = get_layer_parameters(model=model, idx_range=idx_range[i]) # so we only optimize one layer at a time

            #optimizer = optim.SGD(params, lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)

            optimizer = optim.Adam(params, lr=config_dict['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

            optimizer.zero_grad()

            # sigma_optimizer.zero_grad()

            if len(hiddens[i].size()) > 2:

                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))



            

            hx_l, hy_l = hsic_objective(

                    hiddens[i],

                    h_target=h_target.float(),

                    h_data=h_data,

                    sigma=config_dict['sigma'],

            )

            

            

            loss = hx_l - config_dict['lambda_y']*hy_l

            #loss.backward()

            optimizer.step()

            # sigma_optimizer.step()

        # if config_dict['hsic_solve']:

        #     prec1, reorder_list = get_accuracy_hsic(model, data_loader)

        batch_acc.update(prec1)

        batch_loss.update(total_loss)

        batch_hischx.update(hx_l.cpu().detach().numpy())

        batch_hischy.update(hy_l.cpu().detach().numpy())



        # # # preparation log information and print progress # # #



        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] H_hx:{H_hx:.4f} H_hy:{H_hy:.4f}'.format(

                        cepoch = cepoch,  

                        cidx = (batch_idx+1)*config_dict['batch_size'], 

                        tolidx = n_data,

                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 

                        H_hx = batch_hischx.avg, 

                        H_hy = batch_hischy.avg,

                )



        if ((batch_idx+1) % config_dict['log_batch_interval'] == 0):



            batch_log['batch_acc'].append(batch_loss.avg)

            batch_log['batch_loss'].append(batch_acc.avg)

            batch_log['batch_hsic_hx'].append(batch_hischx.avg)

            batch_log['batch_hsic_hy'].append(batch_hischy.avg)



        pbar.set_description(msg)



    return batch_log

class ModelLinear(nn.Module):



    def __init__(self, hidden_width=64, n_layers=5, atype='relu', 

        last_hidden_width=None, model_type='simple-dense', data_code='boston', **kwargs):

        super(ModelLinear, self).__init__()

    

        block_list = []

        

        last_hw = hidden_width

        if last_hidden_width:

            last_hw = last_hidden_width

        

        for i in range(n_layers):

            block = get_primative_block('simple-dense', hidden_width, hidden_width, atype)

            block_list.append(block)



        in_width = 13

        in_ch = 1



        self.input_layer    = makeblock_dense(in_width*in_ch, hidden_width, atype)

        self.sequence_layer = nn.Sequential(*block_list)

        self.output_layer   = makeblock_dense(hidden_width, 1, atype='linear')



        self.in_width = in_width*in_ch



    def forward(self, x):



        output_list = []

        x = x.view(-1, self.in_width)

        x = self.input_layer(x)

        output_list.append(x)

        

        for block in self.sequence_layer:

            x = block(x)

            output_list.append(x)

        x = self.output_layer(x)

        output_list.append(x)



        return x, output_list

def get_activation(atype):



    if atype=='relu':

        nonlinear = nn.ReLU()

    elif atype=='tanh':

        nonlinear = nn.Tanh() 

    elif atype=='sigmoid':

        nonlinear = nn.Sigmoid() 

    

    return nonlinear



def get_primative_block(model_type, hid_in, hid_out, atype):

    if model_type=='simple-dense':

        block = makeblock_dense(hid_in, hid_out, atype) 

    return block



def makeblock_dense(in_dim, out_dim, atype):

    

    layer = nn.Linear(in_dim, out_dim)

    bn = nn.BatchNorm1d(out_dim, affine=False)

    if atype=='linear':

        out = nn.Sequential(*[layer, bn])

    else:

        nonlinear = get_activation(atype)

        out = nn.Sequential(*[layer, bn, nonlinear])

    return out
def get_current_timestamp():

    return strftime("%y%m%d_%H%M%S")



def get_layer_parameters(model, idx_range):



    param_out = []

    param_out_name = []

    for it, (name, param) in enumerate(model.named_parameters()):

        if it in idx_range:

            param_out.append(param)

            param_out_name.append(name)



    return param_out, param_out_name
def sigma_estimation(X, Y):

    """ sigma from median distance

    """

    D = distmat(torch.cat([X,Y]))

    D = D.detach().cpu().numpy()

    Itri = np.tril_indices(D.shape[0], -1)

    Tri = D[Itri]

    med = np.median(Tri)

    if med <= 0:

        med=np.mean(Tri)

    if med<1E-2:

        med=1E-2

    return med



def distmat(X):

    """ distance matrix

    """

    r = torch.sum(X*X, 1)

    r = r.view([-1, 1])

    a = torch.mm(X, torch.transpose(X,0,1))

    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)

    D = torch.abs(D)

    return D



def kernelmat(X, sigma):

    """ kernel matrix baker

    """

    m = int(X.size()[0])

    dim = int(X.size()[1]) * 1.0

    H = torch.eye(m) - (1./m) * torch.ones([m,m])

    Dxx = distmat(X)

    

    if sigma:

        variance = 2.*sigma*sigma*X.size()[1]            

        Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        

        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))

    else:

        try:

            sx = sigma_estimation(X,X)

            Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)

        except RuntimeError as e:

            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(

                sx, torch.max(X), torch.min(X)))



    Kxc = torch.mm(Kx,H)



    return Kxc



def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):

    """

    """

    m = int(x.size()[0])

    Kxc = kernelmat(x, sigma=sigma)

    Kyc = kernelmat(y, sigma=sigma)



    epsilon = 1E-5

    K_I = torch.eye(m)

    Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)

    Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)

    Rx = (Kxc.mm(Kxc_i))

    Ry = (Kyc.mm(Kyc_i))

    Pxy = torch.sum(torch.mul(Rx, Ry.t()))



    return Pxy
#!pip install torchsummary
#from torchsummary import summary

import matplotlib.pyplot as plt





# # # configuration

config_dict = {}

config_dict['batch_size'] = 60

config_dict['learning_rate'] = 0.001

config_dict['lambda_y'] = 50

config_dict['sigma'] = 1

config_dict['task'] = 'hsic-train'

config_dict['log_batch_interval'] = 6



# # # data prepreation

train_loader, test_loader = get_boston_data(batch_size=config_dict['batch_size'])



# # # simple fully-connected model





mdX = {}

mdY = {}

# # # start to train

epochs = 50

for x in range(5,25,5):

    hsicX = []

    hsicY = []

    model = ModelLinear(hidden_width=13,

                    n_layers=x,

                    atype='relu',

                    last_hidden_width=None,

                    model_type='simple-dense',

                    data_code='boston')

    #print(model)



    for cepoch in range(epochs):

        t = hsic_train(cepoch, model, train_loader, config_dict)

        hsicX.append((t['batch_hsic_hx']))

        hsicY.append((t['batch_hsic_hy']))

        

    mdX['Depth ' + str(x)]=hsicX

    mdY['Depth ' + str(x)]=hsicY

    
for k,v in mdX.items():

    plt.plot(range(len(v)),v)

plt.xlabel('Epoch')

plt.ylabel('Average H_hx per epoch')

plt.legend([x for x in mdX.keys()])

plt.show()



for k,v in mdY.items():

    plt.plot(range(len(v)),v)

plt.xlabel('Epoch')

plt.ylabel('Average H_hy per epoch')

plt.legend([x for x in mdY.keys()])

plt.show()
!pip install jovian
import jovian

jovian.commit(project='project-final')