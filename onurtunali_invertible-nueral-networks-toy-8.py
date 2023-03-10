# To install the package and data internet connection must be enabled. Check kernel settings.



!pip install git+https://github.com/onurtunali/FrEIA.git

!wget https://raw.githubusercontent.com/onurtunali/FrEIA/master/examples/data.py
%matplotlib notebook



import torch

import torch.optim



import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time



from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet

from FrEIA.modules import rev_multiplicative_layer, F_fully_connected



import data



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Current device is {device} and Data is imported")
batch_size = 1600

test_split = 10000



pos, labels = data.generate(

    labels='all',

    tot_dataset_size=2**20

)

c = np.where(labels[:test_split])[1]

plt.figure(figsize=(6, 6))

plt.scatter(pos[:test_split, 0], pos[:test_split, 1], c=c, cmap='Set1', s=0.25)

plt.xticks([])

plt.yticks([])

plt.show()

# print(f"{pos[0]}\n{labels[0]}\n{np.where(labels[:test_split])}")
ndim_tot = 16

ndim_x = 2

ndim_y = 8

ndim_z = 2



inp = InputNode(ndim_tot, name='input')



t1 = Node([inp.out0], rev_multiplicative_layer,

          {'F_class': F_fully_connected, 'clamp': 2.0,

           'F_args': {'dropout': 0.0}})



t2 = Node([t1.out0], rev_multiplicative_layer,

          {'F_class': F_fully_connected, 'clamp': 2.0,

           'F_args': {'dropout': 0.0}})



t3 = Node([t2.out0], rev_multiplicative_layer,

          {'F_class': F_fully_connected, 'clamp': 2.0,

           'F_args': {'dropout': 0.0}})



outp = OutputNode([t3.out0], name='output')



nodes = [inp, t1, t2, t3, outp]

model = ReversibleGraphNet(nodes)
# Training parameters

n_epochs = 300

meta_epoch = 12

n_its_per_epoch = 4

batch_size = 1600



lr = 1e-2

gamma = 0.01**(1./120)

l2_reg = 2e-5



y_noise_scale = 3e-2

zeros_noise_scale = 3e-2



# relative weighting of losses:

lambd_predict = 300.

lambd_latent = 300.

lambd_rev = 400.



pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)

pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)



optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.8),

                             eps=1e-04, weight_decay=l2_reg)



scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                            step_size=meta_epoch,

                                            gamma=gamma)





def MMD_multiscale(x, y):

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())



    rx = (xx.diag().unsqueeze(0).expand_as(xx))

    ry = (yy.diag().unsqueeze(0).expand_as(yy))



    dxx = rx.t() + rx - 2.*xx

    dyy = ry.t() + ry - 2.*yy

    dxy = rx.t() + ry - 2.*zz



    XX, YY, XY = (torch.zeros(xx.shape).to(device),

                  torch.zeros(xx.shape).to(device),

                  torch.zeros(xx.shape).to(device))



    for a in [0.2, 0.5, 0.9, 1.3]:

        XX += a**2 * (a**2 + dxx)**-1

        YY += a**2 * (a**2 + dyy)**-1

        XY += a**2 * (a**2 + dxy)**-1



    return torch.mean(XX + YY - 2.*XY)





def fit(input, target):

    return torch.mean((input - target)**2)



loss_backward = MMD_multiscale

loss_latent = MMD_multiscale

loss_fit = fit



test_loader = torch.utils.data.DataLoader(

    torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),

    batch_size=batch_size, shuffle=True, drop_last=True)



train_loader = torch.utils.data.DataLoader(

    torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),

    batch_size=batch_size, shuffle=True, drop_last=True)
def train(i_epoch=0):

    model.train()



    l_tot = 0

    batch_idx = 0

    

    t_start = time()

    

    loss_factor = 600**(float(i_epoch) / 300) / 600

    if loss_factor > 1:

        loss_factor = 1



    for x, y in train_loader:

        batch_idx += 1

        if batch_idx > n_its_per_epoch:

            break



        x, y = x.to(device), y.to(device)

        

        y_clean = y.clone()

        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -

                                                ndim_x, device=device)

        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -

                                                 ndim_y - ndim_z, device=device)



        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)



        x, y = (torch.cat((x, pad_x),  dim=1),

                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),

                          dim=1))

        



        optimizer.zero_grad()



        # Forward step:



        output = model(x)



        # Shorten output, and remove gradients wrt y, for latent loss

        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)



        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])



        output_block_grad = torch.cat((output[:, :ndim_z],

                                       output[:, -ndim_y:].data), dim=1)



        l += lambd_latent * loss_latent(output_block_grad, y_short)

        l_tot += l.data.item()



        l.backward()



        # Backward step:

        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -

                                                 ndim_y - ndim_z, device=device)

        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)



        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *

                            torch.randn(batch_size, ndim_z, device=device))

        y_rev = torch.cat((orig_z_perturbed, pad_yz,

                           y), dim=1)

        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,

                                y), dim=1)

        

        output_rev = model(y_rev, rev=True)

        output_rev_rand = model(y_rev_rand, rev=True)



        l_rev = (

            lambd_rev

            * loss_factor

            * loss_backward(output_rev_rand[:, :ndim_x],

                            x[:, :ndim_x])

        )



        l_rev += 0.50 * lambd_predict * loss_fit(output_rev, x)

        

        l_tot += l_rev.data.item()

        l_rev.backward()



        for p in model.parameters():

            p.grad.data.clamp_(-15.00, 15.00)



        optimizer.step()



#     print('%.1f\t%.5f' % (

#                              float(batch_idx)/(time()-t_start),

#                              l_tot / batch_idx,

#                            ), flush=True)



    return l_tot / batch_idx
for mod_list in model.children():

    for block in mod_list.children():

        for coeff in block.children():

            coeff.fc3.weight.data = 0.01*torch.randn(coeff.fc3.weight.shape)

            

model.to(device)



fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].set_xticks([])

axes[0].set_yticks([])

axes[0].set_title('Predicted labels (Forwards Process)')

axes[1].set_xticks([])

axes[1].set_yticks([])

axes[1].set_title('Generated Samples (Backwards Process)')

fig.show()

fig.canvas.draw()



N_samp = 4096



x_samps = torch.cat([x for x,y in test_loader], dim=0)[:N_samp]

y_samps = torch.cat([y for x,y in test_loader], dim=0)[:N_samp]

c = np.where(y_samps)[1]

y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)

y_samps = torch.cat([torch.randn(N_samp, ndim_z),

                     zeros_noise_scale * torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z), 

                     y_samps], dim=1)

y_samps = y_samps.to(device)

            

try:

#     print('#Epoch \tIt/s \tl_total')

    t_start = time()

    for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):



        scheduler.step()



        # Initially, the l2 reg. on x and z can give huge gradients, set

        # the lr lower for this

        if i_epoch < 0:

            for param_group in optimizer.param_groups:

                param_group['lr'] = lr * 1e-2



#         print(i_epoch, end='\t ')

        train(i_epoch)



        rev_x = model(y_samps, rev=True)

        rev_x = rev_x.cpu().data.numpy()

        

        pred_c = model(torch.cat((x_samps, torch.zeros(N_samp, ndim_tot - ndim_x)),

                                 dim=1).to(device)).data[:, -8:].argmax(dim=1)



        axes[0].clear()

        axes[0].scatter(x_samps[:,0].cpu(), x_samps[:,1].cpu(), c=pred_c.cpu(), cmap='Set1', s=1., vmin=0, vmax=9)

        axes[0].axis('equal')

        axes[0].axis([-3,3,-3,3])

        axes[0].set_xticks([])

        axes[0].set_yticks([])



        

        axes[1].clear()

        axes[1].scatter(rev_x[:,0], rev_x[:,1], c=c, cmap='Set1', s=1., vmin=0, vmax=9)

        axes[1].axis('equal')

        axes[1].axis([-3,3,-3,3])

        axes[1].set_xticks([])

        axes[1].set_yticks([])

        

        fig.canvas.draw()





except KeyboardInterrupt:

    pass

finally:

    print(f"\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")