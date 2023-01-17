import numpy as np



import torch

import torch.nn as nn



from pytorch_zoo import *
# Fake data of examples of different lengths, padded with zeros

data = [[2, 3, 44, 21, 89, 0, 0, 0],

           [45, 22, 89, 0, 0, 0, 0, 0],

           [21, 67, 43, 76, 28, 29, 90, 32],

           [45, 22, 89, 0, 0, 0, 0, 0],

           [45, 22, 62, 80, 89, 0, 0, 0],

           [21, 67, 43, 76, 28, 29, 90, 32]]

data = np.array(data).astype(float)

data = torch.from_numpy(data)
# Create the dataset

train_dataset = torch.utils.data.TensorDataset(data)



# Create the dynamic sampler

sampler = torch.utils.data.RandomSampler(train_dataset)



sampler = DynamicSampler(sampler, batch_size=2, drop_last=False)



# Create the dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)



for epoch in range(10):

    for batch in train_loader:

        batch = trim_tensors(batch)



        # train_batch(...)
# Fake data

logits = torch.rand((4, 3, 128, 128))

labels = torch.ones((4, 3, 128, 128))
loss = lovasz_hinge(logits, labels)

print(loss)
criterion = DiceLoss()

loss = criterion(logits, labels)

print(loss)
class Model(nn.Module):    

    def __init__(self, in_ch, out_ch):

        super(Model, self).__init__()

        

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

        self.SE = SqueezeAndExcitation(out_ch, r=16)



    def forward(self, x):

        x = self.conv(x)

        x = self.SE(x)

        

        return x
# Fake data

x = torch.rand((1, 3, 128, 128))



# Create the model

model = Model(3, 64)



out = model(x)



print(out.shape)
class Model(nn.Module):    

    def __init__(self, in_ch, out_ch):

        super(Model, self).__init__()

        

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

        self.sSE = ChannelSqueezeAndSpatialExcitation(out_ch)



    def forward(self, x):

        x = self.conv(x)

        x = self.sSE(x)

        

        return x
# Fake data

x = torch.rand((1, 3, 128, 128))



# Create the model

model = Model(3, 64)



out = model(x)



print(out.shape)
class Model(nn.Module):    

    def __init__(self, in_ch, out_ch):

        super(Model, self).__init__()

        

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

        self.scSE = ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(out_ch, r=16)



    def forward(self, x):

        x = self.conv(x)

        x = self.scSE(x)

        

        return x
# Fake data

x = torch.rand((1, 3, 128, 128))



# Create the model

model = Model(3, 64)



out = model(x)



print(out.shape)
class Model(nn.Module):    

    def __init__(self, in_ch, out_ch):

        super(Model, self).__init__()

        

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

        self.gaussian_noise = GaussianNoise(0.1)



    def forward(self, x):

        x = self.conv(x)



        if self.training:

            x = self.gaussian_noise(x)

        

        return x
# Fake data

x = torch.rand((1, 3, 128, 128))



# Create the model

model = Model(3, 64)



out = model(x)



print(out.shape)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = CyclicMomentum(optimizer)



train_dataset = torch.utils.data.TensorDataset(data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)



for epoch in range(10):

    for batch in train_loader:

        scheduler.batch_step()



        # train_batch(...)
notify({'value1': 'Notification title', 'value2': 'Notification body'}, key=[YOUR_PRIVATE_KEY_HERE])
seed_environment(42)
gpu_usage(device, digits=4)
print(n_params(model))
save_model(model, fold=0)
model = load_model(model, fold=0)
save(data, 'data.pkl')
data = load('data.pkl')
logits = torch.rand((1, 256))

mask = torch.ones((1, 256))
out = masked_softmax(logits, mask, dim=-1)

print(out.shape)
out = masked_log_softmax(logits, mask, dim=-1)

print(out.shape)