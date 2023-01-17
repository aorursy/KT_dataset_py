import torch

import torchvision

import math

import matplotlib.pyplot as plt
class CosineDecayWithWarmUpScheduler(object):

    def __init__(self,optimizer,step_per_epoch,init_warmup_lr=1e-5,warm_up_steps=1000,max_lr=1e-4,min_lr=1e-6,num_step_down=2000,num_step_up=None,

                T_mul=1,max_lr_decay=None, gamma=1,min_lr_decay=None,alpha=1):

        self.optimizer = optimizer

        self.step_per_epoch = step_per_epoch

        if warm_up_steps != 0:

            self.warm_up = True

        else:

            self.warm_up = False  

        self.init_warmup_lr = init_warmup_lr

        self.warm_up_steps = warm_up_steps

        self.max_lr = max_lr

        if min_lr == 0:

            self.min_lr = 0.1 * max_lr

            self.alpha = 0.1

        else:

            self.min_lr = min_lr

        self.num_step_down = num_step_down

        if num_step_up == None:

            self.num_step_up = num_step_down

        else:

            self.num_step_up = num_step_up    

        self.T_mul = T_mul

        if max_lr_decay == None:

            self.gamma = 1

        elif max_lr_decay == 'Half':

            self.gamma = 0.5

        elif max_lr_decay == 'Exp':

            self.gamma = gamma

        

        if min_lr_decay == None:

            self.alpha = 1

        elif min_lr_decay == 'Half':

            self.alpha = 0.5

        elif min_lr_decay == 'Exp':

            self.alpha = alpha





        self.num_T = 0

        self.iters = 0

        self.lr_list = []

        

        

    def update_cycle(self, lr):

        old_min_lr = self.min_lr

        if lr == self.max_lr or (self.num_step_up == 0 and lr == self.min_lr):

            if self.num_T == 0:

                self.warm_up = False

                self.min_lr /= self.alpha

            self.iters = 0

            self.num_T += 1

            self.min_lr *= self.alpha



        if lr == old_min_lr and self.max_lr * self.gamma >= self.min_lr:

            self.max_lr *= self.gamma

            

    

    def step(self):

        self.iters += 1

        if self.warm_up:

            lr = self.init_warmup_lr + (self.max_lr-self.init_warmup_lr) / self.warm_up_steps * self.iters

        else:

            T_cur = self.T_mul**self.num_T

            if self.iters <= self.num_step_down*T_cur:

                lr = self.min_lr + (self.max_lr-self.min_lr) * (1 + math.cos(math.pi*self.iters/(self.num_step_down*T_cur)))/2

                if lr < self.min_lr:

                    lr = self.min_lr

            elif self.iters > self.num_step_down*T_cur:

                lr = self.min_lr + (self.max_lr-self.min_lr) / (self.num_step_up * T_cur) * (self.iters-self.num_step_down*T_cur)

                if lr > self.max_lr:

                    lr = self.max_lr



        self.update_cycle(lr)

                

        for param_group in self.optimizer.param_groups:

            param_group['lr'] = lr

            self.lr_list.append(lr)
model = torchvision.models.resnet18(pretrained=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=4e-6,num_step_down=6000,

                                          num_step_up = 6000)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=4e-6,num_step_down=6000,

                                          num_step_up = 6000,T_mul=1,max_lr_decay='Half')

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=4e-6,num_step_down=6000,

                                          num_step_up = 6000,T_mul=1,max_lr_decay='Exp',gamma=0.9)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=2e-5,warm_up_steps=4000,max_lr=4e-5,min_lr=1e-5,num_step_down=4000,

                                          num_step_up = 4000,min_lr_decay='Exp',alpha=0.1)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=4e-6,num_step_down=6000,

                                          num_step_up = 0)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=1e-6,num_step_down=6000,

                                          num_step_up = 0, max_lr_decay='Half')

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=2e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=2e-5,num_step_down=6000,

                                          num_step_up = 0, max_lr_decay='Exp',gamma=0.9, min_lr_decay='Exp', alpha=0.5)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=2e-5,num_step_down=6000,

                                          num_step_up = 0, max_lr_decay='Half', min_lr_decay='Exp', alpha=0.5)

for epoch in range(35):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=3000,max_lr=4e-5,min_lr=4e-6,num_step_down=3000,

                                          num_step_up = 3000,T_mul=2)

for epoch in range(60):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-5,warm_up_steps=3000,max_lr=4e-5,min_lr=4e-6,num_step_down=3000,

                                          num_step_up = 3000,T_mul=2,max_lr_decay='Half')

for epoch in range(60):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=2e-5,warm_up_steps=5000,max_lr=4e-5,min_lr=4e-6,num_step_down=5000,

                                          num_step_up = 5000, max_lr_decay='Exp',gamma=0.5, min_lr_decay='Exp', alpha=0.5)

for epoch in range(30):

    for i in range(1000):

        optimizer.step()

        scheduler.step()

plt.figure(figsize=(24,8))

plt.plot(scheduler.lr_list)