# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%load_ext autoreload

%autoreload 2



%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

bs=128



il = ImageList.from_files(path, tfms=tfms)

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))

ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)
nfs = [32,64,128,256]
cbfs = [partial(AvgStatsCallback,accuracy), CudaCallback,

        partial(BatchTransformXCallback, norm_imagenette)]


learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs)


run.fit(1, learn)
class Optimizer():

    def __init__(self, params, steppers, **defaults): #param is a list of list 

        # might be a generator

        self.param_groups = list(params) #this gonna give a list of all the parameter tensors. so all the weights ans all the biases #in fastai we also call a parameter group a lyaer group they are the same thing 

        #so remember that a parameter is pythorch. remember we made a linear layer we had a weight and bias tensor and they are both parameters. it is a paramter tensor

        # ensure params is a list of lists since we then can do parameter groups, and change the learning rate or other things in different groups under training like we wont to change the learning rate on the last to layers 

        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups] #we will check if it is a list of list (self.param_groups[0], list)

            #and if it is not a list of list we will turn ir into one (self.param_groups)- defined param_groups above

        self.hypers = [{**defaults} for p in self.param_groups] #so each parameter group can have its own set of hyperparamters like learinng rate, omentum beta nad adam etc.

        #these hyperparamters are gonna be stored as a dictornary. so there are gonna be one dictornary for each parameter group(param_groups). so for each param_groups (p) has

        #a dictornary and what is in the dictornary is whatever you passed to the constructer (__init__). 

        self.steppers = listify(steppers) #steppers is a function and one look likes in the sgd_step function in hte cell below 



    def grad_params(self):

        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)

            for p in pg if p.grad is not None]



    def zero_grad(self): #so we need a zzero grad, that are gonna go though some parameters and zero them out  

        for p,hyper in self.grad_params():

            p.grad.detach_() #and also remove any gradient computation history 

            p.grad.zero_()



    def step(self): #and we are gonna have a step function that does some kind of a  step 

        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper) #so when step function is called it goes through our paramters and compose togetther our steppers 

            #which is just one thing (sgd_step) and call the parameter(p) 

            #grad_params is just for conviniens 

            #note **hyper is all our hyper paramters since mayde some stepper want to use them like learning rate 
def sgd_step(p, lr, **kwargs): #so laerning rate(lr) comes from the **hyper above in the step function 

    p.data.add_(-lr, p.grad.data) #it does a sgd step which we hva seen before 

    #where p is paramter is it going through one parameter - learning rata and take the gradient to p 

    return p
opt_func = partial(Optimizer, steppers=[sgd_step]) #here we create a optimezer function with the optimazer class and the steppers set = to sgd_step


#export

class Recorder(Callback):

    def begin_fit(self): self.lrs,self.losses = [],[]



    def after_batch(self):

        if not self.in_train: return

        self.lrs.append(self.opt.hypers[-1]['lr'])

        self.losses.append(self.loss.detach().cpu())        



    def plot_lr  (self): plt.plot(self.lrs)

    def plot_loss(self): plt.plot(self.losses)

        

    def plot(self, skip_last=0):

        losses = [o.item() for o in self.losses]

        n = len(losses)-skip_last

        plt.xscale('log')

        plt.plot(self.lrs[:n], losses[:n])



class ParamScheduler(Callback):

    _order=1

    def __init__(self, pname, sched_funcs):

        self.pname,self.sched_funcs = pname,listify(sched_funcs)



    def begin_batch(self):  

        if not self.in_train: return

        fs = self.sched_funcs

        if len(fs)==1: fs = fs*len(self.opt.param_groups)

        pos = self.n_epochs/self.epochs

        for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)

            

class LR_Find(Callback):

    _order=1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):

        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr

        self.best_loss = 1e9

        

    def begin_batch(self): 

        if not self.in_train: return

        pos = self.n_iter/self.max_iter

        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos

        for pg in self.opt.hypers: pg['lr'] = lr

            

    def after_step(self):

        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:

            raise CancelTrainException()

        if self.loss < self.best_loss: self.best_loss = self.loss


sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])


cbfs = [partial(AvgStatsCallback,accuracy),

        CudaCallback, Recorder,

        partial(ParamScheduler, 'lr', sched)]
learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs, opt_func=opt_func)


%time run.fit(1, learn)
run.recorder.plot_loss()
run.recorder.plot_lr()
#add a stepper weight decay 

def weight_decay(p, lr, wd, **kwargs):

    p.data.mul_(1 - lr*wd) #just doing the formular above 

    return p

weight_decay._defaults = dict(wd=0.)
#another stepper for 12_reg

def l2_reg(p, lr, wd, **kwargs):

    p.grad.data.add_(wd, p.data) #add_ in pythorch normally just add the p.data tensor to the p.grad tensor but if you add a scalor(wd-weight decay) they will

    #multiply first to the scalor(wd) will multipy to the p.data tensor and then add it to the p.grad tensor 

    return p

l2_reg._defaults = dict(wd=0.) #so we add a default where we just make the dictornary for weight deca 0, so this is now a function you can call on

#so you can turn off the weight decay (this is used below)
#export

def maybe_update(os, dest, f):

    for o in os: #goes through each of the things in os

        for k,v in f(o).items(): #goes through each of the things in the dictornary 

            if k not in dest: dest[k] = v #check if it is not there, and if it is not it will update it 



def get_defaults(d): return getattr(d,'_defaults',{})


class Optimizer():

    def __init__(self, params, steppers, **defaults):

        self.steppers = listify(steppers)

        maybe_update(self.steppers, defaults, get_defaults) #so we are maybe update going to update our defaults(defined longer up) with whatever self.stepper has

        #and there defaults(get_defaults). And the reason it is maybe update is because if you give a direct weight decay it is not gonna update it 

        #it will only pdate it if it is missing 

        # might be a generator

        self.param_groups = list(params)

        # ensure params is a list of lists

        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]

        self.hypers = [{**defaults} for p in self.param_groups]



    def grad_params(self):

        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)

            for p in pg if p.grad is not None]



    def zero_grad(self):

        for p,hyper in self.grad_params():

            p.grad.detach_()

            p.grad.zero_()



    def step(self):

        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)
#export 

sgd_opt = partial(Optimizer, steppers=[weight_decay, sgd_step]) #note now we have to steppers in the stepper function 


learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs, opt_func=sgd_opt)
model = learn.model
#create teh optimazer with the sgd_opt defined above 

opt = sgd_opt(model.parameters(), lr=0.1) #use sgd optimazer with our models parameters and a learning rate 0.1 

test_eq(opt.hypers[0]['wd'], 0.) #check that the hyperparamter for the weight deacy is 0 

test_eq(opt.hypers[0]['lr'], 0.1) #and the hyperparamter for learning rate is 0.1 
opt = sgd_opt(model.parameters(), lr=0.1, wd=1e-4) #make a new optimazer and give it a weight decay and a new learning rate

test_eq(opt.hypers[0]['wd'], 1e-4) #check if it passes, so it has a weight decay on 1e-04  

test_eq(opt.hypers[0]['lr'], 0.1) #check if it passes, so it has a learning rate on 0.1 
cbfs = [partial(AvgStatsCallback,accuracy), CudaCallback]


learn,run = get_learn_run(nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=partial(sgd_opt, wd=0.01))


run.fit(1, learn)
#momwntum needs mote then just paramters and hyperparamters it uses stats, which is so that momentum knows what the activations was pdatet last time 

#since the formular for momentum is the current momentum times whatever you did list time(last momentum) plus the new step 



class StatefulOptimizer(Optimizer):

    def __init__(self, params, steppers, stats=None, **defaults): 

        self.stats = listify(stats)

        maybe_update(self.stats, defaults, get_defaults)

        super().__init__(params, steppers, **defaults)

        self.state = {} #so when we track every singel parameter for what happen last time, it gets stored in state

        

    def step(self):

        for p,hyper in self.grad_params(): #so step are gonna look at each of our paramters...

            if p not in self.state: #...and it is gonna check to see if that paramter already exsist in our -state dictornary(self.state = {})

                #if it hasnt been initailized then create a state for p and call all the statistics to initialize it.

                self.state[p] = {} #so we will initailiz it wift a empty dictornary 

                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p)) #and then we will update it with the ini_state (defined in below cell block)

            state = self.state[p] #grap the state for each parameter 

            for stat in self.stats: state = stat.update(p, state, **hyper) #call update #note its gonna use the pass arguments in the stat.udate in the below formular to use the momentum 

            compose(p, self.steppers, **state, **hyper) #now we can compose it all #note we also uses our **state 

            self.state[p] = state
#stat is a lot like steppers so when we are going to create a state, this tells it how to do that 

class Stat():

    _defaults = {}

    def init_state(self, p): raise NotImplementedError #

    def update(self, p, state, **kwargs): raise NotImplementedError
class AverageGrad(Stat): #momentum is simply avergeing the gradient #note this just a definesion of a stat class 

    _defaults = dict(mom=0.9)



    def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)} #so we are gonna create a int_state that are gonna create the inited state 

    def update(self, p, state, mom, **kwargs):

        state['grad_avg'].mul_(mom).add_(p.grad.data) #so we take whatever the gradient had before ( state['grad_avg']) we multiply it by monentum(mul_(mom)) and we add the current gradient (p.grad.data)

        return state
#we can now create our momentum stepper 

def momentum_step(p, lr, grad_avg, **kwargs): #note grad_avg is defined above 

    p.data.add_(-lr, grad_avg) #and here is how we use a momentum step which is just the averge gradient times the learning rate 

    return p
sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step,weight_decay],

                  stats=AverageGrad(), wd=0.01) #creating a sgd momentum optimazer 
learn,run = get_learn_run(nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=sgd_mom_opt)


run.fit(1, learn)
#make some experments to see what momentum does 

x = torch.linspace(-4, 4, 200) #create a 200 numbers equarly spaces between -4 and and 4 

y = torch.randn(200) + 0.3 #and lets create another 200 random numbers wth the averge pong of 0.3 

betas = [0.5, 0.7, 0.9, 0.99] #each point will go in for the calulation for momentum and used for plotting 
def plot_mom(f):

    _,axs = plt.subplots(2,2, figsize=(12,8))

    for beta,ax in zip(betas, axs.flatten()):

        ax.plot(y, linestyle='None', marker='.')

        avg,res = None,[]

        for i,yi in enumerate(y):

            avg,p = f(avg, beta, yi, i)

            res.append(p)

        ax.plot(res, color='red')

        ax.set_title(f'beta={beta}')
def mom1(avg, beta, yi, i): 

    if avg is None: avg=yi

    res = beta*avg + yi #this is the momentum function 

    return res,res

#so we plot res for each value of beta 

plot_mom(mom1)
def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2 #called exponentail weightet moving averge 


def mom2(avg, beta, yi, i):

    if avg is None: avg=yi

    avg = lin_comb(avg, yi, beta) #same ass befor but using exponentail weightet moving averge 

    return avg, avg

plot_mom(mom2)
#so lets try with a function

y = 1 - (x/3) ** 2 + torch.randn(200) * 0.1
#where y(0)=0.5 så we get a point that is out of order from the function 

y[0]=0.5
plot_mom(mom2)
def mom3(avg, beta, yi, i):

    if avg is None: avg=0

    avg = lin_comb(avg, yi, beta) #exponentail weighting moving averge from before 

    return avg, avg/(1-beta**(i+1)) #and we have to divide it with (1-beta**(i+1) ->from the formular above to debias it.

#the reason it works is because in debiasing we always start at 0, so even if we start at a abnormal point from for a function it wont effect the exponential weighng moving averge

plot_mom(mom3)
#export

class AverageGrad(Stat):

    _defaults = dict(mom=0.9)

    

    def __init__(self, dampening:bool=False): self.dampening=dampening

    def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, mom, **kwargs):

        state['mom_damp'] = 1-mom if self.dampening else 1. #so if you set dampeping to True it will 1-momentum else we will set it to 1

        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data) #same as before just with dampening 

        return state
#export

class AverageSqrGrad(Stat): #same as AvergeGrad class but we multiply p.grad by it self so we get "kvadroden"(p.grad.data, p.grad.data) of it

    #and note we store them in different names

    _defaults = dict(sqr_mom=0.99)

    

    def __init__(self, dampening:bool=True): self.dampening=dampening

    def init_state(self, p): return {'sqr_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):

        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.

        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)

        return state
#for debias we also need what step we are up to.

#so the below class just count the step  

class StepCount(Stat):

    def init_state(self, p): return {'step': 0}

    def update(self, p, state, **kwargs):

        state['step'] += 1

        return state


#exdebias function note we use formular written above

def debias(mom, damp, step): return damp * (1 - mom**step) / (1-mom)
#export

def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):

    debias1 = debias(mom,     mom_damp, step)

    debias2 = debias(sqr_mom, sqr_damp, step)

    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)

    return p

adam_step._defaults = dict(eps=1e-5)
#so this is our adam optimazer 

def adam_opt(xtra_step=None, **kwargs):

    return partial(StatefulOptimizer, steppers=[adam_step,weight_decay]+listify(xtra_step),

                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)


learn,run = get_learn_run(nfs, data, 0.001, conv_layer, cbs=cbfs, opt_func=adam_opt())
run.fit(3, learn)
#use the formulars from above 

def lamb_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, wd, **kwargs):

    debias1 = debias(mom,     mom_damp, step)

    debias2 = debias(sqr_mom, sqr_damp, step)

    r1 = p.data.pow(2).mean().sqrt()

    step = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + wd*p.data

    r2 = step.pow(2).mean().sqrt()

    p.data.add_(-lr * min(r1/r2,10), step)

    return p

lamb_step._defaults = dict(eps=1e-6, wd=0.)
lamb = partial(StatefulOptimizer, steppers=lamb_step, stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])
learn,run = get_learn_run(nfs, data, 0.003, conv_layer, cbs=cbfs, opt_func=lamb)
run.fit(3, learn)
%load_ext autoreload

%autoreload 2



%matplotlib inline
#export

from exp.nb_09b import *

import time

from fastprogress import master_bar, progress_bar

from fastprogress.fastprogress import format_time
path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)


tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

bs = 64



il = ImageList.from_files(path, tfms=tfms)

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))

ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)
nfs = [32]*4 #this gives a four 32 layers 
# export 

class AvgStatsCallback(Callback):

    def __init__(self, metrics):

        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    

    def begin_fit(self):

        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]

        names = ['epoch'] + [f'train_{n}' for n in met_names] + [

            f'valid_{n}' for n in met_names] + ['time']

        self.logger(names) #just print names at this stage 

    

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        self.start_time = time.time()

        

    def after_loss(self):

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

    

    def after_epoch(self):

        stats = [str(self.epoch)] 

        for o in [self.train_stats, self.valid_stats]:#same as before but here we are storing our stats in a array 

            stats += [f'{v:.6f}' for v in o.avg_stats] 

        stats += [format_time(time.time() - self.start_time)]

        self.logger(stats) #and we are just passing off the array to the logger 
# this makes the process bar when training  

class ProgressCallback(Callback):

    _order=-1

    def begin_fit(self):

        self.mbar = master_bar(range(self.epochs))#create mater bar which are the thing that tracks the epochs 

        self.mbar.on_iter_begin() #tell the mater bar we are starting 

        self.run.logger = partial(self.mbar.write, table=True) #and replace the logger function master bar(mbar).write so it will htlm into that

        

    def after_fit(self): self.mbar.on_iter_end() # and when we are done fitting tell the mater bar we are done 

    def after_batch(self): self.pb.update(self.iter) #after we have done a batch update our process bar 

    def begin_epoch   (self): self.set_pb() #begin epoch

    def begin_validate(self): self.set_pb() #begin validating 

        

    def set_pb(self): #make new progess bar 

        self.pb = progress_bar(self.dl, parent=self.mbar)

        self.mbar.update(self.epoch)
cbfs = [partial(AvgStatsCallback,accuracy),

        CudaCallback,

        ProgressCallback,

        partial(BatchTransformXCallback, norm_imagenette)]
learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs)
learn.fit(2)
%load_ext autoreload

%autoreload 2



%matplotlib inline
#export

from exp.nb_09c import *


#export

make_rgb._order=0
path = datasets.untar_data(datasets.URLs.IMAGENETTE)

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
def get_il(tfms): return ImageList.from_files(path, tfms=tfms)


il = get_il(tfms)
show_image(il[0])
img = PIL.Image.open(il.items[0])#open the original whitout resizing it so to see how it looks in full size 
img
img.getpixel((1,1))
import numpy as np
%timeit -n 10 a = np.array(PIL.Image.open(il.items[0]))
img.resize((128,128), resample=PIL.Image.ANTIALIAS)


img.resize((128,128), resample=PIL.Image.BILINEAR)
img.resize((128,128), resample=PIL.Image.NEAREST)
img.resize((256,256), resample=PIL.Image.BICUBIC).resize((128,128), resample=PIL.Image.NEAREST) #combining two 
%timeit img.resize((224,224), resample=PIL.Image.BICUBIC)
%timeit img.resize((224,224), resample=PIL.Image.BILINEAR)
%timeit -n 10 img.resize((224,224), resample=PIL.Image.NEAREST)
import random
def pil_random_flip(x):

    return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<0.5 else x
il1 = get_il(tfms)# create a itemlist 

il1.items = [il1.items[0]]*64#lets replace the items with just the first item with 64 copies of it 

dl = DataLoader(il1, 8)


x = next(iter(dl))
#export

def show_image(im, ax=None, figsize=(3,3)):

    if ax is None: _,ax = plt.subplots(1, 1, figsize=figsize)

    ax.axis('off')

    ax.imshow(im.permute(1,2,0))



def show_batch(x, c=4, r=None, figsize=None):

    n = len(x)

    if r is None: r = int(math.ceil(n/c))

    if figsize is None: figsize=(c*3,r*3)

    fig,axes = plt.subplots(r,c, figsize=figsize)#go through our batch and show all the images 

    for xi,ax in zip(x,axes.flat): show_image(xi, ax)
show_batch(x)
il1.tfms.append(pil_random_flip)


x = next(iter(dl))

show_batch(x)
class PilRandomFlip(Transform):

    _order=11

    def __init__(self, p=0.5): self.p=p #p is the properbilit for the image to flip in a given direction 

    def __call__(self, x):

        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x


#export

class PilTransform(Transform): _order=11



class PilRandomFlip(PilTransform):

    def __init__(self, p=0.5): self.p=p

    def __call__(self, x):

        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x


del(il1.tfms[-1])

il1.tfms.append(PilRandomFlip(0.8))#note p=0.8 so most of them are flipped 


x = next(iter(dl))

show_batch(x)
PIL.Image.FLIP_LEFT_RIGHT,PIL.Image.ROTATE_270,PIL.Image.TRANSVERSE
img = PIL.Image.open(il.items[0])

img = img.resize((128,128), resample=PIL.Image.NEAREST)

_, axs = plt.subplots(2, 4, figsize=(12, 6))

for i,ax in enumerate(axs.flatten()):

    if i==0: ax.imshow(img)

    else:    ax.imshow(img.transpose(i-1))

    ax.axis('off')
#export

class PilRandomDihedral(PilTransform):

    def __init__(self, p=0.75): self.p=p*7/8 #Little hack to get the 1/8 identity dihedral transform taken into account.

    def __call__(self, x):

        if random.random()>self.p: return x

        return x.transpose(random.randint(0,6))
del(il1.tfms[-1])

il1.tfms.append(PilRandomDihedral())
show_batch(next(iter(dl)))


img = PIL.Image.open(il.items[0])

img.size
img.crop((60,60,320,320)).resize((128,128), resample=PIL.Image.BILINEAR)


cnr2 = (60,60,320,320)

resample = PIL.Image.BILINEAR
%timeit -n 10 img.crop(cnr2).resize((128,128), resample=resample)
img.transform((128,128), PIL.Image.EXTENT, cnr2, resample=resample)
%timeit -n 10 img.transform((128,128), PIL.Image.EXTENT, cnr2, resample=resample)
#export

from random import randint



def process_sz(sz):

    sz = listify(sz)

    return tuple(sz if len(sz)==2 else [sz[0],sz[0]])



def default_crop_size(w,h): return [w,w] if w < h else [h,h]



class GeneralCrop(PilTransform):

    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR): 

        self.resample,self.size = resample,process_sz(size)

        self.crop_size = None if crop_size is None else process_sz(crop_size)

        

    def default_crop_size(self, w,h): return default_crop_size(w,h)



    def __call__(self, x):

        csize = self.default_crop_size(*x.size) if self.crop_size is None else self.crop_size

        return x.transform(self.size, PIL.Image.EXTENT, self.get_corners(*x.size, *csize), resample=self.resample)

    

    def get_corners(self, w, h): return (0,0,w,h)



class CenterCrop(GeneralCrop):

    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):

        super().__init__(size, resample=resample)

        self.scale = scale

        

    def default_crop_size(self, w,h): return [w/self.scale,h/self.scale]

    

    def get_corners(self, w, h, wc, hc):

        return ((w-wc)//2, (h-hc)//2, (w-wc)//2+wc, (h-hc)//2+hc)
il1.tfms = [make_rgb, CenterCrop(128), to_byte_tensor, to_float_tensor]
show_batch(next(iter(dl)))


# export

class RandomResizedCrop(GeneralCrop):

    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4., 4./3.), resample=PIL.Image.BILINEAR): #(3./4., 4./3.) changing the size of the person like the man is a bit thinner on and image and a bit fatter on another

        super().__init__(size, resample=resample)

        self.scale,self.ratio = scale,ratio

    

    def get_corners(self, w, h, wc, hc):

        area = w*h

        #Tries 10 times to get a proper crop inside the image.

        for attempt in range(10):

            area = random.uniform(*self.scale) * area

            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))

            new_w = int(round(math.sqrt(area * ratio)))

            new_h = int(round(math.sqrt(area / ratio)))

            if new_w <= w and new_h <= h:

                left = random.randint(0, w - new_w)

                top  = random.randint(0, h - new_h)

                return (left, top, left + new_w, top + new_h)

        

        # Fallback to squish

        if   w/h < self.ratio[0]: size = (w, int(w/self.ratio[0]))

        elif w/h > self.ratio[1]: size = (int(h*self.ratio[1]), h)

        else:                     size = (w, h)

        return ((w-size[0])//2, (h-size[1])//2, (w+size[0])//2, (h+size[1])//2)


il1.tfms = [make_rgb, RandomResizedCrop(128), to_byte_tensor, to_float_tensor]
show_batch(next(iter(dl)))


# export

from torch import FloatTensor,LongTensor



def find_coeffs(orig_pts, targ_pts):

    matrix = []

    #The equations we'll need to solve.

    for p1, p2 in zip(targ_pts, orig_pts):

        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])

        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])



    A = FloatTensor(matrix)

    B = FloatTensor(orig_pts).view(8, 1)

    #The 8 scalars we seek are solution of AX = B

    return list(torch.solve(B,A)[0][:,0])
# export

def warp(img, size, src_coords, resample=PIL.Image.BILINEAR):

    w,h = size

    targ_coords = ((0,0),(0,h),(w,h),(w,0))

    c = find_coeffs(src_coords,targ_coords)

    res = img.transform(size, PIL.Image.PERSPECTIVE, list(c), resample=resample)

    return res


targ = ((0,0),(0,128),(128,128),(128,0))

src  = ((90,60),(30,280),(310,280),(250,60))
c = find_coeffs(src, targ)

img.transform((128,128), PIL.Image.PERSPECTIVE, list(c), resample=resample)
%timeit -n 10 warp(img, (128,128), src)
%timeit -n 10 warp(img, (128,128), src, resample=PIL.Image.NEAREST)
warp(img, (64,64), src, resample=PIL.Image.BICUBIC)
warp(img, (64,64), src, resample=PIL.Image.NEAREST)


# export

def uniform(a,b): return a + (b-a) * random.random()
class PilTiltRandomCrop(PilTransform):

    def __init__(self, size, crop_size=None, magnitude=0., resample=PIL.Image.NEAREST): 

        self.resample,self.size,self.magnitude = resample,process_sz(size),magnitude

        self.crop_size = None if crop_size is None else process_sz(crop_size)

        

    def __call__(self, x):

        csize = default_crop_size(*x.size) if self.crop_size is None else self.crop_size

        up_t,lr_t = uniform(-self.magnitude, self.magnitude),uniform(-self.magnitude, self.magnitude)

        left,top = randint(0,x.size[0]-csize[0]),randint(0,x.size[1]-csize[1])

        src_corners = tensor([[-up_t, -lr_t], [up_t, 1+lr_t], [1-up_t, 1-lr_t], [1+up_t, lr_t]])

        src_corners = src_corners * tensor(csize).float() + tensor([left,top]).float()

        src_corners = tuple([(int(o[0].item()), int(o[1].item())) for o in src_corners])

        return warp(x, self.size, src_corners, resample=self.resample)
il1.tfms = [make_rgb, PilTiltRandomCrop(128, magnitude=0.1), to_byte_tensor, to_float_tensor]


x = next(iter(dl))

show_batch(x)


# export

class PilTiltRandomCrop(PilTransform):

    def __init__(self, size, crop_size=None, magnitude=0., resample=PIL.Image.BILINEAR): 

        self.resample,self.size,self.magnitude = resample,process_sz(size),magnitude

        self.crop_size = None if crop_size is None else process_sz(crop_size)

        

    def __call__(self, x):

        csize = default_crop_size(*x.size) if self.crop_size is None else self.crop_size

        left,top = randint(0,x.size[0]-csize[0]),randint(0,x.size[1]-csize[1])

        top_magn = min(self.magnitude, left/csize[0], (x.size[0]-left)/csize[0]-1)

        lr_magn  = min(self.magnitude, top /csize[1], (x.size[1]-top) /csize[1]-1)

        up_t,lr_t = uniform(-top_magn, top_magn),uniform(-lr_magn, lr_magn)

        src_corners = tensor([[-up_t, -lr_t], [up_t, 1+lr_t], [1-up_t, 1-lr_t], [1+up_t, lr_t]])

        src_corners = src_corners * tensor(csize).float() + tensor([left,top]).float()

        src_corners = tuple([(int(o[0].item()), int(o[1].item())) for o in src_corners])

        return warp(x, self.size, src_corners, resample=self.resample)


il1.tfms = [make_rgb, PilTiltRandomCrop(128, 200, magnitude=0.2), to_byte_tensor, to_float_tensor]


x = next(iter(dl))

show_batch(x)
[(o._order,o) for o in sorted(tfms, key=operator.attrgetter('_order'))]


#export

import numpy as np



def np_to_float(x): return torch.from_numpy(np.array(x, dtype=np.float32, copy=False)).permute(2,0,1).contiguous()/255.

np_to_float._order = 30


%timeit -n 10 to_float_tensor(to_byte_tensor(img))
%timeit -n 10 np_to_float(img)


il1.tfms = [make_rgb, PilTiltRandomCrop(128, magnitude=0.2), to_byte_tensor, to_float_tensor]


dl = DataLoader(il1, 64)


x = next(iter(dl))


from torch import FloatTensor
def affine_grid_cpu(size): #create a affine grid which is just koordinates of where is every pixsel 

    N, C, H, W = size

    grid = FloatTensor(N, H, W, 2)

    linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1])

    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])

    linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1])

    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])

    return grid
grid = affine_grid_cpu(x.size())


grid.shape
grid[0,:5,:5] #like litaraly where a re the pixsel koordinates from -1 to 1 


m = tensor([[1., 0., 0.], [0., 1., 0.]])

theta = m.expand(x.size(0), 2, 3)


m = tensor([[1., 0., 0.], [0., 1., 0.]])

theta = m.expand(x.size(0), 2, 3)
%timeit -n 10 grid = F.affine_grid(theta, x.size())


%timeit -n 10 grid = F.affine_grid(theta.cuda(), x.size())


def affine_grid(x, size):

    size = (size,size) if isinstance(size, int) else tuple(size)

    size = (x.size(0),x.size(1)) + size

    if x.device.type == 'cpu': return affine_grid_cpu(size) 

    m = tensor([[1., 0., 0.], [0., 1., 0.]], device=x.device)

    return F.affine_grid(m.expand(x.size(0), 2, 3), size)
grid = affine_grid(x, 128)
from torch import stack,zeros_like,ones_like


def rotation_matrix(thetas):

    thetas.mul_(math.pi/180)

    rows = [stack([thetas.cos(),             thetas.sin(),             torch.zeros_like(thetas)], dim=1),

            stack([-thetas.sin(),            thetas.cos(),             torch.zeros_like(thetas)], dim=1),

            stack([torch.zeros_like(thetas), torch.zeros_like(thetas), torch.ones_like(thetas)], dim=1)]

    return stack(rows, dim=1)


thetas = torch.empty(x.size(0)).uniform_(-30,30)
thetas[:5]
m = rotation_matrix(thetas)
m.shape, m[:,None].shape, grid.shape
grid.view(64,-1,2).shape
a = m[:,:2,:2]

b = m[:, 2:,:2]

tfm_grid = (grid.view(64,-1,2) @ a + b).view(64, 128, 128, 2)
%timeit -n 10 tfm_grid = grid @ m[:,None,:2,:2] + m[:,2,:2][:,None,None]


%timeit -n 10 tfm_grid = torch.einsum('bijk,bkl->bijl', grid, m[:,:2,:2]) + m[:,2,:2][:,None,None]
%timeit -n 10 tfm_grid = torch.matmul(grid, m[:,:2,:2].unsqueeze(1)) + m[:,2,:2][:,None,None]


%timeit -n 10 tfm_grid = (torch.bmm(grid.view(64,-1,2), m[:,:2,:2]) + m[:,2,:2][:,None]).view(-1, 128, 128, 2)
grid = grid.cuda()

m = m.cuda()
%timeit -n 10 tfm_grid = grid @ m[:,None,:2,:2] + m[:,2,:2][:,None,None]
%timeit -n 10 tfm_grid = torch.einsum('bijk,bkl->bijl', grid, m[:,:2,:2]) + m[:,2,:2][:,None,None]
%timeit -n 10 tfm_grid = torch.matmul(grid, m[:,:2,:2].unsqueeze(1)) + m[:,2,:2][:,None,None]


%timeit -n 10 tfm_grid = (torch.bmm(grid.view(64,-1,2), m[:,:2,:2]) + m[:,2,:2][:,None]).view(-1, 128, 128, 2)
tfm_grid = torch.bmm(grid.view(64,-1,2), m[:,:2,:2]).view(-1, 128, 128, 2)


tfm_x = F.grid_sample(x, tfm_grid.cpu())


show_batch(tfm_x, r=2)
tfm_x = F.grid_sample(x, tfm_grid.cpu(), padding_mode='reflection')# padding_mode='reflection' removes the black 'hjørner'


show_batch(tfm_x, r=2)
def rotate_batch(x, size, degrees):

    grid = affine_grid(x, size)

    thetas = x.new(x.size(0)).uniform_(-degrees,degrees)

    m = rotation_matrix(thetas)

    tfm_grid = grid @ m[:,:2,:2].unsqueeze(1) + m[:,2,:2][:,None,None]

    return F.grid_sample(x, tfm_grid)


show_batch(rotate_batch(x, 128, 30), r=2)


%timeit -n 10 tfm_x = rotate_batch(x, 128, 30)
%timeit -n 10 tfm_x = rotate_batch(x.cuda(), 128, 30)


from torch import Tensor


from torch.jit import script



@script

def rotate_batch(x:Tensor, size:int, degrees:float) -> Tensor:

    sz = (x.size(0),x.size(1)) + (size,size)

    idm = torch.zeros(2,3, device=x.device)

    idm[0,0] = 1.

    idm[1,1] = 1.

    grid = F.affine_grid(idm.expand(x.size(0), 2, 3), sz) #do the affine grid here instead of above work just fine 

    thetas = torch.zeros(x.size(0), device=x.device).uniform_(-degrees,degrees)

    m = rotation_matrix(thetas)

    tfm_grid = torch.matmul(grid, m[:,:2,:2].unsqueeze(1)) + m[:,2,:2].unsqueeze(1).unsqueeze(2)

    return F.grid_sample(x, tfm_grid)
m = tensor([[1., 0., 0.], [0., 1., 0.]], device=x.device)
%timeit -n 10 tfm_x = rotate_batch(x.cuda(), 128, 30)
def rotate_batch(x, size, degrees):

    size = (size,size) if isinstance(size, int) else tuple(size)

    size = (x.size(0),x.size(1)) + size

    thetas = x.new(x.size(0)).uniform_(-degrees,degrees)

    m = rotation_matrix(thetas)

    grid = F.affine_grid(m[:,:2], size)

    return F.grid_sample(x.cuda(), grid)
%timeit -n 10 tfm_x = rotate_batch(x.cuda(), 128, 30)