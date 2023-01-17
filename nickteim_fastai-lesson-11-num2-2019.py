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
from fastai import *

from fastai.vision import *
path = untar_data(URLs.IMAGENETTE_160)
import PIL,os,mimetypes #PIL python Image Liberary 

Path.ls = lambda x: list(x.iterdir())


path.ls()
(path/'val').ls() #note we have one directory for each category 
path_tench = path/'val'/'n01440764' #look at one category 
img_fn = path_tench.ls()[0] #grap one file name 

img_fn
img = PIL.Image.open(img_fn) #look at the one file name 

img
plt.imshow(img)
import numpy

imga = numpy.array(img) #turning image into an array so we can see that properties it has 

imga.shape
imga[:10,:10,0] #print the image in numbers and note it is of dtype=unit8 which means it contains bits so they are numbers of integers not float 
#export

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/')) #image "udvidelser" that the computer already knows about 

#we use mimetypes for this. 
' '.join(image_extensions) #so here is the images the mimetypes knows exsist and note we want to use this to get only images when we train


def setify(o): return o if isinstance(o,set) else set(listify(o)) #so now we can loop though each file in the directory and see which files there are 

#so the fastets way to check is something is in a list we first put it in a 'set' therefor we create a function called 'setify'

#it simply check if it a set (isinstance(o,set)) else it will it first turnes it into a list and then a set (set(listify(o)))

#just testing it is working 

# test_eq(setify('aa'), {'aa'})

# test_eq(setify(['aa',1]), {'aa',1})

# test_eq(setify(None), set())

# test_eq(setify(1), {1})

# test_eq(setify({1}), {1})
#go thorugh a singel directory and grap the images in that 

def _get_files(p, fs, extensions=None): #get files #p =path and fs=list of files

    p = Path(p) #just makes sure u can use Path 

    res = [p/f for f in fs if not f.startswith('.') #so we go throught the list of files(fs) and makes sure it doesnt start ith '.' because if it does ot is a unit or mac hidden file

           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)] #check if we asked for extensions and if the ektension is in the list of extensions

    return res
#so now we can grap the image files 

t = [o.name for o in os.scandir(path_tench)] #python has function called 'scandir' which will take a the path ('path_tench') and list all the files in that path 

t = _get_files(path, t, extensions=image_extensions) #so here is how we are gonna call _get_files.

t[:3]#and we are just gonna show the first 3 files 

#we can rewrite the code above and put it together and we get this

def get_files(path, extensions=None, recurse=False, include=None):

    path = Path(path)

    extensions = setify(extensions)

    extensions = {e.lower() for e in extensions}

    if recurse: #if recurse sat to True this code will be exsicuted

        res = []

        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)

            if include is not None and i==0: d[:] = [o for o in d if o in include]

            else:                            d[:] = [o for o in d if not o.startswith('.')] #change the list of directories if you want to

            res += _get_files(p, f, extensions)

        return res

    else: #other wise if recurse is False this code wil exsicute

        f = [o.name for o in os.scandir(path) if o.is_file()]

        return _get_files(path, f, extensions)

    

    #one thing to note is that scandir is super fast 
get_files(path_tench, image_extensions)[:3]#use the function above and see we can grap tre files in this way also  
get_files(path, image_extensions, recurse=True)[:3] #here we use recurse because got a few levels of directories strutures before we get to the pictures
all_fns = get_files(path, image_extensions, recurse=True) #so now we want to get all the file names 

len(all_fns)#and we can see we have 13394


%timeit -n 10 get_files(path, image_extensions, recurse=True) #and we can see it takes about 70 ms which is fast
#export just a list that contains a lot of usefull features not inparticular needed

class ListContainer():

    def __init__(self, items): self.items = listify(items)

    def __getitem__(self, idx): #being clled when we are useing 'firkant parantesterne' for correct operation usage 

        if isinstance(idx, (int,slice)): return self.items[idx] #bla bla bla ...

        if isinstance(idx[0],bool): #bla bla bla...

            assert len(idx)==len(self) # bool mask

            return [o for m,o in zip(idx,self.items) if m]

        return [self.items[i] for i in idx]

    def __len__(self): return len(self.items) #return lenght 

    def __iter__(self): return iter(self.items) #iteration

    def __setitem__(self, i, o): self.items[i] = o 

    def __delitem__(self, i): del(self.items[i])

    def __repr__(self): #printing

        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'

        if len(self)>10: res = res[:-1]+ '...]'

        return res
def compose(x, funcs, *args, order_key='_order', **kwargs):

    key = lambda o: getattr(o, order_key, 0) #check to see if the _order is used and then will sort them in line below (sorted)

    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)

    return x



class ItemList(ListContainer): #ListContainer comes from last lesson and it written in the hidden cell above 

    #so besic what ItemList does is making sure we can get any kind of data 

    def __init__(self, items, path='.', tfms=None):

        #items in this case is the filenames and the path (path='.') they came from and optinaly we can use tfms(transformed items)

        super().__init__(items)

        self.path,self.tfms = Path(path),tfms



    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'

    

    #below 'new' function create a new itemlist of the type of items we pass in

    def new(self, items, cls=None):

        if cls is None: cls=self.__class__ #so if cls(class) is not defined set cls = to whatever class the object is 

            #this chould be any class like ImageList or another class that havent been defined yet. (in this case it is ItemList)

        return cls(items, self.path, tfms=self.tfms) #and let pass it in the items that we asked for and pass in our path and our transform

    #so all in all 'new' is gonna create a new ItemList with the same type, with the same path and the same transform but with new items (this is uses longer done in code)

    

    def  get(self, i): return i #overwrite get to return the item itself which in this chase whould be the file name 

    def _get(self, i): return compose(self.get(i), self.tfms) #it will call the get method (defined at buttom) and call open the image and then it wil compose the transforms

    #compose it a concept that where you go through a list of functions (sorted(listify(funcs)) and call the function (x = f(x,) --> means replace x with the result of given function when looping through each function  that 

    #note a deep neural network is just composes function where each layer is a function and we compose them all together.

    #so all in all what is does is it will modify the image with the transform function (tfms).

    

    def __getitem__(self, idx): #when you index into you itemlist 

        res = super().__getitem__(idx) #we will pass this back up to ListContainer and this will either return one item or a list of items 

        if isinstance(res,list): return [self._get(o) for o in res] #if it is a list of items we will call self._get(o) on all of them (for o in res)

        return self._get(res) #if it is a singel item we will just call 



class ImageList(ItemList):

    @classmethod

    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):

        if extensions is None: extensions = image_extensions

        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    

    def get(self, fn): return PIL.Image.open(fn) #when you get something from the ImageList it should open the image 
#So this is our fist trans form 

class Transform(): _order=0 #_order means it is the first thing it should do 



class MakeRGB(Transform):

    def __call__(self, item): return item.convert('RGB') #... so what will do is call pillow. convert and 'RGB', which means if something is not RGB it will turn it into RGB 

    #remembter __call__ wil treat the class function as if it only was a function 



#instead of the class above we can just make it into a function and we will get the same result     

def make_rgb(item): return item.convert('RGB')


il = ImageList.from_files(path, tfms=make_rgb) #so now lets use it, here will only use the function make_rgb 
il #view the itemlist and remember that itemlist inherit from ListContainer which had a __repr__ so we get all the nice printing 
img = il[0]; img#index into it 
il[:1] #we can also use splice since we wrote the function for it in ListContainer 
fn = il.items[0]; fn
fn.parent.parent.name #this is the grandparrent file since it goes through 2 dictories to get to the train folder 
#So now lets make a function that grap the grandparent file 

def grandparent_splitter(fn, valid_name='valid', train_name='train'):

    gp = fn.parent.parent.name #grap the grandparent name  

    return True if gp==valid_name else False if gp==train_name else None



def split_by_func(items, f):

    mask = [f(o) for o in items] #create a mask where you pass it same function (f)

    # `None` values will be filtered out

    f = [o for o,m in zip(items,mask) if m==False] #grap all the thing when it is true (grandparent_splitter) here is f=train

    t = [o for o,m in zip(items,mask) if m==True ]#grap all the thing when it is false (grandparent_splitter) here is t=valid

    return f,t #and it will return them 
splitter = partial(grandparent_splitter, valid_name='val') #so here are the grandparant name for valid called 'val' in imagenette. 

#and so we find all the validation tings 
%time train,valid = split_by_func(il, splitter) #and so we split the data up in training and validation sets 
len(train),len(valid) #and now we see we got 500 valdation things and the rest is for training 
#so now lets use it 

class SplitData():

    def __init__(self, train, valid): self.train,self.valid = train,valid #store the train and valid varibles 

        

    def __getattr__(self,k): return getattr(self.train,k) #dounder get attribute, so if we pass it a attribute it doenst know about it will take it from the training dataset 

    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors

    def __setstate__(self,data:Any): self.__dict__.update(data) 

    

    @classmethod

    def split_by_func(cls, il, f):

        lists = map(il.new, split_by_func(il.items, f)) #so it will call the split_by_func function we defined above 

        #note we are using il.new where new is a function defined in ItemList above. Note that we now have create a training set 

        # and a validation set with the same path and the sam transform and the same type 

        return cls(*lists)



    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n' #we give it a representation when we print it 
sd = SplitData.split_by_func(il, splitter); sd #so now when we call it, we can see we got our training set and or validation set 
from collections import OrderedDict



def uniqueify(x, sort=False): #so to convert labels to numbers (int) we need to know all the posible labels so therefor we just need to find all the uniique things in a list (x=list)

    #so the 2 lines below is how to get the unique thing from a list 

    res = list(OrderedDict.fromkeys(x).keys())

    if sort: res.sort() #if we set it to True return the list to the function called 'sort' from above else

    return res#just return the list 
#so now let create a processor 

class Processor(): 

    def process(self, items): return items #and a processer sinply is just something at can process some items 



    #a category processer is just a processer that is the thing that create a list of all our posible categories

class CategoryProcessor(Processor):

    def __init__(self): self.vocab=None

    

    def process(self, items): #could also use __call__ 

        #The vocab is defined on the first use. 

        if self.vocab is None: #check if there is a vocab yet and if there is not, this must be the training set 

            self.vocab = uniqueify(items) #so we will create a vocab and it is just the unique values of all the items 

            self.otoi  = {v:k for k,v in enumerate(self.vocab)} #and now we want something that goes from object to int

            #so this is reversed mapping so we enumerate the vocab and create a dicornary with the revered mapping(v:k for k,v)

        return [self.proc1(o) for o in items] #so now since we have a vocab we can go through ever item and process them one at the time (proc1)

    def proc1(self, item):  return self.otoi[item]# and process one (proc1) simply means look ind the revered mappong 

   

    #we can also deprocess which takes a lot of indexses (inxs)

    def deprocess(self, idxs):

        assert self.vocab is not None #make sure we have vocab otherwice we cant do anything 

        return [self.deproc1(idx) for idx in idxs] # and then we just deprocess one (deproc1) for each index

    def deproc1(self, idx): return self.vocab[idx] #and deprocess one(deproc1) just looks it up in the vocab (vocab[idx])

    

#so now we can combine it all in together 

class ProgessedItemList(ListContainer):

    def __init__(self, inputs, processor):

        self.processor=processor #contains a processer 

        items=processor.process(inputs) #and the items in it was whatever it waas given after it was processed(inputs)

        super().__init__(items)

        

    def obj(self, idx): #so object(obj) and that is just the thing that is going to...

        res=sef[idx]

        if isinstance(res(tuple,list,Generator)): return self.processor.deprocess(res) #... deprocess the items again

        return self.processor.deproc1(idx)



    #so this is all the stuff we need to label things 
#we fund that for the splitting we needed the grandparant but for the labeling we need the paratens 

def parent_labeler(fn): return fn.parent.name #so this is just a parant labeler 



def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path) #so this is just a function that label things it just 

#call 'f(o)' (function) for each thing in the item list 



#This is a slightly different from what was seen during the lesson,

#   we'll discuss the changes in lesson 11

class LabeledData():

    def process(self, il, proc): return il.new(compose(il.items, proc))



    def __init__(self, x, y, proc_x=None, proc_y=None): #pass it its independed varible(x) and a depended varible(y)

        self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)

        self.proc_x,self.proc_y = proc_x,proc_y

        

    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n' #make it print out nicely 

    def __getitem__(self,idx): return self.x[idx],self.y[idx] #a indexser to grap the x and to grap the y 

    def __len__(self): return len(self.x) #we need a lenthg 

    

    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)

    

    def obj(self, items, idx, procs):

        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)

        item = items[idx]

        for proc in reversed(listify(procs)):

            item = proc.deproc1(item) if isint else proc.deprocess(item)

        return item



    @classmethod

    def label_by_func(cls, il, f, proc_x=None, proc_y=None): #does the labeling  

        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)



def label_by_func(sd, f, proc_x=None, proc_y=None):

    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y) #note that in the validation set will use the trainings set vovab

    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)

    return SplitData(train,valid)
ll = label_by_func(sd, parent_labeler)
assert ll.train.proc_y is ll.valid.proc_y
ll.train.y
ll.train.y.items[0], ll.train.y_obj(0), ll.train.y_obj(slice(2)) #to get the name of a categorieor just some of them 
ll
ll.train[0]
ll.train[0][0] 
ll.train[0][0].resize((128,128)) #for all to be in the same batch they all need the same size so we just use resize 
#so below is a transform that resizes things 

class ResizeFixed(Transform):

    _order=10 #and it has to be after all the ohter transforms 

    def __init__(self,size): #tages a size

        if isinstance(size,int): size=(size,size) #if you passed in a int it will turn it into a tuble 

        self.size = size

        

    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR) #and when you call it, it wil do resize and it will do linear resizing 



    #now they all have the same size we can turn them all into tensors 

def to_byte_tensor(item):

    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))

    w,h = item.size

    return res.view(h,w,-1).permute(2,0,1)

to_byte_tensor._order=20 #it hass to be done after the resizing so it gets a lesser order. note we can add orders to functions aswell 



#since the above turns it into a byte tensor and we need a float tensor we can do like:

def to_float_tensor(item): return item.float().div_(255.) #we divide it since we dont want it between 0 and 255 but between 0 and 1, so we divide it by 255

to_float_tensor._order=30 #it gets a higher order since the above function needs to run first 
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor] #so this is the list of transofrms we have 



il = ImageList.from_files(path, tfms=tfms) #we can pass that to out imagelist 

sd = SplitData.split_by_func(il, splitter) #we can split it 

ll = label_by_func(sd, parent_labeler) # we can label it 


#export

def show_image(im, figsize=(3,3)):

    plt.figure(figsize=figsize)

    plt.axis('off')

    plt.imshow(im.permute(1,2,0))
x,y = ll.train[0]

x.shape
show_image(x)
bs=64
def get_dls(train_ds, valid_ds, bs, **kwargs):

    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),

            DataLoader(valid_ds, batch_size=bs*2, **kwargs)) #since we dont have to do the backward pass, so we have twice as much space

            # therefor we can multipy the batch size with 2.
train_dl,valid_dl = get_dls(ll.train,ll.valid,bs, num_workers=4) #we will use the get dataloader from before(get_dls)

#and pass in the ll.train and ll.valid directly from our labeled list 
x,y = next(iter(train_dl)) #lets grap a minibatch 
x.shape #and here we can se the minibacth 
show_image(x[0])

ll.train.y
y
class DataBunch():

    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None, c=None):

        self.train_dl,self.valid_dl,self.c_in,self.c_out = train_dl,valid_dl,c_in,c_out #c_in is the numbers it need in input and c_out is the correct numbers for are databunch



    @property

    def train_ds(self): return self.train_dl.dataset



    @property

    def valid_ds(self): return self.valid_dl.dataset
def databunchify(sd, bs, c_in=None, c_out=None,c=None, **kwargs):

    dls = get_dls(sd.train, sd.valid, bs, **kwargs)

    return DataBunch(*dls, c_in=c_in, c_out=c_out)



SplitData.to_databunch = databunchify
path = untar_data(URLs.IMAGENETTE_160) #grap the path

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor] #grap the transform 



il = ImageList.from_files(path, tfms=tfms) #grap the itemlist 

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val')) #slit the data 

ll = label_by_func(sd, parent_labeler) #label it 

data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4) #create databunch with 3 chanels in and 10 chanels out and 4 processers(num_worker)
#export

class AvgStatsCallback(Callback):

    def __init__(self, metrics):

        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

        

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

        

    def after_loss(self):

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

    

    def after_epoch(self):

        print(self.train_stats)

        print(self.valid_stats)

        

class Recorder(Callback):

    def begin_fit(self):

        self.lrs = [[] for _ in self.opt.param_groups]

        self.losses = []



    def after_batch(self):

        if not self.in_train: return

        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])

        self.losses.append(self.loss.detach().cpu())        



    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])

        

    def plot(self, skip_last=0, pgid=-1):

        losses = [o.item() for o in self.losses]

        lrs    = self.lrs[pgid]

        n = len(losses)-skip_last

        plt.xscale('log')

        plt.plot(lrs[:n], losses[:n])



class ParamScheduler(Callback):

    _order=1

    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

        

    def begin_fit(self):

        if not isinstance(self.sched_funcs, (list,tuple)):

            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)



    def set_param(self):

        assert len(self.opt.param_groups)==len(self.sched_funcs)

        for pg,f in zip(self.opt.param_groups,self.sched_funcs):

            pg[self.pname] = f(self.n_epochs/self.epochs)

            

    def begin_batch(self): 

        if self.in_train: self.set_param()
class CudaCallback(Callback):

    def begin_fit(self): self.model.cuda() #note we dont have to move it to a device we just use .cuda

    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()
cbfs = [partial(AvgStatsCallback,accuracy),

        CudaCallback] #callback function from last time 
m,s = x.mean((0,2,3)),x.std((0,2,3))

m,s #normalize 
#create function that normalize things that have 3 chanels 

def normalize_chan(x, mean, std):

    return (x-mean[...,None,None]) / std[...,None,None]



_m = tensor([0.47, 0.45,  0.42]) #mean of the imagenette 

_s = tensor([0.27, 0.27, 0.29])#std of the imagenette 

norm_imagenette = partial(normalize_chan, mean=_m, std=_s)
#previus when we wrote the transforming of mnist it will only work with it but here we will make a more generel transformation of the data 

class BatchTransformXCallback(Callback): #transform the indepened verible (X-training data) for a batch 

    _order=2

    def __init__(self, tfm): self.tfm = tfm #you pass it som etransformation function (tfm) which its stores away 

    def begin_batch(self): self.run.xb = self.tfm(self.xb) #begin batch just replaces the current batch (xb) with the result of the transformation (tfm)



def view_tfm(*size): #view_tfm takes and transform the size

    def _inner(x): return x.view(*((-1,)+size))

    return _inner
cbfs.append(partial(BatchTransformXCallback, norm_imagenette)) #add the above to a callback 
nfs = [64,64,128,256] #create a conv net of these layers 
import math

def prev_pow_2(x): return 2**math.floor(math.log2(x))



def get_cnn_layers(data, nfs, layer, **kwargs):

    def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)

    l1 = data.c_in #knows how big the first layer has to be (c_in) 

    l2 = prev_pow_2(l1*3*3) #secound layer takes the 3 by 3 kernel and mulitipy it by next biggest power of two to that number 

    layers =  [f(l1  , l2  , stride=1), #tird layer takes previus output 

               f(l2  , l2*2, stride=2), #and multiply first by 2

               f(l2*2, l2*4, stride=2)] # and then by 4 

    nfs = [l2*4] + nfs #and the next layers is whatever we asked for (nfs is defined above)

    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]

    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten), 

               nn.Linear(nfs[-1], data.c_out)] #note also the c_out is predefined and is how many classes we have in our data 

    return layers



def get_cnn_model(data, nfs, layer, **kwargs):

    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))



def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, **kwargs):

    model = get_cnn_model(data, nfs, layer, **kwargs)

    init_cnn(model)

    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)
#dont think to much about the below code 

def combine_scheds(pcts, scheds):

    assert sum(pcts) == 1.

    pcts = tensor([0] + listify(pcts))

    assert torch.all(pcts >= 0)

    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):

        idx = (pos >= pcts).nonzero().max()

        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])

        return scheds[idx](actual_pos)

    return _inner
#a decorator is a function that returns a function 

def annealer(f):

    def _inner(start, end): return partial(f, start, end)

    return _inner



@annealer #the annealer decorator does the same as the partial 

def sched_lin(start, end, pos): return start + pos*(end-start)


#export

@annealer

def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer

def sched_no(start, end, pos):  return start #no schedular return always start

@annealer

def sched_exp(start, end, pos): return start * (end/start) ** pos



def cos_1cycle_anneal(start, high, end):

    return [sched_cos(start, high), sched_cos(high, end)]



#This monkey-patch is there to be able to plot tensors

torch.Tensor.ndim = property(lambda x: len(x.shape))
sched = combine_scheds([0.3,0.7], cos_1cycle_anneal(0.1,0.3,0.05)) #one cycle scheduling 
#export

class Lambda(nn.Module):

    def __init__(self, func):

        super().__init__()

        self.func = func



    def forward(self, x): return self.func(x) #the only thing it does is



def flatten(x):      return x.view(x.shape[0], -1)
#we are gonna try a few things to fix the above problem 

def get_cnn_layers(data, nfs, layer, **kwargs): #note we use **kwargs since we can get extra arguments from GeneralRelu class

    nfs = [1] + nfs

    return [layer(nfs[i], nfs[i+1], 5 if i==0 else 3, **kwargs) #note we use **kwargs since we can get extra arguments from GeneralRelu class

            for i in range(len(nfs)-1)] + [

        nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c_out)]



def conv_layer(ni, nf, ks=3, stride=2, **kwargs): #note we use **kwargs since we can get extra arguments from GeneralRelu class

    return nn.Sequential(

        nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), GeneralRelu(**kwargs)) #note we use **kwargs since we can get extra arguments from GeneralRelu class



#better Relu 

class GeneralRelu(nn.Module):

    def __init__(self, leak=None, sub=None, maxv=None): #so now we can use subret from the relu(which we found was good about 0.5)

        #and we can handle leaking (leak). And maybe we want a limit so we can use maximum value (maxv)

        super().__init__()

        self.leak,self.sub,self.maxv = leak,sub,maxv



    def forward(self, x): 

        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x) #so if you pass leak(from GeneralRelu) it will

        #use leaky_relu otherwise we will use normal relu (F.relu(x))

        if self.sub is not None: x.sub_(self.sub) #if you want to subrat something: go do that

        if self.maxv is not None: x.clamp_max_(self.maxv) #if you want to use maksimum value: go do that

        return x



def init_cnn(m, uniform=False): #uniform boolien 

    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_ #so some people say uniform is better then normal, so it is an optinal see buttom of this part when used

    for l in m:

        if isinstance(l, nn.Sequential):

            f(l[0].weight, a=0.1)

            l[0].bias.data.zero_()



def get_cnn_model(data, nfs, layer, **kwargs):

    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))
from torch.nn import init
#lets put the model and the layers intot he function below 

def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func = F.cross_entropy): 

    if opt_func is None: opt_func = optim.SGD #grap optmisation function

    opt = opt_func(model.parameters(), lr=lr) #grap the optimazer

    learn = Learner(model, opt, loss_func, data) #grap our learner 

    return learn, Runner(cb_funcs=listify(cbs)) 
learn,run = get_learn_run(nfs, data, 0.2, conv_layer, cbs=cbfs+[

    partial(ParamScheduler, 'lr', sched)

])


#export

def model_summary(run, learn, data, find_all=False):

    xb,yb = get_batch(data.valid_dl, run)

    device = next(learn.model.parameters()).device#Model may not be on the GPU yet

    xb,yb = xb.to(device),yb.to(device)

    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()

    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")

    with Hooks(mods, f) as hooks: learn.model(xb)
model_summary(run, learn, data)


%time run.fit(5, learn)