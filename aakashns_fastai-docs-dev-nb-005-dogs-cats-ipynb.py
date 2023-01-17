# Clean up
!rm -rf data nb_001b.py nb_002.py nb_002b.py nb_002c.py nb_004.py 

# Install additional libraries
!pip -q install dataclasses
!pip -q install fastprogress

# Download old code 
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_001b.py
!wget -q https://gist.githubusercontent.com/aakashns/3e05dc120569bf269ade9f3ac1b1a677/raw/e855471c38b18854e55e84734a991c720503de51/nb_002.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_002b.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_002c.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_003.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_004.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_004a.py
!wget -q https://raw.githubusercontent.com/fastai/fastai_docs/master/dev_nb/nb_004b.py
    
# Download and untar data
!wget http://files.fast.ai/data/dogscats.zip
!unzip -q dogscats.zip

# Create data directories
!mkdir data
!mv dogscats data/dogscats

# Clean up
!rm dogscats.zip
from nb_004b import *
import torchvision.models as tvm
PATH = Path('data/dogscats')
arch = tvm.resnet34
def uniform_int(low:Number, high:Number, size:Optional[List[int]]=None)->FloatOrTensor:
    "Generate int or tensor `size` of ints from uniform(`low`,`high`)"
    return random.randint(low,high) if size is None else torch.randint(low,high,size)

@TfmPixel
def dihedral(x, k:partial(uniform_int,0,8)):
    "Randomly flip `x` image based on k"
    flips=[]
    if k&1: flips.append(1)
    if k&2: flips.append(2)
    if flips: x = torch.flip(x,flips)
    if k&4: x = x.transpose(1,2)
    return x.contiguous()
def get_transforms(do_flip:bool=True, flip_vert:bool=False, max_rotate:float=10., max_zoom:float=1.1, 
                   max_lighting:float=0.2, max_warp:float=0.2, p_affine:float=0.75, 
                   p_lighting:float=0.75, xtra_tfms:float=None)->Collection[Transform]:
    "Utility func to easily create list of `flip`, `rotate`, `zoom`, `warp`, `lighting` transforms"
    res = [rand_crop()]
    if do_flip:    res.append(dihedral() if flip_vert else flip_lr(p=0.5))
    if max_warp:   res.append(symmetric_warp(magnitude=(-max_warp,max_warp), p=p_affine))
    if max_rotate: res.append(rotate(degrees=(-max_rotate,max_rotate), p=p_affine))
    if max_zoom>1: res.append(rand_zoom(scale=(1.,max_zoom), p=p_affine))
    if max_lighting:
        res.append(brightness(change=(0.5*(1-max_lighting), 0.5*(1+max_lighting)), p=p_lighting))
        res.append(contrast(scale=(1-max_lighting, 1/(1-max_lighting)), p=p_lighting))
    #       train                   , valid
    return (res + listify(xtra_tfms), [crop_pad()])  

imagenet_stats = tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225])
imagenet_norm,imagenet_denorm = normalize_funcs(*imagenet_stats)
size=224

tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.2, max_lighting=0.3, max_warp=0.15)
data = data_from_imagefolder(PATH, test='test1', bs=64, ds_tfms=tfms,
                             num_workers=0, tfms=imagenet_norm, size=size)
(x,y) = next(iter(data.valid_dl))

_,axs = plt.subplots(4,4,figsize=(12,12))
for i,ax in enumerate(axs.flatten()): show_image(imagenet_denorm(x[i].cpu()), ax)
(x,y) = next(iter(data.test_dl))

_,axs = plt.subplots(4,4,figsize=(12,12))
for i,ax in enumerate(axs.flatten()): show_image(imagenet_denorm(x[i].cpu()), ax)
x=data.valid_ds[2][0]
_,axes = plt.subplots(2,4, figsize=(12,6))
for i,ax in enumerate(axes.flat): dihedral(x,i).show(ax)
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def create_body(model:Model, cut:Optional[int]=None, body_fn:Callable[[Model],Model]=None):
    "Cut off the body of a typically pretrained model at `cut` or as specified by `body_fn`"
    return (nn.Sequential(*list(model.children())[:cut]) if cut
            else body_fn(model) if body_fn else model)

def num_features(m:Model)->int:
    "Return the number of output features for a model"
    for l in reversed(flatten_model(m)):
        if hasattr(l, 'num_features'): return l.num_features
model = create_body(arch(), -2)
num_features(model)
def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5):
    """Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes.
       `ps` is for dropout and can be a single float or a list for each layer"""
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns): 
        layers += bn_drop_lin(ni,no,True,p,actn)
    return nn.Sequential(*layers)
create_head(512, 2)
LayerFunc = Callable[[nn.Module],None]

def cond_init(m:nn.Module, init_fn:LayerFunc):
    "Initialize the non-batchnorm layers"
    if (not isinstance(m, bn_types)) and requires_grad(m):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)

def apply_leaf(m:nn.Module, f:LayerFunc):
    "Apply `f` to children of m"
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)

def apply_init(m, init_fn:LayerFunc): 
    "Initialize all non-batchnorm layers of model with `init_fn`"
    apply_leaf(m, partial(cond_init, init_fn=init_fn))

def _init(learn, init): apply_init(learn.model, init)
Learner.init = _init

def _default_split(m:Model): 
    "By default split models between first and second layer"
    return split_model(m, m[1])

def _resnet_split(m:Model):  
    "Split a resnet style model"
    return split_model(m, (m[0][6],m[1]))

_default_meta = {'cut':-1, 'split':_default_split}
_resnet_meta  = {'cut':-2, 'split':_resnet_split }

model_meta = {
    tvm.resnet18 :{**_resnet_meta}, tvm.resnet34: {**_resnet_meta}, 
    tvm.resnet50 :{**_resnet_meta}, tvm.resnet101:{**_resnet_meta}, 
    tvm.resnet152:{**_resnet_meta}}

class ConvLearner(Learner):
    "Builds convnet style learners"
    def __init__(self, data:DataBunch, arch:Callable, cut=None, pretrained:bool=True, 
                 lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                 custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None, **kwargs:Any)->None:
        meta = model_meta.get(arch, _default_meta)
        torch.backends.cudnn.benchmark = True
        body = create_body(arch(pretrained), ifnone(cut,meta['cut']))
        nf = num_features(body) * 2
        head = custom_head or create_head(nf, data.c, lin_ftrs, ps)
        model = nn.Sequential(body, head)
        super().__init__(data, model, **kwargs)
        self.split(ifnone(split_on,meta['split']))
        if pretrained: self.freeze()
        apply_init(model[1], nn.init.kaiming_normal_)
learn = ConvLearner(data, arch, metrics=accuracy)
learn.fit_one_cycle(1)
learn.save('0')
learn.load('0')
learn.unfreeze()
lr=1e-4
learn.fit_one_cycle(1, slice(lr/25,lr), pct_start=0.05)
learn.save('1')
learn.load('1')
model = learn.model
def pred_batch(learn:Learner, is_valid:bool=True) -> Tuple[Tensors, Tensors, Tensors]:
    "Returns input, target and output of the model on a batch"
    x,y = next(iter(learn.data.valid_dl if is_valid else learn.data.train_dl))
    return x,y,learn.model(x).detach()
Learner.pred_batch = pred_batch

def get_preds(model:Model, dl:DataLoader, pbar:Optional[PBar]=None) -> List[Tensor]:
    "Predicts the output of the elements in the dataloader"
    return [torch.cat(o).cpu() for o in validate(model, dl, pbar=pbar)]

def _learn_get_preds(learn:Learner, is_test:bool=False) -> List[Tensor]:
    "Wrapper of get_preds for learner"
    return get_preds(learn.model, learn.data.holdout(is_test))
Learner.get_preds = _learn_get_preds

def show_image_batch(dl:DataLoader, classes:Collection[str], rows:int=None, figsize:Tuple[int,int]=(12,15), 
                     denorm:Callable=None) -> None:
    "Show a few images from a batch"
    x,y = next(iter(dl))
    if rows is None: rows = int(math.sqrt(len(x)))
    x = x[:rows*rows].cpu()
    if denorm: x = denorm(x)
    show_images(x,y[:rows*rows].cpu(),rows, classes)
def _tta_only(learn:Learner, is_test:bool=False, scale:float=1.25) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.data.holdout(is_test)
    ds = dl.dataset
    old = ds.tfms
    augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in
               (crop_pad, flip_lr, dihedral, zoom)]
    try:
        pbar = master_bar(range(8))
        for i in pbar:
            row = 1 if i&1 else 0
            col = 1 if i&2 else 0
            flip = i&4
            d = {'row_pct':row, 'col_pct':col, 'is_random':False}
            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
            if flip: tfm.append(flip_lr(p=1.))
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar)[0]
    finally: ds.tfms = old
        
Learner.tta_only = _tta_only
t = list(learn.tta_only(scale=1.35))
preds,y = get_preds(model, data.valid_dl)
accuracy(preds, y)
avg_preds = torch.stack(t).mean(0)
avg_preds.shape, accuracy(avg_preds, y)
accuracy(preds*0.5 + avg_preds*0.5, y)
[(beta,accuracy(preds*beta + avg_preds*(1-beta), y)) for beta in np.linspace(0,1,11)]
def _TTA(learn:Learner, beta:float=0.4, scale:float=1.35, is_test:bool=False) -> Tensors:
    preds,y = learn.get_preds(is_test)
    all_preds = list(learn.tta_only(scale=scale, is_test=is_test))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:            return preds*beta + avg_preds*(1-beta), y

Learner.TTA = _TTA
learn = ConvLearner(data, arch, metrics=accuracy, path=PATH)
learn.load('1')
tta_preds = learn.TTA()
accuracy(*tta_preds)
# Clean up
!rm -rf data
