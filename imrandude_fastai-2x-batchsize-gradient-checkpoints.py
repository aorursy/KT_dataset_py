from fastai import *

from fastai.vision import *



path = untar_data(URLs.PETS)

path_anno = path/'annotations'

path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
bs = 16*5
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet101, metrics=error_rate)
learn.model
from fastai.utils.mem import GPUMemTrace

with GPUMemTrace():

  learn.fit_one_cycle(4)
########################################

## Defaults

########################################

import torch

import torch.nn as nn

from torch.utils.checkpoint import checkpoint, checkpoint_sequential



from fastai.callbacks.hooks import *



def cnn_config(arch):

    "Get the metadata associated with `arch`."

    torch.backends.cudnn.benchmark = True

    return model_meta.get(arch, _default_meta)



def _default_split(m:nn.Module): return (m[1],)

def _resnet_split(m:nn.Module): return (m[0][6],m[1])



_default_meta    = {'cut':None, 'split':_default_split}

_resnet_meta     = {'cut':-2, 'split':_resnet_split }



model_meta = {

    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},

    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},

    models.resnet152:{**_resnet_meta}}

########################################

## Custom Checkpoint

########################################

class CheckpointModule(nn.Module):

    def __init__(self, module, num_segments=1):

        super(CheckpointModule, self).__init__()

        assert num_segments == 1 or isinstance(module, nn.Sequential)

        self.module = module

        self.num_segments = num_segments



    def forward(self, *inputs):

        if self.num_segments > 1:

            return checkpoint_sequential(self.module, self.num_segments, *inputs)

        else:

            return checkpoint(self.module, *inputs)



########################################

# Extract the sequential layers for resnet

########################################

def layer_config(arch):

    "Get the layers associated with `arch`."

    return model_layers.get(arch)



model_layers = {

    models.resnet18 :[2, 2, 2, 2], models.resnet34: [3, 4, 6, 3],

    models.resnet50 :[3, 4, 6, 3], models.resnet101:[3, 4, 23, 3],

    models.resnet152:[3, 8, 36, 3]}
########################################

## Send sequential layers in custom_body to Checkpoint

########################################

def create_body1(arch:Callable, pretrained:bool=True, cut:Optional[Union[int, Callable]]=None):

    "Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)` (function)."

    model = arch(pretrained)

    cut = ifnone(cut, cnn_config(arch)['cut'])

    dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    if cut is None:

        ll = list(enumerate(model.children()))

        cut = next(i for i,o in reversed(ll) if has_pool_type(o))

    if   isinstance(cut, int):

    #Checkpoint - Changes Start

      if (arch.__name__).find("resnet")==0:       # Check if the Model is resnet                                                        

        n = 4                                     # Initial 4 Layers didn't have sequential and were not applicable with Checkpoint

        layers = layer_config(arch)               # Fetch the sequential layer split

        out = nn.Sequential(*list(model.children())[:cut][:n],

                            *[CheckpointModule(x, min(checkpoint_segments, layers[i])) for i, x in enumerate(list(model.children())[:cut][n:])])

        # Join the Initial 4 layers with Checkpointed sequential layers

      else:

        out = nn.Sequential(*list(model.children())[:cut])

      return out

    #Checkpoint - Changes End

    elif isinstance(cut, Callable): return cut(model)

    else:                           raise NamedError("cut must be either integer or a function")

## From base - function renamed

def create_head1(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,

                concat_pool:bool=True, bn_final:bool=False):

    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."

    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]

    ps = listify(ps)

    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps

    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]

    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)

    layers = [pool, Flatten()]

    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):

        layers += bn_drop_lin(ni, no, True, p, actn)

    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

    return nn.Sequential(*layers)



## From base - function renamed

def create_cnn1_model1(base_arch:Callable, nc:int, cut:Union[int,Callable]=None, pretrained:bool=True,

                     lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,

                     bn_final:bool=False, concat_pool:bool=True):

    "Create custom convnet architecture"

    body = create_body1(base_arch, pretrained, cut)

    if custom_head is None:

        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)

        head = create_head1(nf, nc, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)

    else: head = custom_head

    return nn.Sequential(body, head)



## From base - function renamed

def cnn_learner1(data:DataBunch, base_arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,

                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,

                split_on:Optional[SplitFuncOrIdxList]=None, bn_final:bool=False, init=nn.init.kaiming_normal_,

                concat_pool:bool=True, **kwargs:Any)->Learner:

    "Build convnet style learner."

    meta = cnn_config(base_arch)

    model = create_cnn1_model1(base_arch, data.c, cut, pretrained, lin_ftrs, ps=ps, custom_head=custom_head,

        bn_final=bn_final, concat_pool=concat_pool)

    learn = Learner(data, model, **kwargs)

    learn.split(split_on or meta['split'])

    if pretrained: learn.freeze()

    if init: apply_init(model[1], init)

    return learn
## Clear redundant Memory

gc.collect()

import torch

torch.cuda.empty_cache()

learn.purge()

del data

del learn
bs = bs * 2

checkpoint_segments = 4

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

learn = cnn_learner1(data, models.resnet101, metrics=error_rate)
learn.model
from fastai.utils.mem import GPUMemTrace

with GPUMemTrace():

  learn.fit_one_cycle(4)