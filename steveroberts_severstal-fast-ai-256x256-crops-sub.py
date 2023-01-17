NotebookTitle = "Steel Defect Segmentation - Inference"
%reload_ext autoreload

%autoreload 2

%matplotlib inline



import fastai

from fastai.vision import *

from PIL import Image

import zipfile

import io

import cv2

import warnings

warnings.filterwarnings("ignore")



fastai.__version__
nfolds = 1#4

bs = 4

n_cls = 4

noise_th = 2000 #predicted masks must be larger than noise_th (noise threshold)

TEST = '../input/severstal-steel-defect-detection/test_images/'

BASE = '../input/severstal-fast-ai-256x256-crops/models/'



torch.backends.cudnn.benchmark = True
#the code below modifies fast.ai functions to incorporate Hcolumns into fast.ai Dynamic Unet



from fastai.vision.learner import create_head, cnn_config, num_features_model, create_head

from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook, _hook_inner

from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock



class Hcolumns(nn.Module):

    def __init__(self, hooks:Collection[Hook], nc:Collection[int]=None):

        super(Hcolumns,self).__init__()

        self.hooks = hooks

        self.n = len(self.hooks)

        self.factorization = None 

        if nc is not None:

            self.factorization = nn.ModuleList()

            for i in range(self.n):

                self.factorization.append(nn.Sequential(

                    conv2d(nc[i],nc[-1],3,padding=1,bias=True),

                    conv2d(nc[-1],nc[-1],3,padding=1,bias=True)))

                #self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))

        

    def forward(self, x:Tensor):

        n = len(self.hooks)

        out = [F.interpolate(self.hooks[i].stored if self.factorization is None

            else self.factorization[i](self.hooks[i].stored), scale_factor=2**(self.n-i),

            mode='bilinear',align_corners=False) for i in range(self.n)] + [x]

        return torch.cat(out, dim=1)



class DynamicUnet_Hcolumns(SequentialEx):

    "Create a U-Net from a given architecture."

    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, 

                 self_attention:bool=False,

                 y_range:Optional[Tuple[float,float]]=None,

                 last_cross:bool=True, bottle:bool=False, **kwargs):

        imsize = (256,256)

        sfs_szs = model_sizes(encoder, size=imsize)

        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))

        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])

        x = dummy_eval(encoder, imsize).detach()



        ni = sfs_szs[-1][1]

        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),

                                    conv_layer(ni*2, ni, **kwargs)).eval()

        x = middle_conv(x)

        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]



        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]

        hc_c = [x.shape[1]]

        

        for i,idx in enumerate(sfs_idxs):

            not_final = i!=len(sfs_idxs)-1

            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])

            do_blur = blur and (not_final or blur_final)

            sa = self_attention and (i==len(sfs_idxs)-3)

            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, 

                blur=blur, self_attention=sa, **kwargs).eval()

            layers.append(unet_block)

            x = unet_block(x)

            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))

            hc_c.append(x.shape[1])



        ni = x.shape[1]

        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))

        if last_cross:

            layers.append(MergeLayer(dense=True))

            ni += in_channels(encoder)

            layers.append(res_block(ni, bottle=bottle, **kwargs))

        hc_c.append(ni)

        layers.append(Hcolumns(self.hc_hooks, hc_c))

        layers += [conv_layer(ni*len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]

        if y_range is not None: layers.append(SigmoidRange(*y_range))

        super().__init__(*layers)



    def __del__(self):

        if hasattr(self, "sfs"): self.sfs.remove()

            

def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,

        norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 

        blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, 

        last_cross:bool=True, bottle:bool=False, cut:Union[int,Callable]=None, 

        hypercolumns=True, **learn_kwargs:Any)->Learner:

    "Build Unet learner from `data` and `arch`."

    meta = cnn_config(arch)

    body = create_body(arch, pretrained, cut)

    M = DynamicUnet_Hcolumns if hypercolumns else DynamicUnet

    model = to_device(M(body, n_classes=data.c, blur=blur, blur_final=blur_final,

        self_attention=self_attention, y_range=y_range, norm_type=norm_type, 

        last_cross=last_cross, bottle=bottle), data.device)

    learn = Learner(data, model, **learn_kwargs)

    learn.split(ifnone(split_on, meta['split']))

    if pretrained: learn.freeze()

    apply_init(model[2], nn.init.kaiming_normal_)

    return learn



class SegmentationLabelList(SegmentationLabelList):

    def open(self, fn): return open_mask(fn, div=True)

    

class SegmentationItemList(SegmentationItemList):

    _label_cls = SegmentationLabelList



# Setting transformations on masks to False on test set

def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):

    if not tfms: tfms=(None,None)

    assert is_listy(tfms) and len(tfms) == 2

    self.train.transform(tfms[0], **kwargs)

    self.valid.transform(tfms[1], **kwargs)

    kwargs['tfm_y'] = False # Test data has no labels

    if self.test: self.test.transform(tfms[1], **kwargs)

    return self

fastai.data_block.ItemLists.transform = transform



def open_mask(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=ImageSegment,

        after_open:Callable=None)->ImageSegment:

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", UserWarning)

        x = PIL.Image.open(fn).convert(convert_mode)

    if after_open: x = after_open(x)

    x = pil2tensor(x,np.float32)

    return cls(x)
# Prediction with flip TTA

def model_pred(learns, F_save,

        ds_type:fastai.basic_data.DatasetType=DatasetType.Valid, 

        tta:bool=True): #if use train dl, disable shuffling

    

    # put all the models into evaluation mode

    for learn in learns: learn.model.eval();

        

    # get the data that is to be used for testing

    dl = learn.data.dl(ds_type)

    #sampler = dl.batch_sampler.sampler

    #dl.batch_sampler.sampler = torch.utils.data.sampler.SequentialSampler(sampler.data_source)

    name_list = [Path(n).stem for n in dl.dataset.items]

    num_batchs = len(dl)

    t = progress_bar(iter(dl), leave=False, total=num_batchs)

    count = 0

    

    # don't want to calculate gradients during evaluation

    with torch.no_grad():

        for x,y in t:

            x = x.cuda()

            preds = []

            for learn in learns:

                #i, hights, widths, classes

                py = torch.softmax(learn.model(x),dim=1).permute(0,2,3,1).detach()

                if tta:

                    #you can comment some transfromations to save time

                    flips = [[-1],[-2],[-2,-1]]

                    for f in flips:

                        py += torch.softmax(torch.flip(learn.model(torch.flip(x,f)),f),dim=1).permute(0,2,3,1).detach()

                    py /= len(flips) + 1

                preds.append(py)

            py = torch.stack(preds).mean(0).cpu().numpy() # taking average of all preds

            batch_size = len(py)

            for i in range(batch_size):

                taget = y[i].detach().cpu().numpy() if y is not None else None

                F_save(py[i],taget,name_list[count])

                count += 1

    #dl.batch_sampler.sampler = sampler

    

def save_img(data,name,out):

    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]

    out.writestr(name, img)

    

#dice for threshold selection

def dice_np(pred, targs, e=1e-7):

    targs = targs[0,:,:]

    pred = np.dstack([1.0 - pred.sum(-1), pred])

    c = pred.shape[-1]

    pred = np.argmax(pred, axis=-1)

    dices = []

    eps = 1e-7

    for i in range(1,c):

        intersect = ((pred==i) & (targs==i)).sum().astype(np.float)

        union = ((pred==i).sum() + (targs==i).sum()).astype(np.float)

        dices.append((2.0*intersect + eps) / (union + eps))

    return np.array(dices).mean()
def enc2mask(encs, shape=(1600,256)):

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for m,enc in enumerate(encs):

        if isinstance(enc,np.float) and np.isnan(enc): continue

        s = enc.split()

        for i in range(len(s)//2):

            start = int(s[2*i]) - 1

            length = int(s[2*i+1])

            img[start:start+length] = 1 + m

    return img.reshape(shape).T



def mask2enc(mask, n=n_cls):

    pixels = mask.T.flatten()

    encs = []

    for i in range(1,n+1):

        p = (pixels == i).astype(np.int8)

        if p.sum() == 0: encs.append('')

        else:

            p = np.concatenate([[0], p, [0]])

            runs = np.where(p[1:] != p[:-1])[0] + 1

            runs[1::2] -= runs[::2]

            encs.append(' '.join(str(x) for x in runs))

    return encs
stats = ([0.396,0.396,0.396], [0.179,0.179,0.179])

#check https://www.kaggle.com/iafoss/256x256-images-with-defects for stats



data = (SegmentationItemList.from_folder(TEST)

        .split_by_idx([0])

        .label_from_func(lambda x : str(x), classes=[0,1,2,3,4])

        .add_test(Path(TEST).ls(), label=None)

        .databunch(path=Path('.'), bs=bs)

        .normalize(stats))
rles,ids_test = [],[]

learns = []

for fold in range(nfolds):

    learn = unet_learner(data, models.resnet34, pretrained=False)

    learn.model.load_state_dict(torch.load(Path(BASE)/f'fold{fold}.pth')['model'])

    learns.append(learn)



with zipfile.ZipFile('pred.zip', 'w') as archive_out:

    def to_mask(yp, y, id):

        name = id + '.png'

        save_img(yp[:,:,1:],name,archive_out)

        yp = np.argmax(yp, axis=-1)

        for i in range(n_cls):

            idxs = yp == i+1

            if idxs.sum() < noise_th: yp[idxs] = 0

        encs = mask2enc(yp)

        for i, enc in enumerate(encs):

            ids_test.append(id + '.jpg_' + str(i+1))

            rles.append(enc)

    

    model_pred(learns,to_mask,DatasetType.Test)

    

sub_df = pd.DataFrame({'ImageId_ClassId': ids_test, 'EncodedPixels': rles})

sub_df.sort_values(by='ImageId_ClassId').to_csv('submission.csv', index=False)