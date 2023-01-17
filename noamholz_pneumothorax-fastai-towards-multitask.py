import os, sys, time

from datetime import datetime

sys.path.insert(0, '/kaggle/input/siim-acr-pneumothorax-segmentation')

sys.path.insert(0, '/kaggle/input/pneum-scripts')



import pneum_aux

from pathlib import Path

import pandas as pd

from tqdm import tqdm_notebook as tqdm

import glob

import pdb

from IPython.display import FileLink, FileLinks

from scipy import ndimage



import fastai

from fastai.vision import *

from mask_functions import *

import torchvision.transforms as transforms

import pydicom

fastai.__version__

class ParamsManager(object):

    def __init__(self):    self.config = {}

    def __call__(self, key, value, method='override'):

#         assert key not in self.config.keys(), f"a value has already been set for {key}."

        if method == 'override':

            self.config[key] = value

        return value

    def __getitem__(self, key):    return self.config[key]

    def name(self):    return '_'.join([k + '-' + str(self.config[k]) for k in self.config])



    

def save_weight_n_ipynb(learn, work_path, prms):

    modelname = prms.name()

    learn.model_dir = work_path + '/'

    model_path = learn.model_dir + modelname

    ipynb_out_path = model_path + '.ipynb'

    if not os.path.isdir(learn.model_dir): os.mkdir(learn.model_dir)

    %notebook $ipynb_out_path

    learn.save(model_path)

    FileLinks('.')
pred_sig_id = 0

valid_fraction = 0.2



prms = ParamsManager()

prms.config = {'date': datetime.today().strftime('%m%d'), 

                'SEED': 42, 

                'SZ': 512,

                'BS': 4 

               }

   

    

if prms['SEED'] >= 0:

    pneum_aux.seed_everything(prms['SEED'])



#### Paths

# copy pretrained weights for resnet34 to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '/kaggle/input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'

work_path = '/kaggle/working'

csv_path = '/kaggle/input/siim-train-test/siim/train-rle.csv'

saved_dfs_folder = '/kaggle/input/saved-dfs/'





# IMAGE SIZE

if prms['SZ'] <= 256:

    path = Path(f'/kaggle/input/pneumotorax{prms["SZ"]}/data{prms["SZ"]}/data{prms["SZ"]}')

    prms('data','nohisteq')

elif prms['SZ'] == 512:

    path = Path(f'/kaggle/input/pneumothorax_{prms["SZ"]}x{prms["SZ"]}_png/data{prms["SZ"]}')

    prms('data','nohisteq')

#     path = Path(f'/kaggle/input/siimacr-pneumothorax-segmentation-data-512')

#     prms('data','histeq')

elif prms['SZ'] == 1024:

#     path = Path(f'/kaggle/input/pneumothorax_{model_params["SZ"]}x{model_params["SZ"]}_png/data{model_params["SZ"]}')

    path = Path(f'/kaggle/input/siimacr-pneumothorax-segmentation-data-1024')

    prms('data','histeq')

# print(learn.model_dir)



#### define labels df

df, df_bal = pneum_aux.make_dfs(csv_path, saved_dfs_folder)

# prms('data',prms['data'] + '-blncd')  # somehow document that the data is balanced

# model_params['data'] = model_params['data'] + '-blncd'



# check split

pneum_frac_train = sum(df.loc[df['is_valid']==False, 'EncodedPixels'] !='-1')/sum(df['is_valid']==False)

pneum_frac_valid = sum(df.loc[df['is_valid']==True, 'EncodedPixels'] !='-1')/sum(df['is_valid']==True)

valid_frac = df.is_valid.mean()

print(f'pneum examples / tot examples [train] = {pneum_frac_train:.2f}')

print(f'pneum examples / tot examples [test] = {pneum_frac_valid:.2f}')

print(f'validation examples / tot examples = {valid_frac:.2f}')

class BCE_Custom_Loss_MultiTask(nn.Module):



    def __init__(self, num_classes=1, dice_rebal_fact=9, pix_imbal_fact=0):

        super().__init__()

        self.num_classes = num_classes

        self.dice_rebal_fact = dice_rebal_fact

        self.pix_imbal_fact = pix_imbal_fact

        self.__name__ = 'BCE_Custom_Loss'

        self.activefun = 'sigmoid'



    def forward(self, pred, targ_img, targ_meta):

        t_mask = targ_img[:, 0, :, :].type(torch.FloatTensor).contiguous().cpu()  # note that in addition to the channel-index, squeeze() would omit batch-index when bs=1

        p_mask = pred[0][:, 0, :, :].cpu()

        t_gender = targ_meta[:, 0].cpu()

        p_gender = pred[1][:,0].cpu()

        t_pneum = targ_meta[:, 1].cpu()

        p_pneum = pred[1][:,1].cpu()

        

        w_pneum = (1 + self.pix_imbal_fact)

        w_nonpneun = (1 + self.dice_rebal_fact * (1- t_pneum))

        w = w_pneum * t_mask + w_nonpneun[:,None,None] * (1 - t_mask)



        loss_mask = F.binary_cross_entropy_with_logits(p_mask, t_mask, w, reduction='mean')

        loss_gender = F.binary_cross_entropy_with_logits(p_gender, t_gender, reduction='mean')

        loss_pneum = F.binary_cross_entropy_with_logits(p_pneum, t_pneum, reduction='mean')

        return loss_mask + loss_gender + loss_pneum  # TODO: add relative weights to the loss terms





class MultitaskDataset(Dataset):

    '''`Dataset` for joint single and multi-label image classification.'''

    def __init__(self, fns, meta_dict={}):  #labels_gender, labels_pneum, ages, classes_pneum, classes_gender):

        self.x = np.array(fns)

        self.meta_dict = meta_dict

        if len(self.meta_dict) > 0:

            self.classes_gender = meta_dict['classes_gender']

            self.classes_pneum = meta_dict['classes_pneum']



            self.class2idx_gender = {v:k for k,v in enumerate(self.classes_gender)}

            self.y_gender = np.array([self.class2idx_gender[o] for o in meta_dict['labels_gender']], dtype=np.int64)



            self.class2idx_pneum = {v:k for k,v in enumerate(self.classes_pneum)}

            self.y_pneum = np.array([self.class2idx_pneum[o] for o in meta_dict['labels_pneum']], dtype=np.int64)



            self.y_age = meta_dict['ages'][:, None].astype('float32')



            self.c = 1  #self.c_gender + self.c_pneum + self.c_age

    

    def __len__(self): return len(self.x)

    

    def __getitem__(self,i:int): 

        if len(self.meta_dict) > 0:

            img = open_image(self.x[i])

            return img, [open_mask(self.x[i].replace('train', 'masks'), div=True), torch.tensor([self.y_gender[i], self.y_pneum[i], self.y_age[i]]).float()]

        else:

#             oi = open_image(self.x[i]), 

            return [open_image(self.x[i]), torch.tensor([0]).float()]

        

    def __repr__(self): return f'{type(self).__name__} of len {len(self)}'

    

     

def get_dataset(df):

    return MultitaskDataset(str(path/'train') + '/' + df.ImageId + '.png',

                            meta_dict={'labels_gender': df.Sex, 

                            'labels_pneum': df.is_pneum,

                            'ages': df.Age,

                            'classes_pneum': sorted(set(df.is_pneum)),

                            'classes_gender': sorted(set(df.Sex))

                            }

                           )



# The way I constructed MultitaskDataset is not consistent enough with fastai's design, which raises difficulties in applying augmentations to the samples

# Augs are therefore skipped till this will be solved.



# # Setting transformations on masks to False on test set

# def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):

#     if not tfms: tfms=(None,None)

#     assert is_listy(tfms) and len(tfms) == 2

#     self.train.transform(tfms[0], **kwargs)

#     self.valid.transform(tfms[1], **kwargs)

#     kwargs['tfm_y'] = False # Test data has no labels

#     if self.test: self.test.transform(tfms[1], **kwargs)

#     return self

# fastai.data_block.ItemLists.transform = transform



# tfms = ([rotate(degrees=90)],[])





# train_df = df_bal[df_bal.is_valid==False].sample(prms['BS']*30)

# valid_df = train_df.copy()

train_df = df[df.is_valid==False]  #.sample(prms['BS']*30)

valid_df = df[df.is_valid==True]  #.sample(prms['BS']*30)

train_df = pd.concat([train_df, train_df.iloc[:prms['BS'] - (len(train_df) % prms['BS'])]])

valid_df = pd.concat([valid_df, valid_df.iloc[:prms['BS'] - (len(valid_df) % prms['BS'])]])

train_ds = get_dataset(train_df)

valid_ds = get_dataset(valid_df)

test_ds = MultitaskDataset((path/'test').ls(), meta_dict={})

data2 = ImageDataBunch.create(train_ds=train_ds, valid_ds=valid_ds,test_ds=test_ds, path=str(path/'train'), bs=prms['BS']).normalize(imagenet_stats)

#   dl_tfms=get_transforms(flip_vert=True), 

# data2.test_ds = test_ds

# data2.test_dl = DataLoader(test_ds, batch_size=prms['BS'], shuffle=False)

# data2_test = ImageDataBunch.create(test_ds, test_ds, path=str(path/'train'), bs=prms['BS']).normalize(imagenet_stats)

# data2.add_test(test_ds)

#  .add_test((path/'test').ls(), label=None))

print('train n = ',len(data2.train_ds), '. valid n = ', len(data2.valid_ds))

print('train n % bs = ', len(train_df) % prms['BS'], '. valid n % bs = ', len(valid_df) % prms['BS'])

# debug

# for i in range(10):

#     fig, ax = plt.subplots(1,2)

#     ax[0].imshow(data2.train_ds[i][0].data[0,:,:]*0.9 + data2.train_ds[i][1][0].data[0,:,:].float())

# learn.data = data2

# preds, _ = learn.get_preds(ds_type=DatasetType.Test)

def MT_metrics(metric_func, *args, **kwargs):

    @functools.wraps(metric_func)

    def wrapper(inputs, targs_seg, targs_clf, **kwargs):

        n = targs_seg.shape[0]

        input_seg, input_clf = inputs

        input_seg = inputs[0].sigmoid().view(n,-1)  #torch.softmax(input_seg, dim=1)[:,1,...].view(n,-1)

        input_clf = inputs[1].sigmoid().view(n,3,-1)

        targs_seg = targs_seg.view(n,-1)

        targs_clf = targs_clf.view(n,3,-1)

        return metric_func(input_seg, input_clf, targs_seg, targs_clf, **kwargs)

    return wrapper



@MT_metrics

def mydice(input_seg:Tensor, input_clf:Tensor, targs_seg:Tensor, targs_clf:Tensor, thrs=0.9, eps:float=1e-8):

    input_seg = (input_seg > thrs).long()

    input_seg[input_seg.sum(-1) < 0,...] = 0.0 

    intersect = (input_seg * targs_seg).sum(-1).float()

    union = (input_seg + targs_seg).sum(-1).float()

#     return ((2.0*intersect + eps) / (union+eps)).mean()

    u0 = union==0

    intersect[u0], union[u0] = 1, 2

#     pdb.set_trace()

    return (2. * intersect / union)

    

@MT_metrics

def mygender(input_seg:Tensor, input_clf:Tensor, targs_seg:Tensor, targs_clf:Tensor, iou:bool=False, eps:float=1e-8):

    input_gender = (input_clf[:,0] > 0.5).long()

    targs_gender = targs_clf[:,0].long()

    return (input_gender == targs_gender).float().mean()

    

@MT_metrics

def mypneum(input_seg:Tensor, input_clf:Tensor, targs_seg:Tensor, targs_clf:Tensor, iou:bool=False, eps:float=1e-8):

    input_gender = (input_clf[:,1] > 0.5).long()

    targs_gender = targs_clf[:,1].long()

    return (input_gender == targs_gender).float().mean()



# debug metrics

# print(mydice(torch.Tensor([[0,1,2],[3,4,5]]), torch.Tensor([[1,1,1]]).long(), torch.Tensor([[1,1,1]]).long()))

# print(mygender(torch.Tensor([[0,1,2],[3,4,5]]), torch.Tensor([[1,1,1]]).long(), torch.Tensor([[1,1,1]]).long()))
class AccumulateOptimWrapper(OptimWrapper):

    def step(self):           pass

    def zero_grad(self):      pass

    def real_step(self):      super().step()

    def real_zero_grad(self): super().zero_grad()

        

        

def acc_create_opt(self, lr:Floats, wd:Floats=0.):

        "Create optimizer with `lr` learning rate and `wd` weight decay."

        self.opt = AccumulateOptimWrapper.create(self.opt_func, lr, self.layer_groups,

                                         wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

Learner.create_opt = acc_create_opt





def set_BN_momentum(model,momentum=0.1*prms['BS']/64):

    for i, (name, layer) in enumerate(model.named_modules()):

        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):

            layer.momentum = momentum

        

@dataclass

class AccumulateStep(LearnerCallback):

    """

    Does accumlated step every nth step by accumulating gradients

    """

    def __init__(self, learn:Learner, n_step:int = 1):

        super().__init__(learn)

        self.n_step = n_step

        set_BN_momentum(learn.model)

        

        

    def on_epoch_begin(self, **kwargs):

        "init samples and batches, change optimizer"

        self.acc_batches = 0

        

    def on_batch_begin(self, last_input, last_target, **kwargs):

        "accumulate samples and batches"

        self.acc_batches += 1

        

    def on_backward_end(self, **kwargs):

        "step if number of desired batches accumulated, reset samples"

        if (self.acc_batches % self.n_step) == self.n_step - 1:

            for p in (self.learn.model.parameters()):

                 if p.requires_grad: p.grad.div_(self.acc_batches)

    

            self.learn.opt.real_step()

            self.learn.opt.real_zero_grad()

            self.acc_batches = 0

    

    def on_epoch_end(self, **kwargs):

        "step the rest of the accumulated grads"

        if self.acc_batches > 0:

            for p in (self.learn.model.parameters()):

                if p.requires_grad: p.grad.div_(self.acc_batches)

            self.learn.opt.real_step()

            self.learn.opt.real_zero_grad()

            self.acc_batches = 0

           

        

@dataclass

class KeepKernelAlive(LearnerCallback):

    def __init__(self,time_interval):

        self.time_interval = time_interval

    def reset_time(self): self.time0 = time.time()

    def on_epoch_begin(self, **kwargs): self.reset_time()

    def on_batch_begin(self, **kwargs):

        if time.time() - self.time0 > self.time_interval:

            self.reset_time()

            print('Kernel pulse...')          

# @dataclass

# class GetGradNorm(LearnerCallback):

#     """

#     stores & plots grad norm

#     """

#     def __init__(self, learn:Learner, n_step:int = 1):

#         super().__init__(learn)

#         self.gnorms, self.gnorms_tot = [], []

#         self.fig, self.ax = plt.subplots(ncols=2,nrows=1, figsize=(4,4))



#     def on_backward_end(self, **kwargs):

#         "calc total grads norm"

#         parameters = list(filter(lambda p: p.grad is not None, learn.model.parameters()))

#         total_norm = 0

#         for p in parameters:

#             param_norm = p.grad.data.norm(2)

#             total_norm += param_norm.item() ** 2

#         total_norm = total_norm ** (1. / 2)

#         self.gnorms.append(total_norm)

    

#     def on_epoch_begin(self, **kwargs):

#         self.gnorms_tot.append(self.gnorms)

#         self.gnorms = []

        

#     def on_epoch_end(self, **kwargs):

#         "plot the total grads norms"

#         self.ax[0].plot(self.gnorms)

#         plt.show()

    

#     def on_training_end(self, **kwargs):

#         self.ax[1].plot(np.hstack(get_grad_norm.gnorms_tot[2:]));

#         plt.show()

        



# class RunningBatchNorm(nn.Module):

#     def __init__(self, nf, mom=0.1, eps=1e-5):

#         super().__init__()

#         self.mom,self.eps = mom,eps

#         self.mults = nn.Parameter(torch.ones (nf,1,1))

#         self.adds = nn.Parameter(torch.zeros(nf,1,1))

#         self.register_buffer('sums', torch.zeros(1,nf,1,1))

#         self.register_buffer('sqrs', torch.zeros(1,nf,1,1))

#         self.register_buffer('batch', tensor(0.))

#         self.register_buffer('count', tensor(0.))

#         self.register_buffer('step', tensor(0.))

#         self.register_buffer('dbias', tensor(0.))



#     def update_stats(self, x):

#         bs,nc,*_ = x.shape

#         self.sums.detach_()

#         self.sqrs.detach_()

#         dims = (0,2,3)

#         s = x.sum(dims, keepdim=True)

#         ss = (x*x).sum(dims, keepdim=True)

#         c = self.count.new_tensor(x.numel()/nc)

#         mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)

#         self.mom1 = self.dbias.new_tensor(mom1)

#         self.sums.lerp_(s, self.mom1)

#         self.sqrs.lerp_(ss, self.mom1)

#         self.count.lerp_(c, self.mom1)

#         self.dbias = self.dbias*(1-self.mom1) + self.mom1

#         self.batch += bs

#         self.step += 1



#     def forward(self, x):

#         if self.training: self.update_stats(x)

#         sums = self.sums

#         sqrs = self.sqrs

#         c = self.count

#         if self.step<100:

#             sums = sums / self.dbias

#             sqrs = sqrs / self.dbias

#             c    = c    / self.dbias

#         means = sums/c

#         vars = (sqrs/c).sub_(means*means)

#         if bool(self.batch < 20): vars.clamp_min_(0.01)

#         x = (x-means).div_((vars.add_(self.eps)).sqrt())

#         return x.mul_(self.mults).add_(self.adds)

    



#### Test loss function

# from torch.autograd import Variable as V

# x,y = next(iter(data.valid_dl))

# # x,y = V(x).cpu(),V(y)

# x,y = V(x),V(y)



# for i,o in enumerate(y): y[i] = o.cuda()

# learn.model.cuda()



# batch = learn.model(x)

# learn.loss_func(batch, y)

# Extend the unet model to preform classification at the encoder end

class MultiTaskUnetResnet(nn.Module):

    def __init__(self,pretrained_unet_resnet34):

        super(MultiTaskUnetResnet, self).__init__()

        self.unet_resnet34=pretrained_unet_resnet34

        self.clf_layer = AdaptiveConcatPool2d((1,2))

        self.fc = nn.Linear(2*1024, 3)





    def forward(self, x):

        res = x

    

        res.orig = x

        nres = self.unet_resnet34[0](res)

        res.orig = None

        res = nres

        

        res_clf = res.clone()

        res_clf = self.clf_layer(res_clf)

        res_clf = self.fc(res_clf.view(res_clf.shape[0],-1))

        

        for l in (self.unet_resnet34[1:]):

            res.orig = x

            nres = l(res)

            res.orig = None

            res = nres

            

        return res,  res_clf 

# learn.model, _ = convert_layers(learn.model, 41, nn.BatchNorm2d, RunningBatchNorm, convert_weights=False)



# Function to replace layer of a certain type with an alternative

# https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7

def convert_layers(model, num_to_convert, layer_type_old, layer_type_new, convert_weights=False):

    conversion_count = 0

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:

            # recurse

            model._modules[name], num_converted = convert_layers(module, num_to_convert-conversion_count, layer_type_old, layer_type_new, convert_weights)

            conversion_count += num_converted



        if type(module) == layer_type_old and conversion_count < num_to_convert:

            layer_old = module

            layer_new = layer_type_new(module.num_features, mom=0.1, eps=1e-5).cuda()



            if convert_weights == True:

                layer_new.weight = layer_old.weight

                layer_new.bias = layer_old.bias



            model._modules[name] = layer_new

            conversion_count += 1



    return model, conversion_count
arch = models.resnet34

# Create U-Net with a pretrained resnet34 as encoder

learn = unet_learner(data2, arch, metrics=[mydice, mygender, mypneum])  # , opt_func=optim.SGD

prms('arch', 'unet:' + arch.__name__)

learn.loss_func = BCE_Custom_Loss_MultiTask(num_classes=1, dice_rebal_fact=prms('dice_rebal_fact', 0), pix_imbal_fact=prms('pix_imbal_fact',0))

prms('loss', 'BCECostum_dicefact')



learn.model = MultiTaskUnetResnet(learn.model).cuda()
# Find LR

learn.model_dir= '/kaggle/working/'

learn.lr_find(start_lr=1e-09, end_lr=1e-1, stop_div=True, num_it=100)

learn.recorder.plot()
# Fit

# learn.load('/kaggle/input/date0805-seed42-sz512/date-0805_SEED-42_SZ-512_crop_fact-2_BS-8_data-histeq--blncd_arch-unet_resnet34_loss-BCECostum_dicefact9_lr0-0.0001_cycles0-6')

n_acc = 64//prms['BS']

learn.fit_one_cycle(prms('cycles0',1), max_lr=prms('lr0',1e-5), callbacks=[AccumulateStep(learn, n_acc), KeepKernelAlive(time_interval=60*30)])

save_weight_n_ipynb(learn, work_path, prms)

FileLinks('.')

    

print(torch.cuda.get_device_properties(0).total_memory/1024/1024)

print(torch.cuda.memory_allocated()/1024/1024)
# Unfreeze the encoder

# learn.load('/kaggle/input/date0811-seed42-sz512/date-0811_SEED-42_SZ-512_crop_fact-2_BS-8_data-nohisteq_arch-unet_resnet34_dice_rebal_fact-10_pix_imbal_fact-10_loss-BCECostum_dicefact_cycles1-4_lr1-0.0001')

learn.model_dir= '/kaggle/working/'



learn.unfreeze()

learn.lr_find(start_lr=1e-11, end_lr=1e-1, stop_div=True)  # stop_div=False)

learn.recorder.plot()
# Fit unfreezed



# learn.load('/kaggle/input/date0805-seed42-sz512/date-0805_SEED-42_SZ-512_crop_fact-2_BS-8_data-histeq--blncd_arch-unet_resnet34_loss-BCECostum_dicefact9_lr0-0.0001_cycles0-6')

n_acc = 64//prms['BS']

# get_grad_norm = GetGradNorm(learn)

learn.fit_one_cycle(prms('cycles1',1), slice(prms('lr1',1e-8)/30, prms('lr1',1e-8)), callbacks=[AccumulateStep(learn, n_acc), KeepKernelAlive(time_interval=60*40)])

save_weight_n_ipynb(learn, work_path, prms)

FileLinks('.')

    

print(torch.cuda.get_device_properties(0).total_memory/1024/1024)

print(torch.cuda.memory_allocated()/1024/1024)
# get predictions on data withou exhausting RAM



# learn.load('/kaggle/input/1707-blncd-nohisteq-512/modelsdate-0717_SEED-42_SZ-512_BS-8_augs-rotate_data-nohisteq_balanced_arch-unet_resnet34_lr0-0.0001_cycles0-10')

# learn.data.batch_size = 8

if 'preds' in globals():    del preds

if 'ys' in globals():    del ys

    

frac = 1

thrs = np.arange(0.5, 1, 0.01)



def mypreds(prms,dl=data2.valid_dl,frac=frac, thrs=[0.5], batch_size=None):

    if batch_size is not None:

        bs = batch_size

    else:

        bs = prms['BS']

    orig_bs = dl.batch_size

    dl.batch_size = bs

    

    dices = []

    all_preds = []

    example_cases = []

    batch_iter = iter(dl)

    dummy4dice = torch.tensor(np.ones((bs,3,3))*1.)



    for n in tqdm(range(int(len(dl) * frac / bs))):

        batch = next(batch_iter)

        preds_tup = learn.pred_batch(batch=batch)

        preds_batch, ys_batch = preds_tup[0][:,0,:,:].sigmoid().data.clone().detach().cpu(), batch[1][0].long().clone().detach().cpu()

        # store info

        if n % 10 == 0 : example_cases.append((preds_batch, ys_batch, batch))

        all_preds.append(preds_batch)

        for i in thrs:

            preds_i = (preds_batch>i).float()

            dices.append(mydice([preds_i, dummy4dice], ys_batch, dummy4dice.long()))

            

    try:

        print("explaining 'batch' structure:")

        print("'batch' elemnts =",len(batch), ': \n\t1st->', batch[0].shape, '. \n\t2nd->', len(batch[1]), 'elemnts: \n\t\t1ts->', batch[1][0].shape, '. \n\t\t2nd->', batch[1][1].shape)

    except:

        pass

    dl.batch_size = orig_bs

    return dices, np.stack(example_cases,1), all_preds

 

dices, example_cases, _ = mypreds(prms, dl=data2.valid_dl, frac=frac, thrs=thrs)

dices_reshaped=np.stack(dices).T.reshape((-1, len(thrs)))

dices_mean = dices_reshaped.mean(0)

    



## debug

# for i in range(8):  #[1,2,3,5]:

#     fig, axs = plt.subplots(1,3)

#     axs[0].imshow(batch[0][i,0,:,:])

#     axs[1].imshow(preds_batch[i,:,:].sigmoid())

#     axs[2].imshow(ys_batch[i,0,:,:])

# dices = np.array(dices)  # .reshape((-1, len(thrs)))

# dices_reshaped = dices.reshape((-1, len(thrs)))

best_dice = dices_mean.max()

best_thr = thrs[dices_mean.argmax()]



plt.figure(figsize=(8,4))

plt.plot(thrs, dices_mean)

plt.vlines(x=best_thr, ymin=dices_mean.min(), ymax=dices_mean.max())

plt.text(best_thr+0.03, best_dice-0.01, f'DICE = {best_dice:.3f} \n THR = {best_thr: .3f}', fontsize=14);

plt.show()
# Plot some samples



chk_thr = 0.1 if 'best_thr' not in globals() else best_thr

batch_id = 5

ex_preds, ex_ys, ex_imgs = example_cases[0][batch_id].squeeze(), example_cases[1][batch_id], example_cases[2][batch_id][0]

plot_idx = np.argsort(ex_preds[:,:,:].sum((1,2)))  #ex_preds.squeeze().sum((1,2)).sort(descending=True).indices[:]

dummy4dice = torch.tensor(np.ones((1,3,3))*1.)

for idx in plot_idx:

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

    img = ex_imgs[idx,0,:,:].cpu()



    ax0.imshow(img)

    ax1.imshow(ex_preds[idx], vmin=0, vmax=1)

    ex_ys_img = ex_ys.squeeze()[idx].numpy().copy()

    ax1.imshow(ex_ys_img - ndimage.binary_erosion(ex_ys_img, structure=np.ones((8,8))) , alpha=0.5)

    ax2.imshow(ex_preds[idx]>chk_thr, vmin=0, vmax=1)



    img_dice= mydice([ex_preds[idx][None,None,:,:].type(torch.FloatTensor), dummy4dice.float()], ex_ys[idx][None,:,:,:], dummy4dice.float(), thrs=best_thr).numpy()[0]

    ax1.set_title('targs + preds, dice={0:.2f}'.format(img_dice))

    ax2.set_title('preds threshed')
# learn.load('/kaggle/input/1407-unfrz-blncd-model/modelsdate_0713_SEED_42_SZ_512_crop_fact_2_BS_8_cycles0_5_lr0_1e-05_cycles1_5_lr1_1e-05_augs_rotate_arch_unet_resnet34')

# best_thr = 0.75

if 'preds' in globals():    del preds

if 'ys' in globals():    del ys

# Predictions for test set

gc.collect()

_,_, preds = mypreds(prms,dl=data2.test_dl, frac=1, thrs=[best_thr], batch_size=1)

preds = [(item.sigmoid()>best_thr).long().numpy() for sublist in preds for item in sublist]

assert len(preds) == len(learn.data.test_ds), "haven't ran prediction on all test images"

# preds = preds[:,pred_sig_id,:,:].sigmoid()  # .sigmoid()

# preds = (preds>best_thr).long().numpy()

# print(preds.sum())



# Generate rle encodings (images are first converted to the original size)

rles = []

for p in progress_bar(preds):

    im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))

    im = np.asarray(im)

    rles.append(mask2rle(im, 1024, 1024))
prms('finalthr', best_thr)

modelname = prms.name()



if not os.path.isdir(modelname):

    os.mkdir(modelname)





ids = [o.stem for o in (path/'test').ls()]  # TODO: replace with the more correct: ids = [o.stem for o in data.test_ds.items]

sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})

sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'

sub_df.to_csv('submission_' + modelname + '.csv', index=False)

sub_df.head()



save_weight_n_ipynb(learn, work_path, prms)

FileLinks('.')
@dataclass

class get_grad_norm(LearnerCallback):

    """

    stores & plots grad norm

    """

    def __init__(self, learn:Learner, n_step:int = 1):

        super().__init__(learn)

        self.gnorms = []

        self.fig, self.ax = plt.subplots(ncols=2,nrows=1, figsize=(4,4))



    def on_backward_end(self, **kwargs):

        "calc total grads norm"

        parameters = list(filter(lambda p: p.grad is not None, learn.model.parameters()))

        total_norm = 0

        for p in parameters:

            param_norm = p.grad.data.norm(2)

            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)

        self.gnorms.append(total_norm)

    

    def on_epoch_begin(self, **kwargs):

        self.gnorms = []

        

    def on_epoch_end(self, **kwargs):

        "plot the total grads norms"

        self.ax.plot(self.gnorms)

        plt.show()

# # Find optimal threshold

# dices = []

# thrs = np.arange(0, 1.01, 0.01)

# for i in progress_bar(thrs):

#     preds_m = (torch.tensor(preds)>i).long()

#     dices.append(dice_overall(preds_m, ys).mean())

# dices = np.array(dices)