%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

from tqdm.notebook import tqdm_notebook

import cv2

import torch.utils.data

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms, datasets

import torchvision.models as models

import re

import torch

import torch.nn as nn

from typing import Optional

import albumentations as A





torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE = True

KAGGLE = True

PATH = "../input/bengaliai-cv19" if KAGGLE else "../../data/bengali" 

MODELS_PATH = "../input/bengali" if KAGGLE else "../../data/bengali"

CSV_PATH = "" if KAGGLE else "../../data/bengali/"

!dir $PATH

PREFIX = "test" if INFERENCE else "train"

file = PATH + "/{0}_image_data_{1}.parquet"

WIDTH=236

HEIGHT=137

FILES_NUM = 4

FILE_RECORDS = 50210

TRAIN_RECORDS_TOTAL = FILES_NUM * FILE_RECORDS

BS = 8 if INFERENCE else 8

TRAIN_SIZE = int(FILE_RECORDS * 0.9)

VALID_SIZE = int(FILE_RECORDS * 0.1)

np.random.seed(42)
dfs = None

if (not INFERENCE):

    t1 = time.time()

    dfs = [pd.read_parquet(file.format(PREFIX,i)) for i in range(4)]

    print(time.time() - t1)

def get_data(ind):

    df = pd.read_parquet(file.format(PREFIX,ind)) if INFERENCE else dfs[ind] 

    ids = df.iloc[:,0] if INFERENCE else None

    data = df.iloc[:,1:].values.reshape(-1,HEIGHT, WIDTH).astype(np.uint8)          

    return data, ids



train_csv=None

if (not INFERENCE):    

    train_data, ids = get_data(0)

    train_csv = pd.read_csv(PATH + "/train.csv")

    print(train_csv.head())

    print(train_csv['grapheme_root'])

    print(train_csv.max())

    print(train_data.shape)
bc=[255,255,255]



def get_aug(vert_flip = False):

    p_vert_flip = 1 if vert_flip else 0

    train_aug =  A.Compose([

        A.OneOf([A.Blur(blur_limit=10,p=1.0),

                 A.GaussianBlur(blur_limit=15,p=1.0),

                 A.GaussNoise(var_limit=200,p=1.0),#???

                ],p=0.4),

        A.OneOf([A.Cutout(num_holes=9,  max_h_size=20, max_w_size=20, p=1.0, fill_value=255),

                 A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=1.0, fill_value=255),

                 A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,value=bc, p=0.1)

                ],p=0.4),

        A.OneOf([A.RandomBrightness(p=1.0),

                A.RandomContrast(p=1.0),

                A.RandomBrightnessContrast(p=1.0)

               ],p=0.4),

        A.OneOf([A.IAAPiecewiseAffine(p=1.0),

                A.ElasticTransform(sigma=30, alpha=1, alpha_affine=30, 

                                 border_mode=cv2.BORDER_CONSTANT,value=bc, p=1.0)

               ],p=0.4),

        #A.VerticalFlip(p=p_vert_flip),

        A.RandomGamma(p=0.8),

        A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,

                            rotate_limit=30,p=1.0,value=bc,border_mode=cv2.BORDER_CONSTANT), 

    ])

    

    inf_aug = A.VerticalFlip(p=p_vert_flip)

    return inf_aug if INFERENCE else train_aug





    

#imgs = [get_image(train_data,i) for i in range(1000)]



#aug(image=imgs[0])  



if (not INFERENCE and False):

    n_imgs = 8

    fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

    for idx in range(n_imgs):        

        #aug  = A.Blur(blur_limit=10,p=1.0)

        #aug = A.GaussianBlur(blur_limit=15,p=1.0)

        #aug = A.GaussNoise(var_limit=200,p=1.0)#???

        #aug=A.Cutout(num_holes=9,  max_h_size=20, max_w_size=20, p=1.0, fill_value=255)

        #aug = A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=1.0, fill_value=255)

        #aug = A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,value =1, p=0.1)

        #aug =  A.RandomBrightness(p=1.0)

        #aug =  A.RandomContrast(p=1.0)

        #aug =  A.RandomBrightnessContrast(p=1.0)

        #aug = A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=30,p=1.0,border_mode=cv2.BORDER_CONSTANT)

        #aug =  A.IAAPiecewiseAffine(p=1.0)

        #aug = A.ElasticTransform(sigma=30, alpha=1, alpha_affine=30, border_mode=cv2.BORDER_CONSTANT,value=[255,255,255], p=1.0)

        #aug = A.VerticalFlip(p=1.0)

        #aug = A.RandomGamma(p=1.0)

        img = get_image(train_data, idx)

        img0 = train_data[idx]#.reshape(HEIGHT, WIDTH).astype(np.uint8)

        axs[idx,0].imshow(img)

        #axs[idx,0].set_title('Original image')

        axs[idx,0].axis('off')

        axs[idx,1].imshow(get_aug(True)(image=img)['image'])

        #axs[idx,1].set_title('Crop & resize')

        axs[idx,1].axis('off')

    plt.show()
#some code of this cell was taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128

#but I changed a lot I saw more logical....



SIZE=224





def bbox(img):   

    rows = np.any(img, axis=1)    

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] < (255-60))    

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    #img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant',)

    

    return cv2.resize(img,(size,size))



def get_image(data, idx) :

    img = data[idx]

    img = crop_resize(img)

    img = np.reshape(img, (SIZE,SIZE, -1))

    img  = np.repeat(img, 3, 2)    

    return img    



if (not INFERENCE):

    n_imgs = 8

    fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

    for idx in range(n_imgs):        

        img = get_image(train_data, idx)

        img0 = train_data[idx]#.reshape(HEIGHT, WIDTH).astype(np.uint8)

        axs[idx,0].imshow(img0)

        axs[idx,0].set_title('Original image')

        axs[idx,0].axis('off')

        axs[idx,1].imshow(img)

        axs[idx,1].set_title('Crop & resize')

        axs[idx,1].axis('off')

    plt.show()


def my():

    def __init__(self):

        self.mean = m

        self.std = s

        self.n = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    def __call__(self, sample):

        print("!!!called")

        return n(sample)

        



data_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])



class GraphemeDataset(Dataset):

    def __init__(self, labels_csv, data, file_num = 0, transform=data_transform, rev = True):

        self.labels_csv = labels_csv

        self.data = data

        self.transform = data_transform

        self.offset = file_num * FILE_RECORDS

        self.aug = get_aug(rev)

        

        

    def __len__(self):        

        return self.data.shape[0]

    

    def prepare_image(self,img, do_aug=True):

        result = img

        #print(do_aug)

        if do_aug:

            #print("aug",self.aug)

            result = self.aug(image=img)['image']

        result = (result/255).astype(np.float32)

        result = self.transform(result)

        return result

    

    def __getitem__(self,idx):

        image_pure = get_image(self.data, idx)

        #print("img0")

        img0 = self.prepare_image(img = image_pure, do_aug=not INFERENCE)  

        if (INFERENCE):

            #print("img1")

            img1 = self.prepare_image(img = image_pure, do_aug=True)

        #    img2 = self.prepare_image(img = image_pure)

            return img0,img1

                   

        else :    

            labels = self.labels_csv.iloc[idx + self.offset][1:4]

            root = labels[0]

            vowel = labels[1]

            consonant = labels[2]

            return img0, root, vowel, consonant

    

    

def get_dls(ind):

    data, ids = get_data(ind)

    rev_ds = GraphemeDataset(train_csv, data, ind, rev = True)

    if (INFERENCE):

        rev_dl =  DataLoader(rev_ds, batch_size = BS)

        

        return rev_dl, None, ids

    else:

        train_ds, valid_ds = torch.utils.data.random_split(rev_ds, [TRAIN_SIZE, VALID_SIZE])

        train_dl = DataLoader(train_ds, batch_size=BS)

        valid_dl = DataLoader(valid_ds, batch_size=BS)

        return train_dl, valid_dl, ids 

    

if (INFERENCE):

    check_dl, some_dl, ids = get_dls(0)

    print(ids[0])

    x1, x2 = next(iter(check_dl))

    print(x1.shape, x2.shape)

else:    

    check_dl, valid_dl, ids = get_dls(0)

    x, l1, l2, l3 = next(iter(check_dl))

    print(x.shape, l1.shape, l2.shape, l3.shape, type(x[0][0][0][0]))

    


#This code was taken from fastai (at least at some extent)



def requires_grad(m:nn.Module, b:Optional[bool]=None)->Optional[bool]:

    "If `b` is not set return `requires_grad` of first param, else set `requires_grad` on all params as `b`"

    ps = list(m.parameters())

    if not ps: return None

    if b is None: return ps[0].requires_grad

    for p in ps: 

        p.requires_grad=b

        #print(p.requires_grad)



def is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)

def has_pool_type(m):

    if is_pool_type(m): return True

    for l in m.children():

        if has_pool_type(l): return True

    return False



def create_body(model):

    ll = list(enumerate(model.children()))

    cut = next(i for i,o in reversed(ll) if has_pool_type(o))

    

    return nn.Sequential(*list(model.children())[:cut])



class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=None):

        super().__init__()

        sz = sz or (1,1)

        self.ap = nn.AdaptiveAvgPool2d(sz)

        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

    

class Flatten(nn.Module):

    def forward(self, x): return x.view(x.size(0), -1)

    

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):

    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."

    layers = [nn.BatchNorm1d(n_in)] if bn else []

    if p != 0: layers.append(nn.Dropout(p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None: layers.append(actn)

    return layers    



def create_head(is_densenet):    

    #lin_ftrs = [1024, 256, 168 + 11 + 7]

    #lin_ftrs = if [4096, 256, 168 + 11 + 7]

    lin_ftrs = [4416, 256, 168 + 11 + 7] if is_densenet else [4096, 256, 168 + 11 + 7]

    ps = [0.5]#listify(ps)

    bn_final=False

    concat_pool=True

    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps

    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]

    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)

    layers = [pool, Flatten()]

    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):

        layers += bn_drop_lin(ni, no, True, p, actn)

        #layers += bn_drop_lin(ni, no, False, 0, actn)

    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

    head = nn.Sequential(*layers)

    return head



def cond_init(m:nn.Module):

    "Initialize the non-batchnorm layers of `m` with `init_func`."

    if (not isinstance(m, nn.BatchNorm1d)) and requires_grad(m): 

        if hasattr(m, 'weight'): nn.init.kaiming_normal_(m.weight)

        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)



def apply_leaf(m:nn.Module, f):

    "Apply `f` to children of `m`."

    c = list(m.children())

    if isinstance(m, nn.Module): f(m)

    for l in c: apply_leaf(l,f)

        

def apply_init(m):

    "Initialize all non-batchnorm layers of `m` with `init_func`."

    apply_leaf(m, cond_init)



def get_resnet():

    resnet101 = models.resnet101(pretrained=not INFERENCE)

    body = create_body(resnet101)

    head = create_head(False)

    apply_init(head)

    return nn.Sequential(body, head),body



def get_densenet() :

    densenet= models.densenet161(pretrained= not INFERENCE)

    

    head = nn.Linear(2208, 168 + 11 + 7)#create_head()

    apply_init(head)

    children = densenet.children()

    

    body = next(children) 

    head = create_head(True)

    return nn.Sequential(body, head),body

    

m_resnet,body = get_resnet()

m_densenet,body = get_densenet()

print(m_densenet

     )

m = m_resnet#m_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 1



optimizer = torch.optim.Adam(m.parameters(), lr=2e-3)

m.to(device)

criterion = nn.CrossEntropyLoss()



def split(a):

    return a[:, 0:168], a[:, 168:168+11], a[:, 168+11:]



def loss_func(roots168, vowels11, consonants7, outputs):    

    r,v,c = split(outputs)

    root_loss = criterion(r, roots168)

    vowel_loss = criterion(v, vowels11)

    consonant_loss = criterion(c, consonants7)

    

    total_loss = root_loss*2 + vowel_loss + consonant_loss

    #total_loss = root_loss + vowel_loss + consonant_loss

    return total_loss



def acc_func(roots168, vowels11, consonants7, outputs, acc):

    r,v,c = split(outputs)

    acc_c = [0,0,0,0]

    acc_c[0] = (r.argmax(1)==roots168).float().mean()

    acc_c[1] = (v.argmax(1)==vowels11).float().mean()

    acc_c[2] = (c.argmax(1)==consonants7).float().mean()

    acc_c[3] = (2*acc_c[0] + acc_c[1] + acc_c[2])/4

       

    return np.add(acc, acc_c)



def getv(v, length):   

    

    return round((v/length).item(),4)



def getv_list(v, length):

    return [getv(e,length) for e in v]



def to_d(i, r, v,c):

    return i.to(device), r.to(device), v.to(device), c.to(device)



def train_int(train, dl, m, pbar):

    acc = [0,0,0,0]

    loss = 0

    m.train(train)

    for (idx, (inp, r, v, c)) in (enumerate(dl)):

        inp, r, v, c = inp.to(device), r.to(device), v.to(device), c.to(device)           

        outputs = m(inp)

        #print(outputs.shape)

        acc = acc_func(r, v, c, outputs, acc) 

        if (train):    

            total_loss = loss_func(r, v, c, outputs)

            loss += total_loss

            total_loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            

        pbar.update(BS)

        

    return acc, loss    





def train_epoch(num, start3):

    #from tqdm.notebook import tqdm_notebook



    pbar = tqdm_notebook(total = TRAIN_RECORDS_TOTAL)

    print("acc train\tloss train\tacc valid")

    if (start3 != 0):

        m.load_state_dict(torch.load(MODELS_PATH + "/densenet162.2e-3.4.tmp")) 

    for file_num in range(start3, FILES_NUM):

        train_dl, valid_dl, ids = get_dls(file_num)

        

        acc_train, loss_train = train_int(True, train_dl, m, pbar)

        acc_valid, l = train_int(False, valid_dl, m, pbar)            

        

        train_length = len(train_dl)

                

        acc_p = getv(acc_train[3], train_length)

        loss_p = getv(loss_train, train_length)        

        acc_v_p = getv_list(acc_valid, len(valid_dl))

        

        print(acc_p,"\t\t", loss_p, "\t\t",acc_v_p[3], 

              "(",acc_v_p[0], acc_v_p[1], acc_v_p[2],")")

        torch.save(m.state_dict(), MODELS_PATH + "/densenet162.2e-3.4.tmp")

    pbar.close()

    

def train():

    print("Will train a frozen-body model...")

    requires_grad(body, False)

    start1 = 5

    start3 = 2

    m.load_state_dict(torch.load(MODELS_PATH + "/densenet162.2e-3.5.0"))

    for num in range(start1, 5):

        print("Frozen, number " + str(num))

        train_epoch(num, start3)

        start3 = 0

        torch.save(m.state_dict(), MODELS_PATH + "/densenet162.2e-3." + str(num))

        #torch.save(m.state_dict(), MODELS_PATH + "/resnet.2e-3.1.1")

        

    print("Will train an unfrozen-body model...")

    start2 = 5

    start3 = 2

    requires_grad(body, True)  

    for num in range(start2,30):

        print("Unfrozen, number " + str(num))

        train_epoch(num, start3)

        start3 = 0

        torch.save(m.state_dict(), MODELS_PATH + "/densenet162.2e-3.5." + str(num))   

    #torch.save(m.state_dict(), MODELS_PATH + "/model.2e-3.1.1")

    

if (not INFERENCE):    

    train()   

#!pip install line_profiler        

#%load_ext line_profiler        

#%lprun -f GraphemeDataset.__getitem__ train()

#%lprun -f get_image train()

#%lprun -f train train()





    
##INFERENCE!!!!

def handle_single_file(file_num, subm_csv):

    print("file",file_num)    

    test_dl, something, ids = get_dls(file_num)

    m_densenet.train(False)

    m_resnet.train(False)

    

    index = 0

    for (idx, (img, img_flipped)) in (enumerate(test_dl)):

        img = img.to(device)

        #img_flipped = img_flipped.to(device)

        

        outputs1 = m_resnet(img)

        #outputs2 = m_densenet(img_flipped)

        outputs = outputs1#0.5 * (outputs1 + outputs2) 

        r,v,c = split(outputs)

        roots = r.argmax(1).tolist()

        vowels = v.argmax(1).tolist()

        consonants = c.argmax(1).tolist()

        current_bs = img.shape[0]        

        for i in range(current_bs):

            test_name = ids[index] 

            

            subm_csv.write(test_name + "_consonant_diacritic," + str(consonants[i]) + "\n")

            subm_csv.write(test_name + "_grapheme_root," + str(roots[i]) + "\n")

            subm_csv.write(test_name + "_vowel_diacritic," + str(vowels[i]) + "\n")

            index += 1

                



def inference():

    m_resnet.load_state_dict(torch.load(MODELS_PATH + "/resnet101.2e-4.5.21"))

    m_densenet.load_state_dict(torch.load(MODELS_PATH + "/densenet162.2e-3.5.9"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    m_resnet.to(device)

    m_densenet.to(device)

    print("Inference....")

    total_index = 0

    subm_csv = open(CSV_PATH + "submission.csv", 'w')

    subm_csv.write('row_id,target\n')

    for file_num in range(FILES_NUM):

        handle_single_file(file_num,subm_csv)

                            

    subm_csv.flush()        

    subm_csv.close()   

    

if (INFERENCE):

    inference()
#if (INFERENCE):

 #   !cat {CSV_PATH}submission.csv