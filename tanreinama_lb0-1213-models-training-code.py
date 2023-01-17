import pandas as pd

import numpy as np

import json

import os

from multiprocessing import Pool

from tqdm.notebook import tqdm

import gc

import pickle

import joblib

import cv2

import bz2

from PIL import Image

import matplotlib.pyplot as plt
REDUCE_MEM = True

MODEL_FILE_DIR = '../input/imaterialist2020-pretrain-models/'

attr_image_size = (160,160)
to_training = not os.path.isfile(MODEL_FILE_DIR+"maskmodel_%d.model"%attr_image_size[0])
train_df = pd.read_csv("../input/imaterialist-fashion-2020-fgvc7/train.csv")
def rle_to_mask(rle_string,height,width):

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]

        rlePairs = np.array(rleNumbers).reshape(-1,2)

        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img



def mask_to_rle(mask):

    pixels = mask.T.flatten()

    # We need to allow for cases where there is a '1' at either end of the sequence.

    # We do this by padding with a zero at each end when needed.

    use_padding = False

    if pixels[0] or pixels[-1]:

        use_padding = True

        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)

        pixel_padded[1:-1] = pixels

        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2

    if use_padding:

        rle = rle - 1

    rle[1::2] = rle[1::2] - rle[:-1:2]

    return ' '.join(str(x) for x in rle)
max_clz = train_df.ClassId.max()
max_attr = 0

for i in train_df.AttributesIds:

    for a in str(i).split(','):

        if a!='nan':

            a = int(a)

            if a > max_attr:

                max_attr = a
clz_attr = np.zeros((max_clz+1,max_attr+1))

clz_attrid2idx = [[] for _ in range(max_clz+1)]

clz_attr.shape
for c,i in zip(train_df.ClassId,train_df.AttributesIds):

    for a in str(i).split(','):

        if a!='nan':

            a = int(a)

            clz_attr[c,a] = 1

            if not a in clz_attrid2idx[c]:

                clz_attrid2idx[c].append(a)
clz_attr_num = clz_attr.sum(axis=1).astype(np.int64)

clz_attr_num
train_df.head()
def ptoz(obj):

    return bz2.compress(pickle.dumps(obj), 3) if REDUCE_MEM else obj

def ztop(b):

    return pickle.loads(bz2.decompress(b)) if REDUCE_MEM else b

def __getitem__(imgid):

    df = train_df[train_df.ImageId==imgid]

    res = []

    imag = cv2.imread("../input/imaterialist-fashion-2020-fgvc7/train/"+str(imgid)+".jpg")

    for idx in range(len(df)):

        t = df.values[idx]

        cid = t[4]

        mask = rle_to_mask(t[1],t[2],t[3])

        attr = map(int,str(t[5]).split(",")) if str(t[5]) != 'nan' else []

        where = np.where(mask != 0)

        y1,y2,x1,x2 = 0,0,0,0

        if len(where[0]) > 0 and len(where[1]) > 0:

            y1,y2,x1,x2 = min(where[0]),max(where[0]),min(where[1]),max(where[1])

        if y2>y1+10 and x2>x1+10:

            X = cv2.resize(imag[y1:y2,x1:x2], attr_image_size)

            X = ptoz(X)

        else:

            X = None

        mask = cv2.resize(mask, attr_image_size)

        mask = ptoz(mask)

        res.append((cid, mask, attr, X))

    imag = cv2.resize(imag, attr_image_size)

    imag = ptoz(imag)

    return res, imag, imgid
if to_training:

    if os.path.isfile(MODEL_FILE_DIR+"data_cache_%d"%attr_image_size[0]):

        data_cache = joblib.load(MODEL_FILE_DIR+"data_cache_%d"%attr_image_size[0])

    elif REDUCE_MEM:

        data_cache = []

        for i in tqdm(list(set(train_df.ImageId))):

            res, imag, imgid = __getitem__(i)

            for cid, mask, attr, X in res:

                data_cache.append((cid, mask, attr, imag, X, imgid))

        joblib.dump(data_cache, MODEL_FILE_DIR+"data_cache_%d"%attr_image_size[0])

    else:

        with Pool(8) as p:

            tmp = p.map(__getitem__, list(set(train_df.ImageId)))

        data_cache = []

        for res, imag, imgid in tmp:

            for cid, mask, attr, X in res:

                data_cache.append((cid, mask, attr, imag, X, imgid))

        del tmp

        joblib.dump(data_cache, MODEL_FILE_DIR+"data_cache_%d"%attr_image_size[0])

else:

    data_cache = []
import torch

import torch.nn as nn

import torch.nn.functional as F



@torch.jit.script

def mish(input):

    return input * torch.tanh(F.softplus(input))



class Mish(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, input):

        return mish(input)



class SeparableConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):

        super(SeparableConv2d,self).__init__()



        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)

        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)



    def forward(self,x):

        x = self.conv1(x)

        x = self.pointwise(x)

        return x



class Block(nn.Module):

    def __init__(self,in_filters,out_filters,reps,strides=1,activation=None):

        super(Block, self).__init__()



        if out_filters != in_filters or strides!=1:

            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)

            self.skipbn = nn.BatchNorm2d(out_filters)

        else:

            self.skip=None



        act = nn.ReLU() if activation is None else activation

        rep=[]



        rep.append(act)

        rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))

        rep.append(nn.BatchNorm2d(out_filters))

        filters = out_filters



        for i in range(reps-1):

            rep.append(act)

            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))

            rep.append(nn.BatchNorm2d(filters))



        if strides != 1:

            rep.append(nn.MaxPool2d(3,strides,1))

        self.rep = nn.Sequential(*rep)



    def forward(self,inp):

        x = self.rep(inp)



        if self.skip is not None:

            skip = self.skip(inp)

            skip = self.skipbn(skip)

        else:

            skip = inp



        x += skip

        return x



class AttrXception(nn.Module):

    def __init__(self, num_classes=1000):

        super(AttrXception, self).__init__()

        self.num_classes = num_classes



        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=True)

        self.bn1 = nn.BatchNorm2d(64)

        self.mish = Mish()



        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)

        self.bn2 = nn.BatchNorm2d(128)



        self.block1 = Block(128,256,2,2)

        self.block2 = Block(256,256,3,1)

        self.block3 = Block(256,256,3,1)

        self.block4 = Block(256,256,3,1)

        self.block5 = Block(256,256,3,1)

        self.block6 = Block(256,256,3,1)

        self.block7 = Block(256,384,2,2)



        self.conv3 = SeparableConv2d(384,512,3,stride=1,padding=0,bias=True)

        self.fc = nn.Linear(512, num_classes)



    def forward(self, input):

        x = self.conv1(input)

        x = self.bn1(x)

        x = self.mish(x)



        x = self.conv2(x)

        x = self.bn2(x)



        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.block7(x)



        x = self.mish(x)

        x = self.conv3(x)



        x = self.mish(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)

        result = self.fc(x)

        

        return torch.sigmoid(result)



class HourglassNet(nn.Module):

    def __init__(self, depth, channel):

        super(HourglassNet, self).__init__()

        self.depth = depth

        hg = []

        for _ in range(self.depth):

            hg.append([

                Block(channel,channel,3,1,activation=Mish()),

                Block(channel,channel,2,2,activation=Mish()),

                Block(channel,channel,3,1,activation=Mish())

            ])

        hg[0].append(Block(channel,channel,3,1,activation=Mish()))

        hg = [nn.ModuleList(h) for h in hg]

        self.hg = nn.ModuleList(hg)



    def _hour_glass_forward(self, n, x):

        up1 = self.hg[n-1][0](x)

        low1 = self.hg[n-1][1](up1)



        if n > 1:

            low2 = self._hour_glass_forward(n-1, low1)

        else:

            low2 = self.hg[n-1][3](low1)



        low3 = self.hg[n-1][2](low2)

        up2 = F.interpolate(low3, scale_factor=2)

        out = up1 + up2

        return out



    def forward(self, x):

        return self._hour_glass_forward(self.depth, x)



class XceptionHourglass(nn.Module):

    def __init__(self, num_classes):

        super(XceptionHourglass, self).__init__()

        self.num_classes = num_classes



        self.conv1 = nn.Conv2d(3, 128, 3, 2, 1, bias=True)

        self.bn1 = nn.BatchNorm2d(128)

        self.mish = Mish()



        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)

        self.bn2 = nn.BatchNorm2d(256)



        self.block1 = HourglassNet(4, 256)

        self.bn3 = nn.BatchNorm2d(256)

        self.block2 = HourglassNet(4, 256)



        self.sigmoid = nn.Sigmoid()



        self.conv3 = nn.Conv2d(256, num_classes, 1, bias=True)



    def forward(self, input):

        x = self.conv1(input)

        x = self.bn1(x)

        x = self.mish(x)



        x = self.conv2(x)

        x = self.bn2(x)

        x = self.mish(x)



        out1 = self.block1(x)

        x = self.bn3(out1)

        x = self.mish(x)

        out2 = self.block2(x)



        r = self.sigmoid(out1 + out2)

        r = F.interpolate(r, scale_factor=2)

        

        return self.conv3(r)
class AttrDataset(object):

    def __init__(self, chaches, clzid):

        self.clzid = clzid

        self.chaches = [cd for cd in chaches if cd[0]==clzid]



    def __getitem__(self, idx):

        cid, mask, attr, imag, X, imgid = self.chaches[idx]

        mask = ztop(mask)

        imag = ztop(imag)

        if X is None:

            X = imag

        else:

            X = ztop(X)

        y = np.zeros(clz_attr_num[self.clzid])

        for a in attr:

            y[clz_attrid2idx[self.clzid].index(a)] = 1

        return X.transpose((2,0,1)).astype(np.float32), y.astype(np.float32)

        

    def __len__(self):

        return len(self.chaches)



def train_attr_net(clzid, num_epochs=1):

    data = AttrDataset(data_cache, clzid)

    data_loader = torch.utils.data.DataLoader(

        data, batch_size=64, shuffle=True, num_workers=1)



    model = AttrXception(clz_attr_num[clzid])

    model.cuda()

    dp = torch.nn.DataParallel(model)

    loss = nn.BCELoss()



    params = [p for p in dp.parameters() if p.requires_grad]

    optimizer = torch.optim.RMSprop(params, lr=2.5e-4,  momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                                   step_size=6,

                                                   gamma=0.9)

    

    prog = tqdm(list(range(num_epochs)))

    for epoch in prog:

        for i, (X, y) in enumerate(data_loader):

            X = X.cuda()

            y = y.cuda()

            xx = dp(X)



            losses = loss(xx, y)



            prog.set_description("loss:%05f"%losses)

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()



        X, xx, y, losses = None, None, None, None

        torch.cuda.empty_cache()

        gc.collect()

    return model
for clzid in range(len(clz_attr_num)):

    if clz_attr_num[clzid] > 0:

        if not os.path.isfile(MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid)):

            model = train_attr_net(clzid, 32)

            torch.save(model.state_dict(), MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid))
model = None

torch.cuda.empty_cache()

gc.collect()
data_mask = dict()

while len(data_cache) > 0:

    cid, mask, _, imag, _, imgid = data_cache.pop()

    mask = ztop(mask)

    if imgid not in data_mask:

        imag = ztop(imag)

        data_mask[imgid] = [ptoz(imag.transpose((2,0,1)).astype(np.float32)), np.zeros(attr_image_size, dtype=np.int)]

    data_mask[imgid][1][mask!=0] = cid + 1
del data_cache
for k in data_mask.keys():

    data_mask[k][1] = ptoz(data_mask[k][1])

gc.collect()
class MaskDataset(object):

    def __init__(self, keys):

        self.keys = keys



    def __getitem__(self, idx):

        k = self.keys[idx]

        return ztop(data_mask[k][0]), ztop(data_mask[k][1])

        

    def __len__(self):

        return len(self.keys)
def train_mask_net(num_epochs=1):

    data = MaskDataset(list(data_mask.keys()))

    data_loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, num_workers=4)



    model = XceptionHourglass(max_clz+2)

    model.cuda()

    dp = torch.nn.DataParallel(model)

    loss = nn.CrossEntropyLoss()



    params = [p for p in dp.parameters() if p.requires_grad]

    optimizer = torch.optim.RMSprop(params, lr=2.5e-4,  momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                                   step_size=6,

                                                   gamma=0.9)

    for epoch in range(num_epochs):

        total_loss = []

        prog = tqdm(data_loader, total=len(data_loader))

        for i, (imag, mask) in enumerate(prog):

            X = imag.cuda()

            y = mask.cuda()

            xx = dp(X)

            # to 1D-array

            y = y.reshape((y.size(0),-1))  # batch, flatten-img

            y = y.reshape((y.size(0) * y.size(1),))  # flatten-all

            xx = xx.reshape((xx.size(0), xx.size(1), -1))  # batch, channel, flatten-img

            xx = torch.transpose(xx, 2, 1)  # batch, flatten-img, channel

            xx = xx.reshape((xx.size(0) * xx.size(1),-1))  # flatten-all, channel



            losses = loss(xx, y)



            prog.set_description("loss:%05f"%losses)

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()



            total_loss.append(losses.detach().cpu().numpy())



        prog, X, xx, y, losses = None, None, None, None, None,

        torch.cuda.empty_cache()

        gc.collect()

    return model
if to_training:

    model = train_mask_net(64)

    torch.save(model.state_dict(), MODEL_FILE_DIR+"maskmodel_%d.model"%attr_image_size[0])
del data_mask

gc.collect()
class MaskDataset(object):

    def __init__(self, folder):

        self.imgids = [f.split(".")[0] for f in os.listdir(folder)]

        self.folder = folder



    def __getitem__(self, idx):

        imag = cv2.imread(self.folder+self.imgids[idx]+".jpg")

        imag = cv2.resize(imag, attr_image_size)

        return imag.transpose((2,0,1)).astype(np.float32)

        

    def __len__(self):

        return len(self.imgids)
model = XceptionHourglass(max_clz+2)

model.cuda()

model.load_state_dict(torch.load(MODEL_FILE_DIR+"maskmodel_%d.model"%attr_image_size[0]))



dataset = MaskDataset("../input/imaterialist-fashion-2020-fgvc7/test/")



data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=8, shuffle=False, num_workers=4)



predict_imgeid = []

predict_mask = []

predict_rle = []

predict_classid = []

predict_attr = []



model.eval()

prog = tqdm(data_loader, total=len(data_loader))

num_pred = 0

for X in prog:

    X = X.cuda()

    pred = model(X).detach().cpu().numpy()

    for i, mask in enumerate(pred):

        imgid = dataset.imgids[num_pred]

        num_pred += 1

        pred_id = mask.argmax(axis=0) - 1  # -1 is background.

        for clz in set(pred_id.reshape((-1,)).tolist()):

            if clz >= 0:

                maskdata = (pred_id == clz).astype(np.uint8) * 255

                predict_imgeid.append(imgid)

                predict_mask.append(maskdata)

                predict_rle.append("")

                predict_classid.append(clz)

                predict_attr.append([])



prog, X, pred, dataset, data_loader = None, None, None, None, None

torch.cuda.empty_cache()

gc.collect()
import math

def _scale_image(img, long_size):

    if img.shape[0] < img.shape[1]:

        scale = img.shape[1] / long_size

        size = (long_size, math.floor(img.shape[0] / scale))

    else:

        scale = img.shape[0] / long_size

        size = (math.floor(img.shape[1] / scale), long_size)

    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
for clzid in range(len(clz_attr_num)):

    if clz_attr_num[clzid] > 0 and os.path.isfile(MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid)):

        model = AttrXception(clz_attr_num[clzid])

        model.cuda()

        model.eval()

        model.load_state_dict(torch.load(MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid)))

        for i in range(len(predict_classid)):

            if predict_classid[i] == clzid:

                imag = cv2.imread("../input/imaterialist-fashion-2020-fgvc7/test/"+predict_imgeid[i]+".jpg")

                imag = _scale_image(imag, 1024)

                mask = cv2.resize(predict_mask[i], (imag.shape[1],imag.shape[0]), interpolation=cv2.INTER_NEAREST)

                where = np.where(mask!=0)

                y1,y2,x1,x2 = 0,0,0,0

                if len(where[0]) > 0 and len(where[1]) > 0:

                    y1,y2,x1,x2 = min(where[0]),max(where[0]),min(where[1]),max(where[1])

                    if y2>y1+80 and x2>x1+80 and np.sum(mask)/255 > 1000:

                        print("class id=",clzid)

                        plt.subplot(1,2,1)

                        plt.imshow(imag)

                        plt.subplot(1,2,2)

                        plt.imshow(mask)

                        plt.show()

                        break
uses_index = []

for clzid in tqdm(range(len(clz_attr_num))):

    if clz_attr_num[clzid] > 0 and os.path.isfile(MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid)):

        model = AttrXception(clz_attr_num[clzid])

        model.cuda()

        model.eval()

        model.load_state_dict(torch.load(MODEL_FILE_DIR+"attrmodel_%d-%d.model"%(attr_image_size[0],clzid)))

        for i in range(len(predict_classid)):

            if predict_classid[i] == clzid:

                imag = cv2.imread("../input/imaterialist-fashion-2020-fgvc7/test/"+predict_imgeid[i]+".jpg")

                imag = _scale_image(imag, 1024)

                mask = cv2.resize(predict_mask[i], (imag.shape[1],imag.shape[0]), interpolation=cv2.INTER_NEAREST)

                #imag[mask==0] = 255

                where = np.where(mask!=0)

                y1,y2,x1,x2 = 0,0,0,0

                if len(where[0]) > 0 and len(where[1]) > 0:

                    y1,y2,x1,x2 = min(where[0]),max(where[0]),min(where[1]),max(where[1])

                    if y2>y1+80 and x2>x1+80 and np.sum(mask)/255 > 1000:

                        predict_rle[i] = mask_to_rle(mask)

                        X = cv2.resize(imag[y1:y2,x1:x2], attr_image_size).transpose((2,0,1))

                        attr_preds = model(torch.tensor([X], dtype=torch.float32).cuda())

                        attr_preds = attr_preds.detach().cpu().numpy()[0]

                        for ci in range(len(attr_preds)):

                            if attr_preds[ci] > 0.5:

                                uses_index.append(i)

                                predict_attr[i].append(clz_attrid2idx[predict_classid[i]][ci])
predict_attri_str = [",".join(list(map(str,predict_attr[i]))) for i in range(len(predict_classid))]
predict_imgeid = [predict_imgeid[i] for i in set(uses_index)]

predict_mask = [predict_mask[i] for i in set(uses_index)]

predict_rle = [predict_rle[i] for i in set(uses_index)]

predict_classid = [predict_classid[i] for i in set(uses_index)]

predict_attr = [predict_attr[i] for i in set(uses_index)]

predict_attri_str = [predict_attri_str[i] for i in set(uses_index)]
setidlist = set(predict_imgeid)

for i in os.listdir("../input/imaterialist-fashion-2020-fgvc7/test/"):

    id = i.split('.')[0]

    if not id in setidlist:

        predict_imgeid.append(id)

        predict_rle.append("1 1")

        predict_classid.append(0)

        predict_attri_str.append("111,137")
pd.DataFrame({

    "ImageId":predict_imgeid,

    "EncodedPixels":predict_rle,

    "ClassId":predict_classid,

    "AttributesIds":predict_attri_str

}).to_csv("submission.csv", index=False)
"""

for clzid in range(len(clz_attr_num)):

    if os.path.isfile("attrmodel_%d-%d.model"%(attr_image_size[0],clzid)):

        os.remove("attrmodel_%d-%d.model"%(attr_image_size[0],clzid))

os.remove( "maskmodel_%d.model"%attr_image_size[0]))

"""
!head submission.csv