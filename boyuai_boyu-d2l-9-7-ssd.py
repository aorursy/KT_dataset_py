%matplotlib inline



import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision

from torchvision.datasets import ImageFolder

from torchvision import transforms

from torchvision import models

import time

import os

import sys

sys.path.append("../input/")

import d2lpytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print(torch.__version__, device)



def cls_predictor(in_channels, num_anchors, num_classes):

    return nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3,

                     padding=1)
def bbox_predictor(in_channels, num_anchors):

    return nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
def forward(x, block):

    return block(x)



Y1 = forward(torch.zeros(2, 1, 20, 20), cls_predictor(1, 5, 10))

Y2 = forward(torch.zeros(2, 16, 10, 10), cls_predictor(16, 3, 10))

(Y1.shape, Y2.shape)
def flatten_pred(pred):

    return pred.permute(0, 2, 3, 1).reshape(pred.shape[0],-1)



def concat_preds(preds):

    return torch.cat([flatten_pred(p) for p in preds], dim=1)
concat_preds([Y1, Y2])[1][5888]
def down_sample_blk(in_channels, out_channels):

    blk = nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

        nn.BatchNorm2d(out_channels),

        nn.ReLU(),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),

        nn.BatchNorm2d(out_channels),

        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2,2),stride=2)

    )

    return blk
forward(torch.zeros(2, 3, 20, 20), down_sample_blk(3, 10)).shape
def base_net(in_channels):

    blk = nn.Sequential(

        down_sample_blk(in_channels, 16),

        down_sample_blk(16, 32),

        down_sample_blk(32, 64)

        

        )

#     blk = nn.Sequential()

#     blk.add_module("down_sample_block0", down_sample_blk(3, 16))

#     blk.add_module("down_sample_block1", down_sample_blk(16, 32))

#     blk.add_module("down_sample_block2", down_sample_blk(32, 64))

    

    return blk



forward(torch.zeros(2, 3, 256, 256), base_net(in_channels = 3)).shape
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):

    blk = blk.to(device)

    cls_predictor = cls_predictor.to(device)

    bbox_predictor = bbox_predictor.to(device)

    X = X.to(device)

    Y = blk(X)

    anchors = d2l.MultiBoxPrior(Y, sizes=size, ratios=ratio)

    cls_preds = cls_predictor(Y)

    bbox_preds = bbox_predictor(Y)

    return (Y, anchors, cls_preds, bbox_preds)
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],

         [0.88, 0.961]]

ratios = [[1, 2, 0.5]] * 5

num_anchors = len(sizes[0]) + len(ratios[0]) - 1

      
class TinySSD(nn.Module):

    def __init__(self, in_channels, num_classes, **kwargs):

        super(TinySSD, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.in_channels = in_channels

#         self.basenet = nn.ModuleList()

#         self.downSampleBlk1 = nn.ModuleList()

#         self.downSampleBlk2 = nn.ModuleList()

#         self.downSampleBlk3 = nn.ModuleList()

#         self.adaptiveAvgPool2d = nn.ModuleList()

        self.basenet = base_net(in_channels)

        self.downSampleBlk1 = down_sample_blk(in_channels=64, out_channels=128)

        self.downSampleBlk2 = down_sample_blk(in_channels=128, out_channels=128)

        self.downSampleBlk3 = down_sample_blk(in_channels=128, out_channels=128)

        self.adaptiveAvgPool2d = nn.AdaptiveMaxPool2d((1, 1))

        

    def test(self):

        print(base_net(self.in_channels))

        

    def forward(self, X):

        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        # getattr(self, 'blk_%d' % i)即访问self.blk_i

        X.to(device)

        X, anchors[0], cls_preds[0], bbox_preds[0] = blk_forward(

            X, 

            self.basenet,

            sizes[0], 

            ratios[0],

            cls_predictor(64, num_anchors,self.num_classes), 

            bbox_predictor(64, num_anchors)

        )

        X, anchors[1], cls_preds[1], bbox_preds[1] = blk_forward(

            X,

            self.downSampleBlk1,

            sizes[1], 

            ratios[1],

            cls_predictor(128, num_anchors,self.num_classes), 

            bbox_predictor(128, num_anchors)

        )

        X, anchors[2], cls_preds[2], bbox_preds[2] = blk_forward(

            X, 

            self.downSampleBlk2,

            sizes[2], 

            ratios[2],

            cls_predictor(128, num_anchors,self.num_classes), 

            bbox_predictor(128, num_anchors)

        )

        X, anchors[3], cls_preds[3], bbox_preds[3] = blk_forward(

            X, 

            self.downSampleBlk3,

            sizes[3], 

            ratios[3],

            cls_predictor(128, num_anchors,self.num_classes), 

            bbox_predictor(128, num_anchors)

        )

        

        X, anchors[4], cls_preds[4], bbox_preds[4] = blk_forward(

            X, 

            self.adaptiveAvgPool2d,

            sizes[4], 

            ratios[4],

            cls_predictor(128, num_anchors,self.num_classes), 

            bbox_predictor(128, num_anchors)

        )

        # reshape函数中的0表示保持批量大小不变

        return (torch.cat([anchor for anchor in anchors], dim = 1),

                concat_preds(cls_preds).reshape(X.shape[0], -1, self.num_classes+1), concat_preds(bbox_preds))
net = TinySSD(in_channels=3, num_classes=1)

#print(net.test())

X = torch.zeros(32, 3, 256, 256)

anchors, cls_preds, bbox_preds = net(X)



print('output anchors:', anchors.shape)

print('output class preds:', cls_preds.shape)

print('output bbox preds:', bbox_preds.shape)
batch_size = 32

train_iter, test_iter = d2l.load_data_pikachu(data_dir="../input/pikachu", batch_size = batch_size)
net = TinySSD(in_channels=3, num_classes=1).to(device)

lr, num_epochs = 0.001, 5

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
cls_loss = nn.CrossEntropyLoss().to(device)

bbox_loss = nn.L1Loss().to(device)



def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):

    cls = cls_loss(cls_preds, cls_labels)

    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)

    return cls + bbox
def cls_eval(cls_preds, cls_labels):

    # 由于类别预测结果放在最后一维，argmax需要指定最后一维

    return (cls_preds.argmax(dim=-1) == cls_labels).sum()



def bbox_eval(bbox_preds, bbox_labels, bbox_masks):

    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum()
def trainSSD(train_iter, net, loss, optimizer, device, num_epochs):

    net = net.to(device)

    acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0

    print("training on ", device)



    for epoch in range(num_epochs):

        start = time.time()

        for batch in train_iter:

            X = batch["image"]

            X = X.to(device)

            Y = batch["label"]

            # 生成多尺度的锚框，为每个锚框预测类别和偏移量

            anchors, cls_preds, bbox_preds = net(X)

            # 为每个锚框标注类别和偏移量

            bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(

                anchors, Y)

            bbox_labels = bbox_labels.to(device)

            bbox_masks = bbox_masks.to(device)

            cls_labels = cls_labels.to(device)

            # 根据类别和偏移量的预测和标注值计算损失函数

            l = loss(cls_preds.reshape(-1, cls_preds.shape[2]), cls_labels.reshape(-1), bbox_preds, bbox_labels,

                          bbox_masks)

            optimizer.zero_grad()

            l.backward()

            optimizer.step()

            acc_sum += cls_eval(cls_preds, cls_labels)

            n += cls_labels.reshape(-1).shape[0]

            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)

            m += bbox_labels.reshape(-1).shape[0]



        print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (

            epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
trainSSD(train_iter, net, calc_loss, optimizer, device , 5)
# img = image.imread('../img/pikachu.jpg')

# feature = image.imresize(img, 256, 256).astype('float32')

# X = feature.permute(2, 0, 1).expand_dims(axis=0)
# def predict(X):

#     anchors, cls_preds, bbox_preds = net(X.to(device))

#     cls_probs = cls_preds.softmax().permute(0, 2, 1)

#     output = d2l.MultiBoxDetection(cls_probs, bbox_preds, anchors)

#     idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]

#     return output[0, idx]



# output = predict(X)
# d2l.set_figsize((5, 5))



# def display(img, output, threshold):

#     fig = d2l.plt.imshow(img.asnumpy())

#     for row in output:

#         score = row[1].asscalar()

#         if score < threshold:

#             continue

#         h, w = img.shape[0:2]

#         bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]

#         d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')



# display(img, output, threshold=0.3)
# sigmas = [10, 1, 0.5]

# lines = ['-', '--', '-.']

# x = nd.arange(-2, 2, 0.1)

# d2l.set_figsize()



# for l, s in zip(lines, sigmas):

#     y = nd.smooth_l1(x, scalar=s)

#     d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)

# d2l.plt.legend();
# def focal_loss(gamma, x):

#     return -(1 - x) ** gamma * x.log()



# x = nd.arange(0.01, 1, 0.01)

# for l, gamma in zip(lines, [0, 1, 5]):

#     y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,

#                      label='gamma=%.1f' % gamma)

# d2l.plt.legend();