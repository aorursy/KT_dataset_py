# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, subdirs, filenames in os.walk('/kaggle/input'):

    for subdir in subdirs:

        print(subdir, end=' ')



# Any results you write to the current directory are saved as output.
import mxnet as mx, sys, os, time

from mxnet import autograd, nd, init, gluon, image

from mxnet.gluon import nn, data as gdata, loss as gloss

import math
def try_gpu():

    try:

        ctx = mx.gpu()

        _ = nd.array([0], ctx=ctx)

    except mx.base.MXNetError:

        ctx = mx.cpu()

    return ctx
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):

    if not autograd.is_training():

        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)

    else:

        assert len(X.shape) in (2, 4)

        if len(X.shape) == 2:

            mean = X.mean(axis=0)

            var = ((X - mean) ** 2).mean(axis=0)

        else:

            mean = X.mean(axis=(0, 2, 3), keepdims=True)

            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

        X_hat = (X - mean) / nd.sqrt(var + eps)

        moving_mean = moving_mean * momentum + (1 - momentum) * mean

        moving_var = moving_var * momentum + (1 - momentum) * var

    Y = gamma * X_hat + beta

    return Y, moving_mean, moving_var
class BatchNorm(nn.Block):

    def __init__(self, num_features, num_dims, **kwargs):

        super(BatchNorm, self).__init__(**kwargs)

        if num_dims == 2:

            shape = (1, num_features)

        else:

            shape = (1, num_features, 1, 1)

        self.gamma = self.params.get('gamma', shape=shape, init=init.One())

        self.beta = self.params.get('beta', shape=shape, init=init.Zero())

        self.moving_mean = nd.zeros(shape=shape)

        self.moving_var = nd.zeros(shape=shape)

    def forward(self, x):

        if self.moving_mean.context != x.context:

            self.moving_mean = self.moving_mean.as_in_context(x.context)

            self.moving_var = self.moving_var.as_in_context(x.context)

        Y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)

        return Y
def conv_blocks(num_convs, num_channels, kernel_size, strides, padding):

    blk = nn.Sequential()

    for i in range(num_convs):

        blk.add(nn.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding=padding),

                BatchNorm(num_channels, 4), nn.Activation('relu'))

    blk.add(nn.MaxPool2D(pool_size=2, strides=2))

    return blk
class SegmentNet(nn.Block):

    def __init__(self, **kwargs):

        super(SegmentNet, self).__init__(**kwargs)

        self.net = nn.Sequential()

        self.net.add(conv_blocks(2, 32, 5, 1, 2),

                conv_blocks(3, 64, 5, 1, 2),

                conv_blocks(4, 64, 5, 1, 2),

                nn.Conv2D(1024, kernel_size=15, strides=1, padding=7),

                BatchNorm(1024, 4), nn.Activation('relu'))

        self.conv1x1 = nn.Conv2D(1, kernel_size=1, strides=1, padding=0)

        self.batch_1x1 = BatchNorm(1, 4)

        self.sigmoid = nn.Activation('sigmoid')

    

    def forward(self, X):

        features = self.net(X)

        logits = self.conv1x1(features)

        logits_pixels = self.batch_1x1(logits)

        mask = self.sigmoid(logits_pixels)

        return features, logits_pixels, mask
class DecisionNet(nn.Block):

    def __init__(self, **kwargs):

        super(DecisionNet, self).__init__(**kwargs)

        self.net = nn.Sequential()

        self.net.add(nn.MaxPool2D(pool_size=2, strides=2),

                nn.Conv2D(8, kernel_size=5, strides=1, padding=2),

                BatchNorm(8, 4), nn.Activation('relu'),

                nn.MaxPool2D(pool_size=2, strides=2),

                nn.Conv2D(16, kernel_size=5, strides=1, padding=2),

                BatchNorm(16, 4), nn.Activation('relu'),

                nn.MaxPool2D(pool_size=2, strides=2),

                nn.Conv2D(32, kernel_size=5, strides=1, padding=2),

                BatchNorm(32, 4), nn.Activation('relu'))

        self.max_pool1 = nn.GlobalMaxPool2D()

        self.avg_pool1 = nn.GlobalAvgPool2D()

        self.max_pool2 = nn.GlobalMaxPool2D()

        self.avg_pool2 = nn.GlobalAvgPool2D()

        self.dense = nn.Dense(units=2)

        

    def forward(self, features, mask):

        assert len(features.shape) == 4

        data = mx.ndarray.concat(features, mask, dim=1)

        tmp = self.net(data)

        out = mx.ndarray.concat(self.max_pool1(tmp), self.avg_pool1(tmp), self.max_pool2(mask), self.avg_pool2(mask), dim=1)

        out = out.squeeze(axis=(2, 3))

        logits = self.dense(out)

        output = mx.ndarray.argmax(logits, axis=1)

        return logits, output
class combinedModel(nn.Block):

    def __init__(self, ctx, **kwargs):

        super(combinedModel, self).__init__(**kwargs)

        self.segment = SegmentNet()

        self.decision = DecisionNet()

        self.ctx = ctx

        self.init_params()

    

    def init_params(self):

        if os.path.exists('Segment.params'):

            self.segment.load_parameters('Segment.params', ctx=self.ctx)

        else:

            raise Exception('no segment net parameters file is found')

        if os.path.exists('Decision.params'):

            self.decision.load_parameters('Decision.params', ctx=self.ctx)

        else:

            self.decision.initialize(init=init.Normal(sigma=1), ctx=self.ctx)

        

    def forward(self, X):

        seg_features, seg_pixels, seg_mask = self.segment(X)

        logits, label = self.decision(seg_features, seg_mask)

        return logits, label
class DataManager(mx.gluon.data.dataset.Dataset):

    """A dataset for loading image files and labels stored in a list structure.



    Parameters

    ----------

    datalist : list

        a list of Paths contain image and label.



    flag : {0, 1}, default 1

        If 0, always convert loaded images to greyscale (1 channel).

        If 1, always convert loaded images to colored (3 channels).

        

    transform : callable, default None

        A function that takes data and label and transforms them::



            transform = lambda data, label: (data.astype(np.float32)/255, label)



    """

    def __init__(self, datalist, flag=0, transform=None):

        self._data_list = datalist

        self._flag = flag

        self._transform = transform

    

    def __getitem__(self, idx):

        imgfile, labelfile = self._data_list[idx]

        img = image.imread(imgfile, self._flag)

        img = image.imresize(img, IMAGE_SIZE[0], IMAGE_SIZE[1])

        label = image.imread(labelfile, self._flag)

        label = image.imresize(label, IMAGE_SIZE[0]//8, IMAGE_SIZE[1]//8)

        label_pixel = self._image_binarization(label)

        label = label_pixel.sum()

        if label > 0:

            label = 1

        else:

            label = 0

        if self._transform is not None:

            img = self._transform(img)

            label_pixel = self._transform(label_pixel)

        return img, label_pixel, label

    

    def __len__(self):

        return len(self._data_list)

    

    def _image_binarization(self, img, threshold=1):

        img = img > threshold

        return img.astype('float32')
def data_iter(dataloader):

    for img, pix, label in dataloader:

        yield img, pix, label
IMAGE_SIZE=[1280,512]

POSITIVE_KolektorSDD=[['5'], ['6'], ['2'], ['3'], ['5'], ['7'], ['1'], ['2'], ['6'], ['3'],

                        ['4'], ['5'], ['3'], ['7'], ['3'], ['5'], ['5'], ['3'], ['5'], ['4'],

                        ['5'], ['6'], ['6'], ['1'], ['4'], ['5'], ['0'], ['3'], ['0'], ['0'],

                        ['1'], ['2'], ['6'], ['0'], ['5'], ['3'], ['0'], ['0', '1'], ['6', '7'],['5'],

                        ['7'], ['3'], ['1'], ['6'], ['3'], ['7'], ['2'], ['5',], ['2'],['4']]



class Agent(object):

    def __init__(self, **kwargs):

        self.mode = kwargs['mode']

        self.data_dir = kwargs['data_dir']

        self.epoch_num = kwargs['epoch_num']

        self.batch_size = kwargs['batch_size']

        self.ctx = kwargs['ctx']

        self.lr = kwargs['learning_rate']

        self.init_dataset()

    

    def init_dataset(self):

        Positive_train, Negative_train, Positive_valid, Negative_valid = self.get_datalist(self.data_dir)

        transformer = gdata.vision.transforms.Compose([

            gdata.vision.transforms.ToTensor()

        ])

        DataManager_train_Positive = DataManager(Positive_train, transform=transformer)

        DataManager_train_Negative = DataManager(Negative_train, transform=transformer)

        DataManager_test_Positive = DataManager(Positive_valid, transform=transformer)

        DataManager_test_Negative = DataManager(Negative_valid, transform=transformer)

        self.train_pos = gdata.DataLoader(DataManager_train_Positive, self.batch_size, shuffle=True, last_batch='keep')

        self.train_neg = gdata.DataLoader(DataManager_train_Negative, self.batch_size, shuffle=True, last_batch='keep')

        self.test_pos = gdata.DataLoader(DataManager_test_Positive, self.batch_size, shuffle=False, last_batch='keep')

        self.test_neg = gdata.DataLoader(DataManager_test_Negative, self.batch_size, shuffle=False, last_batch='keep')

    

    def get_datalist(self, data_dir, test_ratio=0.4, positive_index=POSITIVE_KolektorSDD):

        example_dirs = [x[1] for x in os.walk(data_dir)][0]

        example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}

        train_test_offset=math.floor(len(example_lists)*(1-test_ratio))

        Positive_examples_train = []

        Negative_examples_train = []

        Positive_examples_valid = []

        Negative_examples_valid = []

        for i in range(len(example_dirs)):

            example_dir = example_dirs[i]

            example_list = example_lists[example_dir]

            # 过滤label图片

            example_list = [item for item in example_list if "label" not in item]

            # 训练数据

            if i < train_test_offset:

                for j in range(len(example_list)):

                    example_image = example_dir + '/' + example_list[j]

                    example_label = example_image.split(".")[0] + "_label.bmp"

                    # 判断是否是正样本

                    index = example_list[j].split(".")[0][-1]

                    if index in positive_index[i]:

                        Positive_examples_train.append([example_image, example_label])

                    else:

                        Negative_examples_train.append([example_image, example_label])

            else:

                for j in range(len(example_list)):

                    example_image = example_dir + '/' + example_list[j]

                    example_label = example_image.split(".")[0] + "_label.bmp"

                    index=example_list[j].split(".")[0][-1]

                    if index in positive_index[i]:

                        Positive_examples_valid.append([example_image, example_label])

                    else:

                        Negative_examples_valid.append([example_image, example_label])

        # concatenate root path

        Positive_examples_train = list(map(lambda x: [os.path.join(data_dir, x[0]), os.path.join(data_dir, x[1])], Positive_examples_train))

        Negative_examples_train = list(map(lambda x: [os.path.join(data_dir, x[0]), os.path.join(data_dir, x[1])], Negative_examples_train))

        Positive_examples_valid = list(map(lambda x: [os.path.join(data_dir, x[0]), os.path.join(data_dir, x[1])], Positive_examples_valid))

        Negative_examples_valid = list(map(lambda x: [os.path.join(data_dir, x[0]), os.path.join(data_dir, x[1])], Negative_examples_valid))

        return Positive_examples_train, Negative_examples_train, Positive_examples_valid, Negative_examples_valid

    

    def train(self):

        trainer = None

        loss = None

        best_score = 0.0

        if self.mode is 'segment':

            best_score = float('inf')

            net = SegmentNet()

            loss = gloss.SigmoidBinaryCrossEntropyLoss()

            if os.path.exists('Segment.params'):

                net.load_parameters('Segment.params', ctx=self.ctx)

            else:

                net.initialize(init=init.Uniform(1), ctx=self.ctx)

            trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': self.lr})

        elif self.mode is 'decision':

            best_score = 0.0

            loss = gloss.SoftmaxCrossEntropyLoss()

            net = combinedModel(ctx=self.ctx)

            trainer = gluon.Trainer(net.decision.collect_params(), 'sgd', {'learning_rate': self.lr})

        print('start training {} net'.format(self.mode))

        for i in range(self.epoch_num):

            iter_loss = 0

            train_pos_iter = data_iter(self.train_pos)

            train_neg_iter = data_iter(self.train_neg)

            try:

                while True:

                    pimg, ppix, plab = next(train_pos_iter)

                    nimg, npix, nlab = next(train_neg_iter)

                    loss_batch = 0

                    with autograd.record():

                        pimg, ppix, plab = pimg.as_in_context(self.ctx), ppix.as_in_context(self.ctx), plab.as_in_context(self.ctx)

                        if self.mode is 'segment':

                            _, y_hat, _ = net(pimg)

                            l = loss(y_hat, ppix).mean()

                        else:

                            y_hat, _ = net(pimg)

                            l = loss(y_hat, plab)

                    l.backward()

                    trainer.step(self.batch_size)

                    iter_loss += l.asscalar()

                    with autograd.record():

                        nimg, npix, nlab = nimg.as_in_context(self.ctx), npix.as_in_context(self.ctx), nlab.as_in_context(self.ctx)

                        if self.mode is 'segment':

                            _, y_hat, _ = net(nimg)

                            l = loss(y_hat, npix).mean()

                        else:

                            y_hat, _ = net(nimg)

                            l = loss(y_hat, nlab)

                    l.backward()

                    trainer.step(self.batch_size)

                    iter_loss += l.asscalar()

            except StopIteration:

                pass

            print('epoch:[{}] ,train_mode:{}, loss: {}'.format(i, self.mode, iter_loss))

            if self.mode is 'segment' and best_score > iter_loss:

                net.save_parameters('Segment.params')

                best_score = iter_loss

                print('The best {} net score: {}, save segment net parameters into file: Segment.params'.format(self.mode, best_score))

            elif self.mode is 'decision':

                test_list = [self.test_pos, self.test_neg]

                acc = self.evaluate_acc(test_list, net)

                print('test acc is: {}'.format(acc))

                if best_score < acc:

                    net.decision.save_parameters('Decision.params')

                    best_score = acc

                    print('The best {} net score: {}, save decision net parameters into file: Decision.params'.format(self.mode, best_score))

                

    def evaluate_acc(self, test_iter_list, net):

        acc_sum, n = 0.0, 0

        for test_iter in test_iter_list:

            for img, pix, label in test_iter:

                img, pix, label = img.as_in_context(self.ctx), pix.as_in_context(self.ctx), label.as_in_context(self.ctx)

                _, y_hat = net(img)

                acc_sum += (y_hat == label.astype('float32')).sum().asscalar()

                n += label.size

        return acc_sum / n

    

    def evaluate_mse(self, test_iter_list, net):

        mse_loss, n = 0.0, 0

        for test_iter in test_iter_list:

            for img, pix, label in test_iter:

                img, pix, label = img.as_in_context(self.ctx), pix.as_in_context(self.ctx), label.as_in_context(self.ctx)

                _, _, y_hat = net(img)

                mse_loss += ((y_hat - pix.astype('float32')) ** 2).mean().asscalar()

                n += label.size

        return mse_loss / n



    def test(self):

        net = combinedModel(ctx=self.ctx)

        test_list = [self.test_pos, self.test_neg]

        acc = self.evaluate_acc(test_list, net)

        print('test acc is: {}'.format(acc))

    

#     def test(self):

#         if self.mode is 'segment':

#             net = SegmentNet()

#             loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

#             if os.path.exists('Segment.params'):

#                 net.load_parameters('Segment.params', ctx=self.ctx)

#             else:

#                 raise Exception('no segment module was found!')

#         elif self.mode is 'decision':

#             print('dummy')

#             return None

#         print('Starting testing on {} net'.format(self.mode))

#         count=0

#         count_TP = 0  # 真正例

#         count_FP = 0  # 假正例

#         count_TN = 0  # 真反例

#         count_FN = 0  # 假反例

#         for pimg, ppix, plabel in self.test_pos:

#             pimg, plabel = pimg.as_in_context(self.ctx), plabel.as_in_context(self.ctx)

#             y_hat = net(pimg)
lr, ctx, epoch_num, batch_size = 0.1, try_gpu(), 100, 1

agent = Agent(mode='decision', data_dir='/kaggle/input/kolektorsdd', epoch_num=epoch_num, batch_size=batch_size, learning_rate=lr, ctx=ctx)

agent.train()