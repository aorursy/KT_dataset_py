%matplotlib inline

import d2l_utils_eg as d2l

from mxnet import autograd, contrib, gluon, image, init, nd

from mxnet.gluon import nn

from mxnet.gluon.model_zoo import vision

class residual_block(nn.Block):

    def __init__(self, c, use_conv1x1=False, strides=1,**kwargs):

        super().__init__(**kwargs)

        self.main_path = nn.Sequential()

        

        self.main_path.add(

             nn.Conv2D(c,kernel_size=3,padding=1, strides=strides),

             nn.BatchNorm(),

             nn.Activation('relu'),

             nn.Conv2D(c,kernel_size=3, padding=1),

             nn.BatchNorm()

        )

        self.conv1x1 = None

        if use_conv1x1:

            self.conv1x1 = nn.Sequential()

            self.conv1x1.add(

                 nn.Conv2D(c,kernel_size=1, strides=strides)

            )

        self.final_relu = nn.Activation('relu')

    def forward(self,x):

        o = self.main_path(x)

        if self.conv1x1:

            return self.final_relu(o+self.conv1x1(x))

        else: 

            return self.final_relu(o+x)

class resnet_block(nn.Block):

    def __init__(self, c, num_residuals, first_block = False, **kwargs):

        super().__init__(**kwargs)

        self.block = nn.Sequential()

        for i in range(num_residuals):

            if i == 0 and not first_block:

                self.block.add(residual_block(c=c, use_conv1x1=True, strides=2))

            else:

                self.block.add(residual_block(c=c))

    def forward(self, x):

        return self.block(x)

        

class Resnet18(nn.Block):

    

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.model = nn.Sequential()

        

        self.model.add(

            nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3),

            nn.BatchNorm(),

            nn.Activation('relu'),

            nn.MaxPool2D(pool_size=3, strides=2,padding=1),

            resnet_block(c = 64, num_residuals=2, first_block=True),

            resnet_block(c = 128, num_residuals=2),

            resnet_block(c = 256, num_residuals=2),

            resnet_block(c = 512, num_residuals=2)

        )

    def forward(self, x):

        return self.model(x)
def base_net():

    net = Resnet18()

    net.initialize(ctx=d2l.try_gpu())

    return net

# m = base_net()
m = base_net()
def cls_predictor(num_anchors, num_classes):

    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,

                     padding=1)

def bbox_predictor(num_anchors):

    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)



def forward(x, block):

    block.initialize()

    return block(x)



def flatten_pred(pred):

    return pred.transpose((0, 2, 3, 1)).flatten()



def concat_preds(preds):

    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(num_channels):

    blk = nn.Sequential()

    for _ in range(2):

        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),

                nn.BatchNorm(in_channels=num_channels),

                nn.Activation('relu'))

    blk.add(nn.MaxPool2D(2))

    return blk

# def base_net():

#     blk = nn.Sequential()

#     for num_filters in [16, 32, 64]:

#         blk.add(down_sample_blk(num_filters))

#     return blk





def base_net():

    net = Resnet18()

    net.initialize(ctx=d2l.try_gpu())

    return net



# forward(nd.zeros((2, 3, 256, 256)), base_net()).shape

def get_blk(i):

    if i == 0:

        blk = base_net()

    elif i == 4:

        blk = nn.GlobalMaxPool2D()

    else:

        blk = down_sample_blk(128)

    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):

    Y = blk(X)

    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)

    cls_preds = cls_predictor(Y)

    bbox_preds = bbox_predictor(Y)

    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],

         [0.88, 0.961]]

ratios = [[1, 2, 0.5]] * 5

num_anchors = len(sizes[0]) + len(ratios[0]) - 1



class TinySSD(nn.Block):

    def __init__(self, num_classes, **kwargs):

        super(TinySSD, self).__init__(**kwargs)

        self.num_classes = num_classes

        for i in range(5):

            # The assignment statement is self.blk_i = get_blk(i)

            setattr(self, 'blk_%d' % i, get_blk(i))

            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors,

                                                      num_classes))

            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))



    def forward(self, X):

        X = X / 255.0

        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        for i in range(5):

            # getattr(self, 'blk_%d' % i) accesses self.blk_i

            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(

                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],

                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))

        # In the reshape function, 0 indicates that the batch size remains

        # unchanged

        return (nd.concat(*anchors, dim=1),

                concat_preds(cls_preds).reshape(

                    (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))
from load_pascal import *

batch_size = 32

train_iter, val_iter = load_pascal_dataset(batch_size=32, edge_size=256)


JSON_PATH = Path("../input/pascal/PASCAL_VOC/PASCAL_VOC")

#     'pascal_train2007.json'

trn_j = json.load((JSON_PATH/"pascal_train2012.json").open())

IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']

FILE_NAME,ID,IMG_ID,CAT_ID,BBOX,WIDTH, HEIGHT = 'file_name','id','image_id','category_id','bbox', 'width', 'height'

cats = {o[ID]:o['name'] for o in trn_j[CATEGORIES]} #cat_id -> cat_string

batch = train_iter.next()

batch.data[0].shape, batch.label[0].shape

edge_size = 256

imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255

axes = d2l.show_images(imgs, 2, 5, scale=2)

for ax, label in zip(axes, batch.label[0][0:10]):

#     print()

    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], labels=cats[label[0][0].asscalar()],colors=['w'])
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()

bbox_loss = gluon.loss.L1Loss()



def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):

    cls = cls_loss(cls_preds, cls_labels)

    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)

    return cls + bbox

def cls_eval(cls_preds, cls_labels):

    # Because the category prediction results are placed in the final

    # dimension, argmax must specify this dimension

    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()



def bbox_eval(bbox_preds, bbox_labels, bbox_masks):

    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()
ctx, net = d2l.try_gpu(), TinySSD(num_classes=20)

net.initialize(init=init.Xavier(), ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd',

                        {'learning_rate': 0.2, 'wd': 5e-4})



num_epochs, timer = 20, d2l.Timer()

animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],

                        legend=['class error', 'bbox mae'])

for epoch in range(num_epochs):

    # accuracy_sum, mae_sum, num_examples, num_labels

    metric = d2l.Accumulator(4)

    train_iter.reset()  # Read data from the start.

    for batch in train_iter:

        timer.start()

        X = batch.data[0].as_in_context(ctx)

        Y = batch.label[0].as_in_context(ctx)

        with autograd.record():

            # Generate multiscale anchor boxes and predict the category and

            # offset of each

            anchors, cls_preds, bbox_preds = net(X)

            # Label the category and offset of each anchor box

            bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(

                anchors, Y, cls_preds.transpose((0, 2, 1)))

            # Calculate the loss function using the predicted and labeled

            # category and offset values

            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,

                          bbox_masks)

        l.backward()

        trainer.step(batch_size)

        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,

                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),

                    bbox_labels.size)

    cls_err, bbox_mae = 1-metric[0]/metric[1], metric[2]/metric[3]

    animator.add(epoch+1, (cls_err, bbox_mae))

print('class err %.2e, bbox mae %.2e' % (cls_err, bbox_mae))

print('%.1f exampes/sec on %s'%(train_iter.num_image/timer.stop(), ctx))
# ! ls "../input/pascal/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
def predict(X):

    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))

    print(f'anchor shape: {anchors.shape}')

    print(f'cls shape: {cls_preds.shape}')

    cls_probs = cls_preds.softmax().transpose((0, 2, 1))

    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)

    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]

    return output[0, idx]

def display(img, output, threshold):

    d2l.set_figsize((5, 5))

    fig = d2l.plt.imshow(img.asnumpy())

    for row in output:

        score = row[1].asscalar()

        if score < threshold:

            continue

        h, w = img.shape[0:2]

        print(row)

        print(cats[int(row[0].asscalar())])

        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]

        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
img = image.imread("../input/pascal/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/006223.jpg")

feature = image.imresize(img, 256, 256).astype('float32')

X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

output = predict(X)

display(img, output, threshold=0.05)

img = image.imread("../input/pascal/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/007914.jpg")

feature = image.imresize(img, 256, 256).astype('float32')

X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

output = predict(X)

display(img, output, threshold=0.1)

img = image.imread("../input/pascal/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/002866.jpg")

feature = image.imresize(img, 256, 256).astype('float32')

X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

output = predict(X)

display(img, output, threshold=0.1)
