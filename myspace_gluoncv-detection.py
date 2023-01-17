# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import ast



# Read wheat detection dataset



data = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

data.head()



# Separate the bbox column

df = data.bbox.str.rstrip(']')

df = df.str.lstrip('[') 

df = df.str.replace(",",'')

data[["x", "y","h", "w"]] = pd.DataFrame([x.split(' ') for x in df.tolist()], index= data.index)

# Add xmin and xmax

xmin = data.iloc[:, 5].astype('float32')

ymin = data.iloc[:, 6].astype('float32')

data['xmax'] = xmin + data.iloc[:, 7].astype('float32')

data['ymax'] = ymin + data.iloc[:, 8].astype('float32')

# Add class name

data['class_name'] = 'wheat'

# Add the extension to the images

data['image_id'] = data['image_id'] + '.jpg'

data.head()





classes = data['class_name'].unique().tolist()



data['class_int'] = data['class_name'].map(lambda x: classes.index(x))

data.head()
import os

import time

from matplotlib import pyplot as plt

import numpy as np

import mxnet as mx

from mxnet import autograd, gluon

import gluoncv as gcv

from gluoncv.utils import download, viz

import cv2



class wheat_data(gluon.data.Dataset):

    def __init__(self, csv_file,img_dir):

        self.data_info = csv_file 

        self.image_arr = self.data_info['image_id'].unique() 

        self.bbox_arr = self.data_info.iloc[:, 3]

        self.img_dir = img_dir

        

    def __getitem__(self, idx):

        image_arr = self.image_arr[idx]

        image = mx.image.imread(f'{self.img_dir}/{image_arr}')

        img_path = f'{self.img_dir}/{image_arr}'

        num_bbox = len(self.bbox_arr)

            

        data = self.data_info[self.data_info['image_id'] == image_arr]

        boxes = data[['x','y','xmax','ymax']].values

        

        image_id = data[['class_int']].values

        img_shape = image.shape

        

        return img_path, img_shape,np.array(boxes), np.array(image_id),idx 

    



    def __len__(self):

        return len(self.image_arr)

        

        
dataset = wheat_data(data,'/kaggle/input/global-wheat-detection/train/')

print(dataset[0])
def write_line(img_path, im_shape, boxes, ids, idx):

    h, w, c = im_shape

    # for header, we use minimal length 2, plus width and height

    # with A: 4, B: 5, C: width, D: height

    A = 4

    B = 5

    C = w

    D = h

    # concat id and bboxes

    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')

    # normalized bboxes (recommanded)

    labels[:, (1, 3)] /= float(w)

    labels[:, (2, 4)] /= float(h)

    # flatten

    labels = labels.flatten().tolist()

    str_idx = [str(idx)]

    str_header = [str(x) for x in [A, B, C, D]]

    str_labels = [str(x) for x in labels]

    str_path = [img_path]

    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'

    return line
with open('/kaggle/working/wheat_train.lst', 'w') as fw:

    for img_path, im_shape, all_boxes, all_ids,i in dataset:

        line = write_line(img_path, im_shape, all_boxes, all_ids, i)

        #print(line)

        fw.write(line)
from gluoncv.data import LstDetection

from gluoncv.utils import download, viz



dataset = LstDetection('wheat_train.lst', root=os.path.expanduser('.'))



image, label = dataset[0]

classes = ['wheat']

ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)

plt.show()
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,

    pretrained_base=False, transfer='voc')
def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):

    from gluoncv.data.batchify import Tuple, Stack, Pad

    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

    width, height = data_shape, data_shape

    # use fake data to generate fixed anchors for target generation

    with autograd.train_mode():

        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))

    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets

    train_loader = gluon.data.DataLoader(

        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),

        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    return train_loader



train_data = get_dataloader(net, dataset, 512, 1, 0)


a = mx.nd.zeros((1,), ctx=mx.gpu(0))

ctx = [mx.gpu(0)]

net.collect_params().reset_ctx(ctx)

trainer = gluon.Trainer(

    net.collect_params(), 'sgd',

    {'learning_rate': 0.0002, 'wd': 0.0005, 'momentum': 0.9})



mbox_loss = gcv.loss.SSDMultiBoxLoss()

ce_metric = mx.metric.Loss('CrossEntropy')

smoothl1_metric = mx.metric.Loss('SmoothL1')



for epoch in range(0, 2):

    ce_metric.reset()

    smoothl1_metric.reset()

    tic = time.time()

    btic = time.time()

    net.hybridize(static_alloc=True, static_shape=True)

    for i, batch in enumerate(train_data):

        batch_size = batch[0].shape[0]

        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)

        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

        with autograd.record():

            cls_preds = []

            box_preds = []

            for x in data:

                cls_pred, box_pred, _ = net(x)

                cls_preds.append(cls_pred)

                box_preds.append(box_pred)

            sum_loss, cls_loss, box_loss = mbox_loss(

                cls_preds, box_preds, cls_targets, box_targets)

            autograd.backward(sum_loss)

        # since we have already normalized the loss, we don't want to normalize

        # by batch-size anymore

        trainer.step(1)

        ce_metric.update(0, [l * batch_size for l in cls_loss])

        smoothl1_metric.update(0, [l * batch_size for l in box_loss])

        name1, loss1 = ce_metric.get()

        name2, loss2 = smoothl1_metric.get()

        if i % 20 == 0:

            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(

                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))

        btic = time.time()



net.save_parameters('/kaggle/working/ssd_512_mobilenet1.0_wheat.params')        
classes = ['wheat']

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)

net.load_parameters('/kaggle/working/ssd_512_mobilenet1.0_wheat.params')





sub = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

img_id = sub['image_id'].values

submission = []



for img in img_id:

     #print(img)

    prediction_string = []

    x, image = gcv.data.transforms.presets.ssd.load_test(f'/kaggle/input/global-wheat-detection/test/{img}.jpg', 224)

    cid, score, bbox = net(x)

    #print(score[0][0].asnumpy().squeeze().astype(float))

    for (x_min,y_min,x_max,y_max),s in zip(bbox[0].asnumpy(),score[0].asnumpy().squeeze().astype(float)):

        x = round(x_min)

        y = round(y_min)

        h = round(x_max-x_min)

        w = round(y_max-y_min)

        prediction_string.append(f"{s} {x} {y} {h} {w}")

    prediction_string = " ".join(prediction_string)

    

    submission.append([img,prediction_string])



sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])

sample_submission.to_csv('/kaggle/working/submission.csv', index=False)



mysub = pd.read_csv('/kaggle/working/submission.csv')

mysub.head()