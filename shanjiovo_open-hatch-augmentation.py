import os

import numpy as np

from matplotlib import pyplot as plt

import cv2

from PIL import Image

import torch

import torch.utils.data

import torchvision.transforms as T

from torchvision.transforms import functional as F

import random 

import base64

import json

from torch.utils.data import DataLoader, random_split,Subset

import pickle

from PIL import Image, ImageEnhance

import math
# 图像增广方法汇总

class Augmentation:

    

    def image_augment(self,img, gtboxes, gtlabels, means=None):

        # 随机改变亮暗、对比度和颜色等

        img = self.random_distort(img)

        # 随机填充

        img, gtboxes = self.random_expand(img, gtboxes, fill=means)

        # 随机裁剪

        img, gtboxes, gtlabels, = self.random_crop(img, gtboxes, gtlabels)

#         # 随机缩放

#         img = self.random_interp(img, size)

        # 随机翻转

        img, gtboxes = self.random_flip(img, gtboxes)

        # 随机打乱真实框排列顺序

        gtboxes, gtlabels = self.shuffle_gtbox(gtboxes, gtlabels)

        return img, gtboxes, gtlabels

    

    

    # 随机改变亮暗、对比度和颜色等

    def random_distort(self,img):

        # 随机改变亮度

        def random_brightness(img, lower=0.5, upper=1.5,thresh=0.5):

            if random.random() > thresh:

                return img

            e = np.random.uniform(lower, upper)

            return ImageEnhance.Brightness(img).enhance(e)

        # 随机改变对比度

        def random_contrast(img, lower=0.5, upper=1.5,thresh=0.5):

            if random.random() > thresh:

                return img

            e = np.random.uniform(lower, upper)

            return ImageEnhance.Contrast(img).enhance(e)

        # 随机改变颜色

        def random_color(img, lower=0.5, upper=1.5,thresh=0.5):

            if random.random() > thresh:

                return img

            e = np.random.uniform(lower, upper)

            return ImageEnhance.Color(img).enhance(e)



        ops = [random_brightness, random_contrast, random_color]

        np.random.shuffle(ops)



        img = Image.fromarray(img)

        img = ops[0](img)

        img = ops[1](img)

        img = ops[2](img)

        img = np.array(img)

        return img

    

    #### 已经可用

    # 随机填充

    def random_expand(self,

                      img,

                  gtboxes,

                  max_ratio=1.5,

                  fill=None,

                  keep_ratio=True,

                  thresh=0.5):

        if random.random() > thresh:

            return img, gtboxes

        if max_ratio < 1.0:

            return img, gtboxes

        

        h, w, c = img.shape

        ratio_x = random.uniform(1, max_ratio)

        if keep_ratio:

            ratio_y = ratio_x

        else:

            ratio_y = random.uniform(1, max_ratio)

        oh = int(h * ratio_y)  ###新的宽高

        ow = int(w * ratio_x)

        off_x = random.randint(0, ow - w) ## 相对左上角的偏置

        off_y = random.randint(0, oh - h)



        out_img = np.zeros((oh, ow, c))   ## 处理图像

        if fill and len(fill) == c:

            for i in range(c):

                out_img[:, :, i] = fill[i] * 255.0

        out_img[off_y:off_y + h, off_x:off_x + w, :] = img



        ## 处理gtboxes

        gtboxes[:, 0] = gtboxes[:, 0] + off_x

        gtboxes[:, 1] = gtboxes[:, 1] + off_y

        gtboxes[:, 2] = gtboxes[:, 2] + off_x

        gtboxes[:, 3] = gtboxes[:, 3] + off_y

        return out_img.astype('uint8'), gtboxes.astype('float32')

    



    

    def random_crop(self,image, boxes, laels,thresh=0.5):

        if random.random() > thresh:

            return image, boxes, laels

        heigth, width, _ = image.shape

        if len(boxes) == 0:

            return image, boxes, laels



        while True:

            mode = random.choice((

                None,

                (0.1, None),

                (0.3, None),

                (0.5, None),

                (0.7, None),

                (0.9, None),

                (None, None)

            ))

            if mode is None:

                return image, boxes, laels



            min_iou, max_iou = mode

            if min_iou is None:

                min_iou = float('-inf')

            if max_iou is None:

                max_iou = float('inf')

            # 多次迭代 

            for _ in range(50):

                scale = random.uniform(0.3, 1.)

                min_ratio = max(0.5, scale*scale)

                max_ratio = min(2, 1./scale/scale)

                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))

                # 随机裁剪的图片大小

                w = int(scale*ratio*width)

                h = int((scale/ratio)*heigth)

                # 从0- width-w中选择一个起始点

                l = random.randrange(width-w)

                t = random.randrange(heigth-h)

                roi = np.array([l, t, l+w, t+h])

                # 计算目标框和待裁剪区域的iou大小 iou太小则重新选择

                iou = self.matrix_iou(boxes, roi[np.newaxis])

                if not (min_iou <= iou.min() and iou.max() <= max_iou):

                    continue



                image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

                # 得到每个目标框中心点

                centers = (boxes[:, :2] + boxes[:, 2:])/2

                # 选择在roi区域的box

                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)

                boxes_t = boxes[mask].copy()

                laels_t = laels[mask].copy()



                # 裁剪区域不存在目标 重新裁剪

                if len(boxes_t) == 0:

                    continue

                # 判断box是否超过roi范围并调整

                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])

                boxes_t[:, :2] -= roi[:2]

                boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])

                boxes_t[:, 2:] -= roi[:2]

                return image_t, boxes_t.astype('float32'), laels_t



    def matrix_iou(self,a,b):

        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])

        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])



        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)

        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)

        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

        return area_i / (area_a[:, np.newaxis] + area_b - area_i)

    

    

    

    # work

    # 随机打乱真实框排列顺序

    def shuffle_gtbox(self,gtbox, gtlabel,thresh=0.5):

        if random.random() > thresh:

            return gtbox, gtlabel

        gt = np.concatenate([gtbox, gtlabel[:, np.newaxis]], axis=1)

        idx = np.arange(gt.shape[0])

        np.random.shuffle(idx)

        gt = gt[idx, :]

        return gt[:, :4].astype('float32'), gt[:, 4]

    

    # 随机翻转

      ##  翻转图片，输入为numpy的array类型

    def random_flip(self,image, boxes,thresh=0.5):

        if random.random() > thresh:

            return image, boxes

        width = image.shape[1]

        image = F.hflip(Image.fromarray(image) )

        boxes[:, 0::2] = width - boxes[:, 2::-2]

        return np.array(image), boxes.astype('float32')
class BasicDataset(torch.utils.data.Dataset):

  def __init__(self, dir_path, augmentation=None,is_trained=False):

    self.augmentation = augmentation

    self.is_trained = is_trained

    self.dataset = []

    for file_name in list(sorted(os.listdir(dir_path))):

        if(file_name.endswith('.json')):

          file_path = os.path.join(dir_path,file_name)

          with open(file_path, 'rb') as f: 

            ### 读取一个样本

            one_data = pickle.load(f)

            one_data['target']['boxes'] = np.array(one_data['target']['boxes'], dtype=np.float32)

            ### 加入，现在都还是numpy类型

            self.dataset.append(one_data)





  

  def __getitem__(self, index):

    item = self.dataset[index]

    

    # 要搞个副本，避免原始数据遭到修改

    image = item['image'].copy()

    target = {k:v.copy() for k,v in item['target'].items()} 

    

#     ## 这个要放在这里转化，因为所在空间是翻了20倍，内存吃不消

    image = cv2.imdecode(image, cv2.IMREAD_COLOR) 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    boxes = target['boxes']

    labels = np.array([True]*len(boxes))

    if(self.is_trained):

        image,boxes,labels = self.augmentation.image_augment(image,boxes,labels)

    

    ## 对数据的随机处理要放在深复制后面，避免原始数据被修改

    ## 这里将numpy转成Tensor

    image = F.to_tensor(image)

    target = {

        'boxes':torch.as_tensor(boxes, dtype=torch.float32),

        'labels':torch.as_tensor(labels,dtype=torch.int64)

    }

    

    return image,target





  def __len__(self):

    return len(self.dataset)
train_path = '../input/airport-dataset-v2/AirportDataset/_开舱门/train'

test_path = '../input/airport-dataset-v2/AirportDataset/_开舱门/test'
augmentation = Augmentation()
# 这里直接是dataset_train、dataset_test

dataset_train = BasicDataset(train_path,augmentation,True)

dataset_test = BasicDataset(test_path,augmentation,False)

print(len(dataset_train),len(dataset_test))
def collate_fn(batch):

  return tuple(zip(*batch)) #这里的意思估计是对batch解压，再转成元组，长度为二。元组的元组

  

train_loader = torch.utils.data.DataLoader(

    dataset_train, batch_size=1, shuffle=True, num_workers=4, # batch_size不能大，用pin_memory也会内存爆炸

    collate_fn=collate_fn)



test_loader = torch.utils.data.DataLoader(

    dataset_test, batch_size=1, shuffle=False, num_workers=4,pin_memory=True,

    collate_fn=collate_fn)
def validate():

  loss = 0

  cnt = 0

  with torch.no_grad():

    for images, targets in test_loader:

      # images, targets = images.to(device), targets.to(device)

      images = list(image.to(device) for image in images)

      targets = [{k:v.to(device) for k,v in t.items()} for t in targets]



      loss_dict = model(images, targets)

      loss += sum(loss for loss in loss_dict.values())

      cnt += len(images)

  loss = loss/cnt

  print('--Loss of the network on the test images: %.3f\n' % loss)

  return loss
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



num_classes = 2



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0001,momentum=0.9,weight_decay=0.0005)



lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
num_epochs = 40

train_list = []

valid_list = []

for epoch in range(num_epochs): 

  itr_loss = 0.0

  epo_loss = 0.0

  accumulation_steps = 16

  itr = 0

  optimizer.zero_grad()

  for i,(images,targets) in enumerate(train_loader):

    images = list(image.to(device) for image in images)

    targets = [{key:val.to(device) for key,val in target.items()} for target in targets]



    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    losses.backward()  

    

    itr_loss += losses.item() 

    epo_loss += losses.item()

    itr = itr+len(images)

    if (itr%accumulation_steps) == 0: 

      optimizer.step() 

      optimizer.zero_grad()

      print("Iteration %d train_loss:%.03f"%(itr,itr_loss/accumulation_steps))

      itr_loss = 0   



  optimizer.step() 

  optimizer.zero_grad()

  if lr_scheduler is not None:

      lr_scheduler.step()

  epo_loss = epo_loss/itr

  train_list.append(epo_loss)

  print('--Epoch %d finished, train_loss:%.3f lr=%f'%(epoch,epo_loss,lr_scheduler.get_last_lr()[0])) 

  ret = validate()

  valid_list.append(ret)
epochs = range(1+1, len(train_list) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, train_list[1:], 'r', label='Training loss')

# b is for "solid blue line"x

plt.plot(epochs, valid_list[1:], 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
torch.save(model, 'open_hatch.pkl')
class NMSTool:

    

    def __init__(self,

             nms_iou_threshold=0.5

        ):

        self.nms_iou_threshold = nms_iou_threshold

    

    

    # NMS算法

    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor

    def nms(self,prediction):

        

        boxes = prediction['boxes']

        scores = prediction['scores']

        labels = prediction['labels']

        

        if(len(scores)==0):

            return prediction



        x1 = boxes[:,0]

        y1 = boxes[:,1]

        x2 = boxes[:,2]

        y2 = boxes[:,3]

        areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积

        _, order = scores.sort(0, descending=True)    # 降序排列



        keep = []

        while order.numel() > 0:       # torch.numel()返回张量元素个数

            if order.numel() == 1:     # 保留框只剩一个

                i = order.item()

                keep.append(i)

                break

            else:

                i = order[0].item()    # 保留scores最大的那个框box[i]

                keep.append(i)



            # 计算box[i]与其余各框的IOU(思路很好)

            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]

            yy1 = y1[order[1:]].clamp(min=y1[i])

            xx2 = x2[order[1:]].clamp(max=x2[i])

            yy2 = y2[order[1:]].clamp(max=y2[i])

            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]



            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]

            idx = (iou <= self.nms_iou_threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]

            if idx.numel() == 0:

                break

            order = order[idx+1]  # 修补索引之间的差值

        

        prediction['boxes'] = prediction['boxes'][keep]

        prediction['scores'] = prediction['scores'][keep]

        prediction['labels'] = prediction['labels'][keep]

        return prediction
nms_tool = NMSTool(0.01)

model.eval()

score_threshold = 0.3

cnt = 0

with torch.no_grad():

  for images, targets in test_loader:

    image = images[0]

    target = targets[0]

    img = image

    image = (image*255).permute([1,2,0]).numpy().astype(np.uint8).copy()

    image.shape

    prediction = model([img.to(device)])[0]



    keep = nms_tool.nms(prediction)



    for i in range(len(prediction['boxes'])):

      box = prediction['boxes'][i]

      score = prediction['scores'][i]

      if(score<score_threshold):

        continue

      cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 4)

    

    

    for i in range(len(target['boxes'])):

      box = target['boxes'][i]

      cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(0, 0, 220), 2)

    



    plt.figure(figsize=(8,8),dpi=150)

    plt.subplot(111)

    plt.imshow(image)

    plt.show()

    cnt = cnt+1

    

    if cnt==50:

        break
class Evaluator:

    """

    功能：

        对行李箱检测器的性能进行评估。

        指标有：precision、recall、mAP

    """



    def voc_eval(self,

                 predictions,

                 targets,

                 ovthresh=0.5,

                 use_07_metric=False

                 ):

        """

        功能：

            对行李箱检测器的性能进行评估。



        参数:

            detector（model）:需要进行评估的检测器。

            images (np.array): 输入图像集。[Batch,Width,Height,Channels]的四维array数组，取值为[0,255]

            targets (list): 输入图像对应的Ground Truth Boxes的信息。为一个list，每个元素又为一个dict。

                其格式同检测器detector检测一张图片时输出的结果格式相同。

            ovthresh (float): IoU的阈值。当IoU大于该值时，检测框才被认为是true



        返回值:

            metric（dict）：一个字典。有3个键，分表表示3个指标：

            metric['recall']召回率、metric['precision']精确率、以及metric['mAP']

        """



        ## 读取GT box

        npos = 0

        class_recs = {}

        for i, target in enumerate(targets):

            det = np.array([False] * len(target['boxes']))

            difficult = np.array([False] * len(target['boxes']))

            npos = npos + sum(~difficult)

            class_recs[i] = {'bbox': target['boxes'].numpy(),

                            'det': det}



  

        image_ids = []

        confidence = []

        BB = []

        for i, prediction in enumerate(predictions):

            for j in range(len(prediction['scores'])):

                image_ids.append(i)

            for score in prediction['scores']:

                confidence.append(score)

            for box in prediction['boxes']:

                BB.append(box)



        image_ids = np.array(image_ids)

        confidence = np.array(confidence)

        BB = np.array(BB)

        '''

        class_recs 图片gt box的信息

        下面三个都是model的预测结果：

        image_ids det box对应的图片ID

        confidence det box的置信度

        BB  det box的四个坐标

        '''

        # sort by confidence 根据det box的置信度重新排序

        sorted_ind = np.argsort(-confidence)

        BB = BB[sorted_ind, :]  # 预测框坐标

        image_ids = [image_ids[x] for x in sorted_ind]  # 各个预测框的对应图片id# 便利预测框，并统计TPs和FPsnd = len(image_ids)



        # go down dets and mark TPs and FPs

        nd = len(image_ids)

        tp = np.zeros(nd)

        fp = np.zeros(nd)

        for d in range(nd):  ##枚举所有det box

            R = class_recs[image_ids[d]]  ## 找到对应的图片

            bb = BB[d, :].astype(float)  ##预测的box，等等，这里究竟有一个还是多个box？好似是只有一个box

            ovmax = -np.inf

            BBGT = R['bbox'].astype(float)  ##gt box， 好像是有多个box。整张图片对应class的所有gt box了



            if BBGT.size > 0:

                # compute overlaps

                # intersection

                ixmin = np.maximum(BBGT[:, 0], bb[0])

                iymin = np.maximum(BBGT[:, 1], bb[1])

                ixmax = np.minimum(BBGT[:, 2], bb[2])

                iymax = np.minimum(BBGT[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin + 1., 0.)

                ih = np.maximum(iymax - iymin + 1., 0.)

                inters = iw * ih



                # union

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +

                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *

                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)



                overlaps = inters / uni

                ovmax = np.max(overlaps)

                jmax = np.argmax(overlaps)



            if ovmax > ovthresh:

                if not R['det'][jmax]:

                    tp[d] = 1.

                    R['det'][jmax] = 1

                else:

                    fp[d] = 1.

            else:

                fp[d] = 1.



        # compute precision recall

        my_rec = np.sum(tp)/npos

        my_prec = np.sum(tp)/nd

        wrong_detected_rate = 1-my_prec

        fp = np.cumsum(fp)  ##当成一维数据并求前缀和

        tp = np.cumsum(tp)

        rec = tp / float(npos)

        # avoid divide by zero in case the first detection matches a difficult

        # ground truth

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = self.__voc_ap(rec, prec, use_07_metric)

        metric = {'recall': my_rec,  'mAP': ap,'precision':my_prec,'wrong_detected_rate':wrong_detected_rate}

        return metric





    def __voc_ap(self,

                 rec,

                 prec,

                 use_07_metric=False):

        """Compute VOC AP given precision and recall. If use_07_metric is true, uses

        the VOC 07 11-point method (default:False).

        """

        if use_07_metric:

            # 11 point metric

            ap = 0.

            for t in np.arange(0., 1.1, 0.1):

                if np.sum(rec >= t) == 0:

                    p = 0

                else:

                    p = np.max(prec[rec >= t])

                    ap = ap + p / 11.

        else:

            # correct AP calculation

            # first append sentinel values at the end

            mrec = np.concatenate(([0.], rec, [1.]))

            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope

            for i in range(mpre.size - 1, 0, -1):

                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points

                # where X axis (recall) changes value

                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec

                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap
images = [x[0] for x in dataset_test]

targets = [x[1] for x in dataset_test]
evaluator = Evaluator()
nms_tool = NMSTool(0.01)

model.eval()

score_threshold = 0.3

predictions = []

with torch.no_grad():

  for img in images:

    prediction = model([img.to(device)])[0]

    

    prediction = nms_tool.nms(prediction)  ## NMS

    boxes = []

    scores = []

    labels = []

    for i in range(len(prediction['labels'])):

      box = prediction['boxes'][i].cpu().numpy()

      score = prediction['scores'][i].cpu().numpy()

      label = prediction['labels'][i].cpu().numpy()

      if(score<score_threshold):      ## 低阈值

        continue

      boxes.append(box)

      scores.append(score)

      labels.append(label)

#       print(score)

    prediction = {'boxes': boxes, 'scores': scores, 'labels': labels}

    predictions.append(prediction)
metric = evaluator.voc_eval(predictions,targets,ovthresh=0.2)
print('查全率: %.3f'%metric['recall'])

print('误检率: %.3f'%metric['wrong_detected_rate'])

# print('mAP: %.3f'%metric['mAP'])