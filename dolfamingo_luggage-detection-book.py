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

from torch.utils.data import DataLoader, random_split

import pickle

import time

import datetime
root_path = '../input/luggage-detection-v3/luggage_detection'

train_path = os.path.join(root_path, "train")

test_path = os.path.join(root_path, "test")
class LuggageDataset(torch.utils.data.Dataset):

  def __init__(self, root, transforms=None):

    self.dataset = []

    self.transforms=transforms

    dir_list = list(sorted(os.listdir(root)))

    for dir_name in dir_list:

      if(dir_name != '31178-14'):

        continue

      dir_path = os.path.join(root, dir_name)

      # print(dir_path)

      for file_name in list(sorted(os.listdir(dir_path))):

        if(file_name.endswith('.json')):

          file_path = os.path.join(dir_path,file_name)

          with open(file_path, 'rb') as f:

            one_data = pickle.load(f)

#             one_data['target'] = {

#                 'boxes':torch.as_tensor(one_data['target']['boxes'], dtype=torch.float32),

#                 'labels':torch.as_tensor(one_data['target']['labels']=='行李箱',dtype=torch.int64)

#             }

          self.dataset.append(one_data)



  def __getitem__(self, index):

    item = self.dataset[index]

    ## 这个要放在这里转化，因为所在空间是翻了20倍，内存吃不消

    image = cv2.imdecode(item['image'], cv2.IMREAD_COLOR) 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     image = F.to_tensor(image)

    return image,item['target']



  def __len__(self):

    return len(self.dataset)
dataset = LuggageDataset(test_path)

images = [x[0] for x in dataset]

targets = [x[1] for x in dataset]
len(images)
class DetectorTool:

    

    def __init__(self,

             nms_iou_threshold=0.5

        ):

        self.nms_iou_threshold = nms_iou_threshold

    

    

    # NMS算法

    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor

    def nms(self,boxes, scores):



        if(len(scores)==0):

            return []



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



        return keep
class LuggageDetector:



    def __init__(self,

                 model_path,

                 score_threshold=0.5,

                 nms_iou_threshold=0.5

            ):

        self.model = torch.load(model_path)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)

        self.score_threshold = score_threshold

        self.nms_tool = NMSTool(nms_iou_threshold)



    def visual(self, image,

               prediction,

               vis = False

               ):

        image = self.__check_image_format(image)  # 检查数据格式

        img = image.copy()

        for i in range(len(prediction['boxes'])):

            box = prediction['boxes'][i]

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)

        if vis:

            plt.figure(figsize=(8,8), dpi=120)

            plt.subplot(111)

            plt.imshow(img)

            plt.show()

        return img





    def __check_image_format(self, image):

        if type(image) is not np.ndarray:

            image = np.array(image)



        assert len(image.shape) == 3, '输入图像的格式为[Width，Height，Channels]格式'

        assert image.shape[2] == 3, '输入图像的格式为[Width，Height，Channels]格式'

        if image.max() <= 1:

            image = (image * 255).astype(np.uint8)

        return image





    def detect_image(self, image):

        self.model.eval()

        with torch.no_grad():

            image = self.__check_image_format(image)  # 确保是[W,H,C]的RGB格式

            img = F.to_tensor(image)  ## 转成tensor格式

            prediction = self.model([img.to(self.device)])[0]

            boxes = []

            scores = []

            labels = []

            for i in range(len(prediction['boxes'])):

                score = prediction['scores'][i].cpu().numpy()

                box = prediction['boxes'][i].cpu().numpy()

                label = prediction['labels'][i].cpu().numpy()

                if (score < self.score_threshold):

                    continue

                boxes.append(box)

                scores.append(score)

                labels.append(label)

            

            scores = np.array(scores)

            boxes = np.array(boxes)

            labels = np.array(labels)

#             print(len(scores))

            keep = self.nms_tool.nms(torch.as_tensor(boxes),torch.as_tensor(scores))

            prediction = {'boxes': boxes[keep], 

                          'scores': scores[keep], 

                          'labels': labels[keep]}

        return prediction



    

    def detect_images(self, images):

        predictions = []

        for image in images:

            prediction = self.detect_image(image)

            predictions.append(prediction)

        return predictions
detector = LuggageDetector('../input/frcnn-model/frcnn-v/frcnn-v7.pkl',

                    score_threshold=0.5,  ## 阈值低于多少的会被过滤掉

                   nms_iou_threshold=0.3) 

print()
# index = 112

# prediction = detector.detect_image(images[index])

# x = detector.visual(images[index],prediction,vis=True)
class Luggage:

    def __init__(self, login_time, box):

        self.id = self.tid_maker()

        self.login_time = login_time

        self.last_online_time = login_time

        self.box = box

        self.is_overtime = False

    

    def tid_maker(self):

        return '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())+''.join([str(random.randint(1,10)) for i in range(5)])
### 具体的detected_box和luggage的正负分配问题，要参考下rcnn系列。 

### 难不成真要搞个二分图匹配最优的？以IoU为边的权值

class Unattended_Detector:

    def __init__(self,

          luggage_detector_path = '../input/frcnn-model/frcnn-v/frcnn-v7.pkl',

          online_time_threshold=2,

          offline_time_tolerance=1,

          IoU_threshold=0.2):

        

        self.luggage_detector = detector = LuggageDetector(luggage_detector_path,

                    score_threshold=0.5,  ## 阈值低于多少的会被过滤掉

                   nms_iou_threshold=0.3) 

        

        self.online_time_threshold = online_time_threshold

        self.offline_time_tolerance = offline_time_tolerance

        self.IoU_threshold = IoU_threshold

        self.online_list = {}

        

        



  # 这里的time是以一帧为单位时间，到实际应用时可能就是改成实际时间

    def detect_unattended(self,current_time,image):

        

        prediction = self.luggage_detector.detect_image(image)

        detected_boxes = prediction['boxes']

        boxes_vis = np.array([False] * len(detected_boxes))

        overtime_boxes = []

        pop_ids = [] ## 因为不能在迭代的过程中改变dict，所以先把id记下来，遍历完再去除

         #尝试为每个在online_list下的luggage分配到新的检测box

        for id in self.online_list.keys():

            luggage = self.online_list[id]

            attach_index = None

            max_IoU = 0

            for index,detected_box in enumerate(detected_boxes):  ##枚举所有检测box，找到IOU最大的，分配给它

                IoU = self.calculate_iou(luggage.box,detected_box)

                if(IoU>self.IoU_threshold and IoU>max_IoU):

                    attach_index = index

                    max_IoU = IoU

                    

            if attach_index!=None:  #分配到box，更新在线信息

                boxes_vis[attach_index] = True

                luggage.last_online_time = current_time

                luggage.box = detected_box

                if (current_time-luggage.login_time)>self.online_time_threshold: ## 如果在线超时，则标记起来

                    luggage.is_overtime = True

                    overtime_boxes.append(luggage.box)

                self.online_list[luggage.id] = luggage ## 因为是传址，所以修改过的都已经被更新了，但还是写一下比较稳妥

            else: ## 如果没有找到匹配

                if (current_time-luggage.last_online_time)>self.offline_time_tolerance: #且离线时长超出阈值，则剔除

                    pop_ids.append(luggage.id)

    

        for id in pop_ids:

            self.online_list.pop(id)



        ## 将没有被分配给online_box的box加入到online_list

        for i,detected_box in enumerate(detected_boxes):

            if boxes_vis[i] == False:

                new_luggage = Luggage(current_time,detected_box)

                self.online_list[new_luggage.id] = new_luggage

        

        return overtime_boxes,detected_boxes







    def calculate_iou(self,bbox1,bbox2):

        intersect_bbox = [0., 0., 0., 0.]

        if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:

            pass

        else:

            intersect_bbox[0] = max(bbox1[0],bbox2[0])

            intersect_bbox[1] = max(bbox1[1],bbox2[1])

            intersect_bbox[2] = min(bbox1[2],bbox2[2])

            intersect_bbox[3] = min(bbox1[3],bbox2[3])

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积

        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积

        area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积

        if area_intersect>0:

            return area_intersect / (area1 + area2 - area_intersect)  # 计算iou

        else:

            return 0
detector = Unattended_Detector()
for current_time,image in enumerate(images):

    boxes,detected_boxes = detector.detect_unattended(current_time,image)

    print(len(boxes))

#     visual(image,boxes,detected_boxes)
def visual( image,

           boxes,detected_boxes

           ):

    img = image.copy()

    for box in boxes:

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 220), 10)

    for box in detected_boxes:

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)

    plt.figure(figsize=(10,10), dpi=150)

    plt.subplot(111)

    plt.imshow(img)

    plt.show()
# def visual( image,

#            online_list

#            ):

#     img = image.copy()

#     for luggage in online_list.values():

#         box = luggage.box

#         if(luggage.is_overtime):

#             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 220), 8)

#         else: 

#             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)

#     plt.figure(figsize=(8,8), dpi=120)

#     plt.subplot(111)

#     plt.imshow(img)

#     plt.show()