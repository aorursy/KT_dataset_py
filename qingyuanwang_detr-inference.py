!pip install ../input/weightedboxesfusion/Weighted-Boxes-Fusion-master/ > /dev/null

!pip install --no-deps '../input/timm-0130/timm-0.1.30-py3-none-any.whl' > /dev/null
import torch

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os

import cv2

import numpy as np

import pandas as pd

from ensemble_boxes import *

import sys

sys.path.append('../input/detrmodel/detr')

from models.resnestbackbone import ResnestBackBone

from models.matcher import HungarianMatcher

from models.detr import SetCriterion



from torch.utils.data import Dataset,DataLoader, ConcatDataset



from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



from tqdm import tqdm
!mkdir -p /root/.cache/torch/hub/

!cp -r ../input/detrmodel/detr /root/.cache/torch/hub/facebookresearch_detr_master
def get_train_transforms():

    return A.Compose(

        [  

          A.OneOf([

                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,

                                     val_shift_limit=0.2, p=0.3), 

                A.RandomBrightnessContrast(brightness_limit=0.2,  

                                           contrast_limit=0.2, p=0.3),

                A.RGBShift(r_shift_limit=20/255, g_shift_limit=20/255, b_shift_limit=20/255,p=0.3),

            ], p=0.2),

            A.OneOf([

                A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                A.Blur(p=0.05),

                A.GaussNoise(var_limit=(0.05, 0.1), mean=0, p=0.3),

                A.ToGray(p=0.05)], p=0.2),



            A.OneOf([

                A.HorizontalFlip(p=0.75), 

                A.VerticalFlip(p=0.75),  

                A.Transpose(p=0.75),                

                A.RandomRotate90(p=0.75)

                ], p=1),         

            #  A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.1), 

             A.Resize(height=1024, width=1024, p=1),

            #  A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.3),

             ToTensorV2(p=5.0),

             ],

             

        p=1.0, bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])

    )

def collate_fn(batch):

    return tuple(zip(*batch))
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
class WheatDataset(Dataset):

    def __init__(self,image_ids,dataframe,transforms=None, dir_train ='/content/datasets/ori/train'):

        self.image_ids = image_ids

        self.df = dataframe

        self.transforms = transforms

        self.dir = dir_train

        

        

    def __len__(self) -> int:

        return self.image_ids.shape[0]

    

    def __getitem__(self,index):

        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        # DETR takes in data in coco format 

        boxes = records[['x', 'y', 'w', 'h']].values

        

        #Area of bb

        area = boxes[:,2]*boxes[:,3]

        area = torch.as_tensor(area, dtype=torch.float32)

        

        # AS pointed out by PRVI It works better if the main class is labelled as zero

        labels =  np.zeros(len(boxes), dtype=np.int32)



        

        if self.transforms:

            sample = {

                'image': image,

                'bboxes': boxes,

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']

            boxes = sample['bboxes']

            labels = sample['labels']

            

            

        #Normalizing BBOXES

            

        _,h,w = image.shape

        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)

        target = {}

        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)

        target['labels'] = torch.as_tensor(labels,dtype=torch.long)

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        

        return image, target, image_id
class DETRModel(nn.Module):

    def __init__(self,num_classes,num_queries):

        super(DETRModel,self).__init__()

        self.num_classes = num_classes

        self.num_queries = num_queries

        

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=False)

        self.in_features = self.model.class_embed.in_features

        

        hidden_dim = self.model.transformer.d_model



        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)

        self.model.num_queries = self.num_queries

        self.model.backbone = ResnestBackBone()

        self.model.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1,groups=4)

        self.model.query_embed = nn.Embedding(num_queries, hidden_dim)





    def forward(self,images):

        if isinstance(images, (list, torch.Tensor)):

            images = nested_tensor_from_tensor_list(images)

        return self.model(images)
def tta_image(image, device):

    image_src = torch.tensor(image.copy(),dtype=torch.float,device = device).permute([2, 0, 1])/255

    image_hf = torch.tensor(cv2.flip(image.copy(), 1) ,dtype=torch.float,device = device).permute([2, 0, 1])/255    

    image_vf = torch.tensor(cv2.flip(image.copy(), 0) ,dtype=torch.float,device = device).permute([2, 0, 1])/255  

    image_hf_vf = torch.tensor(cv2.flip(image.copy(), -1) ,dtype=torch.float,device = device).permute([2, 0, 1])/255

    return [image_src,image_hf,image_vf,image_hf_vf]



def tta_reverse_boxes(boxes):

    boxes = (boxes).detach().cpu().numpy().astype(np.float32)

    boxes[1,:,0] = 1 - boxes[1,:,0] - boxes[1,:,2]

    boxes[2,:,1] = 1 - boxes[2,:,1] - boxes[2,:,3]

    boxes[3,:,0] = 1 - boxes[3,:,0] - boxes[3,:,2]

    boxes[3,:,1] = 1 - boxes[3,:,1] - boxes[3,:,3]

    return boxes



def boxes_to_x1x2y1y2(boxes):

    boxes[:,:,2] = boxes[:,:,0] + boxes[:,:,2]

    boxes[:,:,3] = boxes[:,:,1] + boxes[:,:,3]

    return np.clip(boxes,0,1)



def boxes_to_xywh(boxes):

    boxes[:,2] =  boxes[:,2] - boxes[:,0]

    boxes[:,3] =  boxes[:,3] - boxes[:,1]

    return boxes
def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):

    model.train()

    model.model.backbone.freeze_bn()

    criterion.train()

    

    summary_loss = AverageMeter()

    

    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for step, (images, targets, image_ids) in enumerate(tk0):

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        

        output = model(images)

        for k in output.keys():

            output[k] = output[k].float()



        loss_dict = criterion(output, targets)

        weight_dict = criterion.weight_dict

        

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()



#         losses.backward()

        with amp.scale_loss(losses, optimizer) as scaled_loss:

            scaled_loss.backward()



        optimizer.step()

        if scheduler is not None:

            scheduler.step()

        

        summary_loss.update(losses.item(),BATCH_SIZE)

        tk0.set_postfix(loss=summary_loss.avg, loss_ce=loss_dict['loss_ce'].item(), loss_bbox=loss_dict['loss_bbox'].item(), loss_giou=loss_dict['loss_giou'].item(), loss_card=loss_dict['cardinality_error'].item(), acc=100-loss_dict['class_error'].item())

        

    return summary_loss
def train_pl(epochs, model):

    

    df_train = pd.read_csv('../input/global-wheat-detection/train.csv')

    bboxs = np.stack(df_train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

    for i, column in enumerate(['x', 'y', 'w', 'h']):

        df_train[column] = bboxs[:,i]

    df_train.drop(columns=['bbox'], inplace=True)

    

#     df_pl = pd.read_csv('detr_pl.csv')

#     bboxs = np.stack(df_pl['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

#     for i, column in enumerate(['x', 'y', 'w', 'h']):

#         df_pl[column] = bboxs[:,i]

#     df_pl.drop(columns=['bbox'], inplace=True)

    

    train_dataset_ori = WheatDataset(

    image_ids=df_train['image_id'].unique(),

    dataframe=df_train,

    transforms=get_train_transforms(),

    dir_train ='../input/global-wheat-detection/train'



    )

    

#     train_dataset_pl = WheatDataset(

#     image_ids=df_pl['image_id'].unique(),

#     dataframe=df_pl,

#     transforms=get_train_transforms(),

#     dir_train ='../input/global-wheat-detection/test'

#     )

    

    

#     train_dataset = ConcatDataset([train_dataset_ori, train_dataset_pl])

    train_dataset = train_dataset_ori





    train_data_loader = DataLoader(

    train_dataset,

    batch_size=1,

    shuffle=True,

    num_workers=4,

    collate_fn=collate_fn

    )



    

    weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

    matcher = HungarianMatcher(cost_class=weight_dict['loss_ce'], cost_bbox=weight_dict['loss_bbox'], cost_giou=weight_dict['loss_giou'], multiple_match=1)

    device = torch.device('cuda')

    model = model.to(device)

    criterion = SetCriterion(1, matcher, weight_dict, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality'])

    criterion = criterion.to(device)

    



    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    scheduler = None

    for epoch in range(epochs):

        train_loss = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=scheduler,epoch=epoch)

        torch.save(model.state_dict(), './detr_pl.pt')
def inference(model, device, source = '../input/global-wheat-detection/test/', view_visual=False, pl=False):

    image_file_names =  os.listdir(source)

    results = []

    for file_name in image_file_names:

        image_id = file_name.split('.')[0]

        image = cv2.imread('%s/%s.jpg'%(source,image_id), cv2.IMREAD_COLOR)

        image = cv2.resize(image,(1024,1024))

        image_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image = tta_image(image_raw, device)

        output = model(image)

        del image



        boxes = tta_reverse_boxes(output['pred_boxes'])

        prob   = output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:,:,0]

        boxes, prob, _ = weighted_boxes_fusion(boxes_to_x1x2y1y2(boxes), prob, np.zeros_like(prob),iou_thr=0.4, skip_box_thr=0.1)

        boxes = (boxes_to_xywh(boxes)*1024).astype(np.int32)

        



        if view_visual:

            for box,p in zip(boxes,prob):

                if p >0.5:

                    color = (255,0,0) #if p>0.5 else (0,0,0)

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(image_raw,str(p),(box[0], box[1]),font,0.5,color,1)

                    cv2.rectangle(image_raw,(box[0], box[1]),(box[2]+box[0], box[3]+box[1]),color, 1)

            cv2.imwrite(f'./{image_id}.jpg',cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR))

            

        if not pl:

            prediction_strings = []

            for p,b in zip(prob,boxes):

                if p > 0.5:

                    box = b.astype(np.int32)

                    prediction_strings.append("{0:.4f} {1} {2} {3} {4}".format(p, box[0],box[1],box[2],box[3]))



            r = {

                'image_id':image_id,

                'PredictionString':" ".join(prediction_strings)

                }

            results.append(r)

        else:

            for p,b in zip(prob,boxes):

                if p > 0.5:

                    box = b.astype(np.int32)

                    r = {

                        'image_id':image_id,

                        'width': '1024',

                        'height' : '1024',

                        'bbox' : f'[{box[0]},{box[1]},{box[2]},{box[3]}]',

                        'source': 'detr_pl'

                        }

                    results.append(r)

    return results
def get_model(path = '../input/detrmodel/detr_size1024_q512_val40.pt'):

    model = DETRModel(2, 512)

    model.load_state_dict(torch.load(path,map_location=device))

    model.eval()

    model = model.to(device)

    return model
if len(os.listdir('../input/global-wheat-detection/test/'))<11:

    committing=True

else:

    committing=False

    

model = get_model()

with torch.no_grad():

    results = inference(model,device,view_visual=committing)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head()