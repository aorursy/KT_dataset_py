import pandas as pd



import numpy as np



from PIL import Image, ImageDraw

from glob import glob

import matplotlib.pyplot as plt

import os



import torch

import torchvision

#from torchvision import transforms

from torch.utils.data import DataLoader, Dataset



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from tqdm.notebook import tqdm



import time





import albumentations as A



root_dir = '/kaggle/input/global-wheat-detection/'

drive_dir = '/kaggle/working/'

train_dir = root_dir+'train/'

test_dir = root_dir+'test/'
train_frame = pd.read_csv(root_dir+'train.csv')



unique_image_count = len(train_frame['image_id'].unique())
train_glob = glob(train_dir + '*')

test_glob = glob(test_dir + '*')
print ("Images with BBox {}" .format((unique_image_count)))

print ("Images without BBox {}" .format(len(train_glob) - unique_image_count))
train_frame[['xmin','ymin','w','h']] = pd.DataFrame(train_frame.bbox.str.strip('[]').str.split(',').tolist()).astype(float)



train_frame['xmax'], train_frame['ymax'], train_frame['area'] = train_frame['xmin'] + train_frame['w'], train_frame['ymin'] + train_frame['h'], train_frame['w'] * train_frame['h']
train_frame["class"] = 1

train_frame.describe()
def show_image(image_id ):



    fig, axs = plt.subplots(1,2, figsize = (24,24))

    axs = axs.flatten()

    

    bbox = train_frame[train_frame['image_id'] == image_id]

    

    img_path = os.path.join(train_dir, image_id +'.jpg')

    

    image = Image.open(img_path)

    image2 = torch.from_numpy(np.array(image).astype('float32')) / 255.

    

    print("Image shape{}".format(image2.shape))

    

    

    axs[0].set_title('Original Image')

    axs[0].imshow(image2)

    

    for idx, row in bbox.iterrows():

        x1 = row['xmin']

        y1 = row['ymin']

        x2 = row['xmax']

        y2 = row['ymax']

        label = 'Wheat' if row['class'] == 1 else 'background'

        

        image_wth_bb = ImageDraw.Draw(image)

        image_wth_bb.rectangle([(x1,y1),(x2,y2)],width = 5, outline = 'red')

        image_wth_bb.text([(x1,y1-10)], label)

        

    axs[1].set_title('Image with BoundingBox')

    image_wth_bb = torch.from_numpy(np.array(image).astype('float32')) / 255.

    axs[1].imshow(image_wth_bb)

    

    plt.show()
show_image(train_frame.image_id.unique()[1])
class WheatDetectionDataset(Dataset):

    """Global Wheat Detection Dataset"""

    

    def __init__(self,pd_frame, img_dir, transforms = None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        

        self.globalwheat_frame = pd_frame

        self.image_ids = list(self.globalwheat_frame['image_id'].unique())

        self.img_dir = img_dir

        self.transforms = transforms

        

    

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, idx):

        

        image_id = self.image_ids[idx]

        image_data = self.globalwheat_frame.loc[self.globalwheat_frame['image_id'] == image_id]

        

        b_boxes = torch.as_tensor(np.array(image_data[['xmin', 'ymin', 'xmax', 'ymax']]), dtype = torch.float32)

        area = torch.tensor(np.array(image_data['area']), dtype=torch.int64)

        labels = torch.ones((image_data.shape[0],), dtype=torch.int64)

        crowd = torch.zeros((image_data['class'].shape[0],), dtype=torch.uint8)

        

        target = {}

        

        target['boxes'] = b_boxes

        target['area'] = area

        target['labels'] = labels

        target['crowd'] = crowd

        

        img_path = os.path.join(self.img_dir, image_id + '.jpg')

        

        image = Image.open(img_path)

        image = np.array(image).astype('float32') / 255.

        



        

        if self.transforms:

            #image, target = self.transforms(image, target)

            image_transforms = {

                                'image': image,

                                'bboxes': target['boxes'],

                                'labels': labels

                                }

            image_transforms = self.transforms(**image_transforms)

            image = image_transforms['image']

            

            target['boxes'] = torch.as_tensor(image_transforms['bboxes'], dtype=torch.float32)



       

        #else:

        image = torch.from_numpy(image.transpose(2,0,1))

       

    

        return image, target
def get_train_transform():

    return A.Compose([

       # A.Resize(p=1, height=512, width=512),

        #A.RandomCrop( height=512, width=512,p=.5),

        A.ToGray(p=0.5),

        A.Flip(p=.5),

        A.RandomBrightnessContrast(p=.5),

        A.RandomGamma(p=0.5),

        A.MotionBlur(p=.5),

        A.HueSaturationValue(p=0.5),

        A.GaussNoise(p=.5),

        #A.ShiftScaleRotate(p=0.5),

        A.RandomSunFlare(p=.5),

        A.RandomBrightnessContrast(p=0.3),

        A.GaussNoise(p=.5),

        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3)

        

        #ToTensor()

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
def get_test_transform():

    return A.Compose([

        # A.Resize(512, 512),

        ToTensorV2(p=1.0)

    ])
def get_wheat_dataset_frame(csv):

    globalwheat_frame = pd.read_csv(csv)

    

    globalwheat_frame[['xmin','ymin','w','h']] = pd.DataFrame(globalwheat_frame.bbox.str.strip('[]').str.split(',').tolist()).astype(float)

    globalwheat_frame['xmax'], globalwheat_frame['ymax'], globalwheat_frame['area'] = globalwheat_frame['xmin'] + globalwheat_frame['w'], globalwheat_frame['ymin'] + globalwheat_frame['h'], globalwheat_frame['w'] * globalwheat_frame['h']

    globalwheat_frame["class"] = 1

    

    return globalwheat_frame
#414



def get_model_weight():

    return '../input/gwd-starter-weigths/resnet50_pretrainedGWD-8-colab-1-1e-3-epoch-600 last.pth'

    
def collate_fn(batch):

    return tuple(zip(*batch))


def model_fasterrcnn_resnet50(pretrained = False, pretrained_backbone = True):

    

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,  pretrained_backbone=pretrained_backbone)

    num_classes = 2  

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    

    return model
wheat_frame = get_wheat_dataset_frame(root_dir+'train.csv')

wheatds = WheatDetectionDataset(wheat_frame,train_dir, get_train_transform())
train_dl = DataLoader(wheatds, batch_size = 16, num_workers=8, shuffle = True, collate_fn=collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()

print(device)
def save_checkpoint(epoch,model,optimizer,scheduler, loss, PATH):

    torch.save({

            'epoch': epoch,

            'model': model.state_dict(),

            'optimizer': optimizer.state_dict(),

            'scheduler': scheduler.state_dict(),

            'loss': loss,

            

            }, PATH)
def train(data_loader, epoch, resume_training = False):

        

    model = model_fasterrcnn_resnet50(pretrained = not resume_training)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=7e-04, weight_decay=1e-04)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=7e-04, steps_per_epoch=len(data_loader), epochs=epoch)

    model.parameters

    total_train_loss = []

    initial = 0

    

    if resume_training:

        checkpoint = torch.load(get_model_weight())

        model.load_state_dict(checkpoint['model'])

        total_train_loss = checkpoint['loss']

        optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler.load_state_dict(checkpoint['scheduler'])

        initial = checkpoint['epoch'] + 1

        return total_train_loss, initial

     

        

    





    

    itr = 1

    avg_loss = 0

    for epoch in tqdm(range(initial,epoch)):

        

        print(f'Epoch :{epoch + 1}')

        start_time = time.time()

        train_loss = []

        model.train()

        

        for images, targets in tqdm(data_loader):

            

            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



            loss_dict = model(images, targets)



            losses = sum(loss for loss in loss_dict.values())

            

            loss_value = losses.item()

            

            train_loss.append(loss_value)

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            scheduler.step()

            

            

            if itr % 50 == 0:

                print(f"Iteration #{itr} loss: {loss_value:.4f}")



            itr += 1

    

        



        epoch_train_loss = np.mean(train_loss)

        total_train_loss.append(epoch_train_loss)

        print(f'Epoch train loss is {epoch_train_loss:.4f}')

        time_elapsed = time.time() - start_time

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        

        if epoch % 5 == 0:

            save_checkpoint(epoch, model, optimizer, scheduler, total_train_loss, drive_dir+'resnet50_pretrainedGWD-7smallDS-colab-1-600+.pth')

          

       

        plt.figure(figsize=(12,6))

        plt.title('Train Loss', fontsize= 20)

        plt.plot(total_train_loss)

        plt.xlabel('iterations')

        plt.ylabel('loss') 

        plt.show()



    save_checkpoint(epoch, model, optimizer, scheduler, total_train_loss, drive_dir+'resnet50_pretrainedGWD-7smallDS-colab-1-600'+str(epoch)+'.pth')

#loss, epoch = train(train_dl, 600, True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()



print(device)
class WheatDetectionDataset_Test(Dataset):

    """Global Wheat Detection Dataset"""

    

    def __init__(self,pd_frame, img_dir, transforms = None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        

        self.globalwheat_frame = pd_frame

        self.image_ids = list(self.globalwheat_frame['image_id'].unique())

        self.img_dir = img_dir

        self.transforms = transforms

        

    

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, idx):

        

        image_id = self.image_ids[idx]

        image_data = self.globalwheat_frame.loc[self.globalwheat_frame['image_id'] == image_id]

        

              

        img_path = os.path.join(self.img_dir, image_id + '.jpg')

        

        image = Image.open(img_path)

        image = np.array(image).astype('float32') / 255.

        image = torch.from_numpy(image.transpose(2,0,1))

        

        if self.transforms:

            

            image_transforms = {

                                'image': image

                               }

            

            image_transforms = self.transforms(**image_transforms)

            image = image_transforms['image']

            

        

        

        return image, image_id

            

            

    

    
test_frame = pd.read_csv(root_dir+'/sample_submission.csv')
wheat_detection_test = WheatDetectionDataset_Test(test_frame,test_dir, None )



test_dl = DataLoader(

    wheat_detection_test,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], 

                                                             j[1][2], j[1][3]))



    return " ".join(pred_strings)

detection_threshold = 0.7

results = []



def predict_result(dataloader, sub_csv = False):

    

  

    

    model = model_fasterrcnn_resnet50(pretrained=False, pretrained_backbone=False)

    model.load_state_dict(torch.load(get_model_weight())['model'])

    model.eval()

    model.to(device)

    

    

    for images, image_ids in dataloader:

        

        images = list(image.to(device)for image in images)

        outputs = model(images)

        

        for i, image in enumerate(images):

            

            boxes = outputs[i]['boxes'].data.cpu().numpy()

            scores = outputs[i]['scores'].data.cpu().numpy()

            

            boxes = boxes[scores >= detection_threshold].astype(np.int32)

            scores = scores[scores >= detection_threshold]

            

            image_id = image_ids[i]

                

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            

            if sub_csv:

                result = {

                    'image_id' : image_id,

                    'PredictionString' : format_prediction_string(boxes, scores)

                    

                }

                

                results.append(result)

            

            

        img_path = os.path.join(test_dir, image_id + '.jpg')

        

        image = Image.open(img_path)

        

        

       

        

        for b,s in zip(boxes, scores):

            

            image_wth_bb = ImageDraw.Draw(image)

            image_wth_bb.rectangle([(b[0],b[1]),(b[0]+b[2],b[1]+b[3])],width = 2, outline = 'red')

            image_wth_bb.text([(b[0],b[1])], '{:.2}'.format(s))

        

        image_wth_bb = torch.from_numpy(np.array(image).astype('float32')) / 255.

        plt.figure(figsize=(12,12))

        plt.imshow(image_wth_bb)
predict_result(test_dl, sub_csv=True)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)
results