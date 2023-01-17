import cv2

import os

import ast

import torchvision

import torch  

import numbers

import random

import math



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 





from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import utils

from torchvision.transforms import Compose, functional

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
df = pd.read_csv(DIR_INPUT + '/train.csv')

df_test = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')



df.sample(20)



for i,row in df.iterrows():

    box = ast.literal_eval(row['bbox'])

    xmin = box[0]

    ymin = box[1]

    w = box[2]

    h = box[3]

    df.at[i, 'x0'] = xmin

    df.at[i, 'y0'] = ymin

    df.at[i, 'x1'] = xmin + w

    df.at[i, 'y1'] = ymin + h

    

df = df.drop(columns=['width', 'height', 'source', 'bbox'])

df.head(10)
# Inspecting images

fig = plt.figure(figsize=(32,32))



columns=3

rows=1



for i in range(1, columns*rows + 1):  

    row = np.random.randint(0, len(df))

    img_id = df.iloc[row][['image_id']].values[0]

    img = cv2.imread(DIR_TRAIN + '/' + img_id + '.jpg')

    bboxes = df.loc[df['image_id'] == img_id, ['x0', 'y0', 'x1', 'y1']].values.tolist()

    for bbox in bboxes:  

        bbox = list(map(int, bbox))

        cv2.rectangle(img,

                      (bbox[0], bbox[1]), (bbox[2],bbox[3]),

                      color=(0, 255, 0), thickness=3)

        

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    

plt.show() 
class WheatDataset(Dataset):

    """Dataclass for wheat dataset"""

    

    def __init__(self, dataframe, img_dir, transform=None):

        self.dataframe = dataframe

        self.img_dir = img_dir

        self.images = dataframe['image_id'].unique()

        self.transform = transform

   

    def __getitem__(self, idx):

        target = {}

    

        img_id = self.images[idx]

        #target['image_id'] = img_id



        img_arr = cv2.imread(f'{self.img_dir}/{img_id}.jpg', cv2.IMREAD_COLOR)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        img_arr = img_arr / 255

        

        boxes_array = np.array(self.dataframe.loc[self.dataframe['image_id'] == img_id, ['x0', 'y0', 'x1', 'y1']])

        boxes = torch.tensor(boxes_array, dtype=torch.float32)

        target['boxes'] = boxes

        

        area = []

        for box in boxes:

            width = box[2] - box[0]

            height = box[3] - box[1]

            area.append(width * height) 

        target['area'] = torch.tensor(area) 

      

        labels = torch.ones((len(self.images)), dtype=torch.int64)

        target['labels'] = labels

               

        iscrowd = torch.zeros((len(self.images)), dtype=torch.uint8)

        target['iscrowd'] = iscrowd

        

        if self.transform is not None:

            img_arr = self.transform(img_arr)

            

        return img_arr, target, img_id

            

    def __len__(self):

        return len(self.images)

    
class ToTensor(object):

    """Convert ndarrays in sample to Tensors"""

    

    def __call__(self, img):  

        img_arr = np.array(img).transpose((2,0,1))

        img_arr = torch.tensor(img_arr, dtype=torch.float32)

          

        return img_arr

    

class ToPILImage(object):

    """Convert a tensor or an ndarray to PIL image.

       If "mode" is None there are some assumptions made about the input data:       

         - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.

         - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.

         - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.

         - If the input has 1 channel, the ``mode`` is determined by the data type"""

    

    def __init__(self, mode=None):

        self.mode = mode

        

    def __call__(self, img):

        img_arr = functional.to_pil_image(img, self.mode)

        return img_arr

    

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):

        self.p = p

    

    def __call__(self, img):

        

        if random.random() < self.p:

            img_arr = functional.hflip(img)

            return img_arr

        else:

            return img

       

class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):

        if isinstance(degrees, numbers.Number):

            if degrees < 0:

                raise ValueError("If degrees is a single number, it must be positive.")

            self.degrees = (-degrees, degrees)

        else:

            if len(degrees) != 2:

                raise ValueError("If degrees is a sequence, it must be of len 2.")

            self.degrees = degrees

            

        self.resample = resample

        self.expand = expand

        self.center = center

        self.fill = fill

        

    @staticmethod

    def get_params(degrees):

        angle = random.uniform(degrees[0], degrees[1])

        return angle

    

    def __call__(self, img):

        angle = self.get_params(self.degrees)

        img_arr = functional.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

        return img_arr

        

class Normalize(object):

    """Normalize image to 0 mean and unit variance"""

    

    def __init__(self, mean, std, inplace=False):

        self.mean = mean

        self.std = std

        self.inplace = False

 

    def __call__(self, img):

        normalized_img_arr = functional.normalize(img, self.mean, self.std, self.inplace)     

        return normalized_img_arr



class ColorJitter(object):

    """Randomly change the brightness, contrast, and saturation of an image"""

    

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        self.brightness = self._check_input(brightness, 'brightness')

        self.contrast = self._check_input(contrast, 'contrast')

        self.saturation = self._check_input(saturation, 'saturation')

        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),

                                     clip_first_on_zero=False)



    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):

        if isinstance(value, numbers.Number):

            if value < 0:

                raise ValueError("If {} is a single number, it must be non negative.".format(name))

            value = [center - value, center + value]

            if clip_first_on_zero:

                value[0] = max(value[0], 0)

        elif isinstance(value, (tuple, list)) and len(value) == 2:

            if not bound[0] <= value[0] <= value[1] <= bound[1]:

                raise ValueError("{} values should be between {}".format(name, bound))

        else:

            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))



        # if value is 0 or (1., 1.) for brightness/contrast/saturation

        # or (0., 0.) for hue, do nothing

        if value[0] == value[1] == center:

            value = None

        return value



    @staticmethod

    def get_params(brightness, contrast, saturation, hue):

        """Get a randomized transform to be applied on image"""



        transforms = []



        if brightness is not None:

            brightness_factor = random.uniform(brightness[0], brightness[1])

            transforms.append(Lambda(lambda img: functional.adjust_brightness(img, brightness_factor)))



        if contrast is not None:

            contrast_factor = random.uniform(contrast[0], contrast[1])

            transforms.append(Lambda(lambda img: functional.adjust_contrast(img, contrast_factor)))



        if saturation is not None:

            saturation_factor = random.uniform(saturation[0], saturation[1])

            transforms.append(Lambda(lambda img: functional.adjust_saturation(img, saturation_factor)))



        if hue is not None:

            hue_factor = random.uniform(hue[0], hue[1])

            transforms.append(Lambda(lambda img: functional.adjust_hue(img, hue_factor)))



        random.shuffle(transforms)

        transform = Compose(transforms)



        return transform



    def __call__(self, img):

    

        transform = self.get_params(self.brightness, self.contrast,

                                    self.saturation, self.hue)

        

        transformed_img_arr = transform(img)

        

        return transformed_img_arr

    

class Lambda(object):

    """Apply a user-defined lambda as a transform.



    Args:

        lambd (function): Lambda/function to be used for transform.

    """



    def __init__(self, lambd):

        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"

        self.lambd = lambd



    def __call__(self, img):

        return self.lambd(img)



    def __repr__(self):

        return self.__class__.__name__ + '()'

    

class Averager:

    def __init__(self):

        self.current_value = 0.0

        self.iter = 0.0

        

    def send(self, value):

        self.current_value += value

        self.iter += 1

        

    def value(self):

        if self.iter == 0:

            return 0

        else:

            return 1.0 * self.current_value / self.iter

    

    def reset(self):

        self.current_value = 0.0

        self.iter = 0.0
def initiate_model(num_classes):    

    """Initiates a faster-rcnn pretrained on COCO and changes the head,

       Returns a model that can we used for the wheat-box-dection task"""

    # load a model pre-trained pre-trained on COCO

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier & replace pre-trained head with a new one

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    

    return model



def get_transform(train):

    """Applies all the transformations to dataset class"""

    transform = []

    #transform.append(ToPILImage())

    if train:

        transform.append(RandomHorizontalFlip(0.1))

        transform.append(RandomRotation((-5,5)))

        transform.append(ColorJitter(brightness=(0,0.8), hue=0.5, saturation=0.8))



    transform.append(ToTensor())

    #transform.append(Normalize((0.0,), (1,)))  

    

    return Compose(transform)



def collate_fn(batch):

    # youtube.com/watch?v=eKp5YH9ltnE

    return tuple(zip(*batch))
def train():

    

    NUM_CLASSES = 2

    NUM_EPOCS = 5

    BATCH_SIZE = 1

    

    # Train on GPU or CPU

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Running on: {device}')

    

    # Initiate training and test datasets

    train_data = WheatDataset(df, DIR_TRAIN,transform=get_transform(train=False))

    val_data = WheatDataset(df, DIR_TRAIN, transform=get_transform(train=False))

    

    # Split dataset in train & test (use 250 images for test)

    indices = torch.randperm(len(train_data)).tolist()

    train_data = torch.utils.data.Subset(train_data, indices[:-250])

    val_data = torch.utils.data.Subset(val_data, indices[-250:])

    

    print(f'Number of training examples: {len(train_data)}, Number of validation examples: {len(val_data)}')

    

    # Initiate training and test data loaders

    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, 

                                                   shuffle=True, num_workers=2, collate_fn=collate_fn)

                                               

    

    data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE,

                                                   shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Initiate model

    model = initiate_model(NUM_CLASSES)

    model.to(device)

    

    # Get the learnable parameters of our model

    params = [p for p in model.parameters() if p.requires_grad]

    # Initaite optimizer on learnable parameters

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Set up a dynamic learning rate in order to converage faser

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                               step_size=3,

                                               gamma=0.1)

    

    # Initiate averager class to keep track of losses

    loss_hist = Averager()

    val_loss_hist = Averager()

    # Set to high value on purpose

    least_loss = 10e6

    # Allow model to learn parameters

    model.train()

    

    print(f'Starting training....')

    

    for epoch in range(NUM_EPOCS):

        

        loss_hist.reset()

        val_loss_hist.reset()

        itr = 1

        

        for images,targets,img_id in data_loader_train:    

            

            images = list(image.to(device) for image in images)  

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

           

            #final = []

            #for target in targets:

                #result = {}

                #for k,v in target.items():

                    #result[k] = v.to(device)

                #final.append(result)  



            loss_dict = model(images, targets)   

            

            losses = sum(loss for loss in loss_dict.values())        

            loss_value = losses.item()

            

            loss_hist.send(loss_value)

            

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            

            if itr % 50 == 0:

                print(f'Iteration #{itr}, loss: {loss_hist.value()}')

                

            itr += 1

            

        # Validation            

        for images, targets, img_id in data_loader_val:

            images = list(image.to(device) for image in images)  

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        



            val_loss_dict = model(images, targets) 

            val_losses = sum(loss for loss in val_loss_dict.values())

            val_loss_value = val_losses.item()

            val_loss_hist.send(val_loss_value)

            

        if val_loss_value < least_loss:

            least_loss = val_loss_hist.value()

            lval = round(least_loss, 4)

            torch.save(model.state_dict(), f'fastercnn-wheatdetection-epoch:{epoch}-loss:{lval}.pth')

            torch.save(model.state_dict(), 'best_weights.pth')



        else:

            if lr_scheduler is not None:

                lr_scheduler.step()



        print(f"Epoch #{epoch}, training_loss: {loss_hist.value()}, validation_loss: {val_loss_hist.value()}")

            

    print('Done!')

    

train()
class WheatTestDataset(Dataset):

    """Dataclass for wheat dataset"""

    

    def __init__(self, dataframe, img_dir, transform=None):

        self.dataframe = dataframe

        self.img_dir = img_dir

        self.images = dataframe['image_id'].unique()

        self.transform = transform

   

    def __getitem__(self, idx):

        img_id = self.images[idx]



        img_arr = cv2.imread(f'{self.img_dir}/{img_id}.jpg', cv2.IMREAD_COLOR)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        img_arr = img_arr / 255

        

        if self.transform is not None:

            img_arr = self.transform(img_arr)

            

        return img_arr, img_id

            

    def __len__(self):

        return len(self.images)

    

    

def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
def evaluate():

    

    WEIGHTS_FILE = '/kaggle/working/best_weights.pth'    

    DETECTION_THRESHOLD = 0.5

    BATCH_SIZE = 1

    NUM_CLASSES = 2

    

    results = []

    

    # Initiate test dataset 

    test_dataset = WheatTestDataset(df_test, DIR_TEST, get_transform(train=False))

    

    # Load model

    model = initiate_model(NUM_CLASSES)

    model.load_state_dict(torch.load(WEIGHTS_FILE))

    model.eval()

    

    # Set device (cpu or GPU)

    device = torch.device('cpu') 

    print(f'Evaluating on {device}')

    

    # Dataset itr object

    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,

                                                   shuffle=False, num_workers=2, collate_fn=collate_fn)

    

    itr = 1

    print(f'Starting evaluation...')

    

    for images, img_id  in data_loader_test:

        images = list(image.to(device) for image in images)  

        output = model(images) 

        

        if itr % 5 == 0:

            print(f'Evaluated {itr} images')

            

        itr += 1

        

        for i, image in enumerate(images):



            boxes = output[i]['boxes'].data.cpu().numpy()

            scores = output[i]['scores'].data.cpu().numpy()



            boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)

            scores = scores[scores >= DETECTION_THRESHOLD]

            image_id = img_id[i]



            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



            result = {

                'image_id': image_id,

                'PredictionString': format_prediction_string(boxes, scores)

            }



            results.append(result)



    print(f'Writing submision file...')

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

    test_df.to_csv('submission.csv', index=False)

    sub.to

    

    print(f'Done')

         

evaluate()