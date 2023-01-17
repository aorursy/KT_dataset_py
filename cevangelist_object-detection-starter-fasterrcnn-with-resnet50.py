import torch

from torch.utils.data import Dataset, DataLoader

import torchvision

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm



import pathlib

import os



IS_KAGGLE_ENV = True

DIR_INPUT = pathlib.Path('/kaggle/input/global-wheat-detection')

DIR_WEIGHTS = pathlib.Path('/kaggle/input/resnet50-weights-imagenet-pth')

FILENAME_WEIGHTS = 'resnet50-19c8e357.pth'



if not IS_KAGGLE_ENV:  # My local machine.

    DIR_INPUT = pathlib.Path('.')

    DIR_WEIGHTS = pathlib.Path(f'{str(pathlib.Path.home())}/.cache/torch/hub/checkpoints')



os.listdir(DIR_INPUT)
df_train = pd.read_csv(DIR_INPUT / 'train.csv')

df_train.head()
def df_to_list_of_dict_dataset(df: pd.DataFrame) -> list:

    """

    Transform the training DataFrame to list of dict.

    Each sample contains following keys mainly:

        `image_id`: Unique image id locates on single image.

        `bboxes`: Multiple bounding boxes for the image.

        `labels`: Same number of bboxes labeled with ones,

            for this task is just to predict where foregrounds locate at.

    """

    ds_dict = {}

    for idx, line in df.iterrows():

        if line['image_id'] not in ds_dict:

            ds_dict[line['image_id']] = {

                'image_id': line['image_id'],

                'width': float(line['width']),

                'height': float(line['height']),

                'bboxes': [eval(line['bbox'])],

            }

        else:

            ds_dict[line['image_id']]['bboxes'].append(eval(line['bbox']))



    ds_list = [sample for sample in ds_dict.values()]

    for sample in ds_list:

        sample['bboxes'] = np.asarray(sample['bboxes'], dtype='int64')

        sample['labels'] = np.ones(shape=(len(sample['bboxes']),), dtype='int64')

    return ds_list





ds_list_trn = df_to_list_of_dict_dataset(df_train)

ds_list_trn[0]
class WheatDatasetTrain(Dataset):

    

    def __init__(self, py_data: list, img_dir: pathlib.Path, transforms=None):

        self.py_data = py_data

        self.img_dir = img_dir

        self.transforms = transforms

    

    def __len__(self):

        return len(self.py_data)

    

    def __getitem__(self, item):

        if torch.is_tensor(item):

            item = item.tolist()

        

        sample = self.py_data[item]

        image = cv2.imread(str(self.img_dir / f"{sample['image_id']}.jpg"))[..., ::-1].copy()

        

        if self.transforms is not None:

            sample_to_transform = {'image': image, 'bboxes': sample['bboxes'],

                                   'labels': sample['labels']}

            sample_transformed = self.transforms(image=sample_to_transform['image'],

                                                 bboxes=sample_to_transform['bboxes'],

                                                 labels=sample_to_transform['labels'])

            image = sample_transformed['image'].to(torch.float32) / 255.

            boxes = torch.tensor(sample_transformed['bboxes'], dtype=torch.int64)

            boxes[:, [2, 3]] += boxes[:, [0, 1]]  # from coco format to pascal_voc format

            labels = torch.tensor(sample_transformed['labels'], dtype=torch.int64)

            target = {'boxes': boxes, 'labels': labels}

        else:

            image = torch.from_numpy(image.transpose(2, 0, 1)

                                     .astype('float32')) / 255.

            boxes = torch.tensor(sample['bboxes'], dtype=torch.int64)

            boxes[:, [2, 3]] += boxes[:, [0, 1]]

            labels = torch.tensor(sample['labels'], dtype=torch.int64)

            target = {'boxes': boxes, 'labels': labels}

        return image, target





def collate_fn(batch):

    return tuple(zip(*batch))





image_w, image_h = 1024, 1024

# Using A.RandomResizedCrop instance may lost all the bboxes in some sample,

# while method FasterRCNN.forward() requires at least one bbox for each sample.

# See base class at:

# https://github.com/pytorch/vision/blob/v0.7.0/torchvision/models/detection/generalized_rcnn.py#L64

transform = A.Compose([

    A.RandomBrightnessContrast(p=0.5),

    A.Blur(p=0.5),

    A.VerticalFlip(p=0.5),

    A.HorizontalFlip(p=0.5),

    A.RandomRotate90(p=0.5),

    A.RandomSizedBBoxSafeCrop(image_h, image_w, erosion_rate=0.05, p=0.5),

    ToTensorV2(),

], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))



dataset_trn = WheatDatasetTrain(ds_list_trn, DIR_INPUT / 'train', transform)

loader_trn = DataLoader(dataset_trn, batch_size=8, collate_fn=collate_fn)
def show_augmented(transform, sample_dict, img_dir: pathlib.Path):

    img = cv2.imread(str(img_dir / f"{sample_dict['image_id']}.jpg"))[..., ::-1].copy()

    augmented = transform(image=img, bboxes=sample_dict['bboxes'], labels=sample_dict['labels'])

    img = augmented['image'].numpy().transpose(1, 2, 0)

    img = img[..., ::-1].astype('uint8')

    bboxes = np.asarray(augmented['bboxes'], dtype='int64')

    bboxes[:, [2, 3]] += bboxes[:, [0, 1]]

    bboxes = bboxes.tolist()

    for bbox in bboxes:

        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)

    plt.figure(figsize=(6, 6))

    plt.imshow(img)

    plt.show()





for _ in range(3):

    show_augmented(transform, ds_list_trn[0], DIR_INPUT / 'train')
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=False)



# Pretrained weights are better, although it's pre-trained on ImageNet.

missing, unexpected = backbone.body.load_state_dict(torch.load(str(DIR_WEIGHTS / FILENAME_WEIGHTS)), strict=False)

print(f"Missing: {missing}\nUnexpected in loaded state_dict: {unexpected}")



model = torchvision.models.detection.FasterRCNN(backbone, num_classes=2)  # Including the background

print(model)
def train_n_epochs(model, optimizer, loader, device, lr_scheduler=None, n_epochs=10):

    model = model.to(device)

    model.train()

    losses = []

    for epoch in range(n_epochs):

        loss_epoch = 0.

        loader_len = len(loader)

        for img_list, target in tqdm(loader):

            img_list = [img.to(device) for img in img_list]

            target = list(target)

            for i in range(len(target)):

                target[i]['boxes'] = target[i]['boxes'].to(device)

                target[i]['labels'] = target[i]['labels'].to(device)

            losses_batch = model(img_list, target)

            losses_reduce = sum(l for l in losses_batch.values())



            optimizer.zero_grad()

            losses_reduce.backward()

            optimizer.step()



            loss_epoch += losses_reduce.item()

        if lr_scheduler is not None:

            lr_scheduler.step()

        loss_epoch = np.round(loss_epoch / loader_len, decimals=5)

        losses.append(loss_epoch)

        if loss_epoch <= min(losses):  # Save best only.

            torch.save(model.state_dict(), 'model.pt')

            print(f'Model is serialized in {repr("model.pt")}')

        print(f"Loss: {loss_epoch}")

            

    model.eval()

    return model





device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

model = train_n_epochs(model, optimizer, loader_trn, device, lr_scheduler, 10)
class ImgDatasetTest(Dataset):

    

    def __init__(self, img_dir: pathlib.Path):

        self.img_dir = img_dir

        self.img_names = [fname for fname in os.listdir(self.img_dir)

                          if os.path.splitext(fname)[-1].lower() in ['.jpg', '.png']]

        

    def __len__(self):

        return len(self.img_names)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        img_fname = self.img_names[idx]

        img_id, ext = os.path.splitext(img_fname)

        img = cv2.imread(str(self.img_dir / img_fname))[..., ::-1].astype('float32') / 255.

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        return img_tensor, img_id





dataset_test = ImgDatasetTest(DIR_INPUT / 'test')

loader_test = DataLoader(dataset_test, batch_size=8)

model.load_state_dict(torch.load('model.pt'))
def format_result_string(score, box):

    return f"{score:.4f} {' '.join(str(num) for num in box)}"





pred_threshold = 0.3

results = []

with torch.no_grad():

    for imgs, img_ids in loader_test:

        imgs = imgs.to(device)

        preds = model(imgs)

        for pred, img_id in zip(preds, img_ids):

            sample_pred = {'image_id': img_id, 'PredictionString': ''}

            pred['boxes'] = pred['boxes'].data.cpu().numpy()

            pred['scores'] = pred['scores'].data.cpu().numpy()

            boxes = pred['boxes'][pred['scores'] >= pred_threshold].astype('int64')

            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            scores = pred['scores'][pred['scores'] >= pred_threshold]

            sample_pred['PredictionString'] += ' '.join(

                format_result_string(score, box)

                for score, box in zip(scores, boxes)

            )

            results.append(sample_pred)



submission = pd.DataFrame(results)

submission.to_csv('submission.csv', index=False)

submission
def plot_prediction_from_submission(df: pd.DataFrame, img_dir: pathlib.Path, ext='.jpg'):

    for idx, line in df.iterrows():

        img = cv2.imread(str(img_dir / (line['image_id'] + ext)))

        probas_bboxes = np.array(line['PredictionString'].split(' ')).reshape(-1, 5).astype('float64')

        probas = probas_bboxes[:, 0]

        bboxes = probas_bboxes[:, 1:].astype('int64')

        bboxes[:, [2, 3]] += bboxes[:, [0, 1]]

        for proba, bbox in zip(probas, bboxes):

            # The lower probability predicted, the color of bbox will be more blue,

            # otherwise it will be more red.

            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]),

                          (int(255 - 255 * proba), 0, int(255 * proba)), 3)

        img = img[..., ::-1].copy()

        plt.figure(figsize=(6, 6))

        plt.imshow(img)

        plt.show()





plot_prediction_from_submission(submission, DIR_INPUT / 'test')