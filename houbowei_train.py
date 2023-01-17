# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# %%

# randomly show imgs

from sklearn.model_selection import train_test_split

import random

import os

import cv2

import numpy as np

import pandas as pd

from matplotlib import patches

from matplotlib import pyplot as plt

from PIL import Image

import copy





class dataPreprocess(object):

    """Convert train.csv into Train and Validation parts.



    read csv file â†’ convert bbox â†’ drop columns â†’ train and val image ids



    Attributes:

        train_csv_path: train.csv dir



    """



    def __init__(self, data_path):

        """Read data root path.



        Args:

            data_path: data which contains images and train.csv.

        """

        self.data_path = data_path

        self.data_csv_path = os.path.join(data_path, 'train.csv')

        self.data_df = pd.read_csv(self.data_csv_path)



        # conver bbox from string into xmin, ymin, w, h, (top left point, w, h)

        bboxs = np.stack(self.data_df.bbox.apply(

            lambda x: np.fromstring(x[1:-1], sep=',')))

        for i, column in enumerate(['x', 'y', 'w', 'h']):

            self.data_df[column] = bboxs[:, i]

        self.data_df.drop(columns=['bbox', 'width', 'height'], inplace=True)

        self.img_ids = self.data_df.image_id.unique()



    def random_split_dataset(self, frac=0.999):

        """Random split dataset into train and validation based on frac.



        Args:

            frac: frac as train dataset, (1-frac) as validation set.



        Returns:

            return list(train_ids), list(val_ids)

        """

        # In case you need the original img_ids, we copy the imgs and shuffle.

        img_ids = copy.deepcopy(self.img_ids)

        random.shuffle(img_ids)

        train_index = int(frac * len(img_ids))

        train_ids = img_ids[:train_index]

        val_ids = img_ids[train_index:]

        return train_ids, val_ids



    def __str__(self):

        return self.data_df.describe().__str__() + self.data_df.info().__str__()





def draw_rectangle(img_path, x, y, w, h):

    img = Image.open(img_path)

    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 3)

    return img





def draw_multirectangles(boxes, imgpath):

    img = plt.imread(imgpath)

    for x1, y1, x2, y2 in boxes:

        cv2.rectangle(img, (x1, y1), (x2, y2), 255, 3)

    return img
import glob

import os

import random



import cv2

import numpy as np

import pandas as pd

import PIL

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from matplotlib import pyplot as plt

from PIL import Image

from torchvision import transforms





class WheatDataset(torch.utils.data.Dataset):

    """Global wheat dataset for dataloader.



    Attributes:

        image_ids: list of file names, ['fe133ccb4', ...]

        image_dir: data path, data/train/ or data/test/

        transforms: preprocessing for images from torchvision.

        target_transforms: transform target



    Returns:

        image: PIL RGB format.

        bboxes: [N, 4] x1, y1, x2, y2

        labels: [N, ] [1, ..., 1]

        areas: [N, ] areas float

    """



    def __init__(self, image_ids, image_dir, transforms=None, target_transforms=None):

        super(WheatDataset, self).__init__()

        self.image_ids = image_ids

        self.transforms = transforms

        self.target_transforms = target_transforms

        self.img_dir = image_dir

        meta_data = dataPreprocess(image_dir).data_df

        meta_data['x2'] = meta_data.x + meta_data.w

        meta_data['y2'] = meta_data.y + meta_data.h

        meta_data['area'] = meta_data.w * meta_data.h

        self.dataframe = meta_data.groupby('image_id')



    def __getitem__(self, index):

        image_id = self.image_ids[index]

        data = self.dataframe.get_group(

            image_id).loc[:, ['x', 'y', 'x2', 'y2', 'source', 'area']].values

        bboxes, labels, areas = data[:, :4], data[:, -2], data[:, -1]



        bboxes = torch.from_numpy(bboxes.astype(np.float32))

        labels = torch.ones(labels.shape, dtype=torch.int64)



        # image

        img_path = os.path.join(self.img_dir, 'train', image_id + '.jpg')

        image = Image.open(img_path).convert('RGB')

        # The image is in HWC, we need to convert to CHW.

        if self.transforms:

            image = self.transforms(image)

        if self.target_transforms:

            labels = self.target_transforms(labels)

        return image, bboxes, labels, image_id



    def __len__(self):

        return len(self.image_ids)





class WheatDatasetTest(torch.utils.data.Dataset):

    """Some Information about WheatDatasetTest  """



    def __init__(self, test_dir):

        super(WheatDatasetTest, self).__init__()

        # glob all the test data

        self.image_ids = glob.glob(os.path.join(test_dir, "*.jpg"))



    def __getitem__(self, index):

        img = PIL.Image.open(self.image_ids[index]).convert('RGB')

        img = torchvision.transforms.ToTensor()(img)

        return img, self.image_ids[index]



    def __len__(self):

        return len(self.image_ids)





def collate_fn(batch):

    images, bboxes, labels, imageid = tuple(zip(*batch))

    targets = []

    for box, label in zip(bboxes, labels):

        targets.append({'boxes': box, 'labels': label})

    return torch.stack(images), targets, imageid
import os

import random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from torchvision import transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from matplotlib import pyplot as plt



DATA_DIR = '/kaggle/input/global-wheat-detection/'

# torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataPrerpocessing

data = dataPreprocess(DATA_DIR)

train_ids, val_ids = data.random_split_dataset(1.0)



############## hyperparameters

learning_rate = 1e-3

epochs = 4



# Dataset

tsfm = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor()

])



dataset = WheatDataset(train_ids, DATA_DIR, transforms=tsfm)





dataloader = torch.utils.data.DataLoader(

    dataset, batch_size=16, num_workers=8, collate_fn=collate_fn)



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(

    pretrained=True, progress=True, pretrained_backbone=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# model = nn.DataParallel(model)

model.to(device=device)



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=learning_rate)



# loop over the dataset multiple times



# For saving best model

best_loss = float('inf')



# For early stop

steps = 3

previous_loss = []



for epoch in range(epochs):

    for iteration, data in enumerate(dataloader, 0):

        images, targets, image_ids = data

        images = images.to(device)

        targets = [{'boxes': i['boxes'].to(

            device), 'labels':i['labels'].to(device)} for i in targets]

        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        loss_dict = model(images, targets)

        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        optimizer.step()

        if iteration % 50 == 0:

            print(f"Iteration {iteration} Loss: {loss.mean().item()}")

torch.save(model, '/kaggle/working/final.pth')



print('Finished Training')
def eval_dataset(model_path, test_path='test'):

    """Evaluate Model. plot images



    Args:

        model_path: 'final.pth'.



    Returns:

        return result

    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path)

    model.to(device)

    model.eval()

    eval_dataset = WheatDatasetTest(test_path)

    # Use batch_size == 1 for evaluation, DON't CHANGE

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=10)

    results = []

    submmision = open('submission.csv', 'w')

    lines = []

    for images, image_ids in eval_dataloader:

        images = images.to(device)

        outputs = model(images)

        image_ids = list(map(lambda x: x.split(

            '/')[-1].strip('.jpg'), image_ids))

        for image_id, output_dict in zip(image_ids, outputs):

            boxes = output_dict['boxes'].cpu().detach().numpy()

            scores = output_dict['scores'].cpu().detach().numpy()

            one_line = []

            for score, box in zip(scores, boxes):

                one_line.extend([str(score)] + [str(num) for num in box])

            one_line = image_id + ',' + " ".join(one_line)

            submmision.writelines(one_line+'\n')

            lines.append(one_line)

    # submmision.writelines(lines)

    submmision.close()

    # visualize to verify

    # plt.figure(figsize=(20, 20))

    # for i, res in enumerate(results):

    #     out_dict, img_id = res

    #     img = draw_multirectangles(out_dict['boxes'], img_id)

    #     plt.subplot(2, 5, i+1)

    #     plt.imshow(img)

    # # Generate CSV submmision file.

    # # Format: image_id, score x1, y1, x2, y2

    return results
!ls
!ls /kaggle/input/global-wheat-detection/test
eval_dataset('final.pth', '/kaggle/input/global-wheat-detection/test')