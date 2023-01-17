# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from PIL import Image
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch.nn as nn
import torch.nn.functional as F
import os
import helper
from tqdm import tqdm
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
#     device = "cuda"
    print("Running on the GPU")
else:
    device = torch.device("cpu")
#     device = "cpu"
    print("Running on the CPU")
torch.cuda.set_device(device)
# /kaggle/input/landmark-recognition-2020/train/5/5/5/5556e34494b2761d.jpg
train = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
train.head()
label_freq = train['landmark_id'].value_counts()
label_freq.shape[0]
label_freq.describe()
MIN_NUM_IMAGES = 5
label_freq[label_freq>MIN_NUM_IMAGES].shape[0]
train['landmark_id'].describe()
valid_labels = label_freq[label_freq>MIN_NUM_IMAGES].index
valid_labels
IMG_SIZE = 150
CROP_SIZE = 100
transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.CenterCrop(CROP_SIZE),
                                transforms.ToTensor()])
frame = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
frame = frame.loc[frame['landmark_id'].isin(valid_labels), :]
frame = frame.reset_index(drop=True)
le = LabelEncoder()
frame['landmark_id'] = le.fit_transform(frame['landmark_id'])
class LandmarksDatasetTrain(Dataset):
    """Landmarks dataset.""" 

    def __init__(self, landmarks_frame, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = landmarks_frame
        self.root_dir = root_dir
        self.transform = transform 

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.landmarks_frame.loc[idx, 'id'][0],self.landmarks_frame.loc[idx, 'id'][1], self.landmarks_frame.loc[idx, 'id'][2], self.landmarks_frame.loc[idx, 'id'])
        img_name += ".jpg"
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.loc[idx, 'landmark_id']
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['landmarks'] = torch.tensor(sample['landmarks'])

        return sample
class LandmarksDatasetTest(Dataset):
    """Landmarks dataset.""" 

    def __init__(self, test_img_list, root_dir, transform=None):
        self.test_img_list = test_img_list 
        self.root_dir = root_dir
        self.transform = transform 

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.test_img_list[idx])
        image = Image.open(img_name)
        sample = {'image': image}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
dataset_train = LandmarksDatasetTrain(landmarks_frame = frame,
                                      root_dir='/kaggle/input/landmark-recognition-2020/train',
                                      transform=transform)


test_images = []    
for dirpath, dirname, filenames in os.walk('/kaggle/input/landmark-recognition-2020/test/'):
    for f in filenames:
        if not os.path.basename(dirpath).startswith('.'):
            test_images.append("/kaggle/input/landmark-recognition-2020/test/"+f[0]+"/"+f[1]+"/"+f[2]+"/"+f)

dataset_test = LandmarksDatasetTest(test_img_list=test_images,
                                     root_dir='/kaggle/input/landmark-recognition-2020/test',
                                     transform=transform)

print(f'\ntrain images:')
for i in range(len(dataset_train)):
    sample = dataset_train[i]
    print(type(sample['landmarks']))
    print(i, sample['image'].size(), sample['landmarks'].size())
    
    if i == 3:
        break
    
print(f'\ntest images:')
for i in range(len(dataset_test)):
    sample = dataset_test[i]
    print(i, sample['image'].size())
    
    if i == 3:
        break
train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
frame['landmark_id'].nunique()
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CROP_SIZE*CROP_SIZE*3, 512)
        self.conv1d1 = nn.Conv1d(512, 64, 3, stride=2)
        self.fc2 = nn.Linear(64, 128)
        self.conv1d2 = nn.Conv1d(128, 64, 3, stride=2)
        self.fc3 = nn.Linear(64, 256)
        self.conv1d3 = nn.Conv1d(256, 64, 3, stride=2)
        self.fc4 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 64)
        self.fc8 = nn.Linear(64, frame['landmark_id'].nunique())

    def forward(self, x):
        x = F.relu(self.conv1d1(self.fc1(x)))
        x = F.relu(self.conv1d2(self.fc2(x)))
        x = F.relu(self.conv1d3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
torch.cuda.device_count()
# net.to(device)
net.to(torch.device('cuda:0'))
!nvidia-smi
torch.cuda.get_device_name()
for epoch in range(3): # 3 full passes over the data
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for data in tqdm(train_loader):  # `data` is a batch of data
        X = data['image'].to(device)  # X is the batch of features
        y = data['landmarks'].to(device) # y is the batch of targets.
        optimizer.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1,CROP_SIZE*CROP_SIZE*3))  # pass in the reshaped batch
#         print(np.argmax(output))
#         print(y)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients

    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 
# torch.save(net.state_dict(), '/kaggle/working/pytorch_model')
# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testset:
#         X, y = data
#         output = net(X.view(-1,784))
#         #print(output)
#         for idx, i in enumerate(output):
#             #print(torch.argmax(i), y[idx])
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total += 1

# print("Accuracy: ", round(correct/total, 3))
# # From: https://www.kaggle.com/davidthaler/gap-metric
# def GAP_vector(pred, conf, true, return_x=False):
#     '''
#     Compute Global Average Precision (aka micro AP), the metric for the
#     Google Landmark Recognition competition. 
#     This function takes predictions, labels and confidence scores as vectors.
#     In both predictions and ground-truth, use None/np.nan for "no label".

#     Args:
#         pred: vector of integer-coded predictions
#         conf: vector of probability or confidence scores for pred
#         true: vector of integer-coded labels for ground truth
#         return_x: also return the data frame used in the calculation

#     Returns:
#         GAP score
#     '''
#     x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
#     x.sort_values('conf', ascending=False, inplace=True, na_position='last')
#     x['correct'] = (x.true == x.pred).astype(int)
#     x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
#     x['term'] = x.prec_k * x.correct
#     gap = x.term.sum() / x.true.count()
#     if return_x:
#         return gap, x
#     else:
#         return gap

