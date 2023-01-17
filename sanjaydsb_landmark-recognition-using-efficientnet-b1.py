import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import cv2
import seaborn as sn
import torch
from sklearn.preprocessing import LabelEncoder
directory = '../input/landmark-recognition-2020/'
train_dir = '../input/landmark-recognition-2020/train/*/*/*/*'
test_dir = '../input/landmark-recognition-2020/test/*/*/*/*'
output_dir ='../output/kaggle/working/'
image_dir_train='../input/landmark-recognition-2020/train/'
image_dir_test='../input/landmark-recognition-2020/test/'
os.listdir(directory)
test = pd.read_csv(os.path.join(directory,'sample_submission.csv'))
test['image_']=test.id.str[0]+"/"+test.id.str[1]+"/"+test.id.str[2]+"/"+test.id+".jpg"
test.head()
# train_images = glob.glob(train_dir)
# test_images = glob.glob(test_dir)
# print('Training images : ',len(train_images))
# print('Testing images : ',len(test_images))
# assert test.shape[0]==len(test_images)
train = pd.read_csv(os.path.join(directory,'train.csv'))
train["image_"] = train.id.str[0]+"/"+train.id.str[1]+"/"+train.id.str[2]+"/"+train.id+".jpg"
train["target_"] = train.landmark_id.astype(str)
train.head()
Threshold_count = 150

valid_landmark_df = pd.DataFrame(train['landmark_id'].value_counts()).reset_index()
valid_landmark_df.columns =  ['landmark_id', 'count_']
list_valid_landmarks = list(valid_landmark_df[valid_landmark_df.count_ >= Threshold_count]['landmark_id'].unique())

#or
# y = train.landmark_id.values
# valid_landmark_count = Counter(y).most_common(1000)
# print(valid_landmark_count[:10])
# list_valid_landmarks = [landmark[0] for landmark in valid_landmark_count]
print(train.shape)
train= train[train.landmark_id.isin(list_valid_landmarks)]
train.shape
# for img in range(2):
#     image = mpimg.imread(train_images[img])
#     print(image.shape)
#     plt.imshow(image)
#     plt.axis('Off')
#     plt.show()
# for img in range(2):
#     image = cv2.imread(test_images[img])
#     print(image.shape)
#     plt.imshow(image)
#     plt.axis('Off')
#     plt.show()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import transforms

TRAIN_BS = 32
TEST_BS = 32
# [ s for s in train_images if '.jpg' not in s]
# [ s for s in test_images if '.jpg' not in s]
class createDataset(Dataset):
    def __init__(self, transform, image_dir, df, train_type = True):        
        self.df = df 
        self.image_dir = image_dir    
        self.transform = transform
        self.train_type=train_type
        
    def __len__(self):
        return self.df.shape[0] 
    
    def __getitem__(self,idx):
        image_id = self.df.iloc[idx].id
        image_name = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        self.image = Image.open(image_name).convert('RGB')               
        self.image = self.transform(self.image)
#         print(self.image)
        
#         self.Y = self.df.iloc[idx].landmark_id
        self.Y = torch.Tensor([self.df.iloc[idx].landmark_id]).type(torch.LongTensor)        
        if(self.train_type):
            return {'image':self.image, 
                    'label':self.df.iloc[idx].landmark_id}
        else:
            return {'image':self.image}         
# idx=10

# # test_data = createDataset(transform = transformations , df = test , image_dir = image_dir_test , train = False )

# image_id = train.iloc[idx].id
# image_name = f"{image_dir_train}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
# image = Image.open(image_name)                 
# # image = image)
# print(image)

# #         Y = self.df.iloc[idx].landmark_id
# Y = torch.Tensor([train.iloc[idx].landmark_id]).type(torch.LongTensor)
# image
mean = (0.485, 0.456, 0.406)
std =  (0.229,0.225,0.224)
transformations = transforms.Compose([ transforms.Resize((64,64)),
#                                     transforms.Resize((128,128),interpolation=Image.NEAREST),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)
                                     ]
                                    )                               

label_encoder = LabelEncoder()
le = label_encoder.fit(train.landmark_id.values)
unique_classes = len(le.classes_)
print('Total number of classes ', unique_classes)
train_data = createDataset(transform = transformations , df = train , image_dir = image_dir_train , train_type = True )
train_loader = DataLoader(dataset = train_data, batch_size = TRAIN_BS, shuffle = True)
test_data = createDataset(transform = transformations , df = test , image_dir = image_dir_test , train_type = False )
test_loader = DataLoader(dataset = test_data, batch_size = TEST_BS)
!pip install efficientnet_pytorch
# import efficientnet_pytorch

# class EfficientNetEncoderHead(nn.Module):
#     def __init__(self, depth, num_classes):
#         super(EfficientNetEncoderHead, self).__init__()
#         self.depth = depth
#         self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.output_filter = self.base._fc.in_features
#         self.classifier = nn.Linear(self.output_filter, num_classes)
#     def forward(self, x):
#         x = self.base.extract_features(x)
#         x = self.avg_pool(x).squeeze(-1).squeeze(-1)
#         x = self.classifier(x)
#         return x
# model = EfficientNetEncoderHead(depth=0, num_classes=unique_classes)
# model.cuda()
import torchvision
import torch.nn as nn
import torch
# model = torchvision.models.resnet50(pretrained=True)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b1')
print(model)
from tqdm import tqdm
for param in model.parameters():
    param.requires_grad = False
    

model._fc = nn.Linear(model._fc.in_features, unique_classes)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=0.001)
n_epochs = 1
loss_list = []
activation = nn.Softmax(dim=1)
for epochs in range(n_epochs):    
    for i, data_x_y in enumerate(tqdm(train_loader)):
#         model.train()
        x= data_x_y['image']
        y=data_x_y['label']        
        optimizer.zero_grad()
        yhat =  model(x.cuda())
        loss = criterion(yhat, y.cuda())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print('Epoch ', epochs, 'loss : ',loss.item())
for x_test, y_test in test_loader:        
    x_test = x_test.to('cuda')
    y_test=y_test.to('cuda')

    model.eval()
    z = model(x_test)
    score, label = torch.max(z,1)
    correct += (label == y_test).sum().item()
    test_list.append(y_test)
    lables_list(label)
    scores_list.append(score)        
accuracy = accuracy/ n_test
plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()


