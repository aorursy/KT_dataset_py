# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from skimage import io, transform

import os

#os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos')





# Any results you write to the current directory are saved as output.
# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl





# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import torch.nn as nn

from torchvision import models, transforms

from torch.utils.data import DataLoader, Dataset

import glob

import cv2
import torch

import torch.nn as nn
model = models.mobilenet_v2(pretrained=False)

model.classifier = nn.Sequential(

   nn.Linear(1280,1024,bias=False), nn.ReLU(inplace=True),nn.BatchNorm1d(1024),nn.Dropout(0.4),

   nn.Linear(1024,512,bias=False), nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(0.4),

   nn.Linear(512,256,bias=False), nn.ReLU(inplace=True),nn.BatchNorm1d(256),nn.Dropout(0.4),

   nn.Linear(256,1,bias=False))

model.cuda()

model.load_state_dict(torch.load('/kaggle/input/deepfakemodel/mobilenet-detection.pt'))
"""

PyTorch Video Dataset Class for loading videos using PyTorch

Dataloader. This Dataset assumes that video files are Preprocessed

 by being trimmed over time and resizing the frames.

Mohsen Fayyaz __ Sensifai Vision Group

http://www.Sensifai.com

If you find this code useful, please star the repository.

"""



from __future__ import print_function, division

from skimage import io, transform

from tqdm import tqdm_notebook as tqdm

import cv2

import os

import torch

import numpy as np

import pickle

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import json

import torchvision

import torch.nn as nn

import time

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn.functional as F

from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image

mtcnn = MTCNN(image_size=224, select_largest=False, post_process=False, margin=100,device='cuda:0')





import matplotlib.pyplot as plt



class RandomCrop(object):

    """Crop randomly the frames in a clip.

	Args:

		output_size (tuple or int): Desired output size. If int, square crop

			is made.

	"""



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, clip,top,bottom,left,right):



       



        clip = clip[top:bottom, left:right]

        return clip

    

class Rescale(object):

    """Rescale the image in a sample to a given size.



    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image = sample



        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)

       

        img = transform.resize(image, (new_h, new_w))

        

        return img





class videoDataset(Dataset):

    """Dataset Class for Loading Video"""



    def __init__(self, clipsListFile, rootDir, channels, timeDepth, split, ySize, mean, transform=None):

        """

		Args:

			clipsList (string): Path to the clipsList file with labels.

			rootDir (string): Directory with all the videoes.

			transform (callable, optional): Optional transform to be applied

				on a sample.

			channels: Number of channels of frames

			timeDepth: Number of frames to be loaded in a sample

			xSize, ySize: Dimensions of the frames

			mean: Mean valuse of the training set videos over each channel

		"""

        self.counter = 0 

        self.fcounter = 0

        clipsList = clipsListFile

        

        print(len(clipsList))

        self.clipsList = clipsList

        self.rootDir = rootDir

        self.channels = channels

        self.timeDepth = timeDepth

        self.ySize = ySize

        self.mean = mean

        self.transform = transform

        self.split = split



    def __len__(self):

        #print(len(self.clipsList))

        return len(self.clipsList)



    def readVideo(self, videoFile):

        # Open the video file

        cap = cv2.VideoCapture(videoFile)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        d = 0

        h = 0

        new = []

        failedclip = False

        while d != self.timeDepth:

            ret, frame = cap.read()

            if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #try:

                frame = Image.fromarray(frame)

                face = mtcnn(frame)

                try:

                    d += 1



                    in_transform = transforms.Compose([

                        transforms.Resize((224,224)),

                        

                        transforms.ToTensor(),

                        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                         std=[0.229, 0.224, 0.225])])

                    face = face.permute(1, 2, 0).numpy()

                    face = Image.fromarray(np.uint8(face))

                    frame = in_transform(face)

                    #frame= torch.from_numpy(frame)

                

                    #frame = frame.permute(2,0,1)

                    #frame = transform.resize(frame, (3,224,224))

                    #frame = torch.from_numpy(frame)

                    frame = frame.unsqueeze(1)

                    new.append(frame)

                except:

                    h += 1

                    d-=1

                    if h > total:

                        failed_clip = True

                        break

                 



            else:

                failed_clip = True

                break







                

       

        

              

                    



            #else:

             #   print("Skipped!")

              #  failedClip = True

               # break

        try:

            frames = torch.cat((new[0:self.timeDepth]),dim=1)

            frames = frames.squeeze(1)

        except:

            frames = []

            print(total)

                #except: 

                 #   h += 1

                  #  if h > 150:

                   #     break

                    #pass

        return frames, failedclip

        #frames = frames.squeeze(1)

      

    def __getitem__(self, idx):



        videoFile = os.path.join(self.rootDir, self.clipsList[idx])



        clip,failedclip = self.readVideo(videoFile)

        return clip,failedclip


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

dataset = videoDataset(filenames,'/kaggle/input/deepfake-detection-challenge/test_videos/',3,1,244,244,0.5)

test_loader = DataLoader(dataset,batch_size=5)

def imshow(img):

    npimg = img.numpy() 

    #npimg = npimg[:,1,:,:]

    

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

   

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# obtain one batch of training images

dataiter = iter(test_loader)

images,_ = dataiter.next() # _ for no labels



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(20, 8))

plot_size=4

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])
images,_ = next(iter(test_loader))



print(images.shape)





    
from tqdm import tqdm_notebook as tqdm

model.eval()

predictions = []



z = 0

for i in tqdm(range(400)):

    images, failedclip = dataset.__getitem__(i)

    if failedclip:

        predictions.append([os.path.basename(filenames[i]),0.5])

    else:

        images = images.cuda()

        output = model(images.unsqueeze(0))

    

   

        pred = torch.sigmoid(output).cpu()

        

        #if pred1 == 1:

           # prediction = output[i][pred1].data.cpu().numpy()

        #else:

         #   prediction = 1-output[i][pred1].data.cpu().numpy()

        predictions.append([os.path.basename(filenames[i]), pred.data.cpu().numpy()[0][0]])

        

        





   
import matplotlib.pyplot as plt



submission = pd.DataFrame(predictions, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)



plt.hist(submission.label, 20)

plt.show()
sub = pd.read_csv('submission.csv')

sub.head()