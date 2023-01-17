import numpy as np

import pandas as pd



import os
manifest = pd.read_csv('/kaggle/input/youtube-faces-with-facial-keypoints/youtube_faces_with_keypoints_full.csv')

manifest
with np.load('/kaggle/input/youtube-faces-with-facial-keypoints/youtube_faces_with_keypoints_full_3/youtube_faces_with_keypoints_full_3/Natasa_Micic_0.npz') as data:

    colorImages = data['colorImages']

    boundingBox = data['boundingBox']

    landmarks2D = data['landmarks2D']

    landmarks3D = data['landmarks3D']
import matplotlib.pyplot as plt
fig, _ = plt.subplots(1,1)

plt.axis('off')



# use the created array to output your multiple images. In this case I have stacked 4 images vertically

ax = []



ax.append(fig.add_subplot(2,2,1))

plt.imshow(colorImages[:,:,:,0])

plt.axis('off')

plt.scatter(pd.DataFrame(boundingBox[:,:,0]).iloc[:,0],pd.DataFrame(boundingBox[:,:,0]).iloc[:,1],color='red')

plt.scatter(pd.DataFrame(landmarks2D[:,:,0]).iloc[:,0],pd.DataFrame(landmarks2D[:,:,0]).iloc[:,1],color='blue')

plt.scatter(pd.DataFrame(landmarks3D[:,:,0]).iloc[:,0],pd.DataFrame(landmarks3D[:,:,0]).iloc[:,1],color='orange',s=10)



ax.append(fig.add_subplot(2,2,2))

plt.imshow(colorImages[:,:,:,1])

plt.axis('off')

plt.scatter(pd.DataFrame(boundingBox[:,:,1]).iloc[:,0],pd.DataFrame(boundingBox[:,:,1]).iloc[:,1],color='red')

plt.scatter(pd.DataFrame(landmarks2D[:,:,1]).iloc[:,0],pd.DataFrame(landmarks2D[:,:,1]).iloc[:,1],color='blue')

plt.scatter(pd.DataFrame(landmarks3D[:,:,1]).iloc[:,0],pd.DataFrame(landmarks3D[:,:,1]).iloc[:,1],color='orange',s=10)



ax.append(fig.add_subplot(2,2,3))

plt.imshow(colorImages[:,:,:,2])

plt.axis('off')

plt.scatter(pd.DataFrame(boundingBox[:,:,2]).iloc[:,0],pd.DataFrame(boundingBox[:,:,2]).iloc[:,1],color='red')

plt.scatter(pd.DataFrame(landmarks2D[:,:,2]).iloc[:,0],pd.DataFrame(landmarks2D[:,:,2]).iloc[:,1],color='blue')

plt.scatter(pd.DataFrame(landmarks3D[:,:,2]).iloc[:,0],pd.DataFrame(landmarks3D[:,:,2]).iloc[:,1],color='orange',s=10)



ax.append(fig.add_subplot(2,2,4))

plt.imshow(colorImages[:,:,:,3])

plt.axis('off')

plt.scatter(pd.DataFrame(boundingBox[:,:,3]).iloc[:,0],pd.DataFrame(boundingBox[:,:,3]).iloc[:,1],color='red')

plt.scatter(pd.DataFrame(landmarks2D[:,:,3]).iloc[:,0],pd.DataFrame(landmarks2D[:,:,3]).iloc[:,1],color='blue')

plt.scatter(pd.DataFrame(landmarks3D[:,:,3]).iloc[:,0],pd.DataFrame(landmarks3D[:,:,3]).iloc[:,1],color='orange',s=10)

#plt.show()
from keras.preprocessing.image import load_img, img_to_array

from matplotlib import animation

from IPython.display import HTML

%matplotlib inline



def plot_images(img_list):

  def init():

    img.set_data(img_list[0])

    return (img,)



  def animate(i):

    img.set_data(img_list[i])

    return (img,)



  fig = plt.figure()

  ax = fig.gca()

  img = ax.imshow(img_list[0])

  anim = animation.FuncAnimation(fig, animate, init_func=init,

                                 frames=len(img_list), interval=25, blit=True, repeat=False)

  return anim



imgs = [colorImages[:,:,:,j] for j in range(240)]



HTML(plot_images(imgs).to_html5_video())
import fnmatch

import os



matches = []

for root, dirnames, filenames in os.walk('../input/youtube-faces-with-facial-keypoints'):

    for filename in fnmatch.filter(filenames, '*.npz'):

        matches.append(os.path.join(root, filename))
matches = pd.DataFrame(matches)

matches.columns = ['file_name']
matches = pd.DataFrame([(i,j) for i in manifest.videoID for j in matches.file_name  if i in j])

matches.columns = ['videoID','file_name']

matches
manifest = pd.merge(matches,manifest)

manifest
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(manifest,test_size=0.3)
X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)
from torch.utils.data import Dataset, DataLoader

import torch
class FaceKeypointsDataset(Dataset):

    '''Face Keypoints Dataset'''

    def __init__(self, dataframe):

        self.dataframe = dataframe

    

    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self, idx):

        with np.load(self.dataframe.file_name[idx]) as data:

            img = torch.from_numpy(data['colorImages']/255.)

            labels = torch.from_numpy(data['landmarks2D'])

        return (img,labels)
train_data = FaceKeypointsDataset(X_train)

test_data = FaceKeypointsDataset(X_test)



train_data_loader = DataLoader(train_data,batch_size=1,shuffle=True)

test_data_loader = DataLoader(test_data,batch_size=1,shuffle=False)
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)

        self.pool1 = nn.MaxPool2d(2, 2)



        self.conv2 = nn.Conv2d(32,64,3)

        self.pool2 = nn.MaxPool2d(2, 2)



        self.conv3 = nn.Conv2d(64,128,3)

        self.pool3 = nn.MaxPool2d(2, 2)



        self.conv4 = nn.Conv2d(128,256,3)

        self.pool4 = nn.MaxPool2d(2, 2)



        self.conv5 = nn.Conv2d(256,512,1)

        self.pool5 = nn.MaxPool2d(2, 2)



        self.fc1 = nn.Linear(18432, 1024)

        self.fc2 = nn.Linear(1024, 136)

        

        self.drop1 = nn.Dropout(p = 0.1)

        self.drop2 = nn.Dropout(p = 0.2)

        self.drop3 = nn.Dropout(p = 0.25)

        self.drop4 = nn.Dropout(p = 0.25)

        self.drop5 = nn.Dropout(p = 0.3)

        self.drop6 = nn.Dropout(p = 0.4)

    def forward(self, x):

      

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.drop1(x)

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.drop2(x)

        x = self.pool3(F.relu(self.conv3(x)))

        x = self.drop3(x)

        x = self.pool4(F.relu(self.conv4(x)))

        x = self.drop4(x)

        x = self.pool5(F.relu(self.conv5(x)))

        x = self.drop5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.drop6(x)

        x = self.fc2(x)

        return x
net = Net()
net = net.double()
import torch.optim as optim



criterion = torch.nn.SmoothL1Loss()



optimizer = optim.Adam(net.parameters(), lr = 0.00001)
import torch.nn.functional as F
net.cuda()
for epoch in range(5):  # loop over the dataset multiple times



    running_loss = 0.0

    epoch_loss = []

    t = 0

    for i, data in enumerate(train_data_loader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        inputs = inputs.cuda()

        labels = labels.cuda()

        

        for j in range(inputs.shape[4]):

            t+=1

            inp = inputs[:,:,:,:,j]

            inp = inp.permute(0,3,1,2)

            inp = F.interpolate(inp, size=(224, 224), mode='bicubic', align_corners=False)

            lab = labels[:,:,:,j]



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inp)

            loss = criterion(outputs.view(-1), lab.view(-1))

            loss.backward()

            optimizer.step()



            # print statistics

            running_loss += loss.item()

            epoch_loss.append(loss.item())

            if t % 5000 == 4999:    # print every 5000 mini-batches

                

                samp = int(np.random.randint(low=0,high=len(test_data)-1,size=1))

                test1 = test_data[samp][0][:,:,:,0].unsqueeze(0).permute(0,3,1,2).cuda()

                test1 = F.interpolate(test1, size=(224, 224), mode='bicubic', align_corners=False)

                plt.imshow(np.moveaxis(test1.cpu().detach().numpy()[0,:,:,:],0,-1))

                plt.scatter(pd.DataFrame(net(test1).cpu().detach().numpy().reshape(68,2)).iloc[:,0],

                       pd.DataFrame(net(test1).cpu().detach().numpy().reshape(68,2)).iloc[:,1])

                plt.title(f'epoch {epoch+1}, batch step: {t}, average loss for last 5000 images:{running_loss/5000}');

                plt.show()

                plt.gcf().show()

                running_loss = 0.0

    print(f'Epoch {epoch}, loss: ',np.mean(epoch_loss))



    