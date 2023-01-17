# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import shutil



import numpy as np

import pandas as pd

from PIL import Image



import torch

import torchvision

from torchvision.transforms import transforms



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!wget https://www.dropbox.com/s/loz4ijaxcor3l5l/test_set.csv
class VGGFaceRecognition(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # Step one

        # load pretrained model

#         model = models.resnet18(pretrained = True)

        self.model = torchvision.models.vgg19(pretrained = True)

        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        for param in self.model.parameters():

            param.requires_grad = False

        

        # Second step

        # Add new layers to the pretrained model

        self.classifier = torch.nn.Sequential(

            torch.nn.Linear(25088, 512, bias=True),

            torch.nn.Sigmoid(),

            torch.nn.Linear(512, 256, bias=True),

            torch.nn.Sigmoid(),

            torch.nn.Linear(256, 128, bias=True),

            torch.nn.Tanh(), 

        )



    def forward(self,x):

        x = self.model(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x
# Third step: Specify triplet loss

def triplet_loss(anchor, positive, negative, margin=0.75):

    loss = torch.mean(torch.norm(anchor - positive, dim=1)) - torch.mean(torch.norm(anchor - negative, dim=1)) + margin

    loss = torch.clamp(loss, min=0, max=1)

    return loss.sum()
# Fourth step

# Load images

def read_and_resize(filename):

    img = Image.open(filename)

    transform = transforms.Compose([

     transforms.Resize(299),

     transforms.ToTensor(),

     transforms.Normalize(mean=(0.5,), std=(0.5,))

    ])

    img_t = transform(img)

    return torch.unsqueeze(img_t, 0)
def generate_stochastic_batch(prefix, proper_names, size):

    anchors = []

    positives = []

    negatives = []

    for i in range(size):

        # generate 2 random different names for 2 classes

        double = np.random.choice(proper_names, 2, replace=False)

        # 2 different images names of for anchor and positive

        positive_imgs = np.random.choice(os.listdir(prefix + double[0] + '/'), 2, replace=False)

        # image name for negative

        negative_img = np.random.choice(os.listdir(prefix + double[1] + '/'))

        # full path to anchor

        anchor = prefix + double[0] + '/' + positive_imgs[0]

        # full path to positive

        positive = prefix + double[0] + '/' + positive_imgs[1]

        # full path to negative

        negative = prefix + double[1] + '/' + negative_img

        

        anchors.append(read_and_resize(anchor))

        positives.append(read_and_resize(positive))

        negatives.append(read_and_resize(negative))

        

    return torch.cat(anchors), torch.cat(positives), torch.cat(negatives)
# people with 2 or more images

prefix =  '/kaggle/input/cv-fall-2019-hw-2/dataset/'

proper_people = [name for name in os.listdir(prefix) if len(os.listdir(prefix + name)) > 1]
test_set = pd.read_csv("test_set.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = VGGFaceRecognition()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.to(device)

batch_size = 64
# if number of subsequent times when

# loss == 0 is equal autostop_count_max

# then stop training

autostop_count = 0

autostop_count_max = 10

for i in range(100):

    print(f'Epoch #{i}')

    anchors, positives, negatives = generate_stochastic_batch(prefix, proper_people, batch_size)

    anchors_emb = model(anchors.to(device))

    positives_emb = model(positives.to(device))

    negatives_emb = model(negatives.to(device))

    loss = triplet_loss(anchors_emb,

                        positives_emb, 

                        negatives_emb)

    

    print('\tTrain loss: {:.5f}'.format(loss.data.item()))





    if loss.data.item() == 0:

        autostop_count += 1

    else:

        autostop_count = 0

        

    if autostop_count == autostop_count_max:

        print('Model is trained')

        break

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

        

    

    anchors, positives, negatives = [], [], []

    k = 0

    for index, row in test_set.iterrows():

        if k == batch_size:

            break

        anchors.append(read_and_resize(prefix + row['Anchor']))

        positives.append(read_and_resize(prefix + row['Positive']))

        negatives.append(read_and_resize(prefix + row['Negative']))

        k += 1

        

    anchors = torch.cat(anchors)

    positives = torch.cat(positives)

    negatives = torch.cat(negatives)

    

    loss = triplet_loss(

        model(anchors.to(device)),

        model(positives.to(device)),

        model(negatives.to(device))

    )

    print('\tTest loss: {:.5f}'.format(loss.data.item()))

    
anchors, positives, negatives = [], [], []

k = 0

for index, row in test_set.iterrows():

#     if k == batch_size:

#         break

    anchors.append(read_and_resize(prefix + row['Anchor']))

    positives.append(read_and_resize(prefix + row['Positive']))

    negatives.append(read_and_resize(prefix + row['Negative']))

    k += 1



anchors = torch.cat(anchors)

positives = torch.cat(positives)

negatives = torch.cat(negatives)



loss = triplet_loss(

    model(anchors.to(device)),

    model(positives.to(device)),

    model(negatives.to(device))

)

print('\tTest loss after training: {:.5f}'.format(loss.data.item()))