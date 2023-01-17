import numpy as np 

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input/aptos2019-blindness-detection/'):

    print(dirname)
BASE_PATH='/kaggle/input/aptos2019-blindness-detection/'

train_dataset=pd.read_csv(os.path.join(BASE_PATH,'train.csv'))

test_dataset=pd.read_csv(os.path.join(BASE_PATH,'test.csv'))
train_dataset.head(3)
test_dataset.head(3)
from PIL import Image

from matplotlib.pyplot import imshow

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(32, 32))

columns = 3

rows = 5

for i in range(1,rows*columns+1):

    IMG_PATH=BASE_PATH+'train_images/'

    img=Image.open(os.path.join(IMG_PATH,train_dataset.iloc[i][0]+'.png'))

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()
!pip install --upgrade efficientnet-pytorch
!pip install torchsummary
import PIL

import torch

from time import time

import torchvision

from PIL import Image

import torch.nn as nn

import torch.optim as optim

from torch.utils import data

from torchsummary import summary

from torch.autograd import Variable

import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet
class Dataset(data.Dataset):

    def __init__(self,csv_path,images_path,transform=None):

        self.train_set=pd.read_csv(csv_path)

        self.train_path=images_path

        self.transform=transform

    def __len__(self):

        return len(self.train_set)

    

    def __getitem__(self,idx):

        file_name=self.train_set.iloc[idx][0]+'.png'

        label=self.train_set.iloc[idx][1]

        img=Image.open(os.path.join(self.train_path,file_name))

        if self.transform is not None:

            img=self.transform(img)

        return img,label
params = {'batch_size': 16,

          'shuffle': True

         }

epochs = 100

learning_rate=1e-3
transform_train = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([

        torchvision.transforms.RandomRotation(10),

        transforms.RandomHorizontalFlip()],0.7),

		transforms.ToTensor()])
training_set=Dataset(os.path.join(BASE_PATH,'train.csv'),os.path.join(BASE_PATH,'train_images/'),transform=transform_train)

training_generator=data.DataLoader(training_set,**params)
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=5)
model.to(device)
print(summary(model, input_size=(3, 224, 224)))
PATH_SAVE='./Weights/'

if(not os.path.exists(PATH_SAVE)):

    os.mkdir(PATH_SAVE)
criterion = nn.CrossEntropyLoss()

lr_decay=0.99

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eye = torch.eye(5).to(device)

classes=[0,1,2,3,4]
history_accuracy=[]

history_loss=[]

epochs = 25
"""for epoch in range(epochs):  

    running_loss = 0.0

    correct=0

    total=0

    class_correct = list(0. for _ in classes)

    class_total = list(0. for _ in classes)

    for i, data in enumerate(training_generator, 0):

        inputs, labels = data

        t0 = time()

        inputs, labels = inputs.to(device), labels.to(device)

        labels = eye[labels]

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, torch.max(labels, 1)[1])

        _, predicted = torch.max(outputs, 1)

        _, labels = torch.max(labels, 1)

        c = (predicted == labels.data).squeeze()

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

        accuracy = float(correct) / float(total)

        

        history_accuracy.append(accuracy)

        history_loss.append(loss)

        

        loss.backward()

        optimizer.step()

        

        for j in range(labels.size(0)):

            label = labels[j]

            class_correct[label] += c[j].item()

            class_total[label] += 1

        

        running_loss += loss.item()

        

        print( "Epoch : ",epoch+1," Batch : ", i+1," Loss :  ",running_loss/(i+1)," Accuracy : ",accuracy,"Time ",round(time()-t0, 2),"s" )

    for k in range(len(classes)):

        if(class_total[k]!=0):

            print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))

        

    print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch+1, 100 * correct / total))

        

    if epoch%10==0 or epoch==0:

        torch.save(model.state_dict(), os.path.join(PATH_SAVE,str(epoch+1)+'_'+str(accuracy)+'.pth'))

        

"""
plt.plot(history_accuracy)

plt.plot(history_loss)
model.load_state_dict(torch.load('/kaggle/input/efficient-net/Weights/21_0.9243582741671218.pth'))
model.eval()
test_transforms = transforms.Compose([transforms.Resize(512),

                                      transforms.ToTensor(),

                                     ])
def predict_image(image):

    image_tensor = test_transforms(image)

    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)

    input = input.to(device)

    output = model(input)

    index = output.data.cpu().numpy().argmax()

    return index
submission=pd.read_csv(BASE_PATH+'sample_submission.csv')
submission.head(3)
submission_csv=pd.DataFrame(columns=['id_code','diagnosis'])
IMG_TEST_PATH=os.path.join(BASE_PATH,'test_images/')

for i in range(len(submission)):

    img=Image.open(IMG_TEST_PATH+submission.iloc[i][0]+'.png')

    prediction=predict_image(img)

    submission_csv=submission_csv.append({'id_code': submission.iloc[i][0],'diagnosis': prediction},ignore_index=True)

    if(i%10==0 or i==len(submission)-1):

        print('[',32*'=','>] ',round((i+1)*100/len(submission),2),' % Complete')
submission_csv.to_csv('submission.csv',index=False)