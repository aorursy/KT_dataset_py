# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform



from PIL import Image



%matplotlib inline 
# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()
#Dataset class



class ImageDataset(Dataset):

    



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image

        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale

        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image

        

        if self.transform:            

            image = self.transform(image)                                          # applying transforms, if any

        

        sample = (image, label)        

        return sample
# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.ToTensor()                               

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
#Checking training sample size and label

for i in range(len(trainset)):

    sample = trainset[i]

    print(i, sample[0].size(), " | Label: ", sample[1])

    if i == 9:

        break
# Visualizing some sample data

# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):                                             #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')




# Define your CNN model here, shift it to GPU if needed

class ImageClassifierModel(nn.Module):

    def __init__(self):

        super(ImageClassifierModel, self).__init__()

        # convolutional layer

        self.conv1 = nn.Conv2d(3, 16, 3,stride=1, padding=1)

        # max pooling layer

        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,3,stride=1,padding =1)

        self.fc = nn.Linear(128*7*7,67)

        self.pool2 = nn.MaxPool2d(2,2)

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(32,64,3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(64,128,3, stride=1, padding=1)

        self.pool4 = nn.MaxPool2d(4,4)

        self.dropout = nn.Dropout(0.3)



    def forward(self, x):

        # add sequence of convolutional and max pooling layers

        x = self.conv1(x)

        x= self.pool1(x)

        x= self.relu(x)

        x=self.conv2(x)

        x=self.pool2(x)

        x=self.relu(x)

        x=self.conv3(x)

        x= self.pool3(x)

        x=self.dropout(x)

        x=self.conv4(x)

        x= self.pool4(x)

        x= self.sigmoid(x)

        x=x.view(-1,128*7*7)

        x=self.fc(x)

        return (x)





model1 = ImageClassifierModel()

if train_on_gpu:

    model1.cuda()



from torchvision import models



model2 = torchvision.models.resnet50(pretrained=True)

for param in model2.parameters():

    param.requires_grad = False

num_ftrs = model2.fc.in_features

model2.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())



model3 = torchvision.models.vgg19(pretrained = True)

for param in model3.parameters():

    param.requires_grad = False

n_inputs = model3.classifier[6].in_features

model3.classifier[6] = nn.Sequential( nn.Linear(n_inputs,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())



model4 = torchvision.models.resnet152(pretrained=True)

for param in model4.parameters():

    param.requires_grad = False

num_ftrs = model4.fc.in_features

model4.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())



model5 = torchvision.models.vgg16(pretrained = True)

for param in model5.parameters():

    param.requires_grad = False

n_inputs = model5.classifier[6].in_features

model5.classifier[6] = nn.Sequential( nn.Linear(n_inputs,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())



model6 = torchvision.models.alexnet(pretrained=True)

for param in model6.parameters():

    param.requires_grad = False

n_inputs = model6.classifier[6].in_features

model6.classifier[6] = nn.Sequential( nn.Linear(n_inputs,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())



model7 = torchvision.models.googlenet(pretrained=True)

for param in model7.parameters():

    param.requires_grad = False

n_inputs = model7.fc.in_features

model7.fc = nn.Sequential( nn.Linear(n_inputs,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,67), nn.Sigmoid())







model1.cuda()

model2.cuda()

model3.cuda()

model4.cuda()

model5.cuda()

model6.cuda()

model7.cuda()


# Loss function to be used

criterion = nn.CrossEntropyLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer1 = optim.Adam(model1.parameters(), lr=0.00085)

optimizer2 = optim.Adam(model2.parameters(), lr=0.00085)

optimizer3 = optim.Adam(model3.parameters(), lr=0.00085)

optimizer4 = optim.Adam(model4.parameters(), lr=0.00085)

optimizer5 = optim.Adam(model5.parameters(), lr=0.00085)

optimizer6 = optim.Adam(model6.parameters(), lr=0.00085)

optimizer7 = optim.Adam(model7.parameters(), lr=0.00085)





# Training Loop (You can write your own loop from scratch)

n_epochs = 16 #number of epochs, change this accordingly

model1.train()

model2.train()

model3.train()

model4.train()

model5.train()

model6.train()

model7.train()

for epoch in range(1, n_epochs+1):

    for data, target in trainloader:

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

            

        optimizer1.zero_grad()

        optimizer2.zero_grad()

        optimizer3.zero_grad()

        optimizer4.zero_grad()

        optimizer5.zero_grad()

        optimizer6.zero_grad()

        optimizer7.zero_grad()

        

        output1 = model1(data)

        output2 = model2(data)

        output3 = model3(data)

        output4 = model4(data)

        output5 = model5(data)

        output6 = model6(data)

        output7 = model7(data)

        output = output1+output2+output3+output4+output5+output6+output7 

        

        loss = criterion(output, target)

        

        

        loss.backward()

        

        optimizer1.step()

        optimizer2.step()

        optimizer3.step()

        optimizer4.step()

        optimizer5.step()

        optimizer6.step()

        optimizer7.step()



        

    print(loss.item())

    




#Exit training mode and set model to evaluation mode

model1.eval() # eval mode

model2.eval()

model3.eval()

model4.eval()

model5.eval()

model6.eval()

model7.eval()

print("done")

predictions=[]

# iterate over test data to make predictions

for data1, target1 in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        data1, target1 = data1.cuda(), target1.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output1=model1(data1)

    output2=model2(data1)

    output3=model3(data1)

    output4=model4(data1)

    output5=model5(data1)

    output6=model6(data1)

    output7=model7(data1)

    

    output = 1.2*output1+output2+output3+output4+output5+output6+output7

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



submission['Category'] = predictions             #Attaching predictions to submission file



# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions

transform1 = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.ToTensor()                               

                                ])

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
print(len(submission))


#saving submission file

submission.to_csv('submission49.csv', index=False, encoding='utf-8')

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename= "submission49.csv"):

    csv = submission.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download= "{filename}" href="data: text/csv; base64, {payload}" target="_blank">{title}</a>'

    html = html.format(payload = payload, title=title, filename=filename)

    return HTML(html)

create_download_link(df)

