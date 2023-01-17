import os

import torchvision

import torchvision.transforms as pytrans

from torch.utils.data import DataLoader

import torch.nn as nn

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import matplotlib.image as im_age

import matplotlib.pyplot as plt

import cv2

import os

import torch



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

dirs=os.listdir("/kaggle/input/104-flowers-garden-of-eden/")





tforms=pytrans.Compose(

[pytrans.Resize((192,192)),

pytrans.ToTensor(),

# pytrans.Normalize(mean=[0.485, 0.456, 0.406],

#                    std=[0.229, 0.224, 0.225])

])



dataset0 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-192x192/train/", transform=tforms)

dataset1 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-224x224/train/", transform=tforms)

dataset2 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512/train/", transform=tforms)

dataset3 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-311x311/train/", transform=tforms)



dataset=dataset0+dataset1+dataset2+dataset3
import torch

import matplotlib.pyplot as plt

#dataset0.__dict__["classes"]



LABELS=dataset0.classes



def int_to_name(int_list):

    labels=[]

    for i in int_list:

        labels.append(LABELS[i])

    return labels

        





BATCH_SIZE=80

    

out=dataset0.classes+dataset1.classes+dataset2.classes+dataset3.classes



print("Number of Categories", len( set(out) ) )



data_loader=DataLoader(dataset, batch_size=16, shuffle=True)



imgs,labels=next(iter(data_loader))





l = str( int_to_name(labels.tolist()) )



print(" ")

print("Labels:")

print(l)

fig, ax = plt.subplots(figsize=(16,16))

#fig.suptitle(l, fontsize=16)

#fig.suptitle("HEY", fontsize=16)

ax.set_xticks([])

ax.set_yticks([])

ax.imshow(torchvision.utils.make_grid(imgs[:16], nrow=8).permute(1,2,0))



model = torchvision.models.resnext50_32x4d(pretrained=False, progress=True)



num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 104)
import time



EPOCHS=15

BATCH_SIZE=80



data_loader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



loss_func=nn.CrossEntropyLoss()



model = model.to(device)

model.train()



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(model.parameters(), 

                            lr=0.001, 

                            momentum=0.9, 

                            weight_decay=0.0005)

time0=time.time()

print("Begin training")

correct,lossAvg=[],[]

for epoch in range(EPOCHS):

    total_correct,total_loss,count=0,0,0

    for batch in data_loader:

        #check if targets is a list

        imgs,targets=batch

        imgs = imgs.to(device)

        targets = targets.to(device)



        optimizer.zero_grad()

        

        preds = model(imgs)

        loss=loss_func(preds,targets)

        

        #update averagers

        total_correct+=preds.argmax(dim=1).eq(targets).sum().item()

        total_loss+=loss.item()

        count+=1

        

        #update model

        loss.backward()

        optimizer.step()

    

    avg_loss_epoch=total_loss/count

    lossAvg.append( avg_loss_epoch )

    correct.append( total_correct/(count*BATCH_SIZE) )

    print("END EPOCH #{} avg loss: {}".format(epoch,avg_loss_epoch))

    

print(" ")

print("Training Time:")

print(time.time() - time0)
#x=range(0, EPOCHS)



fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(12, 8), sharey=False)



x=list(range(0,EPOCHS))



ax1.plot(x,correct)

ax1.scatter(x,correct,marker="X",color="tomato")

ax1.text(0.0,0.5, "AVG % Correct per epoch")

ax2.plot(x, lossAvg)

ax2.scatter(x,lossAvg,marker="X",color="tomato")

ax2.text(0.0,1.75, "AVG Loss per epoch" )
import time



VAL_BATCH_SIZE=80



tforms=pytrans.Compose(

[pytrans.Resize((192,192)),

pytrans.ToTensor(),

# pytrans.Normalize(mean=[0.485, 0.456, 0.406],

#                    std=[0.229, 0.224, 0.225])

])



dataset0 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-192x192/val/", transform=tforms)

dataset1 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-224x224/val/", transform=tforms)

dataset2 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512/val/", transform=tforms)

dataset3 = torchvision.datasets.ImageFolder("/kaggle/input/104-flowers-garden-of-eden/jpeg-311x311/val/", transform=tforms)



dataset=dataset0+dataset1+dataset2+dataset3



data_loader=DataLoader(dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)



model = model.to(device)

model.eval()



time0=time.time()

print("Begin validating")

num_correct=[]

for batch in data_loader:

    #check if targets is a list

    imgs,targets=batch

    imgs = imgs.to(device)

    targets = targets.to(device)

    

    preds = model(imgs)

    

    num_correct.append( preds.argmax(dim=1).eq(targets).sum().item()  )

    

print(" ")

print("Eval Time:")

print(time.time() - time0)
sum1 = np.sum( np.array( num_correct ) )



s="Percent Correct from Validation Set: {}%".format( round( (sum1/(len(num_correct)*VAL_BATCH_SIZE))*100, 2) )



print(s)
def int_to_name(int_list):

    labels=[]

    for i in int_list:

        labels.append(LABELS[i])

    return labels





path0="/kaggle/input/104-flowers-garden-of-eden/jpeg-192x192/test/"

path1="/kaggle/input/104-flowers-garden-of-eden/jpeg-224x224/test/"

path2="/kaggle/input/104-flowers-garden-of-eden/jpeg-311x311/test/"

path3="/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512/test/"



def get_pic_list(path):



    dirs=os.listdir(path)

    for i,f in enumerate(dirs):

        dirs[i]=path+f

    return dirs



dirs0=get_pic_list(path0)

dirs1=get_pic_list(path1)

dirs2=get_pic_list(path2)

dirs3=get_pic_list(path3)



img_list=dirs0+dirs1+dirs2+dirs3



random.shuffle(img_list)



random_pic_set = img_list[:16]



labels=[]

tensors=[]

for img in random_pic_set:

    

    img = im_age.imread(img)

    img=cv2.resize(img,(192,192))

    img=torchvision.transforms.functional.to_tensor(img).to(device)

    img=img.reshape((1,3,192,192))

    

    tensors.append(img.reshape((3,192,192)))

    with torch.no_grad():

        preds = model(img)

        labels.append( preds.argmax(dim=1).item() )

    

print( int_to_name(labels) )

imgs=torch.stack(tuple(tensors),dim=0).cpu()

fig, ax = plt.subplots(figsize=(16,16))

ax.set_xticks([])

ax.set_yticks([])

ax.imshow(torchvision.utils.make_grid(imgs[:16], nrow=8).permute(1,2,0))




