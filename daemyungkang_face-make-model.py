import numpy as np # linear algebra
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.patches as patches
import os
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")


import os
prefix = "/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/"
def list_imagenames(path, label):
    images = []
    walk_path = prefix + path
    for dirname, _, filenames in os.walk(walk_path):
        for filename in filenames:
            images.append((os.path.join(walk_path, filename), label))
            
    return images
train_withmask = list_imagenames("Train/WithMask", 0)
train_withoutmask = list_imagenames("Train/WithoutMask", 1)
test_withmask = list_imagenames("Test/WithMask", 0)
test_withoutmask = list_imagenames("Test/WithoutMask", 1)
validation_withmask = list_imagenames("Validation/WithMask", 0)
validataion_withoutmask = list_imagenames("Validation/WithoutMask", 1)

imagenames = []
for c in [train_withmask, train_withoutmask, test_withmask, test_withoutmask, validation_withmask, validataion_withoutmask]:
    imagenames.extend(c)
print(len(imagenames))
options={"with_mask":0,"without_mask":1} # mapping
def drow_image(path, input_info): #function to visualize images
    image_path = input_info[0]
        
    image=plt.imread(os.path.join(image_path))
    fig,ax=plt.subplots(1)
    ax.axis("off")
    fig.set_size_inches(10,5)
    ax.imshow(image)
print(train_withmask[0])
drow_image("Train", train_withmask[0])
import collections
def make_dataset(imagenames): #function to make dataset
    image_tensor=[]
    label_tensor=[]
    for image_info in imagenames:
        label=image_info[1]
        image=Image.open(image_info[0]).convert("RGB")
        image_tensor.append(my_transform(image))
        label_tensor.append(torch.tensor(label))
                
    final_dataset=[[k,l] for k,l in zip(image_tensor,label_tensor)]
    return tuple(final_dataset)
#importing neccessary libraries for deeplearning task..
import torch
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader

my_transform=transforms.Compose([
                                 transforms.Resize((64, 64)),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor()
                                ])

dataset=make_dataset(imagenames) #making a datset
train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size
batch_size=32
trainset,testset=torch.utils.data.random_split(dataset,[train_size,test_size])
train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)
resnet = models.resnet50(pretrained=True)
torch.cuda.is_available()
import torch.nn as nn
n_inputs=resnet.fc.in_features
last_layer=nn.Linear(n_inputs,3)
resnet.fc.out_features=last_layer

if torch.cuda.is_available():
    resnet.cuda()
    
print(resnet.fc.out_features)
if torch.cuda.is_available(): #checking for GPU availability
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
for paramet in resnet.parameters():
    paramet.requires_grad=True
resnet.parameters()
import torch.optim as optim

criterion=nn.CrossEntropyLoss()

optimizer=optim.SGD(resnet.parameters(),lr=0.001)
n_epochs=10

for epoch in range(1,n_epochs+1):
    train_loss = 0.0

    for batch,(data,target) in enumerate(train_loader):


        if torch.cuda.is_available():
            data , target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output=resnet(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch%20==19:
            print("Epoch {}, batch {}, training loss {}".format(epoch, batch+1,train_loss/20))
        train_loss = 0.0
#########Testing##########
test_loss=0.0
acc=0
resnet.eval()

for data,target in test_loader:
    if torch.cuda.is_available():
        data,target=data.cuda(),target.cuda()
    output=resnet(data)
    loss=criterion(output,target)
    test_loss+=loss.item()
    _,pred=torch.max(output,1)
    predicted=pred.numpy()[:,np.newaxis] if not torch.cuda.is_available() else pred.cpu().numpy()[:,np.newaxis]
    actual=target.numpy()[:,np.newaxis] if not torch.cuda.is_available() else target.cpu().numpy()[:,np.newaxis]
    images = data.cpu().numpy()
    acc+=np.sum(predicted==actual)/len(target.cpu().numpy())

Average_loss=test_loss/len(test_loader)
Average_acc=acc/len(test_loader)

print("Avg total loss is {:.6f}".format(Average_loss))
print("Avg accuracy is {:.6f}".format(Average_acc))
torch.save(resnet,open("resnet_model_face_mask","wb")) # saving the trained model.
device = torch.device("cuda")
model=torch.load(open("resnet_model_face_mask","rb"),map_location=device) #loading the model
!pip install mtcnn #installing library for predicting faces
from mtcnn import MTCNN
detect=MTCNN()
def trans(bndbox,newimage):
    a,b,c,d=bndbox["box"]
    image_crop=transforms.functional.crop(newimage, b,a,d,c)
    my_transform=transforms.Compose([transforms.Resize((64,64)),
                                     transforms.RandomCrop((64,64)),
                                     transforms.ToTensor()])(image_crop)
        
    return my_transform
def tag_plot(bndbox,filepath,predicted):
    configut=["with_mask","without_mask","mask_weared_incorrect"]
    x=plt.imread(filepath)
    fig,ax=plt.subplots(1)
    ax.axis("off")
    fig.set_size_inches(15,10)
    for i,j in zip(bndbox,predicted):
        a,b,c,d=i["box"]
        edgecolor='g'
        
        if type(j) == "list":
            j = j[0]
            
        if j != 0:
            edgecolor='r'
        patch=patches.Rectangle((a,b),c,d,linewidth=2, edgecolor=edgecolor,facecolor="none",)
        ax.imshow(x)
        color = "white"
        if j == 0:
            color = "blue"
        ax.text(a, b, configut[j], size=10,
                style='italic',verticalalignment="bottom", horizontalalignment="left",color=color)
        ax.add_patch(patch)
model=model.eval()

def testing(filepath):
    configut=["with_mask","without_mask","mask_weared_incorrect"]
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    newimage=Image.open(filepath).convert("RGB")
    bndbox=detect.detect_faces(img)
    if len(bndbox)==1:
        image_pred=trans(bndbox[0],newimage).unsqueeze(0)
        _, pred=torch.max(model(image_pred.to(device)),1)
        tag_plot(bndbox,filepath,predicted=pred)
    else:
        predicted=[]
        for i in bndbox:
            image_pred=trans(i,newimage).unsqueeze(0)
            _, pred=torch.max(model(image_pred.to(device)),1)
            predicted.append(pred)
        tag_plot(bndbox,filepath,predicted)
!wget https://ichef.bbci.co.uk/news/1024/cpsprodpb/D9C8/production/_111125755_facemask.jpg -O test.jpg
testing("test.jpg")
!wget -O test3.jpg "https://img2.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202008/24/dongascience/20200824060024683skml.jpg"
testing("test3.jpg")
!wget -O test5.jpg "https://storage.googleapis.com/kagglesdsdata/datasets%2F667889%2F1176415%2Fimages%2Fmaksssksksss5.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1599706715&Signature=gYcQT4BLhVidy%2F%2FLTcWI3USmbYSy2c%2F1vwDiCaS6IXOPYICTzk%2FhE6TAyvjNEtJ8fkAWRYqOA2Gkyh1JJHkub7xzaGhUbvF4lEnmtPzOQm28cEjpdoC%2BrSBi%2B%2BAWpNxALUSDq5vpmfQz2NfXO8g3r%2FNcyiuz9zQMTi5RQ3TLHpeP44ladzv2bWMW9IO7IZn752BOPVOyg5U8ws9L39erwhtJjQl%2F%2Fl8uGLNQsyXvWyfGvmRShtY6J9aZvpA1Emhe%2FmIkKmWrE3bi2ZlG%2BGYwgU1qEgmCEBbpyqCd0X6rHrf761D2pD6pjLLiTUmUFmH0eGmxDXOj1L13QqlcAOAK7Q%3D%3D"
testing("test5.jpg")