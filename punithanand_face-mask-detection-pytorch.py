import numpy as np 
import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.patches as patches
from torchvision import torch,datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
path_images=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images")
train=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/train.csv"))
train #quick look into dataset
train=train[train.classname.str.contains("face_with_mask$|face_no_mask")] # filtering the data to mask and no mask bales
train.classname.value_counts()
#lets lookinto the data after filtering
train.head()
# creating the function to visualize images
def draw_box(image_name):
    img=plt.imread(os.path.join(path_images,image_name))
    temp=train[train.name==image_name]
    fig,ax=plt.subplots(1)
    fig.set_size_inches(10,5)
    ax.imshow(img)
    ax.axis('off')
    edgecolor={"face_no_mask":"r","face_with_mask":"b"}
    for i in range(len(temp)):
        a,b,c,d=temp.values[i][1:5]
        patch=patches.Rectangle((a,b),c-a,d-b,linewidth=2, 
                                edgecolor=edgecolor[temp.values[i][5:6][0]],facecolor="none",)
        ax.text(a, b, temp.values[i][5:6][0], style='italic',bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        ax.add_patch(patch)
draw_box(random.choice(train.name.values))
#defining sizes for testing and training the images
train_size=int(len(train)*0.8)
test_size=int(len(train))-train_size
from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
train["labels"]=lbl.fit_transform(train.classname)
train.to_csv("new.csv", header=False)
train_new=pd.read_csv("./new.csv",header=None)
class MaskAndNoMask(Dataset): 
    def __init__(self,dataframe,root_dir,transform=None):
        self.annotation=dataframe
        self.root_dir=root_dir
        self.transform=transform
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir,self.annotation.iloc[index,1])
        new_img=Image.open(img_path).crop((self.annotation.iloc[index,2:6]))
        label=torch.tensor(int(self.annotation.iloc[index,7:8]))
    
        if self.transform:
            image=self.transform(new_img)
            return(image,label)
my_transform=transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomCrop((224,224)),
                                 transforms.ToTensor()])

dataset=MaskAndNoMask(dataframe=train_new,root_dir=path_images,transform=my_transform)

batch_size=32

trainset,testset=torch.utils.data.random_split(dataset,[train_size,test_size])
train_loader=DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)
dataiter=iter(train_loader)
images,labels=dataiter.next()
images=images.numpy()

fig=plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    plt.imshow(np.transpose(images[idx],(1,2,0)))
torch.cuda.empty_cache() 
resnet=models.resnet34(pretrained=True)
for param in resnet.parameters():
    param.requires_grad=False
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("gpu available {}".format(torch.cuda.device_count()))
    print("device name {}".format(torch.cuda.get_device_name(0)))
else:
    device=torch.device("cpu")
    print("No gpu avalable,traing on cpu")
import torch.nn as nn
n_inputs=resnet.fc.in_features
last_layer=nn.Linear(n_inputs,2)
resnet.fc.out_features=last_layer

if torch.cuda.is_available():
    resnet.cuda()

print(resnet.fc.out_features)
for param in resnet.parameters():
    param.requires_grad=True
import torch.optim as optim

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(resnet.parameters(),lr=0.001)
n_epochs=3
epochs=[]
training_loss=[]

for epoch in range(1,n_epochs+1):
    train_loss=0
    epochs.append(epoch)
    
    
    for batch,(data,target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,target=data.cuda(),target.cuda()

        optimizer.zero_grad()
        output=resnet(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
     
        if batch%20==19:
            print("Epoch {}, batch {}, training loss {}".format(epoch,batch+1,train_loss/20))
            training_loss.append(train_loss) 
            train_loss=0.0
test_loss=0
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
    acc+=np.sum(predicted==actual)/len(test_loader)
    
avg_loss=test_loss/len(test_loader)
avg_acc=acc/len(test_loader)

print("Average total loss is {:.6f}".format(avg_loss))
print("Average total accuracy is {:.6f}".format(avg_acc))
torch.save(resnet,open("resnet_face_mask_detect","wb"))
model=torch.load(open("./resnet_face_mask_detect","rb"))
!pip install facenet-pytorch
from facenet_pytorch import MTCNN
mtcnn = MTCNN()
model=model.eval()
class TagImages():
    def __init__(self):
        
        self.filepath=filepath
        img=Image.open(self.filepath)
        boxes, _ = mtcnn.detect(img)
        predictions=[]
        for i in boxes:
            im_pr=img.crop(i)
            predict_im=my_transform(im_pr).unsqueeze(0)
            output=model(predict_im.cuda())
            _,pred=torch.max(output,1)
            predicted=pred.numpy() if not torch.cuda.is_available() else pred.cpu().numpy()
            predictions.append(predicted[0])
        self.boxes=boxes
        self.predictions=predictions
        
    def draw_box_predicted(self,filepath):
        img=plt.imread(self.filepath)
        fig,ax=plt.subplots(1)
        fig.set_size_inches(10,5)
        ax.imshow(img)
        ax.axis('off')
        configuration=["face_no_mask", "face_with_mask"]
        color={"face_no_mask":"r","face_with_mask":"b"}
        for i,j in zip(self.boxes,self.predictions):
            a,b,c,d=i
            patch=patches.Rectangle((a,b),c-a,d-b,linewidth=2, 
                                    edgecolor=color[configuration[j]],facecolor="none",)
            ax.text(a, b, configuration[j],
                    style='italic',bbox={'facecolor': color[configuration[j]], 'alpha': 0.4, 'pad': 10})
            ax.add_patch(patch)
filepath=os.path.join(path_images,random.choice(train.name.values))
TagImages().draw_box_predicted(filepath)
