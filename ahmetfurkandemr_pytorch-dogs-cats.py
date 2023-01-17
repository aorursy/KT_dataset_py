!pip3 install imutils
!pip3 install torchsummary
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import imutils
import cv2
from matplotlib import pyplot as plt
# If the images in the data set are readable,
# So we include it if it's not wrong.
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False
# locations of data
train_data_path = "../input/cat-and-dog/training_set/training_set"
val_data_path = "../input/cat-and-dog/test_set/test_set"
image_test_cat = Image.open("../input/cat-and-dog/test_set/test_set/cats/cat.4001.jpg")
image_test_dog = Image.open("../input/cat-and-dog/test_set/test_set/dogs/dog.4003.jpg")
plt.imshow(image_test_dog)
plt.imshow(image_test_cat)
# operations to convert images into vectors.
img_transforms = transforms.Compose([
    
    # we resize the pictures
    transforms.Resize((150,150)),
    transforms.ToTensor()
    ])


train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)


val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms, is_valid_file=check_image)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

val_data_loader  = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True) 
class demirnet(nn.Module):
    
    def __init__(self):
        
        super(demirnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size= 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size= 3)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size= 3)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 250)
        self.fc2 = nn.Linear(250, 2)



    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.maxpool3(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.maxpool4(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.maxpool5(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

model = demirnet()
device = torch.device("cuda")
model=model.to(device) 
print(model)

print("Model = ",next(model.parameters()).device," da eğitilecek")
summary(model, input_size=(3,150,150)) 
loss_func = nn.CrossEntropyLoss()

opt = optim.Adam(model.parameters(), lr=1e-4)
def metrics_batch(target, output):
    pred = output.argmax(dim=1, keepdim=True)
    
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects
def loss_batch(loss_func, xb, yb,yb_h, opt=None):
    
    loss = loss_func(yb_h, yb)
    
    metric_b = metrics_batch(yb,yb_h)
    
    if opt is not None:
        loss.backward() 
        opt.step() 
        opt.zero_grad() 

    return loss.item(), metric_b
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    
    loss=0.0
    metric=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.type(torch.float).to(device)
        yb=yb.to(device)
        
        yb_h=model(xb)

        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)
        loss+=loss_b
        if metric_b is not None:
            metric+=metric_b
            
    loss/=len_data 
    metric/=len_data
    return loss, metric
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    
    for epoch in range(epochs):

        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
            
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)

        print("epoch: %d, train loss: %.6f, val loss: %.6f, train accuracy: %.6f, val accuracy: %.6f" %(epoch, train_loss, val_loss, train_metric, val_metric))
# Training was done with 20 epoch twice. So he trained with 40 epochs in total.
# 20 + 20
num_epochs=20
train_val(num_epochs, model, loss_func, opt, train_data_loader, val_data_loader)
!pip3 install wget
import wget
wget.download("https://www.dropbox.com/s/cclqhu19skquf0b/_92712149_gettyimages-480164327.jpg?dl=1")
wget.download("https://www.dropbox.com/s/5wu1stob9ys3eyt/images.jpeg?dl=1")

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image)
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)

data_transforms = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])
x = image_loader(data_transforms, "./images.jpeg")
z = model(x)
if z[0][0].item() > z[0][1].item():
    za = "Cat"
else:
    za = "Dog"

a = Image.open("./images.jpeg")
plt.title("Estimation of our model = {}".format(za))
plt.imshow(a)
x = image_loader(data_transforms, "./_92712149_gettyimages-480164327 (1).jpg")
z = model(x)
if z[0][0].item() > z[0][1].item():
    za = "Cat"
else:
    za = "Dog"

a = Image.open("./_92712149_gettyimages-480164327 (1).jpg")
plt.title("Estimation of our model = {}".format(za))
plt.imshow(a)
