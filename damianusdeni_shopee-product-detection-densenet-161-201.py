!pip install jcopdl
!pip install jcopml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from jcopdl.utils.dataloader import MultilabelDataset
# bs = 64
# crop_size = 224

# train_transform = transforms.Compose([
#     transforms.RandomRotation(10),
#     transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),
#     transforms.ColorJitter(brightness=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize(230),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# train_set = datasets.ImageFolder("../input/productdetection/train_train/train_train/", transform=train_transform)
# trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

# val_set = datasets.ImageFolder("../input/productdetection/train_test/train_test/", transform=val_transform)
# valloader = DataLoader(val_set, batch_size=bs, shuffle=True)


# Inception
bs = 64
crop_size = 224

train_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.RandomRotation(10),
#     transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_set = datasets.ImageFolder("../input/productdetection/train_train/train_train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

val_set = datasets.ImageFolder("../input/productdetection/train_test/train_test/",transform=val_transform)
valloader = DataLoader(val_set, batch_size=bs, shuffle=True)
train_set, val_set
print("train_set: ", len(train_set))
print("val_set:",  len(val_set))
print("total_set: ", len(train_set) + len(val_set))
feature, target = next(iter(trainloader))
feature.shape
label2cat = train_set.classes
label2cat
from torchvision.models import densenet121, densenet201, densenet161
from torchvision.models import inception_v3
from jcopdl.layers import linear_block

# dnet = densenet121(pretrained=True)
# dnet201 = densenet201(pretrained=True)
# dnet161 = densenet161(pretrained=True)
# inception = inception_v3(pretrained=True)

# # freeze model
# for param in inception.parameters():
#     param.requires_grad = False
# dnet.classifier, dnet161.classifier, dnet201.classifier
inception
# dnet161
# dnet.classifier = nn.Sequential(
#     nn.Linear(1024, 42),
#     nn.LogSoftmax()
# )

# dnet161.classifier = nn.Sequential(
#     nn.Linear(2208, 42),
#     nn.LogSoftmax()
# )

# dnet201.classifier = nn.Sequential(
#     nn.Linear(1920, 42),
#     nn.LogSoftmax()
# )

inception.fc = nn.Sequential(
    nn.Linear(2048, 42),
    nn.LogSoftmax()
)
# dnet.classifier, dnet161.classifier, dnet201.classifier
inception.fc
# class CustomDensenet121(nn.Module):
#     def __init__(self, output_size):
#         super().__init__()
#         self.dnet = densenet121(pretrained=True)
#         self.freeze()
#         self.dnet.classifier = nn.Sequential(
# #             linear_block(1024, 1, activation="lsoftmax")
#             nn.Linear(1024, output_size),
#             nn.LogSoftmax(dim=1)
#         )
        
#     def forward(self, x):
#         return self.dnet(x)

#     def freeze(self):
#         for param in self.dnet.parameters():
#             param.requires_grad = False
            
#     def unfreeze(self):        
#         for param in self.dnet.parameters():
#             param.requires_grad = True  

            
# class CustomDensenet161(nn.Module):
#     def __init__(self, output_size):
#         super().__init__()
#         self.dnet161 = densenet161(pretrained=True)
#         self.freeze()
#         self.dnet161.classifier = nn.Sequential(
# #             linear_block(1024, 1, activation="lsoftmax")
#             nn.Linear(2208, output_size),
#             nn.LogSoftmax(dim=1)
#         )
        
#     def forward(self, x):
#         return self.dnet161(x)

#     def freeze(self):
#         for param in self.dnet161.parameters():
#             param.requires_grad = False
            
#     def unfreeze(self):        
#         for param in self.dnet161.parameters():
#             param.requires_grad = True  
            
            
# class CustomDensenet201(nn.Module):
#     def __init__(self, output_size):
#         super().__init__()
#         self.dnet201 = densenet201(pretrained=True)
#         self.freeze()
#         self.dnet201.classifier = nn.Sequential(
# #             linear_block(1024, 1, activation="lsoftmax")
#             nn.Linear(1920, output_size),
#             nn.LogSoftmax(dim=1)
#         )
        
#     def forward(self, x):
#         return self.dnet201(x)

#     def freeze(self):
#         for param in self.dnet201.parameters():
#             param.requires_grad = False
            
#     def unfreeze(self):        
#         for param in self.dnet201.parameters():
#             param.requires_grad = True  
            

class Inception(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.inception = inception_v3(pretrained=True)
        self.freeze()
        self.inception.fc = nn.Sequential(
#             linear_block(1024, 1, activation="lsoftmax")
            nn.Linear(2048, output_size),
            nn.LogSoftmax(dim=1)
        )
        self.inception.aux_logits = False
        
    def forward(self, x):
        return self.inception(x)

    def freeze(self):
        for param in self.inception.parameters():
            param.requires_grad = False
            
    def unfreeze(self):        
        for param in self.inception.parameters():
            param.requires_grad = True  
config = set_config({
    "output_size": len(train_set.classes),
    "batch_size": bs,
    "crop_size": crop_size
})
from jcopml.utils import save_model, load_model
# model = CustomDensenet121(config.output_size).to(device)
# model = CustomDensenet161(config.output_size).to(device)
# model = CustomDensenet201(config.output_size).to(device)
model = Inception(config.output_size).to(device)


# load model phase 1
# model = load_model('../input/densenet-model/densenet161_p1.pkl')


criterion = nn.NLLLoss()

# optimizer = optim.AdamW(model.parameters(), lr=0.00070)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
# optimizer = optim.AdamW(model.parameters(), lr=0.00075)

# callback = Callback(model, config, early_stop_patience=2, outdir="model")
callback = Callback(model, config, early_stop_patience=2, outdir="model")
# callback = Callback(model, config, early_stop_patience=2, outdir="model")
model
from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", val_set, valloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break
        
#     save_model(model, "densenet_p1.pkl")
# save model phase 1
save_model(model, "densenet161_p1.pkl")
# load model phase 1
model = load_model('../input/densenet-model/densenet161_p1.pkl')
print('test is on 0.7168')
print('test')
model.unfreeze()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

callback.reset_early_stop()
callback.early_stop_patience = 5
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", val_set, valloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break
        
    save_model(model, "densenet161_p1.pkl")
from jcopml.utils import save_model
save_model(model, "shoppe_product_detection_densenet_p2.pkl")
from jcopml.utils import load_model
# model_densenet = load_model('../input/densenet-model/shoppe_product_detection_densenet_p2.pkl')
# model_densenet_161 = load_model('../input/densenet-model/densenet161_p1.pkl')
submit_set = MultilabelDataset("../input/shopee-code-league-2020-product-detection/test.csv", "../input/productdetection2/resized/test/", transform=val_transform, fname_col = 'filename')
submit_loader = DataLoader(submit_set, batch_size=bs, shuffle=False)
feature, target = next(iter(submit_loader))
feature, target = feature.to(device), target.to(device)
submit_set.classes
all_preds = []
for feature, target in submit_loader:
    feature, target = feature.to(device), target.to(device)
    with torch.no_grad():
        model.eval()
        output = model(feature)
        preds = output.argmax(1)
    all_preds.extend(preds)
all_preds_val = [pred.item() for pred in all_preds]
df = pd.read_csv('../input/shopee-code-league-2020-product-detection/test.csv')
df.head()
df_submit_final = pd.DataFrame({
    "filename": df.filename,
    "category": all_preds_val
})

df_submit_final.head()
df_submit_final.to_csv('submission8.csv', index=False)
# import os
# print(os.listdir("../input"))

# ls '../input/productdetection2/resized/train/'


# images, labels = next(iter(trainloader))
# print(images.shape)
# print(labels)


# import glob
# import cv2


# images = glob.glob('../input/productdetection2/resized/test/*.jpg')
# for i in range(8):
#     image = cv2.imread(images[i])
#     plt.figure(figsize=(12,5))
#     plt.subplot(2,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(image)
