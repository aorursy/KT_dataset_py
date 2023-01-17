import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch
import os

os.listdir("../input/virtual-hack/car_data/car_data")
train_folder="../input/virtual-hack/car_data/car_data/train/"

test_folder ="../input/virtual-hack/car_data/car_data/test/"
import os

len(os.listdir(train_folder))
traindata_per_class = list()

folders=list()

for folder in os.listdir(train_folder):

  folders.append(folder)

  length=len(os.listdir(train_folder+folder))

  print(folder,length)

  traindata_per_class.append(length)
plt.bar(folders,traindata_per_class)
cat_df=pd.DataFrame({"car_category":folders,"number":traindata_per_class}).sort_values("car_category")
cat_df[:50].set_index("car_category")['number'].plot.bar(color="r",figsize=(20,6))

# the data is pretty evenly distributed
folders
from torchvision import transforms

image_transforms={

    "train":

     transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])    

        ]),

     'test':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    

}
# from torch.utils.data.sampler import SubsetRandomSampler



# database_size = 192

# batch_size=32

# validation_pct=0.2

# indices = list(range(database_size))

# split = int(np.floor(validation_pct * dataset_size))

# np.random.shuffle(indices)



# train_indices,validation_indices = indices[split:],indices[:split]

# train_sampler = SubsetRandomSampler(train_indices)

# validation_sampler = SubsetRandomSampler(validation_indices)



# train_loader = torch.utils.data.DataLoader()
from torchvision import datasets

path="../input/virtual-hack/car_data/car_data/"

image_datasets={x:datasets.ImageFolder(os.path.join(path,x),image_transforms[x]) for x in ['train','test']}
dataset_size=len(image_datasets['train'])

train_size = int(0.8 * dataset_size)

val_size = dataset_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(image_datasets['train'], [train_size, val_size])
train_dataset.dataset
val_dataset.dataset
from torch.utils.data import DataLoader

dataloader={

    "train":DataLoader(train_dataset,batch_size=16,shuffle=True),

    "val":DataLoader(val_dataset,batch_size=16,shuffle=True),

    "test":DataLoader(image_datasets['test'],batch_size=1,shuffle=True)

}
trainiter = iter(dataloader['val'])

features,labels=next(trainiter)

features.shape,labels.shape
len(trainiter.dataset)
from torchvision import models

model = models.resnet34(pretrained=True)
for param in model.parameters():

  param.requires_grad=False
import torch.nn as nn

model.fc= nn.Sequential(nn.Linear(512,256),

                        nn.ReLU(),

                        nn.Linear(256,196),

                        nn.LogSoftmax(dim=1))
model = model.to("cuda")

model = nn.DataParallel(model)
from torch import optim

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(),lr=1e-04)
from torch import cuda,optim

train_on_gpu = cuda.is_available()

print(f'Train on gpu: {train_on_gpu}')



# Number of gpus

if train_on_gpu:

    gpu_count = cuda.device_count()

    print(f'{gpu_count} gpus detected.')

    if gpu_count > 1:

        multi_gpu = True

    else:

        multi_gpu = False
# from timeit import default_timer as timer

# import torch

# def train(model,criterion,optimizer,train_loader,valid_loader,save_file_name,max_epochs_stop=3,n_epochs=20

#          ,print_every=2):

    

    

#     epochs_no_improve=0

#     valid_loss_min = np.Inf

    

#     valid_max_acc=0

#     history=[]

    

#     try:

#         print(f'Model has been trained for: {model.epochs} epochs \n')

#     except:

#         model.epochs=0

#         print(f'Training from scratch\n')

    

#     overall_start=timer()

    

#     for epoch in range(n_epochs):

#         train_loss = 0.0

#         valid_loss = 0.0

        

#         train_acc = 0.0

#         valid_acc = 0.0

        

#         model.train()

#         start=timer()

        

#         #training loop

#         for ii,(data,target) in enumerate(train_loader):

            

#             if train_on_gpu:

#                 data,target=data.cuda(),target.cuda()

                

#             optimizer.zero_grad()

            

#             output = model(data)

          

#             loss = criterion(output,target)

#             loss.backward()

            

#             optimizer.step()

            

#             train_loss+=loss.item() * data.size(0)

            

#             _,pred = torch.max(output,dim=1)

#             correct_tensor = pred.eq(target.data.view_as(pred))

            

#             accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            

#             train_acc+=accuracy.item() * data.size(0)

#             print(

#                 f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',

#                 end='\r')

        

#         else:

#             model.epochs+=1

        

#             with torch.no_grad():

#                 model.eval()

                

#                 for data,target in valid_loader:

#                     if train_on_gpu:

#                         data,target=data.cuda(),target.cuda()

                        

#                         output = model(data)

                       

#                         loss = criterion(output,target)

                        

#                         valid_loss+=loss.item() * data.size(0)

                        

#                         _,pred = torch.max(output,dim=1)

#                         correct_tensor = pred.eq(target.data.view_as(pred))

#                         accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

                        

#                         valid_acc+=accuracy.item() * data.size(0)

                        

#                 train_loss = train_loss / len(train_loader.dataset)

#                 valid_loss = valid_loss / len(valid_loader.dataset)

                

#                 train_acc = train_acc / len(train_loader.dataset)

#                 valid_acc = valid_acc / len(valid_loader.dataset)

                

#                 history.append([train_loss,valid_loss,train_acc,valid_acc])

                

#                 if (epoch + 1) % print_every == 0:

#                     print(

#                         f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'

#                     )

#                     print(

#                         f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'

#                     )

#                 if valid_loss < valid_loss_min:

#                     # Save model

#                     torch.save(model.state_dict(), save_file_name)

#                     # Track improvement

#                     epochs_no_improve = 0

#                     valid_loss_min = valid_loss

#                     valid_best_acc = valid_acc

#                     best_epoch = epoch

                    

#                 else:

#                     epochs_no_improve += 1

#                     # Trigger early stopping

#                     if epochs_no_improve >= max_epochs_stop:

#                         print(

#                             f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

#                         )

#                         total_time = timer() - overall_start

#                         print(

#                             f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'

#                         )



#                         # Load the best state dict

#                         model.load_state_dict(torch.load(save_file_name))

#                         # Attach the optimizer

#                         model.optimizer = optimizer



#                         # Format history

#                         history = pd.DataFrame(

#                             history,

#                             columns=[

#                                 'train_loss', 'valid_loss', 'train_acc',

#                                 'valid_acc'

#                             ])

#                         return model, history

                    

                    

#     model.optimizer = optimizer

#     total_time = timer() - overall_start

#     print(

#         f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

#     )

#     print(

#         f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'

#     )

#     # Format history

#     history = pd.DataFrame(

#         history,

#         columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

#     return model, history

# save_file_name = 'resnet34-transfer-1.pth'

# checkpoint_path = 'resnet34-transfer-1.pth'

# optimizer = optim.Adam(model.parameters(),lr=1e-04)

# model, history = train(

#     model,

#     criterion,

#     optimizer,

#     dataloader['train'],

#     dataloader['val'],

#     save_file_name=save_file_name,

#     max_epochs_stop=5,

#     n_epochs=40,

#     print_every=1)
# plt.figure(figsize=(8, 6))

# for c in ['train_loss', 'valid_loss']:

#     plt.plot(

#         history[c], label=c)

# plt.legend()

# plt.xlabel('Epoch')

# plt.ylabel('Average Negative Log Likelihood')

# plt.title('Training and Validation Losses')
# save_file_name = 'resnet34-transfer-1.pth'

# model.load_state_dict(torch.load(save_file_name))
# for param in model.parameters():

#   param.requires_grad=True
# save_file_name = 'resnet34-transfer-2.pth'

# checkpoint_path = 'resnet34-transfer-2.pth'

# optimizer = optim.Adam(model.parameters(),lr=1e-03)

# model, history = train(

#     model,

#     criterion,

#     optimizer,

#     dataloader['train'],

#     dataloader['val'],

#     save_file_name=save_file_name,

#     max_epochs_stop=5,

#     n_epochs=40,

#     print_every=1)
# plt.figure(figsize=(8, 6))

# for c in ['train_loss', 'valid_loss']:

#     plt.plot(

#         history[c], label=c)

# plt.legend()

# plt.xlabel('Epoch')

# plt.ylabel('Average Negative Log Likelihood')

# plt.title('Training and Validation Losses')
# plt.figure(figsize=(8, 6))

# for c in ['train_acc', 'valid_acc']:

#     plt.plot(

#         100 * history[c], label=c)

# plt.legend()

# plt.xlabel('Epoch')

# plt.ylabel('Average Accuracy')

# plt.title('Training and Validation Accuracy')
import torch

checkpoint=torch.load("../input/resnet2/resnet34-transfer-2.pth")



model.load_state_dict(checkpoint)

for param in model.parameters():

    param.requies_grad=False

    

model.eval()    

data = {

    'train':

    datasets.ImageFolder(root=train_folder, transform=image_transforms['train']),

    'test':

    datasets.ImageFolder(root=test_folder, transform=image_transforms['test'])

}
model.class_to_idx = data['train'].class_to_idx

model.idx_to_class = {

    idx: class_

    for class_, idx in model.class_to_idx.items()

}



list(model.idx_to_class.items())[:10]
def predict(image_path, model, topk=5):

    """Make a prediction for an image using a trained model



    Params

    --------

        image_path (str): filename of the image

        model (PyTorch model): trained model for inference

        topk (int): number of top predictions to return



    Returns



    """

    real_class = image_path.split('/')[-2]



    # Convert to pytorch tensor

    img_tensor = process_image(image_path)



    # Resize

    if train_on_gpu:

        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()

    else:

        img_tensor = img_tensor.view(1, 3, 224, 224)



    # Set to evaluation

    

    with torch.no_grad():

        model.eval()

        # Model outputs log probabilities

        out = model(img_tensor)

        ps = torch.exp(out)



        # Find the topk predictions

        topk, topclass = ps.topk(topk, dim=1)

        

        # Extract the actual classes and probabilities

        top_classes = [

            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]

        ]

        top_p = topk.cpu().numpy()[0]



        return img_tensor.cpu().squeeze(),top_classes, top_p,real_class
# import cv2

# i=cv2.imread(test_folder+"Chrysler 300 SRT-8 2010"+"/02408.jpg")

# i.shape
# from PIL import Image

# img, top_p, top_classes,_= predict(test_folder+"Chrysler 300 SRT-8 2010"+"/02408.jpg", model)


image_names=list()



for folder in os.listdir(test_folder):

  for img in os.listdir(test_folder+"/"+folder):

    image_names.append(img.split(".")[0])

     
# image_names
# results=list()

# def test_per_class(model, test_loader, criterion, classes):

    

#     total_class = len(classes)



#     test_loss = 0.0

#     class_correct = list(0. for i in range(total_class))

#     class_total = list(0. for i in range(total_class))



#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

#     with torch.

    

#     model.eval()  # prep model for evaluation



#     for data, target in test_loader:

#         # Move input and label tensors to the default device

#         data, target = data.to(device), target.to(device)

#         # forward pass: compute predicted outputs by passing inputs to the model

#         output = model(data)

#         # calculate the loss

#         loss = criterion(output, target)

#         # update test loss

#         test_loss += loss.item() * data.size(0)

#         print(output.shape)

#         # convert output probabilities to predicted class

#         _, pred = torch.max(output, 1)

        

#         results.append(pred)

#         # compare predictions to true label

        

#         # calculate test accuracy for each object class

       

results=list()

def test_per_class(model, test_loader, criterion, classes,topk=5):

    

    total_class = len(classes)



    test_loss = 0.0

    class_correct = list(0. for i in range(total_class))

    class_total = list(0. for i in range(total_class))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    with torch.no_grad():

      model.eval()  # prep model for evaluation



      for data, target in test_loader:

        # Move input and label tensors to the default device

          data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model

          output = model(data)

          

          ps = torch.exp(output)

          

        # calculate the loss

          loss = criterion(output, target)

        # update test loss

        # convert output probabilities to predicted class

          topk, topclass = ps.topk(1, dim=1)

           

          

        # Extract the actual classes and probabilities

          top_classes = [

            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]

        ]

          top_p = topk.cpu().numpy()[0]

          

          results.append(topclass[0][0]) 



        
test_per_class(model,dataloader['test'],criterion,folders)
df = pd.DataFrame({'Id': image_names , 'Predicted': list(map(int,results))} , columns=['Id', 'Predicted'])
df.to_csv("submission.csv",index=False)
pd.read_csv("submission.csv")