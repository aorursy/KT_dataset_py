# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test set/test set"))

import torch

# Any results you write to the current directory are saved as output.
test_dir="../input/test set/test set/"

path="../input/flower_data/flower_data/"

train_dir=path+"train/"

validation_dir=path+"valid/"
data_size=list()

classes=list()

sizes=list()
for direc in os.listdir(train_dir):

    data_size.append({direc:len(os.listdir(train_dir+direc))})

    classes.append(direc)

    sizes.append(len(os.listdir(train_dir+direc)))

    
import json

data=json.loads(open("../input/cat_to_name.json").read())

names=data.values()
import matplotlib.pyplot as plt

plt.bar(names,sizes)
names=list(names)

names
cat_df=pd.DataFrame({"category":names,"train_data":sizes}).sort_values("category")
cat_df.head()
cat_df.set_index("category")['train_data'].plot.bar(color="r",figsize=(20,6))
max(cat_df['train_data'])
sum(sizes)
classes=list()

train_imgs=list()

for direc in os.listdir(train_dir):

    for img in os.listdir(train_dir+direc):

        classes.append(direc)

        train_imgs.append(train_dir+direc+"/"+img)
full_dataset=list()

classes=pd.Series(classes)

images=pd.Series(train_imgs)

    
full_data=pd.concat([classes,images],axis=1)
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

     'valid':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    

}

     

  

             
path
test_dir="../input/test set/"
from torchvision import datasets

image_datasets={x:datasets.ImageFolder(os.path.join(path,x),image_transforms[x]) for x in ['train','valid']}
image_datasets['test']=datasets.ImageFolder(test_dir,image_transforms['test'])
len(image_datasets['valid'])
from torchvision import datasets

from torch.utils.data import DataLoader



dataloader={

    'train':DataLoader(image_datasets['train'],batch_size=32,shuffle=True,drop_last=True),

    'valid':DataLoader(image_datasets['valid'],batch_size=32,shuffle=True,drop_last=True),

    'test' : DataLoader(image_datasets['test'],batch_size=32,shuffle=True)

    

}
trainiter = iter(dataloader['valid'])

features, labels = next(trainiter)

features.shape, labels.shape
trainiter = iter(dataloader['test'])

features, labels = next(trainiter)

features.shape, labels.shape
from torchvision import models

model = models.resnet34(pretrained=True)
for param in model.parameters():

    param.requires_grad=False
import torch.nn as nn

in_features=512

out_classes=102

model.fc = nn.Sequential(nn.Linear(in_features,256),

                         nn.ReLU(),

                         nn.Dropout(0.5),

                         nn.Linear(256,out_classes),

                         nn.LogSoftmax(dim=1))



model = model.to("cuda")

model = nn.DataParallel(model)
from torch import optim

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(),lr=1e-03)
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
from timeit import default_timer as timer

import torch

def train(model,criterion,optimizer,train_loader,valid_loader,save_file_name,max_epochs_stop=3,n_epochs=20

         ,print_every=2):

    

    

    epochs_no_improve=0

    valid_loss_min = np.Inf

    

    valid_max_acc=0

    history=[]

    

    try:

        print(f'Model has been trained for: {model.epochs} epochs \n')

    except:

        model.epochs=0

        print(f'Training from scratch\n')

    

    overall_start=timer()

    

    for epoch in range(n_epochs):

        train_loss = 0.0

        valid_loss = 0.0

        

        train_acc = 0.0

        valid_acc = 0.0

        

        model.train()

        start=timer()

        

        #training loop

        for ii,(data,target) in enumerate(train_loader):

            

            if train_on_gpu:

                data,target=data.cuda(),target.cuda()

                

            optimizer.zero_grad()

            

            output = model(data)

          

            loss = criterion(output,target)

            loss.backward()

            

            optimizer.step()

            

            train_loss+=loss.item() * data.size(0)

            

            _,pred = torch.max(output,dim=1)

            correct_tensor = pred.eq(target.data.view_as(pred))

            

            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            

            train_acc+=accuracy.item() * data.size(0)

            print(

                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',

                end='\r')

        

        else:

            model.epochs+=1

        

            with torch.no_grad():

                model.eval()

                

                for data,target in valid_loader:

                    if train_on_gpu:

                        data,target=data.cuda(),target.cuda()

                        

                        output = model(data)

                       

                        loss = criterion(output,target)

                        

                        valid_loss+=loss.item() * data.size(0)

                        

                        _,pred = torch.max(output,dim=1)

                        correct_tensor = pred.eq(target.data.view_as(pred))

                        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

                        

                        valid_acc+=accuracy.item() * data.size(0)

                        

                train_loss = train_loss / len(train_loader.dataset)

                valid_loss = valid_loss / len(valid_loader.dataset)

                

                train_acc = train_acc / len(train_loader.dataset)

                valid_acc = valid_acc / len(valid_loader.dataset)

                

                history.append([train_loss,valid_loss,train_acc,valid_acc])

                

                if (epoch + 1) % print_every == 0:

                    print(

                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'

                    )

                    print(

                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'

                    )

                if valid_loss < valid_loss_min:

                    # Save model

                    torch.save(model.state_dict(), save_file_name)

                    # Track improvement

                    epochs_no_improve = 0

                    valid_loss_min = valid_loss

                    valid_best_acc = valid_acc

                    best_epoch = epoch

                    

                else:

                    epochs_no_improve += 1

                    # Trigger early stopping

                    if epochs_no_improve >= max_epochs_stop:

                        print(

                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

                        )

                        total_time = timer() - overall_start

                        print(

                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'

                        )



                        # Load the best state dict

                        model.load_state_dict(torch.load(save_file_name))

                        # Attach the optimizer

                        model.optimizer = optimizer



                        # Format history

                        history = pd.DataFrame(

                            history,

                            columns=[

                                'train_loss', 'valid_loss', 'train_acc',

                                'valid_acc'

                            ])

                        return model, history

                    

                    

    model.optimizer = optimizer

    total_time = timer() - overall_start

    print(

        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

    )

    print(

        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'

    )

    # Format history

    history = pd.DataFrame(

        history,

        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    return model, history

save_file_name = 'resnet34-transfer-1.pth'

checkpoint_path = 'resnet34-transfer-1.pth'



model, history = train(

    model,

    criterion,

    optimizer,

    dataloader['train'],

    dataloader['valid'],

    save_file_name=save_file_name,

    max_epochs_stop=5,

    n_epochs=20,

    print_every=1)
plt.figure(figsize=(8, 6))

for c in ['train_loss', 'valid_loss']:

    plt.plot(

        history[c], label=c)

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Average Negative Log Likelihood')

plt.title('Training and Validation Losses')
plt.figure(figsize=(8, 6))

for c in ['train_acc', 'valid_acc']:

    plt.plot(

        100 * history[c], label=c)

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Average Accuracy')

plt.title('Training and Validation Accuracy')
# model.load_state_dict(torch.load(save_file_name))
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



        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class
train_dir
# def evaluate(model, test_loader, criterion, topk=(1, 5)):

#     """Measure the performance of a trained PyTorch model



#     Params

#     --------

#         model (PyTorch model): trained cnn for inference

#         test_loader (PyTorch DataLoader): test dataloader

#         topk (tuple of ints): accuracy to measure



#     Returns

#     --------

#         results (DataFrame): results for each category



#     """



#     classes = []

#     losses = []

#     # Hold accuracy results

#     acc_results = np.zeros((len(test_loader.dataset), len(topk)))

#     i = 0



#     model.eval()

#     with torch.no_grad():



#         # Testing loop

#         for data, targets in test_loader:



#             # Tensors to gpu

#             if train_on_gpu:

#                 data, targets = data.to('cuda'), targets.to('cuda')



#             # Raw model output

#             out = model(data)

#             # Iterate through each example

#             for pred, true in zip(out, targets):

#                 # Find topk accuracy

#                 acc_results[i, :] = accuracy(

#                     pred.unsqueeze(0), true.unsqueeze(0), topk)

#                 classes.append(model.idx_to_class[true.item()])

#                 # Calculate the loss

#                 loss = criterion(pred.view(1, n_classes), true.view(1))

#                 losses.append(loss.item())

#                 i += 1



#     # Send results to a dataframe and calculate average across classes

#     results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])

#     results['class'] = classes

#     results['loss'] = losses

#     results = results.groupby(classes).mean()



#     return results.reset_index().rename(columns={'index': 'class'})
data = {

    'train':

    datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),

    'test':

    datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])

}
model.class_to_idx = data['train'].class_to_idx

model.idx_to_class = {

    idx: class_

    for class_, idx in model.class_to_idx.items()

}



list(model.idx_to_class.items())[:10]
dataloader
test_dir='../input/test set/'
test_dir
model.load_state_dict(torch.load(save_file_name))
img=test_dir+"test set/gc8.jpg"

img=plt.imread(img)

s=img.shape

t=torch.tensor(img).view((1,s))

print(t)
def process_image(image_path):

    """Process an image path into a PyTorch tensor"""



    image = Image.open(image_path)

    # Resize

    img = image.resize((256, 256))

    if img.mode == 'RGBA':

        rgb_im=img.convert('RGB')

        

        img = rgb_im

        

        

    # Center crop

    width = 256

    height = 256

    new_width = 224

    new_height = 224



    left = (width - new_width) / 2

    top = (height - new_height) / 2

    right = (width + new_width) / 2

    bottom = (height + new_height) / 2

    img = img.crop((left, top, right, bottom))

    

    # Convert to numpy, transpose color dimension and normalize

    img = np.array(img).transpose((2, 0, 1)) / 256



    # Standardization

    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))

    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    

    img = img - means

    img = img / stds



    img_tensor = torch.Tensor(img)



    return img_tensor
from PIL import Image

x = process_image(test_dir+"test set/gc8.jpg")


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
img, top_p, top_classes,_= predict(test_dir+"test set"+"/"+"gc11.png", model)
results=list()



for img in os.listdir(test_dir+"test set"):

    

   try:

    imag, top_p, top_classes,_= predict(test_dir+"test set"+"/"+img, model)

    

    results.append(top_p[0])

    

   except:

    

    print(img)

   
image_names=os.listdir(test_dir+"test set")
import json

category_labels=json.loads(open("../input/cat_to_name.json").read())
results
cat_results=list()

for i in results:

    cat_results.append(category_labels[i])
cat_results
df = pd.DataFrame({'image_name': image_names , 'flower_id': results,'flower_name' : cat_results}, columns=['image_name', 'flower_id', 'flower_name']).sort_values('image_name',ascending=True)
df.to_csv('submission.csv',index=False)
x = process_image(test_dir+"test set/gc8.jpg")
pd.read_csv("submission.csv")
plt.imshow(plt.imread(test_dir+"test set/gc11.png"))