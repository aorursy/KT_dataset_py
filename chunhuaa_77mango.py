import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data

import torch.nn.functional as F

import torchvision

import torchvision.models as models

from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt
model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

model_resnet34 = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
for name, param in model_resnet18.named_parameters():

    if("bn" not in name):

        param.requires_grad = False

        

for name, param in model_resnet34.named_parameters():

    if("bn" not in name):

        param.requires_grad = False
num_classes = 3



model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,512),

                                  nn.ReLU(),

                                  nn.Dropout(),

                                  nn.Linear(512, num_classes))



model_resnet34.fc = nn.Sequential(nn.Linear(model_resnet34.fc.in_features,512),

                                  nn.ReLU(),

                                  nn.Dropout(),

                                  nn.Linear(512, num_classes))
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05,verbose=True)

    for epoch in range(epochs):

        training_loss = 0.0

        valid_loss = 0.0

        model.train()

        for batch in train_loader:

            optimizer.zero_grad()

            inputs, targets = batch

            inputs = inputs.to(device)

            targets = targets.to(device)

            output = model(inputs)

            loss = loss_fn(output, targets)

            loss.backward()

            optimizer.step()

            training_loss += loss.data.item() * inputs.size(0)

        training_loss /= len(train_loader.dataset)

        

        model.eval()

        num_correct = 0 

        num_examples = 0

        for batch in val_loader:

            inputs, targets = batch

            inputs = inputs.to(device)

            output = model(inputs)

            targets = targets.to(device)

            loss = loss_fn(output,targets) 

            valid_loss += loss.data.item() * inputs.size(0)

                        

            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)

            num_correct += torch.sum(correct).item()

            num_examples += correct.shape[0]

            scheduler.step(training_loss)

            

        valid_loss /= len(val_loader.dataset)

        

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,

        valid_loss, num_correct / num_examples))

        
import os

import pandas as pd

import shutil

path = "/kaggle/working/abc_train"

os.mkdir(path)

path = "/kaggle/working/abc_dev"

os.mkdir(path)

path = "/kaggle/working/abc_train/A"

os.mkdir(path)

path = "/kaggle/working/abc_train/B"

os.mkdir(path)

path = "/kaggle/working/abc_train/C"

os.mkdir(path)

path = "/kaggle/working/abc_dev/A"

os.mkdir(path)

path = "/kaggle/working/abc_dev/B"

os.mkdir(path)

path = "/kaggle/working/abc_dev/C"

os.mkdir(path)
df = pd.read_csv('../input/mangooo/mango/train.csv')  

for i in range(df.shape[0]):

    old='../input/mangooo/mango/C1-P1_Train Dev_fixed/C1-P1_Train/'+df["image_id"][i]

    new='/kaggle/working/abc_train/'+df["label"][i]

    shutil.copy(old,new)
df = pd.read_csv('../input/mangooo/mango/dev.csv')  

for i in range(df.shape[0]):

    old='../input/mangooo/mango/C1-P1_Train Dev_fixed/C1-P1_Dev/'+df["image_id"][i]

    new='/kaggle/working/abc_dev/'+df["label"][i]

    shutil.copy(old,new)
batch_size=32

img_dimensions = 224



# Normalize to the ImageNet mean and standard deviation

# Could calculate it for the cats/dogs data set, but the ImageNet

# values give acceptable results here.

img_transforms = transforms.Compose([

    transforms.Resize((img_dimensions, img_dimensions)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )

    ])



img_test_transforms = transforms.Compose([

    transforms.Resize((img_dimensions,img_dimensions)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )

    ])



def check_image(path):

    try:

        im = Image.open(path)

        return True

    except:

        return False



train_data_path = "/kaggle/working/abc_train/"

train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)



validation_data_path = "/kaggle/working/abc_dev/"

validation_data = torchvision.datasets.ImageFolder(root=validation_data_path,transform=img_test_transforms, is_valid_file=check_image)



test_data_path = "/kaggle/working/abc_dev/"

test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_test_transforms, is_valid_file=check_image)



num_workers = 6

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)





if torch.cuda.is_available():

    device = torch.device("cuda") 

else:

    device = torch.device("cpu")
print(f'Num training images: {len(train_data_loader.dataset)}')

print(f'Num validation images: {len(validation_data_loader.dataset)}')

print(f'Num test images: {len(test_data_loader.dataset)}')
def test_model(model):

    correct = 0

    total = 0

    with torch.no_grad():

        for data in test_data_loader:

            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('correct: {:d}  total: {:d}'.format(correct, total))

    print('accuracy = {:f}'.format(correct / total))
model_resnet18.to(device)

optimizer = optim.Adam(model_resnet18.parameters(), lr=0.001)



train(model_resnet18, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=20, device=device)
test_model(model_resnet18)
#model_resnet18.to(device)

#optimizer = optim.Adagrad(model_resnet18.parameters(),lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

#train(model_resnet18, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=2, device=device)



model_resnet34.to(device)

optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)

train(model_resnet34, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=5, device=device)
test_model(model_resnet34)
import os

def find_classes(dir):

    classes = os.listdir(dir)

    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx



def make_prediction(model, filename):

    labels, _ = find_classes('/kaggle/working/abc_dev')

    img = Image.open(filename)

    img = img_test_transforms(img)

    img = img.unsqueeze(0)

    prediction = model(img.to(device))

    prediction = prediction.argmax()

    print(labels[prediction])

    return(labels[prediction])

    

make_prediction(model_resnet18, '/kaggle/working/abc_dev/A/00060.jpg')

make_prediction(model_resnet18, '/kaggle/working/abc_dev/B/00027.jpg')

make_prediction(model_resnet18, '/kaggle/working/abc_dev/C/00051.jpg')
df1 = pd.read_csv('../input/test-mango-c/test_example.csv')  

for i in range(1600):

    p='../input/mangooo/mango/C1-P1_Test/C1-P1_Test/'

    df1['label'][i]=make_prediction(model_resnet18, p+df1['image_id'][i])
df1
df1.to_csv("test_example.csv")
df1 = pd.read_csv('../input/test-mango-c/test_example.csv')  

for i in range(1600):

    p='../input/mangooo/mango/C1-P1_Test/C1-P1_Test/'

    df1['label'][i]=make_prediction(model_resnet34, p+df1['image_id'][i])
df1.to_csv("test_example34.csv")
'''torch.save(model_resnet18.state_dict(), "./model_resnet18.pth")

torch.save(model_resnet34.state_dict(), "./model_resnet34.pth")





# Remember that you must call model.eval() to set dropout and batch normalization layers to

# evaluation mode before running inference. Failing to do this will yield inconsistent inference results.



resnet18 = torch.hub.load('pytorch/vision', 'resnet18')

resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))

resnet18.load_state_dict(torch.load('./model_resnet18.pth'))

resnet18.eval()



resnet34 = torch.hub.load('pytorch/vision', 'resnet34')

resnet34.fc = nn.Sequential(nn.Linear(resnet34.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))

resnet34.load_state_dict(torch.load('./model_resnet34.pth'))

resnet34.eval()

'''
# Test against the average of each prediction from the two models

'''

models_ensemble = [resnet18.to(device), resnet34.to(device)]

correct = 0

total = 0

with torch.no_grad():

    for data in test_data_loader:

        images, labels = data[0].to(device), data[1].to(device)

        predictions = [i(images).data for i in models_ensemble]

        print(predictions)

        avg_predictions = torch.mean(torch.stack(predictions), dim=0)

        _, predicted = torch.max(avg_predictions, 1)



        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        

print('accuracy = {:f}'.format(correct / total))

print('correct: {:d}  total: {:d}'.format(correct, total))

'''
#models_ensemble
