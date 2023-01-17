import numpy as np #

import pandas as pd 

import matplotlib.pyplot as plt



import random

import shutil 

import torch

import torchvision

import os

import copy

from tqdm import tqdm

import zipfile



from torchvision import transforms, models





random.seed(6)

np.random.seed(6)

torch.manual_seed(6)

torch.cuda.manual_seed(6)

torch.backends.cudnn.deterministic = True
print(os.listdir("../input/platesv2"))
with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:

   # Extract all the contents of zip file in current directory

   zip_obj.extractall('/kaggle/working/')

    

    

print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
data_root = '/kaggle/working/plates/'

print(os.listdir(data_root))
train_dir = 'train'



class_names = ['cleaned', 'dirty']



for class_name in class_names:

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)



for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):

        dest_dir = os.path.join(train_dir, class_name) 

        shutil.copy(os.path.join(source_dir, file_name),

                    os.path.join(dest_dir, file_name))
aug_lvl_1 = transforms.Compose([              #!!!

    transforms.RandomHorizontalFlip(p=0.75),                                

    transforms.TenCrop(224, vertical_flip=True),

    transforms.Lambda(

        lambda crops: torch.stack([aug_lvl_2(crop) for crop in crops]))

])    





aug_lvl_2 = transforms.Compose([              #!!!            

    transforms.ColorJitter(

        brightness = 0.175,   

        contrast   = 0.175,   

        saturation = 0.195,   

        hue        = (0.1, 0.25)),  

    transforms.RandomRotation(360),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    transforms.Lambda(

        lambda x: x[np.random.permutation(3), :, :])                            

])



train_transforms = transforms.Compose([  

    transforms.Resize((224, 224)),

    transforms.TenCrop(224),

    transforms.Lambda(

        lambda crops: torch.stack([aug_lvl_1(crop) for crop in crops]))

])





"""

Сначала картинка уменьшается до размеров 224x224, затем разбивается

на 10 картинок, а потом каждая из них разбивается еще на 10 картинок в aug_lvl_1.

В aug_lvl_2 рандомно изменяется яркость, контрасность и насыщение картинки

с последующей транформацией в тензор и его нормализацией.

Каждая картинка из train трансформируется в тензор размерности [10, 10, 3, 224, 224]

"""

def create_target_set(variable, size_target, dtype = torch.long): #Эта функия нужна, чтобы "размножить" метки. 



    target = torch.tensor([variable], dtype = dtype, requires_grad=False)

    target = torch.nn.functional.pad(target,

                                     (size_target//2, size_target//2 - 1),

                                     "constant",

                                     variable)

    

    return target



def augmentation(dir_data, dir_transforms, batch_size, shuffle, num_workers):  #Ключевая функция в данном решении. Она позволяет

                                                                               #40 картинок получить 4000 тензоров!

    data = []

    target = []



    #Создание набора данных

    dataset = torchvision.datasets.ImageFolder(dir_data, dir_transforms)



    #Изменение размерностей тензоров и "размножение" меток

    for dt, trgt in tqdm(dataset):

        data.append(dt.resize_(100, 3, 224, 224))

        target.append(create_target_set(trgt, 100))



    #Объединение тензоров

    dtst = list(

        zip(torch.cat(data , dim = 0), torch.cat(target, dim = 0))

        )

    

    #Создание dataloader

    dataloader = torch.utils.data.DataLoader(

        dtst,

        batch_size = batch_size,

        shuffle = shuffle,

        num_workers = num_workers

        )

    

    return dataloader

batch_size = 25



train_dataloader = augmentation(

    train_dir,

    train_transforms,

    batch_size = batch_size,

    shuffle = True,

    num_workers = batch_size

    )

print(len(train_dataloader))

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

def show_input(input_tensor, title=''):

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()

    plt.pause(0.001)





X_batch, y_batch = next(iter(train_dataloader))



for x_item, y_item in zip(X_batch, y_batch):

    show_input(x_item, title=class_names[int(y_item)])

def train_model(model, loss, optimizer, scheduler, num_epochs):

    loss_hist = []

    accuracy_hist = []



    for epoch in range(num_epochs):

        print('Epoch {}/{}:'.format(epoch + 1, num_epochs), flush=True)



        dataloader = train_dataloader

        model.train()  

            

        running_loss = 0.

        running_acc  = 0.



        #Перебор данных

        for inputs, labels in tqdm(dataloader):

            inputs = inputs.to(device)

            labels = labels.to(device)



            optimizer.zero_grad()

            

            #Предсказание

            preds = model(inputs)

            loss_value = loss(preds, labels)

            preds_class = preds.argmax(dim=1)



            #Вычисление и шаг градиента

            loss_value.backward()

            optimizer.step()



            #Статистика

            running_loss += loss_value.item()

            running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()



        scheduler.step() #Переместил данный объект в конец программы согласно рекомендации в документации pytorch



        epoch_loss = running_loss / len(dataloader)

        epoch_acc = running_acc / len(dataloader)



        loss_hist.append(epoch_loss)

        accuracy_hist.append(epoch_acc)



        print('\nLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), flush=True)



    return loss_hist, accuracy_hist, model

model = torchvision.models.vgg13_bn(pretrained=True)



for param in model.parameters():

    param.requires_grad = True  



model.classifier = torch.nn.Sequential(   

    torch.nn.Linear(512 * 7 * 7, 8),      

    torch.nn.ReLU(True),                  

    torch.nn.Dropout(0.5),                

    torch.nn.Linear(8, 8),

    torch.nn.ReLU(True),

    torch.nn.Dropout(0.5),

    torch.nn.Linear(8, 2)

)
#for param in model.parameters():

#    param.requires_grad = False



#model.fc = torch.nn.Linear(model.fc.in_features, 2)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

loss_history = []

accuracy_histoty = []



loss_history, accuracy_histoty, trained_model = train_model(model, loss, optimizer, scheduler, num_epochs = 5);

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 ,7))



ax1.plot(accuracy_histoty)

ax1.set_title('Train Accuracy')

fig.tight_layout()



ax2.plot(loss_history)

ax2.set_title('Train Loss');

fig.tight_layout()
test_dir = 'test'

shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

test_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])





test_dataset = ImageFolderWithPaths(test_dir, test_transforms)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

model.eval()



test_predictions = []

test_img_paths = []

for inputs, labels, paths in tqdm(test_dataloader):

    inputs = inputs.to(device)

    labels = labels.to(device)

    with torch.set_grad_enabled(False):

        preds = trained_model(inputs)

    test_predictions.append(

        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    test_img_paths.extend(paths)

    

test_predictions = np.concatenate(test_predictions)

inputs, labels, paths = next(iter(test_dataloader))



for img, pred in zip(inputs, test_predictions):

    show_input(img, title=pred)

submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

print(submission_df)
#submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test2/unknown/test2/unknown/test2/unknown/', '')

#submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test2', '')

#submission_df['id'] = submission_df['id'].map(lambda x: x.lstrip('aAbBcC').rstrip('aAbBcC'))

submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')

print(submission_df)
submission_df.set_index('id', inplace=True)

submission_df.head(n=6)
submission_df.to_csv('submission.csv')
!rm -rf train val test