import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



import zipfile

with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:

   zip_obj.extractall('/kaggle/working/')

    

print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
import shutil

from tqdm import tqdm

from IPython.display import FileLink



import torch

import torchvision 

import matplotlib.pyplot as plt

%matplotlib inline

import copy



from torchvision import transforms,models



!pip install git+https://github.com/aleju/imgaug

from imgaug import augmenters as iaa

import imgaug as ia



import PIL

import matplotlib as mpl



import warnings

warnings.filterwarnings("ignore")
data_root = '/kaggle/working/plates/'

print(os.listdir(data_root))
train_dir = 'train'

val_dir = 'val'



class_names = ['cleaned', 'dirty']



for dir_name in [train_dir, val_dir]:

    for class_name in class_names:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):

        if i % 6 != 0:

            dest_dir = os.path.join(train_dir, class_name) 

        else:

            dest_dir = os.path.join(val_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
mpl.rcParams['axes.grid'] = False

mpl.rcParams['image.interpolation'] = 'nearest'

mpl.rcParams['figure.figsize'] = 45, 75



mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])



def show_dataset(dataset, n=6):

    img = np.vstack((np.hstack((np.asarray(dataset[i][0].permute(1, 2, 0).numpy() * std + mean )for _ in range(n)))

                   for i in range(len(dataset))))

    

    plt.imshow(img)

    plt.axis('off')
class ImgAugTransform: 

  def __init__(self):

    self.aug = iaa.Sequential([

        #iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),

        #iaa.Fliplr(0.5),

        #iaa.Affine(rotate=(-20, 20), mode='symmetric'),

        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),

        iaa.MultiplyHue(mul = (-1,1)),

        #iaa.ChannelShuffle(p = 0.5),

    ])

      

  def __call__(self, img):

    img = np.array(img)

    return self.aug.augment_image(img)



img_transforms = ImgAugTransform()
train_transforms = torchvision.transforms.Compose([

    transforms.RandomChoice(transforms = [transforms.RandomRotation(degrees = 60),transforms.RandomRotation(degrees = 90)]),

    transforms.RandomChoice(transforms = [transforms.CenterCrop((224,224)),transforms.RandomCrop((224,224))]),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



train_transforms_aug = torchvision.transforms.Compose([

    ImgAugTransform(),

    lambda x: PIL.Image.fromarray(x),

    transforms.RandomChoice(transforms = [transforms.RandomRotation(degrees = 60),transforms.RandomRotation(degrees = 90)]),

    transforms.RandomChoice(transforms = [transforms.CenterCrop((224,224)),transforms.RandomCrop((224,224))]),

    transforms.RandomChoice(transforms = [transforms.RandomHorizontalFlip()]),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])







val_transforms = torchvision.transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
train_dataset = torchvision.datasets.ImageFolder(train_dir,train_transforms)

show_dataset(train_dataset)
train_dataset = torchvision.datasets.ImageFolder(train_dir,train_transforms_aug)



val_dataset = torchvision.datasets.ImageFolder(val_dir,val_transforms)



show_dataset(train_dataset)
batch_size = 8



train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,

                                               num_workers = 0,shuffle = True)

val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size = batch_size,

                                             num_workers = 0,shuffle = False)
def train_model(model, loss, optimizer, scheduler, num_epochs,early_stop):

    

    loss_history = []

    acc_history = []

    

    best_acc = 0.

    best_loss = 1000000

    

    best_acc_val = 0.

    best_loss_val = 1000000

    

    improve_count = 0

    for epoch in range(num_epochs):

        print('\nEpoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

                scheduler.step()

                model.train()  

            else:

                dataloader = val_dataloader

                model.eval()



            running_loss = 0.

            running_acc = 0.

            running_clean = 0.

            running_recall = 0.



            for inputs, labels in dataloader:

                inputs = inputs.to(device)

                labels = labels.to(device)



                

                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)

                    loss_value = loss(preds, labels)

                    preds_class = preds.argmax(dim=1)



                    if phase == 'train':

                        loss_value.backward()

                        optimizer.step()



                running_loss += loss_value.item()

                running_acc += (preds_class == labels.data).float().mean()

                if sum(labels.data) > 0:

                    running_recall += (preds_class[labels.data == 1] == 1).float().mean()



            epoch_loss = running_loss / len(dataloader)

            epoch_acc = running_acc / len(dataloader)

            epoch_recall = running_recall / len(dataloader)

            

            if phase == 'train':

                

                improve_count+=1 

                loss_history.append(epoch_loss)

                acc_history.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_recall), flush=True,end = ' ')

            

            if(phase == 'train' and best_loss>epoch_loss and epoch_recall <0.96):

                

                improve_count = 0

                

                best_loss = epoch_loss    

                best_acc = epoch_acc

                

                best_model_train = copy.deepcopy(model)

                save_epoch_train = epoch

                

                print('| save')

                

            elif(phase == 'val' and best_loss_val>epoch_loss):

                

                best_loss_val = epoch_loss    

                best_acc_val = epoch_acc

                

                best_model_val = copy.deepcopy(model)

                save_epoch_val = epoch

                

                print('| save')

                

            elif (phase == 'val' and best_acc_val == epoch_acc and best_loss_val > epoch_loss):

                

                best_loss_val = epoch_loss    

                best_acc_val = epoch_acc

                

                best_model_val = copy.deepcopy(model)

                save_epoch_val = epoch

                print('| save')

            

            else: print('')

            

            if(phase == 'val' and improve_count == early_stop):

                print('\nLoss does not decrease {} epochs, learning is stopped'.format(early_stop))

                

                torch.save(best_model_train.state_dict(), "best_model_train.pth")

                torch.save(best_model_val.state_dict(), "best_model_val.pth")

                

                print('\n Saved model from the {}th epoch with the best loss on train'.format(save_epoch_train))

                print('\n Saved model from the {}th epoch with the best loss on val'.format(save_epoch_val))

                

                return loss_history,acc_history

            

    print('\nSaved model from the {}th epoch with the best loss on train'.format(save_epoch_train))

    print('\nSaved model from the {}th epoch with the best loss on val'.format(save_epoch_val))

    

    return loss_history,acc_history
vgg16 = models.vgg16(pretrained=True)



for param in vgg16.features.parameters(): # we train the weights only of the last layers

    param.requires_grad = False



vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, 2)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg16 = vgg16.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(vgg16.parameters(), lr=1.0e-2)



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
train_model(vgg16, loss, optimizer, scheduler, num_epochs=40,early_stop = 10);
model = models.vgg16(pretrained=False)



model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)



#model_keys = torch.load('best_model_train.pth')

model_keys = torch.load('/kaggle/input/vgg-models/vgg_91.pth') # i load the model that gave the best result in public leaderboard



model.load_state_dict(model_keys)



model = model.to(device)
# FileLink(r'best_model_train.pth')

# FileLink(r'best_model_val.pth')
test_dir = 'test'

shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

    

test_dataset = ImageFolderWithPaths('/kaggle/working/test', val_transforms)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
model.eval()

test_predictions = []

test_img_paths = []

for inputs, labels, paths in tqdm(test_dataloader):

    inputs, labels = inputs.to(device), labels.to(device)

    with torch.set_grad_enabled(False):

        preds = model(inputs)

    test_predictions.append(

        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    test_img_paths.extend(paths)

    

test_predictions = np.concatenate(test_predictions)
submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})

submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')

submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

submission_df.set_index('id', inplace=True)

submission_df.head(n=6)
submission_df.label.map({'dirty':1,'cleaned':0}).mean()
submission_df.to_csv('submission.csv')
!rm -rf train val test 