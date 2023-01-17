# Libraries

import os

from timeit import default_timer as timer



import random

import matplotlib.pyplot as plt

import numpy



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#PyTorch, of course

import torch

import torch.nn as nn



#We will need torchvision transforms for data augmentation

from torchvision import transforms



### utilities

# tool to print a nice summary of a network, similary to keras' summary

#from torchsummary import summary



# library to do bash-like wildcard expansion

import glob



# others

import numpy as np

import random

from PIL import Image

from IPython.display import display

from tqdm import tqdm_notebook





# a little helper function do directly display a Tensor

def display_tensor(t):

    trans = transforms.ToPILImage()

    display(trans(t))
#Import Pytorch

import torch

import torchvision



#Little commonly used shortcut

import torch.nn as nn



#We need the display function from IPython for Jupyter Notebook/Colab

from IPython.display import display



#A package to make beautiful progress bars :) 

from tqdm import tqdm_notebook



from PIL import Image
#read images

def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        img = Image.open(os.path.join(folder,filename))

        images.append(img)

    return images



train_set =  load_images_from_folder("/kaggle/input/polytech-ds-2019/polytech-ds-2019/training")

val_set =  load_images_from_folder("/kaggle/input/polytech-ds-2019/polytech-ds-2019/validation")

#check

for i in range(20,21):

    display(train_set[i])

    print(train_set[i].filename)

    print(train_set[i].size)
import re



def remwithre(text, there=re.compile(re.escape('_')+'.*')):

    return there.sub('', text)
#read train labels

train_labels=list(train_set)

for i in range(0,len(train_set)):

    train_labels[i]=int(remwithre(train_set[i].filename.replace('/',' ').split()[-1]))
#read validation labels

val_labels=list(val_set)

for i in range(0,len(val_set)):

    val_labels[i]=int(remwithre(val_set[i].filename.replace('/',' ').split()[-1]))
#display(train_labels)
#display(val_labels)
del(train_set)

del(val_set)
#count number of instances for each label

#Get only the label for each image as a list

t = dict([(i, [x for x in train_labels].count(i)) for i in range(0,11)])

v = dict([(i, [x for x in val_labels].count(i)) for i in range(0,11)])



labels = [i+1 for i in range(11)]



x = np.arange(len(labels))  # the label locations

width = 0.4  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, list(t.values()), width, label='Training set')

rects2 = ax.bar(x + width/2, list(v.values()), width, label='Validation set')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Number of instances')

ax.set_title('Number of instances of each class in training and validation sets')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()
#count number of instances for each label

#Get only the label for each image as a list

labels = [x for x in val_labels]



#Use the "count" method from Python lists

for x in range(0,11):

    print("Number of", x, ":", labels.count(x))
class FoodDataset(torch.utils.data.Dataset):

  

    def __init__(self, img_dir, labelz=None, dataset_for_training=False):

    

        super().__init__()



        # store directory names

        self.img_dir = img_dir

    

        # use glob to get all image names

        self.img_names = [x for x in os.listdir(img_dir)]

        

        # store the labels

        self.labels = labelz

        

        # Add augmented data

        if dataset_for_training == True:

            # Train uses data augmentation

            self.transform = transforms.Compose([

                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

                transforms.RandomRotation(degrees=15),

                transforms.ColorJitter(),

                transforms.RandomHorizontalFlip(),

                transforms.CenterCrop(size=224),  # Image net standards

                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406],

                                     [0.229, 0.224, 0.225])  # Imagenet standards

            ])

        else:  

            # Validation and test sets do not use augmentation

            self.transform = transforms.Compose([

                transforms.Resize(size=256),

                transforms.CenterCrop(size=224),

                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406],

                                     [0.229, 0.224, 0.225])  # Imagenet standards

            ])

  

    def __len__(self):

        return len(self.img_names)

    

    def __getitem__(self,i):

        return self._read_img(i)

  

    def _read_img(self, i):

        img = Image.open(self.img_dir + "/" + self.img_names[i])

        

        return self.transform(img),self.labels[i]
food_data_train = FoodDataset("/kaggle/input/polytech-ds-2019/polytech-ds-2019/training",train_labels, dataset_for_training=True)
food_data_train[1]
#food_data_val = FoodDataset("/kaggle/input/polytech-ds-2019/polytech-ds-2019/validation",val_labels,dataset_for_training=True)

food_data_val = FoodDataset("/kaggle/input/polytech-ds-2019/polytech-ds-2019/validation",val_labels,dataset_for_training=False)
food_data_val[1]
display_tensor(food_data_train[2][0])
display_tensor(food_data_val[2][0])
#check

for i in range(20,21):

    display_tensor(food_data_train[i][0])

    print(food_data_train[i][1])

    print(food_data_train[i][0].shape)
cd_train_dl = torch.utils.data.DataLoader(food_data_train, batch_size=20, shuffle=True, num_workers=4)



cd_val_dl = torch.utils.data.DataLoader(food_data_val, batch_size=20, shuffle=True, num_workers=4)
## Print the length of the dataloader

print(len(cd_train_dl))



## Print a batch

#print(next(iter(cd_train_dl)))
 #next(iter(cd_train_dl))[1]
n_classes = 11

def create_model():

    model = torchvision.models.vgg16(pretrained=True)

    n_inputs = model.classifier[6].in_features # get number of features from model

    

    # Freeze model weights

    for param in model.parameters():

        param.requires_grad = False



    # Add on classifier

    model.classifier[6] = nn.Sequential(

        nn.Linear(n_inputs, 256), 

        nn.ReLU(), 

        nn.Dropout(0.4),

        nn.Linear(256, n_classes),                   

        nn.LogSoftmax(dim=1))

    return model

LEARNING_RATE = 0.001
##RE-RUN THIS CODE TO GET A "NEW" NETWORK



## Create an instance of our network

model = create_model()



## Move it to the GPU

model = model.cuda()



# Negative log likelihood loss

criterion = nn.CrossEntropyLoss()



# adam

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
def train(model,

          criterion,

          optimizer,

          train_loader,

          valid_loader,

          save_file_name,

          max_epochs_stop=3,

          n_epochs=20,

          print_every=2):



    # Early stopping intialization

    epochs_no_improvement = 0

    valid_loss_min = np.Inf



    valid_max_acc = 0

    history = []



    # Number of epochs already trained (if using loaded in model weights)

    try:

        print(f'Model has been trained for: {model.epochs} epochs.\n')

    except:

        model.epochs = 0

        print(f'Starting Training from Scratch.\n')



    overall_start = timer()



    # Main loop

    for epoch in range(n_epochs):



        # keep track of training and validation loss each epoch

        train_loss = 0.0

        valid_loss = 0.0



        train_acc = 0

        valid_acc = 0



        # Set to training

        model.train()

        start = timer()



        # Training loop

        for ii, (data, target) in enumerate(train_loader):

            # Tensors to gpu

            

            #if train_on_gpu:

            data, target = data.cuda(), target.cuda()



            # Clear gradients

            optimizer.zero_grad()

            # Predicted outputs are log probabilities

            output = model(data)



            # Loss and backpropagation of gradients

            loss = criterion(output, target)

            loss.backward()



            # Update the parameters

            optimizer.step()



            # Track train loss by multiplying average loss by number of examples in batch

            train_loss += loss.item() * data.size(0)



            # Calculate accuracy by finding max log probability

            _, pred = torch.max(output, dim=1)

            correct_tensor = pred.eq(target.data.view_as(pred))

            # Need to convert correct tensor from int to float to average

            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiply average accuracy times the number of examples in batch

            train_acc += accuracy.item() * data.size(0)



            # Track training progress

            print(

                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',

                end='\r')



        # After training loops ends, start validation

        else:

            model.epochs += 1



            # Don't need to keep track of gradients

            with torch.no_grad():

                # Set to evaluation mode

                model.eval()



                # Validation loop

                for ii, (data, target) in enumerate(valid_loader):

                    # Tensors to gpu

                    data = data.cuda()

                    target = target.cuda()



                    # Forward pass

                    output = model(data)



                    # Validation loss

                    loss = criterion(output, target)

                    # Multiply average loss times the number of examples in batch

                    valid_loss += loss.item() * data.size(0)



                    # Calculate validation accuracy

                    _, pred = torch.max(output, dim=1)

                    correct_tensor = pred.eq(target.data.view_as(pred))

                    accuracy = torch.mean(

                        correct_tensor.type(torch.FloatTensor))

                    # Multiply average accuracy times the number of examples

                    valid_acc += accuracy.item() * data.size(0)



                # Calculate average losses

                train_loss = train_loss / len(train_loader.dataset)

                valid_loss = valid_loss / len(valid_loader.dataset)



                # Calculate average accuracy

                train_acc = train_acc / len(train_loader.dataset)

                valid_acc = valid_acc / len(valid_loader.dataset)



                history.append([train_loss, valid_loss, train_acc, valid_acc])



                # Print training and validation results

                if (epoch + 1) % print_every == 0:

                    print(

                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'

                    )

                    print(

                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'

                    )



                # Save the model if validation loss decreases

                if valid_loss < valid_loss_min:

                    # Save model

                    torch.save(model.state_dict(), save_file_name)

                    # Track improvement

                    epochs_no_improvement = 0

                    valid_loss_min = valid_loss

                    valid_best_acc = valid_acc

                    best_epoch = epoch



                # Otherwise increment count of epochs with no improvement

                else:

                    epochs_no_improvement += 1

                    # Trigger early stopping

                    if epochs_no_improvement >= max_epochs_stop:

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



    # Attach the optimizer

    model.optimizer = optimizer

    # Record overall time and print out stats

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
save_file_name = 'vgg16-transfer-4.pt'

checkpoint_path = 'vgg16-transfer-4.pth'



model, history = train(

    model,

    criterion,

    optimizer,

    cd_train_dl,

    cd_val_dl,

    save_file_name=save_file_name,

    max_epochs_stop=13,

    n_epochs=13,

    print_every=1)
class TestFoodDataset(torch.utils.data.Dataset):

  

    def __init__(self, img_dir):

    

        super().__init__()

    

        # store directory names

        self.img_dir = img_dir

    

        # use glob to get all image names

        self.img_names = [x for x in os.listdir(img_dir)]

        

        # PyTorch transforms

        self.transform = transforms.Compose([torchvision.transforms.Resize((299,299)),transforms.ToTensor()])

  

    def __len__(self):

        return len(self.img_names)

    

    def __getitem__(self,i):

        return self._read_img(i)

  

    def _read_img(self, i):

        img = Image.open(self.img_dir + "/" + self.img_names[i])

        return self.transform(img),self.img_names[i].replace('.jpg','')
food_data_test = TestFoodDataset("/kaggle/input/polytech-ds-2019/polytech-ds-2019/kaggle_evaluation")
len(food_data_test)

print(food_data_test[0])
cd_test_dl = torch.utils.data.DataLoader(food_data_test, batch_size=1, shuffle=False, num_workers=4)
# Evaluation loop

model.eval()

prediction=[]

img_codes=[]

with torch.no_grad():

    for data in cd_test_dl:

        imagess, img_code = data

        images=imagess.cuda()

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        #print(predicted)

        prediction.append(predicted)

        img_codes.append(img_code)
len(prediction)
len(img_codes)
type(prediction[0])
print(img_codes[0][0])
print(prediction[0].cpu().detach().numpy())

#print(img_codes[0][0])
#Id=np.array(img_codes)

#Category=np.array(prediction)

Id=np.empty(len(img_codes),dtype=np.int)

Category=np.empty(len(img_codes),dtype=np.int)

#index=np.empty(len(img_codes),dtype=np.int)

for i in range(0,len(img_codes)):

    Id[i]=np.int(img_codes[i][0])

    Category[i]=np.int(np.int(prediction[i].cpu().detach().numpy()))

    #index[i]=np.int(i+1)
matrix=np.column_stack((Id,Category))
for i in range (0,2):

    print(type(matrix[0][i]))
matrix[0:9]
import csv

headers=['Id','Category']



submission = open("/kaggle/working/submissionX.csv",'w')

wr = csv.writer(submission,delimiter=',')

wr.writerow(headers)

for item in matrix:

    wr.writerow(item)

submission.close()

!less /kaggle/working/submissionX.csv | head -n 10
!sed -i 's/,/, /g'  /kaggle/working/submissionX.csv

!sed -i 's/Id, /Id,/g'  /kaggle/working/submissionX.csv
import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from IPython.display import FileLink

FileLink('/kaggle/working/submissionX.csv')