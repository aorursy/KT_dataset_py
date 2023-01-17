import matplotlib.pyplot as plt

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.models as models

import torchvision.transforms as transforms



import time

import os

import PIL.Image as Image

from IPython.display import display
# Configuration



DEBUG = False



# training

EPOCHS = 16

MOMENTUM = 0.9

LEARNING_RATE = 0.01

BATCH_SIZE = 4

THREADS = 0



USE_CUDA=False



# file paths

IMAGES_PATH = "../input/hyperviddataset"

TRAINING_PATH = "/kaggle/working/train_file.csv"

VALIDATION_PATH = "/kaggle/working/test_file.csv"

TEST_PATH = "/kaggle/working/test_file.csv"

MEAN_STD_PATH="/kaggle/working/mean_devstd.txt"

RESULTS_PATH="/kaggle/working/results"





import torch

import time

import os

import PIL.Image as Image

from IPython.display import display



device = torch.device("cpu",index=0)



print(device)



print(torch.device(0))
import numpy as np

import torch

import os



from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms



from PIL import Image

from os import path

from glob import glob

import random

print(IMAGES_PATH)



dirs = glob(IMAGES_PATH + "/*/")



#print("dirs:"+str(dirs))



num_classes = {}



i = 0

for d in dirs:

    d = d.replace(IMAGES_PATH, "")

    d = d.replace("/", "")

    if " " in d:

        d = d.replace(" ", "_")

    num_classes[d] = i

    i+=1



print ("Classes: ")

print (num_classes)

print ("")



# read mean and dev. standard pre-computed

m = 0

s = 0

if os.path.isfile(MEAN_STD_PATH):

    m_s = open(MEAN_STD_PATH, "r").read()

    if "," in m_s:

        m_s = m_s.replace("\n", "")

        m_s = m_s.replace("tensor", "")

        m_s = m_s.replace("(", "")

        m_s = m_s.replace(")", "")

        m_s = m_s.split(",")

        m = torch.Tensor( [float(m_s[0]), float(m_s[1]), float(m_s[2])] )

        s = torch.Tensor( [float(m_s[3]), float(m_s[4]), float(m_s[5])] )



def get_class(idx):

    #print (num_classes)

    for key in num_classes:

        if idx == num_classes[key]:

            return key



def preprocessing():

    train_csv = ""

    test_csv  = ""

    train_csv_supp = []

    test_csv_supp = []

    class_files_training = []

    class_files_testing  = []



    for key in num_classes:

        if " " in key:

            os.rename(IMAGES_PATH+"/"+key, IMAGES_PATH+"/"+key.replace(" ", "_"))

            key = key.replace(" ", "_")



        class_files = glob(IMAGES_PATH+"/"+str(key)+"/*")

        class_files = [w.replace(IMAGES_PATH+"/"+str(key)+"/", "") for w in class_files]

        class_files.sort()



        class_files_training = class_files[: int(len(class_files)*.66)] # get 66% class images fo training

        class_files_testing = class_files[int(len(class_files)*.66)+1 :] # get 33% class images fo testing



        for f in class_files_training:

            if "," in f or "#" in f or " " in f:

                tmp_f = f.replace(",", "")

                tmp_f = tmp_f.replace("#", "")

                tmp_f = tmp_f.replace(" ", "_")

                os.rename(IMAGES_PATH+"/"+key+"/"+f, IMAGES_PATH+"/"+key+"/"+tmp_f)

                f = tmp_f

            train_csv_supp.append(f + ","+str(key))



        for f in class_files_testing:

            if "," in f or "#" in f or " " in f:

                tmp_f = f.replace(",", "")

                tmp_f = tmp_f.replace("#", "")

                tmp_f = tmp_f.replace(" ", "_")

                os.rename(IMAGES_PATH+"/"+key+"/"+f, IMAGES_PATH+"/"+key+"/"+tmp_f)

                f = tmp_f

            test_csv_supp.append(f + ","+str(key))



    random.shuffle(train_csv_supp)

    random.shuffle(test_csv_supp)



    for t in train_csv_supp:

        train_csv += t + "\n"

    

    for t in test_csv_supp:

        test_csv += t + "\n"



    train_csv_file = open(TRAINING_PATH, "w+")

    train_csv_file.write(train_csv)

    train_csv_file.close()



    test_csv_file = open(VALIDATION_PATH, "w+")

    test_csv_file.write(test_csv)

    test_csv_file.close()



    # Algorithms to calculate mean and standard_deviation

    print("Loading dataset...")

    dataset = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=transforms.ToTensor())

    print("Calculating mean & dev std...")

    

    m = torch.zeros(3) # Mean

    s = torch.zeros(3) # Standard Deviation

    for sample in dataset:

        image,label=sample

        m += image.sum(1).sum(1)

        s += ((image-m.view(3,1,1))**2).sum(1).sum(1)

    m /= len(dataset)*256*144    

    s = torch.sqrt(s/(len(dataset)*256*144))



    print("Calculated mean and standard deviation!")

    str_m = str(m[0])+","+str(m[1])+","+str(m[2])

    str_s = str(s[0])+","+str(s[1])+","+str(s[2])

    file = open(MEAN_STD_PATH, "w+")

    file.write(str(str_m)+","+str(str_s))

    file.close()

#preprocessing()

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')





def pil_loader(path):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, 'rb') as f:

        img = Image.open(f)

        return img.convert('RGB')





def accimage_loader(path):

    import accimage

    try:

        return accimage.Image(path)

    except IOError:

        # Potentially a decoding problem, fall back to PIL.Image

        return pil_loader(path)





def default_loader(path):

    from torchvision import get_image_backend

    if get_image_backend() == 'accimage':

        return accimage_loader(path)

    else:

        return pil_loader(path)



    

class LocalDataset(Dataset):



    def __init__(self, base_path, txt_list, transform=None,target_transform=None):

        self.base_path=base_path

        self.images = np.loadtxt(txt_list,delimiter=',',dtype='str') # use np.genfrom() instead of np.loadtxt() to skip errors



        self.transform = transform

        self.target_transform=target_transform



    def __getitem__(self, index):

        f,c = self.images[index]



        image_path = path.join(self.base_path + "/" + str(c), f)

        

        sample = default_loader(image_path)

        if self.transform is not None:

            sample = self.transform(sample)

            

        target = num_classes[c]

        

        if self.target_transform is not None:

            target = self.target_transform(target)

        

        return (sample,target)



    def __len__(self):

        return len(self.images)

preprocessing()
train_tfms = transforms.Compose([transforms.Resize((400, 400)),

                                 transforms.RandomHorizontalFlip(),

                                 transforms.RandomRotation(15),

                                 transforms.ToTensor(),

                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tfms = transforms.Compose([transforms.Resize((400, 400)),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])





training_set = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=train_tfms)

validation_set = LocalDataset(IMAGES_PATH, VALIDATION_PATH, transform=train_tfms)



trainloader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True)

testloader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False)



print("testloader ok!")
#dataset = torchvision.datasets.ImageFolder(root=IMAGES_PATH, transform = train_tfms)

#trainloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, num_workers = 2)



#dataset2 = torchvision.datasets.ImageFolder(root=IMAGES_PATH, transform = test_tfms)

#testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False, num_workers = 2)

def train_model(model, criterion, optimizer, scheduler, n_epochs = 5):

    

    losses = []

    accuracies = []

    test_accuracies = []

    # set the model to train mode initially

    print("model.train start")

    model.train()

    print("model.train end")

    for epoch in range(n_epochs):

        since = time.time()

        running_loss = 0.0

        running_correct = 0.0

        for i, data in enumerate(trainloader, 0):

            print(i)

            # get the inputs and assign them to cuda

            #print(data)

            #inputs=Variable(data['image'])

            #labels=Variable(data['label'])

            inputs, labels = data

            #print(inputs)

            #print(type(inputs))

            #print(type(labels))

            #inputs = inputs.to(device).half() # uncomment for half precision model

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            

            

            # forward + backward + optimize

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            

            # calculate the loss/acc later

            running_loss += loss.item()

            running_correct += (labels==predicted).sum().item()



        epoch_duration = time.time()-since

        epoch_loss = running_loss/len(trainloader)

        epoch_acc = 100/32*running_correct/len(trainloader)

        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))

        

        losses.append(epoch_loss)

        accuracies.append(epoch_acc)

        

        # switch the model to eval mode to evaluate on test data

        model.eval()

        test_acc = eval_model(model)

        test_accuracies.append(test_acc)

        

        # re-set the model to train mode after validating

        model.train()

        scheduler.step(test_acc)

        since = time.time()

    print('Finished Training')

    return model, losses, accuracies, test_accuracies



print("def train_model ok1")
def eval_model(model):

    correct = 0.0

    total = 0.0

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            images, labels = data

            #images = images.to(device).half() # uncomment for half precision model

            images = images.to(device)

            labels = labels.to(device)

            

            outputs = model_ft(images)

            _, predicted = torch.max(outputs.data, 1)

            

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    test_acc = 100.0 * correct / total

    print('Accuracy of the network on the test images: %d %%' % (

        test_acc))

    return test_acc

print("def eval_model ok")
model_ft = models.resnet34(pretrained=True)

num_ftrs = model_ft.fc.in_features



# replace the last fc layer with an untrained one (requires grad by default)

model_ft.fc = nn.Linear(num_ftrs, 196)

model_ft = model_ft.to(device)



# uncomment this block for half precision model

"""

model_ft = model_ft.half()





for layer in model_ft.modules():

    if isinstance(layer, nn.BatchNorm2d):

        layer.float()

"""

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)



"""

probably not the best metric to track, but we are tracking the training accuracy and measuring whether

it increases by atleast 0.9 per epoch and if it hasn't increased by 0.9 reduce the lr by 0.1x.

However in this model it did not benefit me.

"""

lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

print("download pretrained model ok!")
model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=10)