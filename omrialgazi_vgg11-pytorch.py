import os
import numpy as np 
import pandas as pd 
from math import sqrt, ceil, trunc
from random import shuffle
import random
import cv2


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize

import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

import time

root_path = "../input/train/"
SEED = 448
random.seed(SEED)


dtype = torch.cuda.FloatTensor

lr = 0.0002
resize_target = 200
batch_size = 32
limit_dataset = 1
split_train_val = 0.8

def conv_calc(in_size, model, pool_f = 2, pool_s = 2, conv_f = 3, conv_s = 1, conv_p = 1):
    out = in_size
    for layer in model:
        if layer =='M':
            out = trunc(((out - pool_f)/ pool_s) + 1)
        else:
            out = ((out - conv_f + (2 * conv_p))/conv_s) + 1
            channels = layer
    output = out * out * channels
    print (f'Image size after conv is {out}*{out}, channels:{channels}, final vector size:{output}')
    return  int(output)



class FeatExtract:
    """Grabs file locations, splits to test&val
    Also shuffles"""
    def __init__(self, path, limit = None, shuffled = False, split = 0.8):
        self.path = path
        self.limit = limit
        self.shuffled = shuffled
        self.mapping = {}
        self.all_files =[]
        self.list_all_files()
        if self.shuffled:
            shuffle(self.all_files)
        
        
        #Split to train and validate
        self.split = int(len(self.all_files) * split)
        self.train = self.all_files[:self.split]
        self.val = self.all_files[self.split:]
         
    def list_all_files(self):
        for label, directory in enumerate(os.listdir(self.path)):
            self.mapping[label] = directory
            tmp_list = [[self.path + directory+'/'+file,label] for file in os.listdir(self.path + directory)]
            if self.shuffled:
                shuffle(tmp_list)
            if self.limit:
                tmp_list=tmp_list[:int(len(tmp_list)*self.limit)]
            self.all_files.extend(tmp_list)
        self.n_features = label + 1
        
    

class Seeds(nn.Module):
    """Reads through a DB one by one, perform transforms"""
    def __init__(self, file_list, segment = False, transform = None):
        self.file_list = file_list
        self.transform = transform 
        self.segment = segment
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        item = self.file_list[idx][0]
        label = self.file_list[idx][1]
        img = Image.open(item).convert('RGB')
        if self.segment:
            img = segment_plant(img)
        img = adjust_colors(img)
        if self.transform:
            img = self.transform (img)
        else:
            img =  transforms.functional.resize(img, (resize_target,resize_target))
            img =  transforms.functional.to_tensor(img)
        return img, label


class AccLossPlotter:
    """Perform fwd pass on the loaded set to get loss and accuracy
    Also - plots!"""
    def __init__(self, loader):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.loader = loader
        
    def add_epoch(self, loss):
        self.train_loss.append(loss)
        
    def evaluate(self, model):
        acc, loss = check_accuracy(model, self.loader)
        self.val_acc.append(acc)
        self.val_loss.append(loss)
        
    def plot(self):
        curcurr=range(len(self.val_loss))
        plt.plot(curcurr,self.train_loss,'.r',label='Train loss')
        plt.plot(curcurr,self.val_loss,'+b',label='Validation loss')
        plt.plot(curcurr,self.val_acc,'*g',label='Validation accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Epochs')
        plt.show()


data_transform = transforms.Compose([
        transforms.transforms.Resize((resize_target,resize_target)),
        transforms.transforms.ColorJitter(brightness=1, contrast=0.5, saturation=0.5, hue=0),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.RandomRotation(180),
        transforms.transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])


def adjust_colors(img):
    img = transforms.functional.adjust_brightness(img, 2)
    img = transforms.functional.adjust_contrast(img, 1.1)
    img = transforms.functional.adjust_saturation(img, 1.1)
    return img


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def segment_plant(image):
    image = np.array(image)
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    output = transforms.functional.to_pil_image (output)
    return output



class RandomCrop:
    def __init__(self, image, label, n_crops, crop_size):  
        mask = create_mask_for_plant(image)
        if image.shape[0] > 300:
            image = resize(image, (300,300))
            mask = resize(mask,(300,300))
        self.label = label
        self.image = image
        self.mask = mask
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.crops = []
    
    def create_crops(self):
        #Divide image to a grid where each square is the size of crop_size
        if int(self.image.shape[0]) < 1 + self.crop_size * 2:
            self.crops = [[self.image, self.label] for im in range(self.n_crops)]
        else:
            for row in np.linspace(0,self.image.shape[0] - self.crop_size, int(self.image.shape[0]/self.crop_size), dtype = int):
                for col in  np.linspace(0,self.image.shape[1] - self.crop_size, int(self.image.shape[1]/self.crop_size), dtype = int):

                    #Check for each square crop, the intensity of relevant pixels, and add them to crops list
                    green_pixels_in_crop = sum(sum(self.mask[row:row+self.crop_size,col:col+self.crop_size]/255))
                    self.crops.append((self.image[row:row+self.crop_size, col:col+self.crop_size], green_pixels_in_crop ))

                    #Sort the list by most relevant and take top n_crops
                    self.crops = sorted(self.crops, key=lambda x:x[1], reverse = True)
                    self.crops = [[x[0], self.label] for x in self.crops[:self.n_crops]] #Returns images and label
                    if len(self.crops) < self.n_crops:
                        pass
                        #print ('problem with crops, shape image is', self.image.shape)
                        #print ('we got %s crops but %s requested'%(len(self.crops),self.n_crops))


    def get_crop(self):
        for i in self.crops:
            yield i



'''
#for testing

all_images = FeatExtract(root_path, limit = 1, shuffled = True, split = 1)
disp_images = Seeds(all_images.train)
im = disp_images[0][0]
label = disp_images[0][1]
print (im.shape)
tim1=time.time()
im2 = random_crop(im, n_crops = 4, crop_size = 51)
print(time.time()-tim1)
print (len(im2))
plt.imshow(im2[0]) 
'''
#show_images_range(0,4,disp_images)

def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True).cuda()
        y = Variable(y, volatile =True).cuda()
        
        scores = model(x_var)
        scores = scores.float()
        loss = loss_fn(scores, y)
        
        _, preds = scores.data.cpu().max(1)
        preds = preds.cuda()
        
        y = y.data.cuda()
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc, loss.data[0]



def train(model, loss_fn, optimizer, flex_lr_optim, num_epochs = 1, print_every = 10):
    for epoch in range(num_epochs):
        start_epoch = time.time()
        print('Starting epoch %d / %d' % (epoch + 1 , num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y).cuda()
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch time in seconds: %.2f' %(time.time() - start_epoch))
        plt.imshow(np.array(x[0].cpu()).transpose(1,2,0))
        valoss_plotter.add_epoch(loss.data[0])
        valoss_plotter.evaluate(model)
        flex_lr_optim.step(valoss_plotter.val_loss[-1])
    valoss_plotter.plot()
    return 

trainer = FeatExtract(root_path, limit = False, shuffled = True, split = split_train_val)

loader_train = DataLoader(Seeds(trainer.train, segment = False, transform = data_transform), batch_size=batch_size)

loader_val = DataLoader(Seeds(trainer.val, segment = False), batch_size=batch_size)

#Our loss function!
loss_fn=nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss()

#model = models.vgg11_bn(num_classes = 12)
model = models.resnet101(num_classes = 12)
#Adjust our model to work with our own image size
'''model.classifier._modules['0'] = nn.Linear(in_features=conv_calc(
    resize_target,[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']),
                                           out_features=4096, bias=True)
'''
model = model.cuda()

valoss_plotter = AccLossPlotter(loader_val)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

flex_lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, verbose = True)

train(model, loss_fn, optimizer, flex_lr_optimizer , num_epochs = 40)

valoss_plotter.plot()

test_path = '../input/test/'
test_files = [test_path + filename for filename in os.listdir(test_path)]
test_predictions =[]
model.eval()
for path in test_files:
    img = Image.open(path).convert('RGB')    
    #img = segment_plant(img)
    img = adjust_colors(img)
    img = transforms.functional.resize(img, (resize_target,resize_target))
    img = np.array(img).transpose(2,0,1)
    img = np.expand_dims(img, axis = 0)
    img = torch.from_numpy(img)


    x_var = Variable(img.type(dtype), volatile=True).cuda()
    
    scores = model(x_var)

    _, preds = scores.data.cpu().max(1)
    test_predictions.append(preds)

test_predictions = np.array(test_predictions)
csv_results = pd.DataFrame([trainer.mapping[pred] for pred in test_predictions], os.listdir(test_path))[0]
csv_results = csv_results.reset_index(level = 0)
csv_results.columns = ['file', 'species']
csv_results.to_csv('preds1.csv', header = True, index = False)



#Finding optimal learning rate
'''
model.train()
results = []
for lr in np.geomspace(0.00001,0.0004,6):
    reset(model)
    print (lr)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss = train(model, loss_fn, optimizer, num_epochs = 2)
    acc = check_accuracy(model, loader_val)
    results.append((loss, acc))
'''
"""
#Getting some stats on the image sizes
all_images = FeatExtract(root_path, limit = 1, shuffled = True, split = 1)
disp_images = Seeds(all_images.train)


sizes = np.array([])
label = np.array([])
for num, image in enumerate(disp_images):
    sizes = np.append(sizes,image[0].shape[1])
    label = np.append(label, image[1])

plt.subplot (2, 1 ,1 )
plt.hist(label, bins = 25)
plt.title ('Image label distribution')
plt.subplot (2, 1, 2)
plt.hist(sizes, range=(40,1000), bins = 25)
plt.title ('All images')
plt.tight_layout()
plt.show()

sizes = None
label = None
"""

"""
#Getting a visualization of image processing 

all_images = FeatExtract(root_path, limit = 1, shuffled = True, split = 1)
disp_images = Seeds(all_images.train)
sample_images = [disp_images[img] for img in range(20)]
image = sample_images[0]
image_mask = create_mask_for_plant(image[0])
image_segmented = segment_plant(image[0])
image_sharpen = sharpen_image(image_segmented)

plt.imshow(image[0])

fig = plt.figure(figsize = (25,25))
fig.add_subplot(3,1,1)
plt.imshow(image_mask)
fig.add_subplot(3,1,2)
plt.imshow(image_segmented)
fig.add_subplot(3,1,3)
plt.imshow(image_sharpen)





def show_images_range(start,end,input_x, fig_size = 25):
    #displays images from ^start till end$ in input_x - list of images
    all_pics = end - start
    
    subplot_size = ceil(sqrt(all_pics))
    
    input_x = [input_x[x] for x in range(start, end+1)]

    fig=plt.figure(figsize=(fig_size, fig_size))
    
    plt.axis('off')

    for image in range(1, all_pics + 1):
        fig.add_subplot(subplot_size, subplot_size, image)
        plt.imshow(input_x[image][0])
    plt.tight_layout()

    
"""




def show_images_range(start,input_x, fig_size = 25):
    #displays images from ^start till end$ in input_x - list of images
    img_arr = []
    print((input_x))
    subplot_size = ceil(sqrt(start))
    for i, x in enumerate(input_x):
        img_arr.append(x)
        if i>start:
            break
    
    fig=plt.figure(figsize=(fig_size, fig_size))
    plt.axis('off')

    for image in range(1, len(img_arr)):
        fig.add_subplot(subplot_size, subplot_size, image)
        plt.imshow(img_arr[image])
    plt.tight_layout()
show_images_range(25,loader_train)