import copy, cv2, json, os, time, torch, torchvision



from collections import OrderedDict



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import numpy as np

import pandas as pd

from PIL import Image

from torch import nn

from torch import optim

from torch.autograd import Variable

import torch.nn.functional as F

from torch.optim import lr_scheduler

import torch.utils.data as data

from torch.utils.data.dataset import Dataset

from torch.utils.data.dataloader import DataLoader

from torchvision import models

from torchvision import transforms

import torchvision.datasets as datasets



print(os.listdir("../input"))

%load_ext autoreload

%autoreload 2

%matplotlib inline



from matplotlib.pyplot import figure
print(len(os.listdir('../input/train')))

print(len(os.listdir('../input/test')))

train = pd.read_csv('../input/train.csv')

train.head(3)
train.groupby('Id').count().sort_values('Image',ascending = False).head(10)
train.groupby('Id').count().plot.hist(range = (0,10))
id_map = train[['Id']].drop_duplicates()

id_map.index = list(range(0,len(id_map)))

whale_dict = id_map['Id'].to_dict()

id_dict = dict((v,k) for k,v in whale_dict.items())

#id_dict
with Image.open('../input/test/0027089a4.jpg') as img:

    fig, ax = plt.subplots()

    ax.imshow(img)
def whale_list(whale_id,df = train):

    return df[df['Id']==whale_id]['Image'].tolist()
image_list = whale_list('w_23a388d')[:1]

for img_path in image_list:

    with Image.open('../input/train/' + img_path) as img:

        fig, ax = plt.subplots()

        ax.imshow(img)
class HW_Dataset(Dataset):

    def __init__(self,filepath, csv_path,transform=None):

        self.file_path = filepath

        self.df = pd.read_csv(csv_path)

        self.transform = transform

        self.image_list = [x for x in os.listdir(self.file_path)]

        

    def __len__(self):

        return(len(self.image_list))

    

    def __getitem__(self,idx):

        img_path = os.path.join(self.file_path,self.df.Image[idx])

        label = self.df.Id[idx]

        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        

        return img, label



def label_to_id(label):

    x = [id_dict[i] for i in label]

    return torch.tensor(x)



# Process a PIL image for use in a PyTorch model

def process_image(image):

    #img_transform = transforms.Compose([

#        transforms.ToTensor()])

    img_transform = transform

    pil_image = Image.open(image)

    pil_image = img_transform(pil_image).float()

    np_image = np.array(pil_image)    

    return np_image



def imshow(image, ax=None, title=None):

    if ax is None:

        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension

    # but matplotlib assumes is the third dimension

    image = image.transpose((1, 2, 0))

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
#old 256

dims = 128



transform = transforms.Compose([

                              transforms.Resize((dims, dims)),

                              transforms.ToTensor(),

                              transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                     std=[0.229, 0.224, 0.225])])



dim1 = max(int(dims**2/2),5005)

dim2 = max(int((dims**2)/4),5005)

print(dim1)

print(dim2)
train_dataset = HW_Dataset('../input/train/','../input/train.csv', transform)

test_dataset = HW_Dataset('../input/test/','../input/train.csv', transform)

len(test_dataset.image_list)
train_dataset[0][0].size()
# example

a = list(range(10))

b = list(range(5,15))

np.setdiff1d(a,b) #in a and not in b
test_size = .2

n = len(train_dataset)



np.random.seed(0)

a = list(range(n))



train_index = np.random.choice(a,replace=False,size = int(n*(1-test_size)))

test_index = np.setdiff1d(a,train_index)

print(train_index.size)

print(test_index.size)
data_train = copy.deepcopy(train_dataset)

data_train.image_list = [train_dataset.image_list[i] for i in train_index]



data_test = copy.deepcopy(train_dataset)

data_test.image_list = [train_dataset.image_list[i] for i in test_index]



gen_train = DataLoader(data_train,batch_size=16, shuffle=True)

gen_test = DataLoader(data_test,batch_size=16, shuffle=True)

full_train_generator = DataLoader(train_dataset,batch_size=16, shuffle=True)

full_test_generator = DataLoader(test_dataset,batch_size=16, shuffle=True)



print(len(gen_train))

print(len(gen_test))

print(len(full_train_generator))

print(len(full_test_generator))
print(len(train_dataset))

print(len(test_dataset))

print(len(data_train))

print(len(data_test))
model = models.vgg11(pretrained=True)



classifier = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(dim1, dim2)),                        

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(dim2, 5005)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))



model.classifier = classifier



criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.003)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
epochs = 14

training_data = full_train_generator

#training_data = gen_train



x = time.time()

image_count = len(training_data)

updates = 4

progress_printer = int(image_count/updates)



for e in range(epochs):

    running_loss, i = 0, 0



    for image, label in training_data:            

        label = label_to_id(label)

        image, label = image.to(device), label.to(device)

        i +=1 

        if i % progress_printer == 0:

            print('{:.0f}% complete'.format(i/image_count*100))

            print('Epoch Rate: {}'.format(round((time.time() - x)*updates/60),2))

            x = time.time()

        log_ps = model(image)

        loss = criterion(log_ps, label)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    

    print('epoch {}: loss: {}'.format(e,running_loss))
image_path = "../input/train/002b4615d.jpg"

showthis = imshow(process_image(image_path))
image_list = whale_list('w_23a388d')[:1]

for img_path in image_list:

    with Image.open('../input/train/' + img_path) as img:

        fig, ax = plt.subplots()

        ax.imshow(img)
def predict(image_path, model, n=5):

    with torch.no_grad():

        model.eval()

        np_array = process_image(image_path)

        if np_array.shape[0] == 1:

            #print(np_array.shape)

            np_array = np.repeat(np_array[:, :], 3, axis=0)

            #print(np_array.shape)

        tensor = torch.from_numpy(np_array)

        model = model.cuda()

        inputs = Variable(tensor.float().cuda())

        inputs = inputs.unsqueeze(0)

        output = model.forward(inputs)  

        predictions = torch.exp(output).data.topk(n)

        probabilities = predictions[0].cpu()

        classes = predictions[1].cpu()

        classes_np = classes.numpy()[0]

        classes = [whale_dict[classes_np[i]] for i in list(range(len(classes_np)))]        

        return probabilities.numpy()[0], classes



def predict_image(image_path,n = 5):

    probabilities, classes = predict(image_path, model)

    fig = plt.figure(figsize=(8,8))

    ax1 = plt.subplot2grid((20,10), (0,0), colspan=10, rowspan=10)

    ax2 = plt.subplot2grid((20,10), (10,2), colspan=5, rowspan=8)

    image = Image.open(image_path)

    ax1.imshow(image)

    y_pos = np.arange(n)

    ax2.set_yticks(y_pos)

    ax2.set_yticklabels(classes)

    ax2.invert_yaxis()

    ax2.set_xlabel('Probability')

    ax2.barh(y_pos, probabilities)

    plt.show()

    return classes
rank = 500

ranked_whales = train.groupby('Id').count().sort_values('Image',ascending = False).reset_index()

whale = ranked_whales['Id'][rank-1]

print(whale)



print(ranked_whales[rank-2:rank+1])

#whale = 'w_23a388d'



for image_path in whale_list(whale,train.loc[test_index])[:2]:    

    print(image_path)

    image_path = '../input/train/' + image_path

    x = predict_image(image_path)

    print(x[0] == whale)

    print(x)

    
image_list = copy.deepcopy(test_dataset.image_list)

i = 0



jpegs = []

predictions = []



#for the_jpeg in test_dataset.image_list:

for the_jpeg in image_list:  

    i +=1

    if i % 1000 == 0:

        print(i)

    image_path = '../input/test/' + the_jpeg

    x, y = predict(image_path,model)

    preds = ' '.join(y)

    jpegs.append(the_jpeg)

    predictions.append(preds)    



jpegs

predictions



whale_predictions = pd.DataFrame({'Image':jpegs,'Id':predictions})

print(whale_predictions.shape)

whale_predictions.head()
whale_predictions.to_csv('second_submission_attempt.csv',index=False)