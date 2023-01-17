from __future__ import print_function, division



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler 

from torch.autograd import Variable

import numpy as np

import torchvision

import torch.functional as F

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import seaborn as sns

plt.ion()
os.listdir('../input/back-ground-for-xnature/gdxray')
os.listdir('../input/xnaturev2/xnature/XNature')
# Data augmentation and normalization for training

# Just normalization for validation

data_transforms =transforms.Compose([

            transforms.Resize((224,224)),

            transforms.ToTensor()

            ])







data_dir = '../input/xnaturev2/xnature/XNature'

# loading datasets with PyTorch ImageFolder

image_dataset = datasets.ImageFolder(data_dir,

                                          data_transforms)

# defining data loaders to load data using image_datasets and transforms,

# here we also specify batch size for the mini batch

dataloader =  torch.utils.data.DataLoader(image_dataset, batch_size=4,

                                             shuffle=True, num_workers=4)



dataset_size = len(image_dataset)

class_names = image_dataset.classes



use_gpu = torch.cuda.is_available()
import matplotlib.pyplot as plt

import numpy as np





def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()



    

count = 0

start = time.time()



for batch_id, (images,labels) in enumerate(dataloader): 

    count += 1

    imshow(torchvision.utils.make_grid(images, nrow = 4))

    

    if count == 5:

          break
# Creating data indices for training and validation splits:

validation_split = 0.2

batch_size = 16



indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))



# Shuffle the dataset

np.random.shuffle(indices)



train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, 

                                           sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,

                                                sampler=valid_sampler)



dataloaders = {'train': train_loader, 'test': validation_loader}

dataset_sizes = {'train': len(train_indices), 'test':len(val_indices)}

def train_model(model, criterion, optimizer, num_epochs=10):

    since = time.time()



    best_model_wts = model.state_dict()

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'test']:

            if phase == 'train':

                #scheduler.step()

                model.train(True)  # Set model to training mode

            else:

                model.train(False)  # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for data in dataloaders[phase]:

                # get the inputs

                inputs, labels = data



                # wrap them in Variable

                if use_gpu:

                    inputs = Variable(inputs.cuda())

                    labels = Variable(labels.cuda())

                else:

                    inputs, labels = Variable(inputs), Variable(labels)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                outputs = model(inputs)

                

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)



                # backward + optimize only if in training phase

                if phase == 'train':

                    loss.backward()

                    optimizer.step()



                # statistics

                running_loss += loss.data.to('cpu')

                running_corrects += torch.sum(preds == labels.data).to('cpu').numpy()

            

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]

            



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'test' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = model.state_dict()

                state = {'model':model_ft.state_dict(),'optim':optimizer_ft.state_dict()}

                torch.save(state,'point_resnet_best.pth')



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best test Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
model_ft = models.resnet18(pretrained=True) # loading a pre-trained(trained on image net) resnet18 model from torchvision models

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))      # changing the last layer for this dataset by setting last layer neurons to 200 as this dataset has 200 categories

 

if use_gpu:                                 # if gpu is available then use it

    model_ft = model_ft.cuda()       

criterion = nn.CrossEntropyLoss()           # defining loss function



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft,num_epochs=1)
for batch_id, (images,labels) in enumerate(validation_loader): 

    correct = True

    outputs = model_ft(images.cuda())

    _, predicted = torch.max(outputs, 1)

    

    preddictions = predicted.cpu().numpy()

    



    for i in range(len(preddictions)):

        if preddictions[i] != labels[i].cpu().numpy():

            correct = False

    

    if not correct:

        imshow(torchvision.utils.make_grid(images, nrow = 8))     

        for i in range(len(preddictions)):

            print(class_names[preddictions[i]], end='\t')

!wget -c -O anaconda.sh 'https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh'

!bash anaconda.sh -b

!cd /usr/bin/ && ln -sf /content/anaconda3/bin/conda conda

!yes y | conda install faiss-gpu -c pytorch
import time

import os 

import annoy



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from PIL import Image, ImageOps





import torch

import torchvision
from sklearn.neighbors import NearestNeighbors



def retrieve_images(index_embeddings, querry_embeddings, index_labels, querry_labels, k=5):

    '''

    Retriev k images form the index for each querry image and compute mean values of p@k (precision at rank k)

    

    args:

        index_embedding (dict): keys are image names, and values are the associated embeddings

        querry_embedding (dict): keys are image names, and values are the associated embeddings

        index_labels (dict): keys are image names, and values are a list of labels present in the image

        querry_labels (dict): keys are image names, and values a list of labels present in the image

        k (int): The rank to consider when computing precision@k

        

    outputs:

        retrieval_results (dict): key is a querry image name, and value is a k-size list of index retrieved images

        precision_at_k (float): precision at rank k

    '''

    d = 512

    nb = len(index_embeddings)

    nq = len(querry_embeddings)



    xb = np.zeros((nb,d),dtype=np.float32)

    yb = nb*[None]

    index_names = nb*[None]



    xq = np.zeros((nq,d),dtype=np.float32)

    yq = nq*[None]

    querry_names = nq*[None]



    for ii, image in enumerate(index_embeddings.keys()):

        xb[ii,:] = index_embeddings[image]

        yb[ii] = index_labels[image]

        index_names[ii] = image

    

    for ii, image in enumerate(querry_embeddings.keys()):

        xq[ii,:] = querry_embeddings[image]

        yq[ii] = querry_labels[image]

        querry_names[ii] = image



    '''

    # Building Index  

    index = faiss.IndexFlatL2(d)

    index.add(xb)

    # Searching

    D, I = index.search(xq, k) 

    '''



    

    # Building Index  

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(xb)

    D, I = nbrs.kneighbors(xq)



    retrieval_results = {}



    # Compute precision

    precision_at_k = 0

    for querry in range(nq):

        neighbours = I[querry]

        retrievad_images = []

        are_correct_retrievals = []

    

        relevant_retrievals = 0

        for neighbour in neighbours:

            relevant_retrievals += (len(set(yq[querry]) & set(yb[neighbour]))!=0)

            retrievad_images.append(index_names[neighbour])

            are_correct_retrievals.append((len(set(yq[querry]) & set(yb[neighbour]))!=0))

    

        retrieval_results[querry_names[querry]] = (retrievad_images,are_correct_retrievals)

    

        precision_at_k += relevant_retrievals/(k*nq)

    

    print('mean p@k: ', precision_at_k)

    

    return retrieval_results, precision_at_k





def visualize_retrieval(retrieval_results, k=5, nbr_querries=5):

    '''

    Produces a plot of retrieved images for a fixed number of querry images,

    coorect retrievals are bounded in green and uncorrect ones are bounded in red.

    args:

        retrieval_results (dict): key is paths to querry image, and value is a tuple of two lists

                                the first list contain path to retrieved images and the second 

                                store a boolean that represnets if the retrieval is correct

        k (int): Number of image to retreival for each querry image

        nbr_querries (int): The number of querry images to show

    '''

    

    count = 0

    for querry_image_name,(retrieved_images_names, are_correct_retrievals) in retrieval_results.items():

        w=10

        h=10

        fig=plt.figure(figsize=(16, 16))

        columns = k+1

        rows = 1

        querry_image = Image.open(querry_image_name)

        fig.add_subplot(rows, columns, 1)

        plt.imshow(querry_image,cmap='gray')

        for ii, retrieved_image_name in enumerate(retrieved_images_names):

            retrieved_image = Image.open(retrieved_image_name)

            retrieved_image = retrieved_image.convert('RGB')

            if are_correct_retrievals[ii]:

                retrieved_image = ImageOps.expand(retrieved_image,

                                                  border=int(retrieved_image.size[0]*0.1),

                                                  fill='rgb(0,255,0)')

            else:

                retrieved_image = ImageOps.expand(retrieved_image,

                                                  border=int(retrieved_image.size[0]*0.1),

                                                  fill='rgb(255,0,0)')

                

            fig.add_subplot(rows, columns, ii+2)

            plt.imshow(retrieved_image, cmap='gray')

   

        count+=1

        if count==nbr_querries:

            break

        

        plt.show()

        



def get_embedding_of_image(image_name, model):

    '''

    args:

        image_name (str): Path to the image of interest

        model (torch.nn.Module): THe model to use for embedding computing

    '''

    model.eval()

    image_pil = Image.open(image_name).resize([256,256])



    image_array = np.array(image_pil)/(255)



    image_rgb = np.stack((image_array,)*3, axis=-1)



    # swap color axis because

    # numpy image: H x W x C

    # torch image: C X H X W 

    image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1)))



    image_tensor = image_tensor.expand(1,-1,-1,-1)



    embeddings = model(image_tensor.float().cuda())

    

    return embeddings.cpu().detach().numpy()
data_dir = '../input/xnaturev2/xnature/XNature'

images = {}



for dir_ in os.listdir(data_dir):

    images_path = os.path.join(data_dir,dir_)

    for file in os.listdir(images_path):

        images[os.path.join(images_path, file)] = dir_

        

images = {'path': list(images.keys()),

           'labels': list(images.values())}



df = pd.DataFrame.from_dict(images)

df_train, df_test =train_test_split(df, test_size = 0.2)
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms







## Transforms

class Rescale(object):

    """Rescale the image in a sample to a given size.



    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, int)

        self.output_size = output_size



    def __call__(self, sample):

        sample['querry_image'] = sample['querry_image'].resize([self.output_size,self.output_size])

        sample['image1'] = sample['image1'].resize([self.output_size,self.output_size])

        sample['image2'] = sample['image2'].resize([self.output_size,self.output_size])



        return sample

    

class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):   

        sample['querry_image'] = np.array(sample['querry_image'])/(255)

        sample['image1'] = np.array(sample['image1'])/(255)

        sample['image2'] = np.array(sample['image2'])/(255)

        

        ## Check if images are grayscale or RGBA

        if len(sample['querry_image'].shape)==3:

            sample['querry_image'] = sample['querry_image'][:,:,0]

            

        if len(sample['image1'].shape)==3:

            sample['image1'] = sample['image1'][:,:,0]

            

        if len(sample['image2'].shape)==3:

            sample['image2'] = sample['image2'][:,:,0]

        

  

        

        sample['querry_image'] = np.stack((sample['querry_image'],)*3, axis=-1)

        sample['image1'] = np.stack((sample['image1'],)*3, axis=-1)

        sample['image2'] = np.stack((sample['image2'],)*3, axis=-1)



        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W 

        sample['querry_image'] = torch.from_numpy(sample['querry_image'].transpose((2, 0, 1)))

        sample['image1'] = torch.from_numpy(sample['image1'].transpose((2, 0, 1)))

        sample['image2'] = torch.from_numpy(sample['image2'].transpose((2, 0, 1)))

        

        return sample

      



class XrayDataSet(Dataset):

    def __init__(self, dataframe, transform = None):

        '''

        args: 

        dataframe (pd.DataFrame): dataframe of index images, paths and labels

        transform (callable, optional): Optional transform to be applied

                on a sample.

        '''

        self.df = dataframe

        self.transform = transform

    

    def __len__(self):

        return len(self.df)

    

    def get_image_from_idx(self, idx):

        '''

        Output:

            image (PIL Image): The image in self.df[idx]

            image_labels (list): List of the labels present in image

        '''

        image_path, image_labels = self.df.iloc[idx].values

        image = Image.open(image_path)

        return image, image_labels, image_path

        

    

    def __getitem__(self, idx):

        '''

        return a sample of NIH chest X-rays dataset

        A sample is compsed of tree images, and the labels associated ot each of them

        '''

        # Get the querry image

        querry_path, querry_labels = self.df.iloc[idx].values

        querry_image  = Image.open(querry_path)

              

            

        # Get random similar image

        similar_images = self.df[self.df['labels'] == querry_labels]

        idx_sim = np.random.randint(0,len(similar_images), 1)[0]

        path1, labels1 = similar_images.iloc[idx_sim].values

        image1  = Image.open(path1)

        

        # Get random similar image

        different_images = self.df[self.df['labels'] != querry_labels]

        idx_diff = np.random.randint(0,len(different_images), 1)[0]

        path2, labels2 = different_images.iloc[idx_diff].values

        image2  = Image.open(path2)

        

        sample = {'querry_image': querry_image,

                  'querry_labels': querry_labels,

                  'image1': image1,

                  'labels1': labels1,

                  'image2': image2,

                  'labels2': labels2,

                  'querry_image_name': querry_path}

        

        

        if self.transform:

            sample = self.transform(sample)

            

        return sample
batch_size = 32



train_dataset = XrayDataSet(df_train, transform = transforms.Compose([Rescale(256), ToTensor()]))





test_dataset = XrayDataSet(df_test, transform = transforms.Compose([Rescale(256), ToTensor()]))





TrainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

TestLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()



    

count = 0

start = time.time()

Loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for sample in Loader:

    count += 1



    imshow(torchvision.utils.make_grid(sample['querry_image'], nrow = 4))

  

    

    imshow(torchvision.utils.make_grid(sample['image1'], nrow = 4))



    

    imshow(torchvision.utils.make_grid(sample['image2'], nrow = 4))



    

    if count%5 == 0:

        print(count)

        break

        

    print('-------------------------------------------')

 

    

end = time.time()

print(end - start) 
import torch

import torchvision.models as models

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim





def l2n(x, eps=1e-6):

    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)



class GEM_Net(nn.Module):

    #TO DO: Add a validation metric: (mAP, r@k) 

    def __init__(self, p=2):

        # TODO: Define the netwrok with other feature extractors and choose, how many layers to train, chosse the optimizerr

        super(GEM_Net, self).__init__()

        self.encoder = models.resnet18(pretrained=True)

        modules = list(self.encoder.children())[:-2]

        self.encoder = nn.Sequential(*modules)

    

        ct = 0

        for child in self.encoder.children():

            ct += 1

            if ct < 7:

                for param in child.parameters():

                    param.requires_grad = True

        

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

        self.encoder.to(self.device)

        self.p = p

        self.optimizer = optim.SGD(self.encoder.parameters(), lr=0.001, momentum=0.9)

        

    def forward(self, x, eps=1e-6):

        features = F.relu(self.encoder(x))

        gem = F.avg_pool2d(features.clamp(min=eps).pow(self.p), (features.size(-2), features.size(-1))).pow(1./self.p)

        gem_normal = l2n(gem).squeeze(-1).squeeze(-1)

    

        return gem_normal

    

    def get_embedding(self, loader):

        self.eval()

        "Get a dictionnary that stores embeddings for each image"

        embeddings_dict = {}

        labels_dict = {}

        loss = 0.



        for sample in loader:

            querry_image = sample['querry_image'].float().to(self.device)

            image1 = sample['image1'].float().to(self.device)

            image2 = sample['image2'].float().to(self.device)

                

            querry_embedding = self(querry_image)

            embedding1 = self(image1)

            embedding2 = self(image2)

                

            loss_sim = torch.norm(querry_embedding-embedding1, dim=1)

            loss_diff = torch.norm(querry_embedding-embedding2, dim=1)

                

                

            querry_labels = sample['querry_labels']

            labels1 = sample['labels1']

            labels2 = sample['labels2']

                

            querry_labels = list(map(lambda string: string.split('|'), querry_labels))

            labels1 = list(map(lambda string: string.split('|'), labels1))

            labels2 = list(map(lambda string: string.split('|'), labels2))

   

                

            dynamic_triplet_loss = F.relu(loss_sim - loss_diff + 1).mean()

            

            loss += dynamic_triplet_loss.cpu().data.numpy()

            

            names = sample['querry_image_name']

            embeddings = querry_embedding.detach().cpu().numpy()

            for ii,name in enumerate(names):

                embeddings_dict[name] = embeddings[ii,:]

                labels_dict[name] = querry_labels[ii]

            

        

        return(embeddings_dict, labels_dict, loss)

                

        

    

    def test_on_loader(self, querry_loader, index_loader, rank=5):

        " Test the model on a dataloader"

        """

        args: 

            index_loader(torch.utils.data.DataLoder): index images loader generated from ChestDataSet object

            querry_loader(torch.utils.data.DataLoder): querry images loader generated from ChestDataSet object

            rank (int): The number of images retrived for each query by our retrieval system

        

        outputs: 

            Loss (float): Value oof the loss function on the data provided by the loader

            p@k (float): precision at rank k 

        """

        self.eval()

        print("compute the index set embedding")

        index_embedding, index_labels, index_loss = self.get_embedding(index_loader)

        print("compute the querry set embedding")

        querry_embedding, querry_labels, querry_loss = self.get_embedding(querry_loader)

        

        print('querry_loss: ',querry_loss,'. index_loss: ', index_loss)

        retrieval_results, precision_at_k = retrieve_images(index_embedding, querry_embedding, index_labels, querry_labels)

         

        return(retrieval_results, precision_at_k)

        

    def train_on_loader(self, train_loader, val_loader,  epochs=1, validations_per_epoch=1, hist_verbosity=1, verbosity=1):

        " Train the model with dynamic triplet loss"

        """

        args: 

            train_loader(torch.utils.data.DataLoder): train loader generated from ChestDataSet object

            val_loader(torch.utils.data.DataLoder): validation loader generated from ChestDataSet object

            epochs (int): Number of epochs

            validations_per_epoch (int): Number of validation to perform at each epoch

            hist_verbosity (int): if 1 compute history by epoch, if 2 compute history by batch

            verbosity (int): Controls the logs of the training. The number of epochs before performing a test

        

        outputs: 

            hist (list): history of train_loss and validation loss (To Do)

        """

        assert hist_verbosity in [1,2] , "hist verbosity should be 1 or 2"

        

        for epoch in range(epochs):

            print('epoch', epoch ,'\\', epochs,':')

            for batch_id, sample in enumerate(train_loader):

                self.optimizer.zero_grad()

                

                querry_image = sample['querry_image'].float().to(self.device)

                image1 = sample['image1'].float().to(self.device)

                image2 = sample['image2'].float().to(self.device)

                

                querry_embedding = self(querry_image)

                embedding1 = self(image1)

                embedding2 = self(image2)

                

                loss_sim = torch.norm(querry_embedding-embedding1, dim=1)

                loss_diff = torch.norm(querry_embedding-embedding2, dim=1)

                

                

                querry_labels = sample['querry_labels']

                labels1 = sample['labels1']

                labels2 = sample['labels2']

            

                dynamic_triplet_loss = F.relu(loss_sim - loss_diff + 1).mean()

                

                dynamic_triplet_loss.backward()  

                self.optimizer.step()

                

            if epoch%verbosity==0:

                pass

                #retrieval_results, precision_at_k= self.test(val_loader, train_loader)
gem_net = GEM_Net()



start = time.time()

gem_net.train_on_loader(TrainLoader, TestLoader, epochs=3,verbosity=5)

end = time.time()

print("Train time is: ",end-start)



'''

start = time.time()

retrieval_results, precision_at_k = gem_net.test_on_loader(TestLoader, TrainLoader)

end = time.time()

print("Test time is: ",end-start)

'''
start = time.time()

index_embedding, index_labels, index_loss = gem_net.get_embedding(loader=TrainLoader)

end = time.time()



print('Building Index Embeddings in:', end-start)
start = time.time()

querries_embedding, querries_labels, _ = gem_net.get_embedding(loader=TestLoader)

end = time.time()



print('Building querries Embeddings in:', end-start)
retrieval_results, precision_at_k = retrieve_images(index_embedding, querries_embedding, index_labels, querries_labels)

visualize_retrieval(retrieval_results,nbr_querries=10)
image_name = '../input/xnaturev2/xnature/XNature/Fruits/N0001_0001.png'

embedding = get_embedding_of_image(image_name, gem_net)



querry_embedding = {image_name: embedding}

querry_labels = {image_name: ['Fruits']}



retrieval_results, precision_at_k = retrieve_images(index_embedding, querry_embedding,

                                                   index_labels, querry_labels)

visualize_retrieval(retrieval_results,nbr_querries=100)

import cv2



def image_to_tensor(image_name):

    """

    args:

        image_name(str): path to a .png file 

    outputs:

        tensor of shape (1,3,256,256)

    """

    image_pil = Image.open(image_name).resize([256,256])



    image_array = np.array(image_pil)/(255)



    image_rgb = np.stack((image_array,)*3, axis=-1)



    # swap color axis because

    # numpy image: H x W x C

    # torch image: C X H X W 

    image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1)))



    image_tensor = image_tensor.expand(1,-1,-1,-1)



    return image_tensor.float().cuda()
image_name = '../input/xnaturev2/xnature/XNature/Gun/B0049_0001.png'

image = image_to_tensor(image_name)



embedding = get_embedding_of_image(image_name, gem_net)



querry_embedding = {image_name: embedding}

querry_labels = {image_name: ['Gun']}



retrieval_results, precision_at_k = retrieve_images(index_embedding, querry_embedding,

                                                   index_labels, querry_labels)



visualize_retrieval(retrieval_results,nbr_querries=100)



features_blobs = []

def hook_feature(module, input, output):

    features_blobs.append(output.data.cpu().numpy())

      

gem_net._modules.get('encoder').register_forward_hook(hook_feature)



embeddings = []

embedding = gem_net(image).detach()

embeddings.append(embedding)



for ii, result in enumerate(retrieval_results[image_name][0]):

    image_ = image_to_tensor(result)

    embedding = gem_net(image_).detach()

    embeddings.append(embedding)



#residuals = torch.abs(torch.cat(embeddings[2:3]).mean(dim=0) - embeddings[0].squeeze())

residuals = torch.abs(embeddings[2] - embeddings[0])

#residuals = F.softmax(residuals,dim=0)

residuals /= residuals.sum()

residuals = residuals.reshape((1,512,1,1))

feature_map = torch.Tensor(features_blobs[0]).cuda()

heat_map = ((1-residuals)*feature_map).sum(dim=1).squeeze().cpu().numpy()

heat_map = np.uint8(255*(heat_map))

heat_map = cv2.resize(heat_map,(256,256))

heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)/255



img = Image.open(image_name).resize([256,256])

img = np.array(img)

img = np.stack((img,)*3, axis=-1)/255



result = heat_map * 0.3 + img * 0.7



red = result[:,:,0].copy()

blue = result[:,:,2].copy()



result[:,:,0] = blue

result[:,:,2] = red



plt.imshow(result)