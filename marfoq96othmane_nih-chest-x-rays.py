!wget -c -O anaconda.sh 'https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh'

!bash anaconda.sh -b

!cd /usr/bin/ && ln -sf /content/anaconda3/bin/conda conda

!yes y | conda install faiss-gpu -c pytorch
import time

import os 

import faiss



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from PIL import Image



import torch

import torchvision

import faiss 

import json



def save_retrieval_results(retrieval_results, path="retrieval_results.json"):

    '''

    args:

        retrieval_results (dict): key is an querry image name and value is a list

                                  of the retrieved images.

        path (str): path on wich to save the dictionnary

    '''

    json_file = json.dumps(retrieval_results)

    f = open(path,"w")

    f.write(file)

    f.close()

    

def load_retrival_results(path="retrieval_results.json"):

    '''

    args:

        path (str): path to load the dictionnary from

    '''

    return json.load(open(path))





def retrieve_images(index_embedding, querry_embedding, index_labels, querry_labels, k=5):

    '''

    Retriev k images form the index for each querry image and compute mean values of r@k (recall at rank k)

    

    args:

        index_embedding (dict): keys are image names, and values are the associated embeddings

        querry_embedding (dict): keys are image names, and values are the associated embeddings

        index_labels (dict): keys are image names, and values are a list of labels present in the image

        querry_labels (dict): keys are image names, and values a list of labels present in the image

        k (int): The rank to consider when computing recall@k

        

    outputs:

        retrieval_results (dict): key is a querry image name, and value is a k-size list of index retrieved images

        recall_at_k (float): recall at rank k

    '''

    d = 512

    nb = len(index_embedding)

    nq = len(querry_embedding)



    xb = np.zeros((nb,d),dtype=np.float32)

    yb = nb*[None]

    index_names = nb*[None]



    xq = np.zeros((nq,d),dtype=np.float32)

    yq = nq*[None]

    querry_names = nq*[None]



    for ii, image in enumerate(index_embedding.keys()):

        xb[ii,:] = index_embedding[image]

        yb[ii] = index_labels[image]

        index_names[ii] = image

    

    for ii, image in enumerate(querry_embedding.keys()):

        xq[ii,:] = querry_embedding[image]

        yq[ii] = querry_labels[image]

        querry_names[ii] = image



    # Building Index  

    index = faiss.IndexFlatL2(d)

    index.add(xb)



    retrieval_results = {}



    # Searching

    D, I = index.search(xq, k) 



    # Comute recall

    recall_at_k = 0

    for querry in range(nq):

        neighbours = I[querry]

        retrievad_images = []

    

        relevant_retrievals = 0

        for neighbour in neighbours:

            relevant_retrievals += (len(set(yq[querry]) & set(yb[neighbour]))!=0)

            retrievad_images.append(index_names[neighbour])

    

        retrieval_results[querry_names[querry]] = retrievad_images

    

        recall_at_k += relevant_retrievals/(k*nq)

    

    print('mean r@k: ', recall_at_k)

    

    return retrieval_results, recall_at_k



def LevensteinDistance(a, b):

    """

    Levenstein Distance to mesure the distance between two strings (or two lists of strings)

    args :

    a, b : List of strings

    """

    if len(a) == 0:

        return len(b)



    if len(b) == 0:

        return len(a)





    if a[-1] == b[-1]:

        return min([LevensteinDistance(a[:-1], b)+1, LevensteinDistance(a, b[:-1])+1, LevensteinDistance(a[:-1], b[:-1])]) 

    else:

        return min([LevensteinDistance(a[:-1], b)+1, LevensteinDistance(a, b[:-1])+1, LevensteinDistance(a[:-1], b[:-1])+1])

    
os.listdir('../input/data')
labels = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4,

          'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 10,

          'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}





root_path = '../input/data'

Data_Entry_file=os.path.join(root_path,'Data_Entry_2017.csv')





with open(os.path.join(root_path, 'test_list.txt')) as f:

    test_list = f.readlines()



with open(os.path.join(root_path, 'train_val_list.txt')) as f:

    train_val_list = f.readlines()

    

test_list = set([line.strip() for line in test_list])

train_val_list = set([line.strip() for line in train_val_list]) 



# Delete the images with no findings

Data_Entry = pd.read_csv(Data_Entry_file)

Data_Entry = Data_Entry[['Image Index', 'Finding Labels']]

Data_Entry = Data_Entry.set_index('Image Index')

Data_Entry = Data_Entry[Data_Entry['Finding Labels']!= 'No Finding']



all_images = set(Data_Entry.index)



train_val_list = list(all_images & train_val_list)

test_list = list(all_images & test_list)





# We will create a dictionnary with keys being the names of

# images and the values the paths to them

dict_images = {}



for dir in os.listdir(root_path):

    if dir[:6] == 'images':

        path = os.path.join(root_path, os.path.join(dir,'images'))

        for file in os.listdir(path):

            dict_images[file] = os.path.join(path, file)

            
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

      



class ChestDataSet(Dataset):

    def __init__(self, Data_Entry_file, dict_images, image_list, transform = None):

        '''

        args: 

        Data_Enty_file (str): pah to csv file containing the image names and their labels

        dict_images (dict): keys are the images names, and the values are the paths to them

        images_list (list): list of images to consider 

                    (train_val_list for train and test_list for test list)

        transform (callable, optional): Optional transform to be applied

                on a sample.

        '''

        self.Data_Entry_file = Data_Entry_file

        

        self.Data_Entry = pd.read_csv(Data_Entry_file)

        self.Data_Entry = self.Data_Entry[['Image Index', 'Finding Labels']]

        self.Data_Entry = self.Data_Entry.set_index('Image Index')

        self.Data_Entry = self.Data_Entry[self.Data_Entry['Finding Labels']!='No Finding']

        

        self.dict_images = dict_images

        self.image_list = image_list

        self.transform = transform

    

    def __len__(self):

        return len(self.image_list)

    

    def get_image_from_idx(self, idx):

        '''

        Output:

            image (PIL Image): The image in self.image_list[idx]

            image_labels (list): List of the labels present in image

        '''

        image_labels = self.Data_Entry.loc[self.image_list[idx]].values[0]

        image_path = self.dict_images[self.image_list[idx]]

        image = Image.open(image_path)

        return image, image_labels

    

    def get_image_from_name(self, image_name):

        '''

        Output:

            image (PIL Image): The image in self.image_list[idx]

            image_labels (list): List of the labels present in image

        '''

        image_labels = self.Data_Entry.loc[image_name].values[0]

        image_path = self.dict_images[image_name]

        image = Image.open(image_path)

        return image, image_labels

        

    

    def __getitem__(self, idx):

        '''

        return a sample of NIH chest X-rays dataset

        A sample is compsed of tree images, and the labels associated ot each of them

        '''

        # Get the querry image

        querry_image, querry_labels = self.get_image_from_idx(idx)

              

            

        # Get random similar image

        similar_images = self.Data_Entry[self.Data_Entry['Finding Labels'] == querry_labels]

        idx_sim = np.random.randint(0,len(similar_images), 1)[0]

        similar_image_name = similar_images.iloc[idx_sim].name

        image1, labels1 = self.get_image_from_name(similar_image_name)

        

        # Get random different image

        different_images = self.Data_Entry[self.Data_Entry['Finding Labels'] != querry_labels]

        idx_diff = np.random.randint(0,len(different_images), 1)[0]

        different_image_name = different_images.iloc[idx_diff].name

        image2, labels2 = self.get_image_from_name(different_image_name)

        

        sample = {'querry_image': querry_image,

                  'querry_labels': querry_labels,

                  'image1': image1,

                  'labels1': labels1,

                  'image2': image2,

                  'labels2': labels2,

                  'querry_image_name': train_dataset.image_list[idx]}

        

        

        if self.transform:

            sample = self.transform(sample)

            

        return sample

batch_size = 32



train_dataset = ChestDataSet(Data_Entry_file=os.path.join(root_path,'Data_Entry_2017.csv'), 

                            dict_images=dict_images, image_list = train_val_list,

                            transform = transforms.Compose([Rescale(256), ToTensor()]))





test_dataset = ChestDataSet(Data_Entry_file=os.path.join(root_path,'Data_Entry_2017.csv'), 

                            dict_images=dict_images, image_list = test_list,

                            transform = transforms.Compose([Rescale(256), ToTensor()]))





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



    print(sample['querry_image'].size())

    

    imshow(torchvision.utils.make_grid(sample['querry_image'], nrow = 4))

    print(sample['querry_labels'])

    

    imshow(torchvision.utils.make_grid(sample['image1'], nrow = 4))

    print(sample['labels1'])

    

    imshow(torchvision.utils.make_grid(sample['image2'], nrow = 4))

    print(sample['labels2'])

    

    if count%10 == 0:

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

                    param.requires_grad = False

        

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

   

            labels_distance = np.zeros(len(querry_labels))

            for ii in range(len(querry_labels)):

                labels_distance[ii] = LevensteinDistance(labels1[ii], labels2[ii])

                    

            labels_distance = torch.tensor(labels_distance).float().to(self.device)

                

            dynamic_triplet_loss = F.relu(loss_sim - loss_diff + labels_distance).mean()

            

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

            R@k (float): Recall at rank k 

        """

        self.eval()

        print("compute the index set embedding")

        index_embedding, index_labels, index_loss = self.get_embedding(index_loader)

        print("compute the querry set embedding")

        querry_embedding, querry_labels, querry_loss = self.get_embedding(querry_loader)

        

        print('querry_loss: ',querry_loss,'. index_loss: ', index_loss)

        retrieval_results, recall_at_k = retrieve_images(index_embedding, querry_embedding, index_labels, querry_labels)

         

        return(retrieval_results, recall_at_k)

        

    def train_on_loader(self, train_loader, val_loader,  epochs=1, validations_per_epoch=1, hist_verbosity=1, verbosity=1):

        " Train the model with dynamic triplet loss"

        """

        args: 

            train_loader(torch.utils.data.DataLoder): train loader generated from ChestDataSet object

            val_loader(torch.utils.data.DataLoder): validation loader generated from ChestDataSet object

            epochs (int): Number of epochs

            validations_per_epoch (int): Number of validation to perform at each epoch

            hist_verbosity (int): if 1 compute history by epoch, if 2 compute history by batch

            verbosity (int): Controls the putput of the model

        

        outputs: 

            hist (list): history of train_loss and validation loss (To Do)

        """

        assert hist_verbosity in [1,2] , "hist verbosity should be 1 or 2"

        

        total_train_loss = 0.

        for epoch in range(epochs):

            start = time.time()

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

                

                querry_labels = list(map(lambda string: string.split('|'), querry_labels))

                labels1 = list(map(lambda string: string.split('|'), labels1))

                labels2 = list(map(lambda string: string.split('|'), labels2))

   

                labels_distance = np.zeros(len(querry_labels))

                for ii in range(len(querry_labels)):

                    labels_distance[ii] = LevensteinDistance(labels1[ii], labels2[ii])

                    

                labels_distance = torch.tensor(labels_distance).float().to(self.device)

                

            

                dynamic_triplet_loss = F.relu(loss_sim - loss_diff + labels_distance).mean()

                

                total_train_loss += dynamic_triplet_loss.detach().cpu().numpy()

                if batch_id %50==0:

                    print('batch_id: ', batch_id, '. total_train_loss: ', total_train_loss/((batch_id+1)*batch_size))

                    

                dynamic_triplet_loss.backward()  

                self.optimizer.step()

              

            end = time.time()

            print("training time for epoch ", epoch,'/',epochs,' is: ', int(end-start))

            print("dynamic_triplet_loss on train set is:", total_train_loss)

            

            '''

            start = time.time()

            retrieval_results, recall_at_k= self.test_on_loader(val_loader, train_loader)

            end = time.time()

            print("test time for epoch ", epoch,'/',epochs,' is: ', int(end-start))

            print('r@k on test set:', recall_at_k)

            '''

            

            print('the model is saved to ',"gem_net_"+str(epoch)+".pth")

            self.save_model(path="gem_net_"+str(epoch)+".pth")

            

            

    def save_model(self, path="gem_net.pth"):

        torch.save(self.state_dict(), path)

    



def load_model(path="gem_net.pth"):

    model = GEM_Net()

    model.load_state_dict(torch.load(path))

    

    return model

                

                

        
gem_net = load_model(path='../input/gem-net-weight/gem_net_9.pth')



start = time.time()

gem_net.train_on_loader(TrainLoader, TestLoader, epochs = 10)

retrieval_results, recall_at_k = gem_net.test_on_loader(TestLoader, TrainLoader)

end = time.time()

print(end-start)
