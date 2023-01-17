%matplotlib inline



import os



import torch

import torch.nn.functional as F

import torch.nn as nn



from torch import optim

from torch.autograd import Variable



import torchvision

import torchvision.datasets as dset

import torchvision.models as models

import torchvision.transforms as transforms

import torchvision.utils

from torch.utils.data import DataLoader,Dataset



import matplotlib.pyplot as plt



import pandas as pd

import numpy as np



import random



from PIL import Image

import PIL.ImageOps    
class Img2Vec():



    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):

        """ Img2Vec

        :param cuda: If set to True, will run forward pass on GPU

        :param model: String name of requested model

        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git

        :param layer_output_size: Int depicting the output size of the requested layer

        """

        self.device = torch.device("cuda" if cuda else "cpu")

        self.layer_output_size = layer_output_size

        self.model_name = model

        

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)



        self.model = self.model.to(self.device)



        self.model.eval()



        self.scaler = transforms.Resize((224, 224))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                              std=[0.229, 0.224, 0.225])

        self.to_tensor = transforms.ToTensor()



    def get_vec(self, file_name, tensor=False,bbox = [0,224,0,224]):

        """ Get vector embedding from PIL image

        :param img: PIL Image

        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array

        :returns: Numpy ndarrayim

        """

        image = Image.open(os.path.join(file_name)).crop(bbox).convert('RGB')

        image = self.normalize(self.to_tensor(self.scaler(image))).unsqueeze(0).to(self.device)



        if self.model_name == 'alexnet':

            my_embedding = torch.zeros(1, self.layer_output_size)

        else:

            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)



        def copy_data(m, i, o):

            my_embedding.copy_(o.data)



        h = self.extraction_layer.register_forward_hook(copy_data)

        h_x = self.model(image)

        h.remove()



        if tensor:

            return my_embedding

        else:

            if self.model_name == 'alexnet':

                return my_embedding.numpy()[0, :]

            else:

                return my_embedding.numpy()[0, :, 0, 0]



    def _get_model_and_layer(self, model_name, layer):

        """ Internal method for getting layer from model

        :param model_name: model name such as 'resnet-18'

        :param layer: layer as a string for resnet-18 or int for alexnet

        :returns: pytorch model, selected layer

        """

        if model_name == 'resnet-18':

            model = models.resnet18(pretrained=True)

            if layer == 'default':

                layer = model._modules.get('avgpool')

                self.layer_output_size = 512

            else:

                layer = model._modules.get(layer)



            return model, layer



        elif model_name == 'alexnet':

            model = models.alexnet(pretrained=True)

            if layer == 'default':

                layer = model.classifier[-2]

                self.layer_output_size = 4096

            else:

                layer = model.classifier[-layer]



            return model, layer



        else:

            raise KeyError('Model %s was not found' % model_name)

def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.

    This function computes the average prescision at k between two lists of

    items.

    Parameters

    ----------

    actual : True identity

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0

    for i,p in enumerate(predicted):

        score = 0.0

        if p == actual :

            score = 1.0/(i+1.0)

            break

    return score



def mapk(actual, predicted, k=10):

    """

    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists

    of lists of items.

    Parameters

    ----------

    actual : True identity

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The mean average precision at k over the input lists

    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
def imshow(img,text=None,should_save=False):

    fig = plt.figure(figsize=(20, 80))

    if text:

        plt.text(75, 8, text, style='italic',fontweight='bold',

            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(img.permute(1, 2, 0)) 



def show_plot(iteration,loss):

    plt.plot(iteration,loss)

    plt.show()
class ContrastiveLoss(torch.nn.Module):

    """

    Contrastive loss function.

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    """



    def __init__(self, margin=2.0):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin



    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +

                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))





        return loss_contrastive
class SiameseNetworkEmbeddingDataset(Dataset):

    

    def __init__(self,df,img2vec=None,genEmbed = False):

        self.img2vec = img2vec

        self.df = df

        self.genEmbed = genEmbed

        

    def __getitem__(self,idx):

        # not selecting 'new_whale' for anchor image.

        img0_idx = random.choice(self.df[self.df.Id != 'new_whale'].index.values)

        

        # we need to make sure approx 50% of images are in the same class

        should_get_same_class = random.randint(0,1)

        if should_get_same_class:

            img1_idx = random.choice(self.df[self.df.Id == self.df.Id[img0_idx]].index.values) 

        else:

            img1_idx = random.choice(self.df[self.df.Id != self.df.Id[img0_idx]].index.values)

            

        img0_embedding = self.df.loc[img0_idx,'embedding']

        img1_embedding = self.df.loc[img1_idx,'embedding']

        

        if self.genEmbed :

            print("a")

            img0_embedding = img2vec.get_vec(self.df.loc[img0_idx,'Image'], tensor=False) 

            img1_embedding = img2vec.get_vec(self.df.loc[img1_idx,'Image'], tensor=False)

        

        return img0_embedding, img1_embedding , torch.from_numpy(np.array([int(self.df.Id[img0_idx] != self.df.Id[img1_idx])],dtype=np.float32))

    

    def __len__(self):

        return(self.df.shape[0])
class SiameseNetworkEmbedding(nn.Module):

    def __init__(self):

        super(SiameseNetworkEmbedding, self).__init__()



        self.fc1 = nn.Sequential(

            nn.Linear(512, 512),

            nn.ReLU(inplace=True),



            nn.Linear(512, 5))



    def forward_once(self, x):

        output = self.fc1(x)

        return output



    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)

        return output1, output2
def createEmbeddingFiles() :

    img2vec = Img2Vec()

    train_full = pd.read_csv("../input/humpback-whale-identification/train.csv")

    test_df = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")

    train_full = pd.read_csv("../input/humpback-whale-identification/train.csv")



    id_counts = train_full.Id.value_counts()



    valid_df = train_full.loc[train_full.Id.isin(id_counts[id_counts>5].index.values),:].sample(frac=0.3)



    train_df = train_full.loc[~train_full.index.isin(valid_df.index.values),:]



    test_df = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")



    bbox_df = pd.read_csv("../input/metadata/bounding_boxes.csv")



    def getEmbedding(file_path,x):

        file_name = os.path.join(file_path,x)

        bbox = bbox_df.loc[bbox_df.Image==x,:].values[0,1:]

        return(img2vec.get_vec(file_name,tensor=False,bbox=bbox).squeeze())



    train_df_embed = train_df.assign(embedding = train_df['Image'].apply(lambda x : getEmbedding('../input/humpback-whale-identification/train/',x)))

    valid_df_embed = valid_df.assign(embedding = valid_df['Image'].apply(lambda x : getEmbedding('../input/humpback-whale-identification/train/',x)))

    test_df_embed = test_df.assign(embedding = test_df['Image'].apply(lambda x : getEmbedding('../input/humpback-whale-identification/test/',x)))



    pickle.dump(train_df_embed,open( "train_df_embed.p",'wb'))

    pickle.dump(valid_df_embed,open( "valid_df_embed.p",'wb'))

    pickle.dump(test_df_embed,open( "test_df_embed.p",'wb'))
import pickle

train_df = pickle.load( open( "../input/humpback-whale-identification-embedding/train_df_embed.p",'rb'))

valid_df = pickle.load( open( "../input/humpback-whale-identification-embedding/valid_df_embed.p",'rb'))

test_df = pickle.load( open( "../input/humpback-whale-identification-embedding/test_df_embed.p",'rb'))
train_dataset = SiameseNetworkEmbeddingDataset(train_df)



train_dataloader = DataLoader(train_dataset,

                        shuffle=True,

                        num_workers=0,

                        batch_size=64)



net = SiameseNetworkEmbedding().cuda()

criterion = ContrastiveLoss()

optimizer = optim.Adam(net.parameters(),lr = 0.0005 )



counter = []

loss_history = [] 

iteration_number= 0
net.train()

for epoch in range(0,50):

    for i, data in enumerate(train_dataloader,0):

        img0, img1 , label = data

        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        optimizer.zero_grad()

        output1,output2 = net(img0,img1)

        loss_contrastive = criterion(output1,output2,label)

        loss_contrastive.backward()

        optimizer.step()

        if i %100 == 0 :

            print("Epoch number {} \t Iteration number {} \t Current loss {}\n".format(epoch,iteration_number,loss_contrastive.item()))

            iteration_number +=10

            counter.append(iteration_number)

            loss_history.append(loss_contrastive.item())

show_plot(counter,loss_history)
net.eval()



train_df = train_df.assign(pred=train_df.loc[:,'embedding'].apply(lambda x : (net.forward_once(torch.from_numpy(x).squeeze().cuda()).detach().cpu().numpy())))

valid_df = valid_df.assign(pred = valid_df.loc[:,'embedding'].apply(lambda x : (net.forward_once(torch.from_numpy(x).squeeze().cuda()).detach().cpu().numpy())))

test_df = test_df.assign(pred = test_df.loc[:,'embedding'].apply(lambda x : (net.forward_once(torch.from_numpy(x).squeeze().cuda()).detach().cpu().numpy())))
from sklearn.metrics.pairwise import euclidean_distances

distance_mat_valid = pd.DataFrame(euclidean_distances(np.stack(train_df.pred.values), np.stack(valid_df.pred.values)),columns = valid_df.Image.values,index=train_df.Image.values)

distance_mat_test = pd.DataFrame(euclidean_distances(np.stack(train_df.pred.values), np.stack(test_df.pred.values)),columns = test_df.Image.values,index=train_df.Image.values)
Id_Df = train_df[['Image','Id']].set_index('Image')



def getTopFiveIdValid(x):

    sortedIds = Id_Df.loc[distance_mat_valid.loc[:,x].sort_values().index.values,'Id'].values

    topFiveIds = sortedIds[np.sort(np.unique(sortedIds, return_index=True)[1])[:5]]

    return(topFiveIds)



def getTopFiveIdTest(x):

    sortedIds = Id_Df.loc[distance_mat_test.loc[:,x].sort_values().index.values,'Id'].values

    topFiveIds = ' '.join(sortedIds[np.sort(np.unique(sortedIds, return_index=True)[1])[:5]])

    return(topFiveIds)



test_df = test_df.assign(Id = test_df.loc[:,'Image'].apply(getTopFiveIdTest))

valid_df = valid_df.assign(topFiveIds =  valid_df.loc[:,'Image'].apply(getTopFiveIdValid))
mapk(valid_df.Id, valid_df.topFiveIds, k=5)
test_df[['Image','Id']].to_csv('submission_3.csv',index=False)