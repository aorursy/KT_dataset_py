%%capture
# install missing packages
!pip install networkx
!pip install urllib3
!pip install pytorch_lightning
!pip install dgl 
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx

import nilearn as nl
import nilearn.plotting as nlplt
from nilearn.image import index_img
from nilearn import image
from nilearn import plotting
import nibabel as nib
import h5py
import matplotlib.pyplot as plt


from os import listdir
from os.path import isfile, join

import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint

import seaborn as sns
import datetime

%%capture
!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
# assign paths to variables

mask_filepath = '../input/trends-assessment-prediction/fMRI_mask.nii'
example_subject_filepath = '../input/trends-assessment-prediction/fMRI_train/10004.mat'
fmri_filepath = '../input/trends-assessment-prediction/fMRI_train/'
scores_filepath = '../input/trends-assessment-prediction/train_scores.csv'
smri_filepath = 'ch2better.nii'
fnc_filepath = "../input/trends-assessment-prediction/fnc.csv"
icn_numbers_filepath = "../input/trends-assessment-prediction/ICN_numbers.csv"
def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with the version 7.3 flag. Return the subject
    niimg, using a mask niimg as a template for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg


mask_niimg = nl.image.load_img(mask_filepath)
subject_niimg = load_subject(example_subject_filepath, mask_niimg)

print(f"Image type is {type(subject_niimg)}")
print("Image shape is %s" % (str(subject_niimg.shape)))
num_components = subject_niimg.shape[-1]
print("Detected {num_components} spatial maps".format(num_components=num_components))
# nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filepath, view_type='filled_contours', draw_cross=False, title='All %d spatial maps' % num_components, threshold='auto')
# grid_size = int(np.ceil(np.sqrt(num_components)))
# fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
# [axi.set_axis_off() for axi in axes.ravel()]
# row = -1
# for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
#     col = i % grid_size
#     if col == 0:
#         row += 1
#     nlplt.plot_stat_map(cur_img, bg_img=smri_filepath, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)
# data = subject_niimg.get_fdata()
# m,n = data.shape[::2]
# data_new = data.transpose(0,3,1,2).reshape(m,-1,n)
# data_new.shape
class fmri_dataset_generator(Dataset):
    """fmri dataset."""
    
    def __init__(self, subject_image_paths, mask_file):
        """
        Args:
            subject_image_paths (string): Subject images
            mask_file (string): The mask niimg object used for nifti headers.
        """
        
        self.subject_image_paths = subject_image_paths
        self.mask_file = mask_file
        
    def __len__(self):
        return len(self.subject_image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.subject_image_paths[idx]
        img_data = load_subject(img_name, mask_niimg)
        
        np_data = []
        for img in image.iter_img(img_data):
            np_data.append(img.get_fdata())

        img_data_conv = torch.unsqueeze(torch.FloatTensor(np_data),1)

        return img_data_conv

def my_collate_fn(batch):
    data=[]
    for i in batch:
        data.append(i)
    data_ = torch.cat(data, dim=0)
    return data_, data_
  
root_dir = "../input/trends-assessment-prediction/fMRI_train/"
all_mat_files = ["".join([root_dir, f]) for f in listdir(root_dir)]
len(all_mat_files)
validation_split = 0.1

k = round(len(all_mat_files) * validation_split)
indicies = list(range(0,len(all_mat_files)))

validation_ind = random.sample(indicies, k)
train_ind = list(set(indicies) - set(validation_ind))
print(f"Train dataset has {len(train_ind)} examples.")
print(f"Validation dataset has {len(validation_ind)} examples.")
# select file paths for these datasets
validation_datafiles = [all_mat_files[i] for i in validation_ind]
train_datafiles = [all_mat_files[i] for i in train_ind]
train_dataset = fmri_dataset_generator(train_datafiles, mask_file=mask_niimg)
val_dataset = fmri_dataset_generator(validation_datafiles, mask_file=mask_niimg)
train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=my_collate_fn)
# A single batch
iter(train_dataloader).next()[0].size()
IMAGE_WIDTH = 53
IMAGE_HEIGHT = 63
IMAGE_DEPTH = 52

debug_model = False

class AutoEncoder(nn.Module):
   
    def __init__(self):
        super().__init__()
       
        # Encoder specification
        self.enc_cnn_1 = nn.Conv3d(1, 2, kernel_size=3)
        self.enc_cnn_2 = nn.Conv3d(2, 4 , kernel_size=3)
        
        self.enc_pool = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.enc_linear_1 = nn.Linear(6776, 25)
        self.enc_linear_2 = nn.Linear(25, 10)
        
        self.dec_cnn_1 = nn.Conv3d(2, 1, kernel_size=3, padding = [2,2,2])
        self.dec_cnn_2 = nn.Conv3d(4, 2 , kernel_size=3, padding = [2,2,2])
        
        self.dec_unpool = nn.MaxUnpool3d(2, stride=2)

        self.dec_linear_1 = nn.Linear(10, 25)
        self.dec_linear_2 = nn.Linear(25, 6776)
       
    def forward(self, images):
        code, pool_par = self.encode(images)
        if debug_model == True: print("encode complete") 
        out = self.decode(code, pool_par)
        if debug_model == True: print("decode complete")
        return out, 
   
    def encode(self, images):
        
        if debug_model == True: print(f"start encode: {images.shape}")
        
        code = self.enc_cnn_1(images)
        if debug_model == True: print(f"after cnn_1: {code.shape}")
        
        pool1, indices1 = self.enc_pool(code)
        if debug_model == True: print(f"after 1st max_pool3d: {pool1.shape}")
        code = F.selu(pool1)
        
        code = self.enc_cnn_2(code)
        if debug_model == True: print(f"after cnn_2: {code.shape}")
        
        pool2, indices2 = self.enc_pool(code)
        if debug_model == True: print(f"after 2nd max_pool3d: {pool2.shape}")
        code = F.selu(pool2)
        
        code = pool2.view([images.size(0), -1])
        if debug_model == True: print(f"after view: {code.shape}")
        
        code = F.selu(self.enc_linear_1(code))
        if debug_model == True: print(f"after linear_1: {code.shape}")
        
        code = self.enc_linear_2(code)
        if debug_model == True: print(f"after linear_2: {code.shape}")
        
        #required for unpool
        pool_par = {"P1": [indices1], "P2": [indices2]}
        
        return code, pool_par
   
    def decode(self, code, pool_par):
        
        if debug_model == True: print(f"start decode:{code.shape}") 
        
        out = self.dec_linear_1(code)
        if debug_model == True: print(f"after dec_linear_1:{out.shape}")
        
        out = F.selu(self.dec_linear_2(out))
        if debug_model == True: print(f"after dec_linear_2:{out.shape}")
        
        out = out.view([out.size(0), 4, 11, 14, 11]) 
        if debug_model == True: print(f"after view:{out.shape}")
           
        out = self.dec_unpool(out, pool_par['P2'][0], output_size = [53, 4, 23, 28, 23])
        out = F.selu(out)
        if debug_model == True: print(f"after 1st unpool:{out.shape}")
        
        out = self.dec_cnn_2(out)
        if debug_model == True: print(f"after dec_cnn_2:{out.shape}")
        
        out = self.dec_unpool(out, pool_par['P1'][0], output_size = [53, 2, 51, 61, 50])
        out = F.selu(out)
        if debug_model == True: print(f"after 2nd unpool:{out.shape}")
        
        out = self.dec_cnn_1(out)
        if debug_model == True: print(f"after dec_cnn_1:{out.shape}")
        out = F.selu(out)
        
        return out
USE_GPU = True
    
if USE_GPU and torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print(device)
debug_train = True
debug_sample = 100

# Hyperparameters
num_epochs = 5
bs = 1
lr = 0.002
optimizer_cls = optim.Adam

train_loss_values = []
val_loss_values = []

# Load data
train_dataloader = DataLoader(train_dataset, batch_size = bs, collate_fn=my_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=my_collate_fn)

# Instantiate model
autoencoder = AutoEncoder()
autoencoder.to(device)
loss_fn = nn.MSELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

begin_time = datetime.datetime.now()

# Training loop
for epoch in range(num_epochs):
    
    train_loss=0.0
    val_loss=0.0
    print("Epoch %d \n" % epoch)
   
    for i, train_images in enumerate(train_dataloader):
        
        if i==debug_sample:
            break
        
        print(f"train_subject:{i}")
        
        train_out = autoencoder(Variable(train_images[0]))
        
        optimizer.zero_grad()
        t_loss = loss_fn(train_out[0], Variable(train_images[0]))
        t_loss.backward()
        optimizer.step()
        
        train_loss += t_loss.item()
        
    with torch.no_grad():
        for i, val_images in enumerate(val_dataloader):
            
            if i==debug_sample:
                break
                
            print(f"val_subject:{i}")
            
            autoencoder.eval()
            
            val_out = autoencoder(Variable(val_images[0]))
            v_loss = loss_fn(val_out[0], Variable(val_images[0]))
            val_loss += v_loss.item()
            
#     epoch_loss = running_loss / len(dataloaders['train'])
    epoch_train_loss = train_loss / debug_sample
    epoch_val_loss = val_loss / debug_sample
    train_loss_values.append(epoch_train_loss)
    val_loss_values.append(epoch_val_loss)
    
    plt.plot(np.array(train_loss_values), 'r', np.array(val_loss_values), 'b')
#     plt.show()
        
    ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5', monitor='valloss', verbose = 1)
#     print(f"\nTrain Loss = %.3f\n\n" % loss.data.item())
#     print(f"\nVal Loss = %.3f\n\n" % val_loss.data.item())

print(datetime.datetime.now() - begin_time)

loss_values
embeddings = []
for i, images in enumerate(dl):
    if i == 1:
        break
    out, _ = autoencoder.encode(Variable(images[0]))
    embeddings.append(out)
    
print(embeddings)
    
embeddings[0].shape
fnc_df = pd.read_csv(fnc_filepath)
icn_numbers_df = pd.read_csv(icn_numbers_filepath)

titles = list(fnc_df.head(0))[1:]
titles[:5]
temp_i=[]
temp_j=[]
for t in titles:
    i = t[t.find("(")+1:t.find(")")]
    j = t[t.find("(", -5)+1:t.find(")", -1)]
    temp_i.append(int(i))
    temp_j.append(int(j))
temp_i[:5]

# this is the lookup table
icn_numbers = icn_numbers_df.to_dict()['ICN_number']
icn_numbers.items()
rev_icn_numbers = {v:k for k,v in icn_numbers.items()}
print(rev_icn_numbers)

temp_i = [rev_icn_numbers.get(item,item) for item in temp_i]
temp_j = [rev_icn_numbers.get(item, item) for item in temp_j]
corr_structure = list(zip(temp_i, temp_j))
# every subject has 1378 edges
len(corr_structure)

# create a graph for the example subject
G=nx.Graph()
G.add_edges_from(corr_structure)

for i in G.nodes():
    img = index_img(subject_niimg,i)
    G.nodes[i]['image']=img
    
nx.draw(G)
all_fmri_files = [f for f in listdir(fmri_filepath) if isfile(join(fmri_filepath, f))]
# how many subject files do we have
len(all_fmri_files)
# for cnt, f in enumerate(all_mat_files):
#     if cnt == 1:
#         break
#     subject_path =my_path+f
#     subject_niimg = load_subject(subject_path, mask_niimg)

# e = index_img(subject_niimg,2)
# d = index_img(subject_niimg,2).get_fdata()

# re-initiate the embedding dataloader 
my_dataset = fmri_dataset(fmri_filepath, mask_file=mask_niimg)
dl = DataLoader(my_dataset, batch_size=1, collate_fn=my_collate_fn)
# replace nan in targets with mean values 
scores_df = pd.read_csv(scores_filepath)
scores_df_cleaned = scores_df.fillna(scores_df.mean())
scores_df_cleaned.to_csv("scores_df_cleaned", index=False)
# create a data generator
# 1. Get input            : input_path -> image
# 2. Get output           : input_path -> label
# 3. Pre-process input    : image -> pre-processing step -> image
# 4. Get generator output : ( batch_input, batch_labels )

def get_input(input_path, mask_niimg):
    subject_niimg = load_subject(input_path, mask_niimg)
    return(subject_niimg)

def get_output(input_path, label_file = None ):
    subject_id = input_path.split('/')[-1].split('.')[0]
    labels = label_file[label_file['Id']==int(subject_id)]
    labels = labels.values[0][1:] # remove ID
    return(labels)

def collate_fn(batch):
    data=[]
    for i in batch:
        data.append(i)
    data_ = torch.cat(data, dim=0)
    return data_

def get_image_embedding(subject_niimg, mask_niimg):
    np_data = []
    for img in image.iter_img(subject_niimg):
        np_data.append(img.get_fdata())
    img_data_conv = torch.unsqueeze(torch.FloatTensor(np_data),1)
    emb, _ = autoencoder.encode(Variable(img_data_conv))
    return emb

def preprocess_input(input_path, image_emb, fnc_file, corr_structure):
    subject_id = input_path.split('/')[-1].split('.')[0]
    correlations = fnc_file[fnc_file['Id']==int(subject_id)]
#     edge_attr = [{'attr1': i} for i in correlations.values[0]]
    edge_attr = [i for i in correlations.values[0]]
    edge_attr.pop(0)
#     edge_attrs = dict(zip(corr_structure, edge_attr))
    
#     G = nx.Graph()
#     G.add_edges_from(corr_structure)
#     nx.set_edge_attributes(G, edge_attrs)
#     for i in G.nodes:
#         img = image_emb[i]
#         G.nodes[i]['image']=img
    
    jj = dgl.DGLGraph()
    jj.add_nodes(53)
    jj.add_edges(temp_i, temp_j)
    jj.edata['attr1'] = torch.as_tensor(edge_attr)
    jj.ndata['image'] = image_emb
    
    return jj

def image_generator(files, mask_file, fnc_file, label_file, batch_size = 1):
    
    while True:
          # Select files (paths/indices) for the batch
          batch_paths  = np.random.choice(a = files, size = batch_size)
          batch_input  = []
          batch_output = [] 
          
          # Read in each input, perform preprocessing and get labels
          for input_path in batch_paths:
              input_img = get_input(input_path, mask_niimg)
              output = get_output(input_path, label_file)
              emb = get_image_embedding(input_img, label_file)
            
              input = preprocess_input(input_path, image_emb=emb, fnc_file = fnc_df, corr_structure=corr_structure)
              batch_input += [ input ]
              batch_output += [ output ]
                
          # Return a tuple of (input, output) to feed the network
          yield( batch_input, batch_output )
            

subject_paths = [fmri_filepath+i for i in all_fmri_files]
label_df = pd.read_csv("./scores_df_cleaned")

img_gen = image_generator(files = subject_paths, mask_file = mask_niimg, fnc_file = fnc_df, label_file = label_df, batch_size = 1)
samp=next(img_gen)

# graph with embeddings as node attributes
nx.draw(samp[0][0].to_networkx())
samp[1][0]
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import math

debug_dgl_model = True

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(20, 16)
        self.conv2 = GraphConv(16, 8)
        self.classify = nn.Linear(8, 5)

    def forward(self, g):
        
        
        # Use node embedding vector featue to initialise hidden state.
        h = g[0].ndata['image']
        
#         if debug_dgl_model == True: print(f"feat vectors: {h.shape}")
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g[0], h))
        
#         if debug_dgl_model == True: print(f"after first covn: {h.shape}")
        
        h = F.relu(self.conv2(g[0], h))
        
#         if debug_dgl_model == True: print(f"after second covn: {h.shape}")
                
        g[0].ndata['h'] = h
        
#         if debug_dgl_model == True: print(f"hidden state shape: {g[0].ndata['h'].shape}")
        
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g[0], 'h')
        
#         if debug_dgl_model == True: print(f"av graph nodes: {hg}")
            
        lin_out = self.classify(hg)
#         if debug_dgl_model == True: print(f"lin out: {lin_out}")
        
        return lin_out


# Create model
model = Classifier(1, 20, 5)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(5):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(img_gen):
        prediction = model(bg)
#         print(prediction)
#         print(label)
        loss = loss_func(prediction,  torch.FloatTensor(label))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)