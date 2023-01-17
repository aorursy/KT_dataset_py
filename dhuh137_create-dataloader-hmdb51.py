import os

import numpy as np

from tqdm.notebook import tqdm

import torch

import torchvision

from torch.utils.data import DataLoader, TensorDataset

import pickle

import tarfile

import shutil



normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



def tensors(base):

    x, y = [], []

    for c in tqdm(os.listdir(base)):

        class_path = os.path.join(base,c)

        for vid in os.listdir(class_path):

            y.append(int(c))

            vid = torch.tensor(np.load(os.path.join(class_path,vid))[:10])

            n_vid = []

            for f in vid:

                n_vid.append(normalize(f.reshape(3,64,64)))

            x.append(torch.stack(n_vid))

    return torch.stack(x),torch.tensor(y)



def make_tarfile(output_filename, source_dir):

    with tarfile.open(output_filename, "w:gz") as tar:

        tar.add(source_dir, arcname=os.path.basename(source_dir))



import gc

        

i = 1



if not os.path.exists(f"../working/fold_{i}"):

    os.mkdir(f"../working/fold_{i}")



base = f"../input/hmdb51/HMDB51_NUMPY/fold_{i}/training/0"

x,y = tensors(base)

tdl = DataLoader(TensorDataset(x,y), batch_size=128, shuffle=True)

with open(f"../working/fold_{i}/tdl_hmdb51.pkl",'wb') as f:

    pickle.dump(tdl, f)



base = f"../input/hmdb51/HMDB51_NUMPY/fold_{i}/validation/0"

x,y = tensors(base)

vdl = DataLoader(TensorDataset(x,y), batch_size=128, shuffle=True)

with open(f"../working/fold_{i}/vdl_hmdb51.pkl",'wb') as f:

    pickle.dump(vdl, f)



tdl = None

vdl = None

del tdl

del vdl

gc.collect()



# make_tarfile(f"../working/fold_{i}.tar.gz",f"../working/fold_{i}")

# shutil.rmtree(f"../working/fold_{i}")    