import os

import numpy as np

from tqdm.notebook import tqdm



base = "../input/ucf101/fold_1/training/0"

x, y = [], []



for c in tqdm(os.listdir(base)):

    class_path = os.path.join(base,c)

    for vid in os.listdir(class_path):

        y.append(int(c))

        x.append(np.load(os.path.join(class_path,vid))[:10])
import torch

from torch.utils.data import DataLoader, TensorDataset

import pickle



x = torch.tensor(x)

y = torch.tensor(y)

tdl = DataLoader(TensorDataset(x,y), batch_size=128, shuffle=True)



with open("../working/tdl_ucf101.pkl",'wb') as f:

    pickle.dump(tdl, f)
base = "../input/ucf101/fold_1/validation/0"

x, y = [], []



for c in tqdm(os.listdir(base)):

    class_path = os.path.join(base,c)

    for vid in os.listdir(class_path):

        y.append(int(c))

        v= np.load(os.path.join(class_path,vid))

        x.append(v[np.arange(0,len(v),len(v)//10)[:10]])
x = torch.tensor(x)

y = torch.tensor(y)

vdl = DataLoader(TensorDataset(x,y), batch_size=128, shuffle=True)



with open("../working/vdl_ucf101.pkl",'wb') as f:

    pickle.dump(vdl, f)