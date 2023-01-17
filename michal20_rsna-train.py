# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





!pip install ../input/modelsdefs/efficientnet_pytorch/ > /dev/null

package_path = '../input/rsnascripts'

import sys

sys.path.append(package_path)
!ls ../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection

import torch

import torch.optim as optim

from model import EfficientFPN

from model import load_net

from train import Trainer





model = EfficientFPN(encoder_name='efficientnet-b4', classes=6, use_pretrained=0, 

                         grayscale=True, use_se_block=True, learn_window=True,

                         use_stn = False) #TODO - evaluate losses



state = torch.load('../input/rsna-models/last.pth', map_location=lambda storage, loc: storage)

load_net(state, model)

#model.load_state_dict(state["state_dict"])

device = torch.device("cuda:0")

model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)





train_df_path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'

data_folder = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train"



model_trainer = Trainer(model, batch_size = {"train": 16, "val": 8}, 

                        data_folder=data_folder, train_df_path=train_df_path, optimizer = optimizer, 

                        train_width=256, run_dir='.', base_lr=1e-5,

                        num_epochs = 10, num_workers=8, 

                        from_epoch=0, finetune = False, load_pickle=False,

                        balance = False, use_tqdm=False, tensorboard=False, use_ohem=True)

model_trainer.start()
from IPython.display import FileLink

FileLink(r'model.pth')

FileLink(r'last.pth')