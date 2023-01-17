!pip install pymap3d==2.1.0

!pip install -U l5kit
# Basic imports

import os

import numpy as np

import pandas as pd

from l5kit.data import ChunkedDataset, LocalDataManager
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# local data manager

dm = LocalDataManager()

# set dataset path

dataset_path = dm.require('scenes/test.zarr')

# load the dataset; this is a zarr format, chunked dataset

chunked_dataset = ChunkedDataset(dataset_path)

# open the dataset

chunked_dataset.open()

#print(chunked_dataset)
mask_arr = np.load("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")

mask = mask_arr['arr_0']
sub_df = pd.read_csv("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv")
# Define the indicies of the coordinates we will replace later

x_coord_indices = ["coord_x00","coord_x01","coord_x02","coord_x03","coord_x04","coord_x05","coord_x06","coord_x07","coord_x08","coord_x09","coord_x010","coord_x011","coord_x012","coord_x013","coord_x014","coord_x015","coord_x016","coord_x017","coord_x018","coord_x019","coord_x020","coord_x021","coord_x022","coord_x023","coord_x024","coord_x025","coord_x026","coord_x027","coord_x028","coord_x029","coord_x030","coord_x031","coord_x032","coord_x033","coord_x034","coord_x035","coord_x036","coord_x037","coord_x038","coord_x039","coord_x040","coord_x041","coord_x042","coord_x043","coord_x044","coord_x045","coord_x046","coord_x047","coord_x048","coord_x049"]

y_coord_indices = ["coord_y00","coord_y01","coord_y02","coord_y03","coord_y04","coord_y05","coord_y06","coord_y07","coord_y08","coord_y09","coord_y010","coord_y011","coord_y012","coord_y013","coord_y014","coord_y015","coord_y016","coord_y017","coord_y018","coord_y019","coord_y020","coord_y021","coord_y022","coord_y023","coord_y024","coord_y025","coord_y026","coord_y027","coord_y028","coord_y029","coord_y030","coord_y031","coord_y032","coord_y033","coord_y034","coord_y035","coord_y036","coord_y037","coord_y038","coord_y039","coord_y040","coord_y041","coord_y042","coord_y043","coord_y044","coord_y045","coord_y046","coord_y047","coord_y048","coord_y049"]
velocities = chunked_dataset.agents['velocity'][mask]
# Define the time steps taken (i.e 5 seconds at 10Hz)

delta_t = np.arange(0.1,5.1,0.1)



# Get the distance moved in x and y

delta_x = delta_t*velocities[:,0].reshape(-1,1)

delta_y = delta_t*velocities[:,1].reshape(-1,1)
sub_df[x_coord_indices] = delta_x

sub_df[y_coord_indices] = delta_y
sub_df.to_csv('submission.csv', index=False)