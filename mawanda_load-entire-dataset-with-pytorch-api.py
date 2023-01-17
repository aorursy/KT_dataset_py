import os

from h5py import File as h5File

from tqdm import tqdm

import numpy as np

import pandas as pd

import nibabel as nib

from torch.utils.data import Dataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms
class TReNDS_dataset(Dataset):

    def __init__(

            self, 

            mat_folder, 

            sbm_path, 

            train_scores_path=None, 

            fnc_path=None, 

            ICN_numbers_path=None,

            mask_path=None,

            transform=None

        ):

        super().__init__()  # Inherit from Dataset torch class, therefore initialize it

        print('Loading dataset...')

        # Load data

        # Store the paths to the .mat file as a dictionary {patientID: complete_path_to_file}

        self.mat_paths = {int(filename.split('.')[0]): os.path.join(mat_folder, filename) for filename in os.listdir(mat_folder)}



        if fnc_path:

            fnc = pd.read_csv(fnc_path)  # There are no NaN values here (ref: https://www.kaggle.com/rftexas/trends-in-dept-understanding-eda-lgb-baseline)

            self.fnc = {Id: np.array(fnc.loc[fnc['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()}

        else:

            self.fnc = None



        sbm = pd.read_csv(sbm_path)  # There are no NaN values here (as before)

        self.sbm = {Id: np.array(sbm.loc[sbm['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()}



        ICN_num = pd.read_csv(ICN_numbers_path)  # There are no NaN values

        self.ICN_num = np.array(ICN_num['ICN_number']).squeeze()



        # Check if dataset is for training or for submission

        if train_scores_path:

            train_scores = pd.read_csv(train_scores_path)

            train_scores.fillna(train_scores.mean(), inplace=True)  # Look for NaN values and replace them with column mean

            self.labels = {Id: np.array(train_scores.loc[train_scores['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()}

        else:

            self.labels = None



        # Test code to verify if there are all the labels for each type of data

        # fnc_keys = list(fnc['Id'])

        # sbm_keys = list(sbm['Id'])

        # print(len(pt_keys), len(fnc_keys), len(sbm_keys))

        # fnc_missing = []

        # sbm_missing = []

        # for k in pt_keys:

        #     if k not in fnc_keys:

        #         fnc_missing.append(k)

        #     if k not in sbm_keys:

        #         sbm_missing.append(k)

        # print(fnc_missing, sbm_missing)



        self.mask = np.array(nib.load(mask_path))



        # Prepare num_to_id in order to address the indexes required from torch API

        self.__num_to_id = {i: k for i, k in enumerate(self.mat_paths.keys())}

        # Create reverse order to have control over dataset patients IDs and indexes

        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}



        print('Dataset loaded!')



        self.transform = transform



    def __len__(self):

        # Return the length of the dataset

        return len(self.mat_paths.keys())



    def __getitem__(self, item):

        # Get the ID corresponding to the item (an index) that torch is looking for.

        ID = self.__num_to_id[item]



        # Retrieve all information from the Dataset initialization

        

        # brain = torch.load(self.mat_paths[ID]) # This is the loading in case the tensors has been saved as torch ones.

        # This is the loading in case of saving the tensors as compressed bytes numpy array

        # brain = np.copy(np.frombuffer(zlib.decompress(open(self.pt_paths[ID], 'rb').read()), dtype='float64').reshape(53, 52, 63, 53))  

        # brain = None  # In case of shallow networks which doesn't need the brain images

        brain: np.ndarray = np.array(h5File(self.mat_paths[ID], 'r')['SM_feature'], dtype='float32')  # Load as float32, I think float64 is not needed.



        sbm = self.sbm[ID]

        # Create sample

        sample = {

            'ID': ID,

            'sbm': sbm,

            'brain': brain

        }

        if self.fnc:

            sample['fnc'] = self.fnc[ID]

        # Add labels to the sample if the dataset is the training one.

        if self.labels:

            sample['label'] = self.labels[ID]



        # Transform sample (if defined)

        return self.transform(sample) if self.transform is not None else sample
class ToTensor:

    def __call__(self, sample):

        sbm = torch.tensor(sample['sbm']).float()

        ID = sample['ID']



        new_sample = {**sample, 'sbm': sbm, 'ID': ID}



        if sample['brain'] is not None:

            # Define use of brain images - which are not necessary for shallow networks

            new_sample['brain'] = torch.tensor(sample['brain']).float()

        sample_keys = list(sample.keys())

        # These are labels which can be skept in case they are not needed.

        tries_keys = ['fnc', 'label']

        for tk in tries_keys:

            if tk in sample_keys:

                new_sample[tk] = torch.tensor(sample[tk]).float()



        return new_sample
# Define all the needed paths

base_path = '/kaggle/input/trends-assessment-prediction/'  # In case the working directory is different from the playing one

train_mat_folder = os.path.join(base_path, 'fMRI_train')

test_pt_folder = os.path.join(base_path, 'fMRI_test')

fnc_path = os.path.join(base_path, 'fnc.csv')

sbm_path = os.path.join(base_path, 'loading.csv')

ICN_num_path = os.path.join(base_path, 'ICN_numbers.csv')

train_scores_path = os.path.join(base_path, 'train_scores.csv')

mask_path = os.path.join(base_path, 'fMRI_mask.nii')



# Define transformations

trans = transforms.Compose([ToTensor()])



dataset = TReNDS_dataset(train_mat_folder, sbm_path, train_scores_path, fnc_path, ICN_num_path, mask_path, transform=trans)

dataloader = DataLoader(dataset, batch_size=24, shuffle=True, pin_memory=True, num_workers=1)



for batch in tqdm(dataloader, desc='Reading dataset...'):

    brain = batch['brain']  # Notice that this is still in CPU. If needed, it can be sent to the CUDA device with .to('cuda:0')

    print(brain.shape)

    break