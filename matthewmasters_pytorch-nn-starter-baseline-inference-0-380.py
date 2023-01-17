import numpy as np

import pandas as pd

import json

import torch

from tqdm import tqdm

from torch import nn

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
WEIGHTS_PATH = '../input/weights-5/weights (5).pth'

FEATURE_SIZE = 21

BATCH_SIZE = 256
def one_hot(categories, string):

    encoding = np.zeros((len(string), len(categories)))

    for idx, char in enumerate(string):

        encoding[idx, categories.index(char)] = 1

    return encoding



def featurize(entity):

    sequence = one_hot(list('ACGU'), entity['sequence'])

    structure = one_hot(list('.()'), entity['structure'])

    loop_type = one_hot(list('BEHIMSX'), entity['predicted_loop_type'])

    features = np.hstack([sequence, structure, loop_type])

    return features 



def char_encode(index, features, feature_size):

    half_size = (feature_size - 1) // 2

    

    if index - half_size < 0:

        char_features = features[:index+half_size+1]

        padding = np.zeros((int(half_size - index), char_features.shape[1]))

        char_features = np.vstack([padding, char_features])

    elif index + half_size + 1 > len(features):

        char_features = features[index-half_size:]

        padding = np.zeros((int(half_size - (len(features) - index))+1, char_features.shape[1]))

        char_features = np.vstack([char_features, padding])

    else:

        char_features = features[index-half_size:index+half_size+1]

    

    return char_features
class VaxDataset(Dataset):

    def __init__(self, path, test=False):

        self.path = path

        self.test = test

        self.features = []

        self.targets = []

        self.ids = []

        self.load_data()

    

    def load_data(self):

        with open(self.path, 'r') as text:

            for line in text:

                records = json.loads(line)

                features = featurize(records)

                

                for char_i in range(records['seq_scored']):

                    char_features = char_encode(char_i, features, FEATURE_SIZE)

                    self.features.append(char_features)

                    self.ids.append('%s_%d' % (records['id'], char_i))

                        

                if not self.test:

                    targets = np.stack([records['reactivity'], records['deg_Mg_pH10'], records['deg_Mg_50C']], axis=1)

                    self.targets.extend([targets[char_i] for char_i in range(records['seq_scored'])])

                    

    def __len__(self):

        return len(self.features)

    

    def __getitem__(self, index):

        if self.test:

            return self.features[index], self.ids[index]

        else:

            return self.features[index], self.targets[index], self.ids[index]
test_dataset = VaxDataset('../input/stanford-covid-vaccine/test.json', test=True)

test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=4, drop_last=False, pin_memory=True)
class Flatten(nn.Module):

    def forward(self, x):

        batch_size = x.shape[0]

        return x.view(batch_size, -1)



class VaxModel(nn.Module):

    def __init__(self):

        super(VaxModel, self).__init__()

        self.layers = nn.Sequential(

            nn.Dropout(0.2),

            nn.Conv1d(14, 32, 1, 1),

            nn.PReLU(),

            nn.BatchNorm1d(32),

            nn.Dropout(0.2),

            nn.Conv1d(32, 1, 1, 1),

            nn.PReLU(),

            Flatten(),

            nn.Dropout(0.2),

            nn.Linear(FEATURE_SIZE, 32),

            nn.PReLU(),

            nn.BatchNorm1d(32),

            nn.Dropout(0.2),

            nn.Linear(32, 3),

        )

    

    def forward(self, features):

        return self.layers(features)
model = VaxModel().cuda()
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv', index_col='id_seqpos')
model.load_state_dict(torch.load(WEIGHTS_PATH))

model.eval()

for features, ids in tqdm(test_dataloader):

    features = features.cuda().permute(0,2,1).float()

    predictions = model(features)

    sub.loc[ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = predictions.detach().cpu().numpy()
sub.head()
sub.to_csv('submission.csv')