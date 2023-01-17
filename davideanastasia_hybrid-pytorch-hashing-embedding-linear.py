!pip install h3
import pandas as pd

import numpy as np



# import seaborn as sns

# import matplotlib

#

# from matplotlib import pyplot as plt



import gc

import math

import random

import mmh3



from h3 import h3



import gzip

from tqdm import tqdm



pd.set_option('display.max_columns', 250)  # or 1000

# pd.set_option('display.max_rows', None)  # or 1000

# pd.set_option('display.max_colwidth', -1)  # or 199
CITY = [

    'Atlanta',

    'Boston',

    'Chicago',

    'Philadelphia'

]



INPUTS = [

    'RowId',

    'IntersectionId',

    'Latitude',

    'Longitude',

    'EntryStreetName',

    'ExitStreetName',

    'EntryHeading',

    'ExitHeading',

    'Hour',

    'Weekend',

    'Month',

    'Path',

    'City'

]



OUTPUTS = [

    'TotalTimeStopped_p20',

    'TotalTimeStopped_p40',

    'TotalTimeStopped_p50',

    'TotalTimeStopped_p60',

    'TotalTimeStopped_p80',

    'TimeFromFirstStop_p20',

    'TimeFromFirstStop_p40',

    'TimeFromFirstStop_p50',

    'TimeFromFirstStop_p60',

    'TimeFromFirstStop_p80',

    'DistanceToFirstStop_p20',

    'DistanceToFirstStop_p40',

    'DistanceToFirstStop_p50',

    'DistanceToFirstStop_p60',

    'DistanceToFirstStop_p80'

]



LABELS = [

    "TotalTimeStopped_p20",

    "TotalTimeStopped_p50",

    "TotalTimeStopped_p80",

    "DistanceToFirstStop_p20",

    "DistanceToFirstStop_p50",

    "DistanceToFirstStop_p80"

]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler

from sklearn.impute import SimpleImputer

N_BINS = 10000



def get_id(value, n_components):

    return mmh3.hash(value, signed=False) % n_components
h3_precision = 7

compass_brackets = {

    "N": 0,

    "NE": 1,

    "E": 2,

    "SE": 3,

    "S": 4,

    "SW": 5,

    "W": 6,

    "NW": 7

}



missing_values = {

    'EntryStreetName': 'MISSING',

    'ExitStreetName': 'MISSING'

}



road_encoding = {

    'Road': 1,

    'Street': 2,

    'Avenue': 2,

    'Drive': 3,

    'Broad': 3,

    'Boulevard': 4

}



city_center_encoding = {

    'Atlanta':  (math.radians(33.753746), math.radians(-84.386330)),

    'Boston': (math.radians(42.361145), math.radians(-71.057083)),

    'Chicago': (math.radians(41.881832), math.radians(-87.623177)),

    'Philadelphia': (math.radians(39.952583), math.radians(-75.165222))

}



R = 6371  # radius of the earth in km

def get_geo_distance(lat1, lng1, lat2, lng2):

    x = (lng2 - lng1) * math.cos(0.5*(lat2 + lat1))

    y = lat2 - lat1

    return R * math.sqrt(x*x + y*y)



def get_distance_from_city_center(lat1, lng1, city):

    # already in radians, as stored as such in the hashset

    lat2, lng2 = city_center_encoding[city]

    

    return get_geo_distance(math.radians(lat1), math.radians(lng1), lat2, lng2)



def get_road_type(street_name):

    for road_type in road_encoding.keys():

        if road_type in street_name:

            return road_encoding[road_type]

    return 0



def compute_rotation(entry_heading, exit_heading):

    entry_idx = compass_brackets[entry_heading]

    exit_idx = compass_brackets[exit_heading]

    

    return exit_idx - entry_idx + 8 if entry_idx > exit_idx else exit_idx - entry_idx



def get_hour_group(x):

    if x < 8:

         return "midnight"

    elif x < 12:

         return "morning"

    elif x < 16:

         return "afternoon"

    elif x < 19:

         return "evening"

    else:

         return "midnight"



def transform_dataframe(df, xf=None):

    df = df.copy().fillna(value=missing_values)

    

    print('build distance from center')

    intersections = df[['City', 'IntersectionId', 'Latitude', 'Longitude']].drop_duplicates()

    intersections['DistanceFromCenter'] = intersections.apply(lambda x: get_distance_from_city_center(x[2], x[3], x[0]), axis=1)

    

    df = pd.merge(df, intersections[['City', 'IntersectionId', 'DistanceFromCenter']], on=['City', 'IntersectionId'])

    

    print('remove rare months')

    df.loc[df['Month'] == 1, 'Month'] = 12

    df.loc[df['Month'] == 5, 'Month'] = 6

    

    print('build cross-features')

    df['f1'] = df['EntryStreetName'].map(lambda x: get_id(x, N_BINS))

    df['f2'] = df['ExitStreetName'].map(lambda x: get_id(x, N_BINS))

    df['f5'] = (df['City'] + ':' + df['IntersectionId'].map(str)).map(lambda x: get_id(x, N_BINS))

    df['f7'] = df[['Latitude', 'Longitude']].apply(lambda x: h3.geo_to_h3(x[0], x[1], h3_precision), axis=1).map(lambda x: get_id(x, N_BINS))

    df['f8'] = (df['EntryStreetName'] + ':' + df['ExitStreetName']).map(lambda x: get_id(x, N_BINS))

    df['HourGroup'] = df['Hour'].map(lambda x: get_hour_group(x))

    

    df['f3'] = df['EntryHeading'] + ':' + df['ExitHeading']

    df['f4'] = df[['EntryHeading', 'ExitHeading']].apply(lambda x: compute_rotation(x[0], x[1]), axis=1)

    df['f6'] = df['Hour'].map(str) + ':' + df['Weekend'].map(str)

    

    df['EntryType'] = df['EntryStreetName'].map(lambda x: get_road_type(x))

    df['ExitType'] = df['ExitStreetName'].map(lambda x: get_road_type(x))

      

    df["SameStreetExact"] = (df["EntryStreetName"] == df["ExitStreetName"]).astype(int)

    

    if xf is None:

        y = df[LABELS].values

        df = df.drop(columns=OUTPUTS)

        

        print('build transformers')

        

        xf = ColumnTransformer(transformers=[

            ('cat_02', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['EntryHeading', 'ExitHeading', 'City', 'Month', 'f3', 'f4', 'f6', 'EntryType', 'ExitType', 'HourGroup']),

            ('num_01', StandardScaler(), ['f4', 'Latitude', 'Longitude']),

            ('num_02', SimpleImputer(strategy='constant', fill_value=0), ['Weekend', 'SameStreetExact']),

            ('num_03', StandardScaler(), ['DistanceFromCenter'])

        ], verbose=True)

        xf.fit(df)

    else:

        y = None



    print('apply transform')

    X1 = xf.transform(df)

    X2 = df[['f1', 'f2', 'f5', 'f7', 'f8']].values

    ids = df['RowId'].values

    

    return X1, X2, y, ids, xf, df

import torch



from torch.autograd import Variable 

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device('cpu')

device
class EarlyStopManager:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_filename='checkpoint.pt'):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta

        self.checkpoint_filename = checkpoint_filename



    def __call__(self, val_loss, model):

        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.checkpoint_filename)

        self.val_loss_min = val_loss
class MSEAccumulator:

    def __init__(self):

        self._n_samples = 0

        self._sq_e = 0.0

        

    def add(self, n_samples, partial_mse):

        self._n_samples += n_samples

        self._sq_e += (partial_mse * n_samples)

        

    def mse(self):

        return self._sq_e/self._n_samples
class MyDataset(Dataset):

    def __init__(self, X1, X2, y, row_id):

        if y is not None:

            assert(X1.shape[0] == y.shape[0])

            self._y = torch.from_numpy(y)

        else:

            self._y = torch.zeros(X1.shape[0], 6)

        

        assert(X1.shape[0] == row_id.shape[0])

        assert(X2.shape[0] == row_id.shape[0])

        self._X1 = X1

        self._X2 = X2

        self._row_id = torch.from_numpy(row_id)



    def __len__(self):

        return self._X1.shape[0]



    def __getitem__(self, idx):       

        return self._X1[idx], self._X2[idx], self._y[idx], self._row_id[idx]

class Net(torch.nn.Module):

    def __init__(self, lin_input_size, embs_input_size, embs_input_cats, lin_output_size, embs_output_size):

        super(Net, self).__init__()

#         self.input1 = torch.nn.Linear(lin_input_size, lin_output_size)

        self.input2 = torch.nn.Embedding(embs_input_size, embs_output_size)

#         self.input2.weight.data.uniform_(-1.0/embs_output_size, 1.0/embs_output_size)

        self.net = torch.nn.Sequential(

#             torch.nn.BatchNorm1d(lin_input_size + (embs_output_size * embs_input_cats)),

            torch.nn.Linear(lin_input_size + (embs_output_size * embs_input_cats), 256),

            torch.nn.Tanh(),

            torch.nn.Linear(256, 6),

#             torch.nn.ReLU()

        )

        

    def forward(self, x1, x2):

#         print('x2/1', x2.shape)

        x2 = self.input2(x2)

        batch_size, height, width = x2.shape

        

#         print('x2/2', x2, x2.shape)

        x2 = x2.reshape(batch_size, -1)

#         print('x2/3', x2, x2.shape)

        

#         print('x1/1', x1.shape)

#         x1 = self.input1(x1)

#         print('x1/2', x1.shape)

        

        x3 = torch.cat([x1, x2], 1)

#         print('x3/1', x3.shape)

        x3 = self.net(x3)

#         print('x3/2', x3.shape)

        return x3
def run_training(lin_input_size, embs_input_cats, dl_train, dl_val, max_epochs=50):

    model = Net(lin_input_size=lin_input_size, 

                embs_input_size=N_BINS,

                embs_input_cats=embs_input_cats,

                lin_output_size=128,

                embs_output_size=64).to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.00005, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # lr=0.005) # , weight_decay=0.01)

    

    early_stop_manager = EarlyStopManager(verbose=True, patience=5)

    for epoch in range(max_epochs):

        model.train()

        mse_acc_train = MSEAccumulator()

        for ix, (_X1, _X2, _y, _row_id) in enumerate(dl_train):

            _X1 = Variable(_X1).to(device).float()

            _X2 = Variable(_X2).to(device)

            _y = Variable(_y).to(device).float()



            # zero parameters

            optimizer.zero_grad() 



            # forward + loss + backward + step

            _y_pred = model(_X1, _X2)

            loss = criterion(_y_pred, _y)

            loss.backward()

            optimizer.step()



            mse_acc_train.add(_y.shape[0], loss.item())

            if (ix + 1) % 500 == 0:

                print('train/ epoch {}, batch {}, samples {}, loss {:.4f} {:.4f}'.format(

                    epoch, 

                    ix, 

                    (epoch * train_data.shape[0]) + ((ix + 1) * _y.shape[0]),

                    math.sqrt(loss.item()),

                    math.sqrt(mse_acc_train.mse())

                ))



        model.eval()

        mse_acc_val = MSEAccumulator()

        for ix, (_X1, _X2, _y, _row_id) in enumerate(dl_val):

            _X1 = Variable(_X1).to(device).float()

            _X2 = Variable(_X2).to(device)

            _y = Variable(_y).to(device).float()



            _y_pred = model(_X1, _X2)

            loss = criterion(_y_pred, _y)



            mse_acc_val.add(_y.shape[0], loss.item())



        train_loss = math.sqrt(mse_acc_train.mse())

        valid_loss = math.sqrt(mse_acc_val.mse())

        print('summary/ epoch {}, train loss: {:.4f}, validation loss: {:.4f}'.format(

            epoch, train_loss, valid_loss))



        early_stop_manager(valid_loss, model)

        if early_stop_manager.early_stop:

            print('train/ early stop!')

            break



    model.load_state_dict(torch.load('checkpoint.pt'))

    

    return model
def write_submission(out_file, model, dl_test):    

    for ix, (_X1, _X2, _, _row_id) in enumerate(dl_test):

        _X1 = Variable(_X1).to(device).float()

        _X2 = Variable(_X2).to(device)

        _y_pred = model(_X1, _X2)



        for row_id, preds in zip(_row_id, _y_pred):        

            for pred_id, pred in enumerate(preds.cpu().detach().numpy()):

    #             print(row_id, pred_id, preds, pred)

                out_file.write('{}_{},{:.6f}\n'.format(row_id, pred_id, max(pred, 0.0)))
def get_train_validation_splits(num_train):

    indices = list(range(num_train))

    np.random.shuffle(indices)

    split = int(np.floor(0.10 * num_train))

    # split = 75000

    return indices[split:], indices[:split]

# train_idx, valid_idx = indices[-1000:], indices[:1000]
def build_submission(train_data, test_data):

    with open('submission.csv', 'w') as out_file:

        out_file.write('TargetId,Target\n')

        for curr_city in CITY:

            print('city: {}'.format(curr_city))

            curr_train_data = train_data[train_data['City'] == curr_city]

            curr_test_data = test_data[test_data['City'] == curr_city]



            print(curr_train_data.shape, curr_test_data.shape)



            X1_train, X2_train, y_train, ids_train, xf, train_df = transform_dataframe(curr_train_data)

            print('train', X1_train.shape, X2_train.shape, y_train.shape, ids_train.shape)



            X1_test, X2_test, y_test, ids_test, _, test_df = transform_dataframe(curr_test_data, xf)

            print('test', X1_test.shape, X2_test.shape, y_test, ids_test.shape)



            train_idx, valid_idx = get_train_validation_splits(X1_train.shape[0])

            print('splits', len(train_idx), len(valid_idx))



            ds_train = MyDataset(X1=X1_train, X2=X2_train, y=y_train, row_id=ids_train)

            dl_train = DataLoader(ds_train, batch_size=128, sampler=SubsetRandomSampler(train_idx))

            dl_val = DataLoader(ds_train, batch_size=128, sampler=SubsetRandomSampler(valid_idx))



            print('start training')

            model = run_training(lin_input_size=X1_train.shape[1],

                                 embs_input_cats=X2_train.shape[1],

                                 dl_train=dl_train,

                                 dl_val=dl_val,

                                 max_epochs=250)



            ds_test = MyDataset(X1=X1_test, X2=X2_test, y=None, row_id=ids_test)

            dl_test = DataLoader(ds_test, batch_size=1000, shuffle=False)



            print('write submission file')

            write_submission(out_file, model, dl_test)
DATA_FOLDER = '/kaggle/input/bigquery-geotab-intersection-congestion/'

# DATA_FOLDER = 'bigquery-geotab-intersection-congestion/'
train_data = pd.read_csv(DATA_FOLDER + 'train.csv')

test_data = pd.read_csv(DATA_FOLDER + 'test.csv')



print(train_data.shape)

print(test_data.shape)
# Sample input dataset?



# train_data = train_data.sample(n=10000)

# test_data = test_data.sample(n=10000)



# print(train_data.shape)

# print(test_data.shape)
%%time

build_submission(train_data, test_data)