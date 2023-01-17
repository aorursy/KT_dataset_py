import numpy as np

import pandas as pd

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error

import gc

from tqdm import tqdm_notebook as tqdm

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



from kaggle.competitions import nflrush



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '/kaggle/input/nfl-big-data-bowl-2020/train.csv'

train = pd.read_csv(train_path)

print(train.shape)
use_cols = [

    'GameId', 

    'PlayId', 

    'Team',

    'Yards',

    'X',

    'Y',

    'PossessionTeam',

    'HomeTeamAbbr',

    'VisitorTeamAbbr',

    'Position',

]

train = train[use_cols]

train.head()
"""

Find an offense team.

ref: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112314#latest-648026

"""

def fe_is_offence(row):

    if row["Team"] == "home":

        if row["HomeTeamAbbr"] == row["PossessionTeam"]:

            return 1

        else:

            return 0

    elif row["Team"] == "away":

        if row["VisitorTeamAbbr"] == row["PossessionTeam"]:

            return 1

        else:

            return 0



def fe_is_offence_from_position(row, off_team):

    if row["Team"] == off_team:

        return 1

    else:

        return 0

        

# def run_fe_is_offence(df):

#     df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)

    

#     if (df['is_offence'].values == 0).all():

#         off_team = df[df['Position']=='QB']['Team'].values[0]

#         df['is_offence'] = df.apply(lambda row: fe_is_offence_from_position(row, off_team), axis=1)



"""

bugfix

"""

def run_fe_is_offence(df):

    df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)

    

    check_is_offence = df.groupby('PlayId')['is_offence'].nunique()

    is_offence_not_found_idx = check_is_offence[check_is_offence!=2].index

    not_found_df = df[df['PlayId'].isin(is_offence_not_found_idx)]

    found_df = df[~df['PlayId'].isin(is_offence_not_found_idx)]

#     print('is_offence found: {}'.format(len(found_df)))

#     print('is_offence not found: {}'.format(len(not_found_df)))



    for u_play_id in not_found_df['PlayId'].unique():

        tmp_df = not_found_df[not_found_df['PlayId']==u_play_id]

        pos_list = [pos for pos in tmp_df['Position'].unique() if pos in ['QB', 'RB', 'WR', 'TE']]

        

        if len(pos_list) > 0:

            off_team = tmp_df[tmp_df['Position']==pos_list[0]]['Team'].values[0]

#         else:

#             print('Offence position not found')

#             import pdb;pdb.set_trace()



        target_idx = not_found_df.query('PlayId==@u_play_id and Team==@off_team').index

        not_found_df.loc[target_idx, 'is_offence'] = 1

    

    df = pd.concat([found_df, not_found_df], sort=False)

#     print('done df: {}'.format(df.shape))

    return df
def run_group_fe(df, group_key, aggs):

    

    group_df = df.groupby(group_key).agg(aggs)



    new_cols = [col[0]+'_'+col[1] for col in group_df.columns]

    group_df.columns = new_cols

    group_df.reset_index(inplace=True)

        

    return group_df



def adjust_group_df(group_df, is_train):

    offence_df = group_df[group_df['is_offence']==1]

    deffence_df = group_df[group_df['is_offence']==0]



    del group_df['is_offence']

    del offence_df['is_offence']

    del deffence_df['is_offence']

    

    if is_train:

        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]

        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]

    else:

        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]

        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]

        

    offence_df.columns = off_cols

    deffence_df.columns = deff_cols

    if is_train: del deffence_df['Yards']

    

    adjusted_group_df = pd.merge(offence_df, deffence_df, on=['GameId', 'PlayId'])

    

    return adjusted_group_df
train = run_fe_is_offence(train)
train.head()
train_group_key = ['GameId', 'PlayId', 'is_offence', 'Yards']

aggs = {

    'X': ['mean', 'max', 'min', 'median'],

    'Y': ['mean', 'max', 'min', 'median'],

}

is_train = True

group_df = run_group_fe(train, train_group_key, aggs)

adjusted_group_df = adjust_group_df(group_df, is_train)
print(adjusted_group_df.shape)

adjusted_group_df.head()
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
def getNFLY(df):

    y = np.zeros(shape=(df.shape[0], 199))

    for i, yard in enumerate(df['Yards'].values):

        y[i, yard+99:] = np.ones(shape=(1, 100-yard))

    return y
def generate_dataloader(df, y_val, batch_size, train_idx, valid_idx):

    train_x, train_y = df.iloc[train_idx].values, y_val[train_idx]

    valid_x, valid_y = df.iloc[valid_idx].values, y_val[valid_idx] 

    

    train_x = torch.from_numpy(train_x)

    train_y = torch.from_numpy(train_y)

    valid_x = torch.from_numpy(valid_x)

    valid_y = torch.from_numpy(valid_y)

    

    train_dataset = TensorDataset(train_x, train_y)

    valid_dataset = TensorDataset(valid_x, valid_y)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, valid_dataset, train_loader, valid_loader
"""

ref: https://github.com/Bjarten/early-stopping-pytorch

"""

class EarlyStopping:

    def __init__(self, patience=2, verbose=False):

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf



    def __call__(self, val_loss, model, save_name):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model, save_name)

        elif score < self.best_score:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model, save_name)

            self.counter = 0



    def save_checkpoint(self, val_loss, model, save_name):

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ...')

            print("Save model: {}".format(save_name))

        torch.save(model.state_dict(), save_name)

        self.val_loss_min = val_loss
class NFL_NN(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()



        self.fc1 = nn.Linear(in_features, 216)

        self.bn1 = nn.BatchNorm1d(216)

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(216, 512)

        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 216)

        self.relu3 = nn.ReLU()

        self.dout3 = nn.Dropout(0.2)

        self.out = nn.Linear(216, out_features)

        self.out_act = nn.Sigmoid()

        

    def forward(self, input_):

        a1 = self.fc1(input_)

        bn1 = self.bn1(a1)

        h1 = self.relu1(bn1)

        a2 = self.fc2(h1)

        h2 = self.relu2(a2)

        a3 = self.fc3(h2)

        h3 = self.relu3(a3)

        dout3 = self.dout3(h3)

        a5 = self.out(dout3)

        y = self.out_act(a5)

        return a5
def model_eval(model, dataset, data_loader, out_features, batch_size):

    model.eval()



    preds = np.zeros((len(dataset), out_features))

    with torch.no_grad():

        for i, eval_x_batch in enumerate(data_loader):

                eval_values = eval_x_batch[0].float()

                pred = model(eval_values)

                preds[i * batch_size:(i + 1) * batch_size] = pred

                

    return preds
def run_train_nn(train_df, epoch, batch_size):

    oof_crps_list = []



    fold = GroupKFold(n_splits=5)



    train_y = getNFLY(train_df)



    oof_preds = np.ones((len(train_df), train_y.shape[1]))



    feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]



    print('use feats: {}'.format(len(feats)))



    for n_fold, (train_idx, valid_idx) in enumerate(fold.split(train_df, train_y, groups=train_df['GameId'])):

        print('Fold: {}'.format(n_fold+1))



        early_stopping = EarlyStopping(patience=2, verbose=True)



        train_dataset, valid_dataset, train_loader, valid_loader = generate_dataloader(train_df[feats], train_y, batch_size, train_idx, valid_idx)



        print('train: {}, valid: {}'.format(len(train_dataset), len(valid_dataset)))



        in_features = train_df[feats].shape[1]

        out_features = train_y.shape[1]

        model = NFL_NN(in_features, out_features)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)



        for idx in range(epoch):

            print('Training epoch {}'.format(idx+1))

            train_batch_loss_sum = 0



            for param in model.parameters():

                param.requires_grad = True



            model.train()

            for x_batch, y_batch in tqdm(train_loader):

                y_pred = model(x_batch.float())

                loss = torch.sqrt(criterion(y_pred.float(), y_batch.view((len(y_batch), out_features)).float()))

                train_batch_loss_sum += loss.item()



                del x_batch

                del y_batch



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                torch.cuda.empty_cache()

                gc.collect()



            train_epoch_loss = train_batch_loss_sum / len(train_loader)



            valid_y_pred = model_eval(model, valid_dataset, valid_loader, out_features, batch_size)

            valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))



            oof_preds[valid_idx] = valid_y_pred



            print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))



            model_save_name = 'checkpoint_fold_{}.pt'.format(n_fold+1)

            early_stopping(valid_crps, model, model_save_name)



            if early_stopping.early_stop:

                print("Early stopping")

                break

                

        oof_crps_list.append(-early_stopping.best_score)



        del model, criterion, optimizer

        gc.collect()



    print('DONE OOF MEAN CRPS: {:.5f}'.format(np.mean(oof_crps_list)))

    print('DONE OOF ALL CRPS: {:.5f}'.format(np.sum(np.power(oof_preds - train_y, 2))/(199*len(oof_preds))))

        

    return oof_preds
seed_everything(1234)

epoch = 10

batch_size = 1012

oof_preds = run_train_nn(adjusted_group_df, epoch, batch_size)
def min_max_scaler(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))



def scale_predict(preds):

    y_pred = preds.copy()

    adjust_preds = np.zeros((len(y_pred), y_pred.shape[1]))

    for idx, pred in enumerate(y_pred):

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

        x = min_max_scaler(pred)

        adjust_preds[idx, :] = x



    adjust_preds[:, -1] = 1

    adjust_preds[:, 0] = 0

    return adjust_preds
def run_nn_nfl_inference(test_df, model_path, batch_size):

    feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]

    

    test = torch.from_numpy(test_df[feats].values)

    test_dataset = TensorDataset(test)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    

    in_features = test_df[feats].shape[1]

    out_features = 199

    model = NFL_NN(in_features, out_features)

    model.load_state_dict(torch.load(model_path))

    nfl_pred = model_eval(model, test_dataset, test_loader, out_features, batch_size)

    del model

    gc.collect()

    return nfl_pred
def run_nn_nfl_pipeline(df, sample, models, batch_size):

    nfl_pred = np.zeros((len(df), 199))

    for idx, path in enumerate(models):

        nfl_pred += run_nn_nfl_inference(df, path, batch_size)/len(models)



    adjust_nfl_pred = scale_predict(nfl_pred)



    preds_df = pd.DataFrame(data=adjust_nfl_pred[0].reshape(1, 199), columns=sample.columns)

    env.predict(preds_df)

    

    return preds_df
nn_model_path_list = [

    'checkpoint_fold_1.pt', 'checkpoint_fold_2.pt', 'checkpoint_fold_3.pt', 'checkpoint_fold_4.pt', 'checkpoint_fold_5.pt', 

]



env = nflrush.make_env()

result_df = None

is_train = False

test_group_key = ['GameId', 'PlayId', 'is_offence']



for (test_df, sample_prediction_df) in tqdm(env.iter_test()):

    

    test_df = run_fe_is_offence(test_df)

    test_group_df = run_group_fe(test_df, test_group_key, aggs)

    test_adjusted_group_df = adjust_group_df(test_group_df, is_train)

    

    tmp_result_df = run_nn_nfl_pipeline(test_adjusted_group_df, sample_prediction_df, nn_model_path_list, batch_size)

    

    if result_df is None:

        result_df = tmp_result_df

    else:

        result_df = pd.concat([result_df, tmp_result_df], sort=False)

        

result_df.drop_duplicates(inplace=True)



env.write_submission_file()
print(result_df.shape)

result_df.head(30)