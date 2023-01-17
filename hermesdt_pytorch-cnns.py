!pip install torchsummary > /dev/null
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_test = pd.read_csv("../input/X_test.csv")

X_train = pd.read_csv("../input/X_train.csv")

y_train = pd.read_csv("../input/y_train.csv")

sample = pd.read_csv("../input/sample_submission.csv")
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def feature_extraction(df):

    df['orientation'] = df['orientation_X'] + df['orientation_Y'] + df['orientation_Z']+ df['orientation_W']

    df['angular_velocity'] = df['angular_velocity_X'] + df['angular_velocity_Y'] + df['angular_velocity_Z']

    df['linear_acceleration'] = df['linear_acceleration_X'] + df['linear_acceleration_Y'] + df['linear_acceleration_Z']

    df['velocity_to_acceleration'] = df['angular_velocity'] / df['linear_acceleration']

    df['velocity_linear_acceleration'] = df['linear_acceleration'] * df['angular_velocity']

    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    df['euler_x'] = nx

    df['euler_y'] = ny

    df['euler_z'] = nz

    

    df['total_angular_velocity'] = (df['angular_velocity_X'] ** 2 + df['angular_velocity_Y'] ** 2 + df['angular_velocity_Z'] ** 2) ** 0.5

    df['total_linear_acceleration'] = (df['linear_acceleration_X'] ** 2 + df['linear_acceleration_Y'] ** 2 + df['linear_acceleration_Z'] ** 2) ** 0.5

    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']

    

    df['total_angle'] = (df['euler_x'] ** 2 + df['euler_y'] ** 2 + df['euler_z'] ** 2) ** 5

    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']

    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']

    return df

X_test = feature_extraction(X_test)

X_train = feature_extraction(X_train)
X_train.info()
X_train.describe()
cols = ["orientation_X",

"orientation_Y",

"orientation_Z",

"orientation_W",

"angular_velocity_X",

"angular_velocity_Y",

"angular_velocity_Z",

"linear_acceleration_X",

"linear_acceleration_Y",

"linear_acceleration_Z"]



num_cols = len(cols)

fig, axes = plt.subplots(nrows=np.ceil(len(cols)/3).astype(np.int), ncols=3)

for idx, ax in enumerate(np.array(axes).flatten()):

    if idx < len(cols):

        ax.hist(X_train[cols[idx]], bins=100)

        ax.set_title(cols[idx])



fig.set_figheight(7)

fig.set_figwidth(10)

fig.set_tight_layout(True)

plt.show()
#sns.pairplot(data=X_train[cols].sample(n=200))
X_train[["row_id", "series_id", "measurement_number"]].iloc[120:140]
y_train.head()
X_train.groupby(["series_id"]).series_id.count().nunique()
nunique_series_ids = X_train.series_id.nunique()

nunique_series_ids
nunique_surfaces = y_train.surface.nunique()

nunique_surfaces
cols = orig_cols = X_train.drop(["row_id", "series_id", "measurement_number"], axis=1).columns

# cols = orig_cols = [

# "orientation_X",

# "orientation_Y",

# "orientation_Z",

# "orientation_W",

# "angular_velocity_X",

# "angular_velocity_Y",

# "angular_velocity_Z",

# "linear_acceleration_X",

# "linear_acceleration_Y",

# "linear_acceleration_Z",

# 'orientation',

# 'angular_velocity',

# 'linear_acceleration',

# 'velocity_to_acceleration',

# 'velocity_linear_acceleration',

# ]



num_cols = len(orig_cols)
import torch

from torch import nn

from torch.nn import functional as F

from torch.utils import data

from fastai.tabular.data import Learner, DataBunch

from torchsummary import summary

from fastai import metrics

from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()

X_scaler.fit(X_train[cols])

X_train[cols] = X_scaler.transform(X_train[cols])
X_test[cols] = X_scaler.transform(X_test[cols])
y_train.surface.value_counts()
#others = ['fine_concrete', 'carpet', 'hard_tiles']

#y_train.loc[y_train.surface.isin(others),'surface'] = 'Other'

vcounts = y_train.surface.value_counts()

surfaces = i2s = vcounts.index.values

surfaces
# surfaces_weights = (1-(vcounts.values) / len(y_train))

surfaces_weights = np.ones(len(surfaces))

# surfaces_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 5]

s2i = {s:i for i,s in enumerate(surfaces)}

surfaces, surfaces_weights
#    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

#    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

#    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

#    

#    for col in raw_frame.columns[3:]:

#        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

#        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

#        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

#        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

#        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

#        

#        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

#        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
cols = [c + t for c in orig_cols for t in [""]]

num_cols = len(cols)



class Dataset(data.Dataset):

    def __init__(self, X, y):

        self.y = Dataset.convert_target(y.surface)

        self.X = Dataset.convert_df(X)

    

    @classmethod

    def convert_df(cls, df):

        batches = []

        for serie_id, group in df.groupby(["series_id"]):

            batch_data = []

            for col in orig_cols:

                v = group[col].values

                if col not in []:

                    batch_data.append(v)

                

            batches.append(batch_data)

        return np.array(batches)

    

    @classmethod

    def convert_target(cls, target):

        idxs = [s2i[s] for s in target.values]

        num_classes = len(surfaces)



        # return np.eye(num_classes)[idxs]

        return np.array(idxs)

    

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx):

        x = self.X[idx].reshape(1, num_cols, 128)

        return torch.Tensor(x), torch.Tensor([self.y[idx]]).long()



# cols = [c + t for c in orig_cols for t in ["", "_step1", "_step2", "_avg2"]]

# cols = set(cols) - {'orientation_X'}

# num_cols = len(cols)
np.random.seed(42)

valid_size = int(nunique_series_ids*0.2)



valid_series_ids = np.random.choice(X_train.series_id.unique(), replace=False, size=valid_size)

train_series_ids = np.setdiff1d(X_train.series_id.unique(), valid_series_ids)

# train_series_ids = np.concatenate([train_series_ids, extra_carpet_ids])



valid_idx = np.argwhere(X_train.series_id.isin(valid_series_ids)).reshape(-1)

train_idx = np.argwhere(X_train.series_id.isin(train_series_ids)).reshape(-1)

# train_idx = np.concatenate(

#     train_idx,

#     np.argwhere(X_train.series_id.isin(extra_carpet_ids)).reshape(-1)

# ).reshape(-1)

# np.setdiff1d(np.arange(len(X_train)), valid_idx)



valid_y_idx = np.argwhere(y_train.series_id.isin(valid_series_ids)).reshape(-1)

train_y_idx = np.argwhere(y_train.series_id.isin(train_series_ids)).reshape(-1)

# np.setdiff1d(np.arange(len(y_train)), valid_y_idx)





train_ds = Dataset(X_train.iloc[train_idx], y_train.iloc[train_y_idx])

train_dl = data.DataLoader(train_ds, batch_size=32, shuffle=True)



valid_ds = Dataset(X_train.iloc[valid_idx], y_train.iloc[valid_y_idx])

valid_dl = data.DataLoader(valid_ds, batch_size=32, shuffle=True)
y_train.iloc[train_y_idx].surface.value_counts()
train_count = y_train.iloc[train_y_idx].surface.value_counts()

valid_count = y_train.iloc[valid_y_idx].surface.value_counts()[train_count.index]

pd.DataFrame({"train": train_count/len(train_y_idx), "valid": valid_count/len(valid_y_idx)}, index=train_count.index).plot.bar()

plt.title("% of surfce samples per set")

plt.show()
db = DataBunch(train_dl, valid_dl)
torch.manual_seed(32)

class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)



class Model(nn.Module):

    def __init__(self, num_cols, num_cats):

        super().__init__()

        self.num_cats = num_cats

        self.num_cols = num_cols

        self.convs = nn.Sequential(

            # nn.Conv1d(num_cols, 32, 2, stride=2, dilation=1),

            nn.Conv2d(1, 4, (1,8), stride=(1,2), dilation=1),

            nn.ELU(),

            nn.Dropout2d(0.1),

            

            nn.Conv2d(4, 4, (1,2), stride=(1,2), dilation=1),

            nn.ELU(),

            nn.Dropout2d(0.3),



            nn.Conv2d(4, 8, (1,2), stride=(1,2), dilation=1),

            nn.ELU(),

            nn.Dropout2d(0.2),

            # nn.AvgPool2d(kernel_size=(1,4), )

            #nn.BatchNorm2d(256),

            # nn.LeakyReLU(),

        )

        

        self.convs2 = nn.Sequential(

            nn.Conv2d(1, 32, (num_cols,1), stride=(1,2), dilation=1),

            nn.Tanh(),

        )

        

        self.head = nn.Sequential(

            nn.Linear(4928, 1000),

            nn.Dropout2d(0.1),

            nn.BatchNorm1d(1000),

            nn.ReLU(inplace=True),

            nn.Linear(1000, 50),

            nn.Dropout2d(0.2),

            nn.BatchNorm1d(50),

            nn.ReLU(inplace=True),

            nn.Linear(50, num_cats),

            nn.Softmax(dim=1)

            # nn.LogSigmoid()

        )

    

    def forward(self, x):

        # output = self.layers(x)

        bs = x.size(0)

        c1 = self.convs(x).view(bs, -1)

        c2 = self.convs2(x).view(bs, -1)

        c = torch.cat([c1, c2], dim=1)

        # c = c1

        return self.head(c.view(bs, -1))

        # return x.view(bs, -1)



class CrossEntropyLoss(nn.Module):

    def forward(self, input, target):

        return F.cross_entropy(input, target.view(-1),

                               weight=torch.Tensor(surfaces_weights).cuda())



class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduction=False):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduction  = reduction 

    def forward(self, inputs, targets):

        t = torch.zeros_like(inputs)

        #print(t.scatter_(1, targets, 1))

        #print(inputs.size(), targets.size())

        targets = t

        if self.logits:

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)

        else:

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduction:

            return torch.mean(F_loss)

        else:

            return F_loss



model = Model(num_cols, len(surfaces)).cuda()
summary(model, input_size=(1, num_cols, 128))
learner = Learner(db, model,

                  loss_func=CrossEntropyLoss(),

                  # loss_func=FocalLoss(reduction=False, logits=True, alpha=0.25, gamma=1),

                  wd=0.1,

                  metrics=[metrics.accuracy, metrics.fbeta])
learner.lr_find(start_lr=1e-10, end_lr=10)

learner.recorder.plot()
learner.fit(15, 1e-3)
learner.recorder.plot_losses()
# learner.lr_find(start_lr=1e-10, end_lr=10)

# learner.recorder.plot()
# learner.fit(4, 1e-6)
# learner.recorder.plot_losses()
accuracy_df = pd.DataFrame({"accuracy": []})

for s in surfaces:

    y_idxs = np.argwhere((y_train.surface == s) & y_train.series_id.isin(valid_series_ids)).reshape(-1)

    y = y_train.iloc[y_idxs]

    X = X_train[X_train.series_id.isin(y.series_id)]

    d = Dataset.convert_df(X).reshape(-1, 1, num_cols, 128)

    preds = model(torch.Tensor(d).cuda())

    targets = torch.Tensor(Dataset.convert_target(y.surface)).long().cuda()

    accuracy = "%.4f" % metrics.accuracy(preds, targets).data.item()

    # print("accuracy for", s, accuracy)

    accuracy_df.loc[s] = {"accuracy": accuracy}

train_count = y_train.iloc[train_y_idx].surface.value_counts()

valid_count = y_train.iloc[valid_y_idx].surface.value_counts()[train_count.index]

pd.DataFrame({

    "train": train_count/len(train_y_idx),

    "valid": valid_count/len(valid_y_idx),

    "accuracy": accuracy_df.loc[train_count.index].accuracy.astype(np.float)

}, index=train_count.index).plot.bar()

plt.title("% of surfce samples per set")

plt.show()
def score(X, y, convert=True):

    if convert:

        X = Dataset.convert_df(X_train)

        y = Dataset.convert_target(y)

        

    preds = model(torch.Tensor(X.reshape(-1, 1, num_cols, 128)).cuda())

    targets = torch.Tensor(y).long().cuda()

    return metrics.accuracy(preds, targets)



max_score = score(X_train, y_train.surface)

max_score
diffs = []

for idx,col in enumerate(cols):

    d = train_ds.X.copy()

    new_order = np.random.choice(np.arange(128), replace=False, size=128)

    d[:,0] = d[:,idx,new_order]

    diffs.append(max_score - score(d, train_ds.y, convert=False))

order = np.argsort(diffs)

sorted_diffs = np.array(diffs)[order]

sorted_cols = np.array(cols)[order]
plt.figure(figsize=(10, 8), dpi=80)

plt.barh(np.arange(len(sorted_diffs)), sorted_diffs)

plt.yticks(np.arange(len(sorted_cols)), sorted_cols)

plt.tight_layout()

plt.show()
t = Dataset.convert_df(X_test).reshape(-1, 1, num_cols, 128)
probs, idxs = torch.exp(model(torch.Tensor(t).cuda())).max(dim=1)
preds = [i2s[i] for i in idxs.cpu().detach().numpy()]
train_count = y_train.iloc[train_y_idx].surface.value_counts()

valid_count = y_train.iloc[valid_y_idx].surface.value_counts()[train_count.index]

pd.DataFrame({

    "train": train_count/len(train_y_idx),

    "valid": valid_count/len(valid_y_idx),

}, index=train_count.index).plot.bar()

plt.title("% of surfce samples per set")

plt.show()
train_count = pd.value_counts(preds)[train_count.index]

pd.value_counts

pd.DataFrame({

    "train": train_count/len(train_y_idx),

}, index=train_count.index).plot.bar()

plt.title("% of surfce samples per set")

plt.show()
submission =  pd.read_csv("../input/sample_submission.csv")
submission['surface'] = preds
submission.to_csv("submission.csv", index=False)
X_test.head()
surfaces
preds = torch.exp(

    model(torch.Tensor(Dataset.convert_df(X_train).reshape(-1, 1, num_cols, 128)).cuda())

)

targets = torch.Tensor(Dataset.convert_target(y_train.surface)).long().cuda()

pred_ids = preds.max(dim=1)[1].cpu().detach().numpy()

correct_ids = np.argwhere((y_train.surface == 'tiled') & (pred_ids == 4)).reshape(-1)



series_ids = correct_ids[40:44]

plt.figure(figsize=(14, 10), dpi=80)

for idx, s_id in enumerate(series_ids):

    plt.subplot(len(series_ids), 2, idx*2+1)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_X)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_Y)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_Z)

    plt.legend()

    plt.title(y_train.loc[s_id].surface)



    plt.subplot(len(series_ids), 2, idx*2+2)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_X)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_Y)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_Z)

    plt.legend()

    plt.title(y_train.loc[s_id].surface)

plt.tight_layout(True)

plt.show()
preds = torch.exp(

    model(torch.Tensor(Dataset.convert_df(X_train).reshape(-1, 1, num_cols, 128)).cuda())

)

targets = torch.Tensor(Dataset.convert_target(y_train.surface)).long().cuda()

pred_ids = preds.max(dim=1)[1].cpu().detach().numpy()

incorrect_ids = np.argwhere((y_train.surface == 'tiled') & (pred_ids != 4)).reshape(-1)



series_ids = incorrect_ids[40:44]

plt.figure(figsize=(14, 10), dpi=80)

for idx, s_id in enumerate(series_ids):

    plt.subplot(len(series_ids), 2, idx*2+1)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_X)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_Y)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].angular_velocity_Z)

    plt.legend()

    plt.title(y_train.loc[s_id].surface)



    plt.subplot(len(series_ids), 2, idx*2+2)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_X)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_Y)

    plt.plot(np.arange(128), X_train[X_train.series_id == s_id].linear_acceleration_Z)

    plt.legend()

    plt.title(y_train.loc[s_id].surface)

plt.tight_layout(True)

plt.show()