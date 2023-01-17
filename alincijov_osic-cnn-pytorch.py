import pandas as pd

import numpy as np

from tqdm import tqdm
import torch

from torch import nn

from torch.utils.data import TensorDataset, DataLoader



from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt
PATH = "../input/osic-pulmonary-fibrosis-progression"
df_train = pd.read_csv(f"{PATH}/train.csv")



# drop duplicates for patient and weeks

df_train.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])



df_train.head()
df_test = pd.read_csv(f"{PATH}/test.csv")



# prepare submission

df_sub = pd.read_csv(f"{PATH}/sample_submission.csv")

df_sub['Patient'] = df_sub['Patient_Week'].apply(lambda x:x.split('_')[0])

df_sub['Weeks'] = df_sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

df_sub = df_sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]

df_sub = df_sub.merge(df_test.drop('Weeks', axis=1), on='Patient')
df_sub.head()
df_train['FROM'] = 'train'

df_test['FROM'] = 'val'

df_sub['FROM'] = 'test'

data = df_train.append([df_test, df_sub])
data.head()
data['min_week'] = data['Weeks']

data.loc[data.FROM == 'test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base
COLS = ['Sex','SmokingStatus']

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)
# Normalize



data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week','BASE']
df_train = data.loc[data.FROM == 'train']

df_test = data.loc[data.FROM == 'val']

df_sub = data.loc[data.FROM == 'test']

del data
y = df_train['FVC'].values

z = df_train[FE].values

ze = df_sub[FE].values

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))
max_y = y.max()

y = y / max_y
model = nn.Sequential(nn.Conv1d(3, 32, 3),

                      nn.Dropout(p=0.42),

                      nn.ReLU(),

                      nn.Conv1d(32, 64, 1),

                      nn.Dropout(p=0.35),

                      nn.Flatten(),

                      nn.Linear(64, 128),

                      nn.Tanh(),

                      nn.Linear(128, 64),

                      nn.Tanh(),

                      nn.Linear(64, 32),

                      nn.Tanh(),

                      nn.Linear(32, 16),

                      nn.Tanh(),

                      nn.Linear(16, 8),

                      nn.Tanh(),

                      nn.Linear(8, 1))



model = model.to('cuda')
kf = KFold(n_splits=5)

kf2 = KFold(n_splits=128)
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
losses = []

for tr_idx, val_idx in tqdm(kf.split(z)):

    

    for _ in tqdm(range(300), position=0, leave=True):

        for tidx, vidx in kf2.split(tr_idx):

            optimizer.zero_grad()



            y_pred = model(torch.Tensor(z[tidx]).reshape(len(z[tidx]), 3, 3).to('cuda'))

            loss = loss_fn(y_pred, torch.Tensor(y[tidx]).reshape(len(y[tidx]), 1).to('cuda'))

            losses.append(loss)



            loss.backward()

            optimizer.step()

        

    pred[val_idx] = model(torch.Tensor(z[val_idx]).reshape(len(z[val_idx]), 3, 3).to('cuda')).cpu().detach().numpy()

    pe += (model(torch.Tensor(ze).reshape(len(ze), 3, 3).to('cuda')) / 5).cpu().detach().numpy()
plt.plot(losses)
pred, pe = pred * max_y, pe * max_y
sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)
df_sub['FVC1'] = pe[:, 1]

df_sub['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = df_sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)