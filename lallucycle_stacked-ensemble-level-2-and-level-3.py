import os

import random

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge, ARDRegression

from sklearn.svm import LinearSVR, NuSVR, SVR

from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse
import warnings

warnings.filterwarnings("ignore")
files = [i for i in os.listdir('../input') if ('new-d' in i) or ('25-class-new' in i)]

sorted(files)
df1 = pd.read_csv('../input/'+'cnn-1-pt-regression-swish-activation-new-d'+'/OOF_preds.csv').sort_values(by='row')['preds']

df2 = pd.read_csv('../input/'+'1-pt-regression-multi1-25-class-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df3 = pd.read_csv('../input/'+'fcnn-cms-1-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df4 = pd.read_csv('../input/'+'1-pt-regression-swiss-activation-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df5 = pd.read_csv('../input/'+'cnn-1-pt-regression-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df6 = pd.read_csv('../input/'+'cnn-1-pt-regression-multi1-25-class-new'+'/OOF_preds.csv').sort_values(by='row')['preds']

df7 = 1/pd.read_csv('../input/'+'pt-regression-new-loss-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df8 = 1/pd.read_csv('../input/'+'pt-regression-multi1-new-dat'+'/OOF_preds.csv').sort_values(by='row')['preds']

df9 = 1/pd.read_csv('../input/'+'pt-regression-swiss-activation-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df10 = 1/pd.read_csv('../input/'+'cnn-pt-regression-multi1-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

df11 = 1/pd.read_csv('../input/'+'cnn-pt-regression-swiss-activation-new-dat'+'/OOF_preds.csv').sort_values(by='row')['preds']

# df12 = 1/pd.read_csv('../input/'+'cnn-pt-regression-new-loss-new-data'+'/OOF_preds.csv').sort_values(by='row')['preds']

# df13 = pd.read_csv('../input/'+'pt-class-focal-loss-new-data'+'/OOF_preds.csv').sort_values(by='row')[['0-10', '10-30', '30-100', '100-inf']]

# df14 = pd.read_csv('../input/'+'cnn-pt-class-focal-loss-new-data'+'/OOF_preds.csv').sort_values(by='row')[['0-10', '10-30', '30-100', '100-inf']]



labels = pd.read_csv('../input/'+'cnn-1-pt-regression-swish-activation-new-d'+'/OOF_preds.csv').sort_values(by='row')['true_value']

rows = pd.read_csv('../input/'+'cnn-1-pt-regression-swish-activation-new-d'+'/OOF_preds.csv').sort_values(by='row')['row']



# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14], axis = 1)

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], axis = 1)
len(df1), len(df2), len(df3), len(df4), len(df5), len(df6), len(df7), len(df8), len(df9), len(df10), len(df11)
df.head(5)
sns.heatmap(df.corr())
shuffled_list = list(range(len(df)))

random.Random(242).shuffle(shuffled_list)

shuffled_list = np.array_split(np.array(shuffled_list), 10)
OOF_preds = pd.DataFrame()

OOF_preds['row'] = []

OOF_preds['true_value'] = []

OOF_preds['preds'] = []
model_names = ['bayesian_ridge', 'bagging_regressor', 'Ridge', 'SVR',  'NuSVR'][:3]
X_preds = np.zeros((len(df), len(model_names)))

for i in tqdm(range(10)):

    for j, model in enumerate([BayesianRidge(), BaggingRegressor(Ridge()), Ridge(tol=10e-05),SVR(max_iter=5000, tol=10e-05)][:3]):

    

        X_train = df.iloc[np.concatenate([shuffled_list[j] for j in range(10) if j not in (i,100)])]

        Y_train = labels.iloc[np.concatenate([shuffled_list[j] for j in range(10) if j not in (i,100)])]



        X_test = df.iloc[shuffled_list[i]]

        Y_test = labels.iloc[shuffled_list[i]]



        model.fit(X_train.to_numpy(), Y_train.to_numpy().reshape(-1,1))

        

        X_preds[shuffled_list[i], j] = model.predict(X_test.to_numpy()).reshape(-1)
sns.heatmap(pd.DataFrame(X_preds, columns = model_names).corr())
OOF_preds = pd.DataFrame()

OOF_preds['row'] = rows.to_list()

OOF_preds['true_value'] = labels.to_list()

OOF_preds['preds'] = X_preds.mean(axis = 1)
OOF_preds = OOF_preds.sort_values(by = 'row').reset_index(drop = True)

OOF_preds.to_csv('OOF_preds.csv')
df = pd.read_csv('OOF_preds.csv').drop(columns = ['Unnamed: 0'])

df = df.sort_values(by = 'row').reset_index(drop = True)

df['True_pT'] = 1/df['true_value']

df['Predicted_pT'] = 1/df['preds']
df
MAE1 = []

dx = 0.5

for i in tqdm(range(int(2/dx),int(150/dx))):

    P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]

    try:

        p = mae(P['True_pT'],P['Predicted_pT'])

    except:

        p=0

    MAE1.append(p)

MAE1 = MAE1[:146]

plt.plot([i*dx for i in range(int(75/dx))],[0]*int(int(75/dx)-len(MAE1))+MAE1,label = 'FCNN')
