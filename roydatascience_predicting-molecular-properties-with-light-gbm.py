import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold,KFold

import warnings

import os

from six.moves import urllib

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('seaborn')

from scipy.stats import norm, skew

import plotly

from plotly.plotly import iplot

import plotly.graph_objs as go
os.listdir('../input/')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(pd.read_csv('../input/train.csv'))

test = reduce_mem_usage(pd.read_csv('../input/test.csv'))

mulliken = reduce_mem_usage(pd.read_csv('../input/mulliken_charges.csv'))

dipole = reduce_mem_usage(pd.read_csv('../input/dipole_moments.csv'))

structure = reduce_mem_usage(pd.read_csv('../input/structures.csv'))

potential = reduce_mem_usage(pd.read_csv('../input/potential_energy.csv'))

magnetic = reduce_mem_usage(pd.read_csv('../input/magnetic_shielding_tensors.csv'))

scaler = reduce_mem_usage(pd.read_csv('../input/scalar_coupling_contributions.csv'))

sub = reduce_mem_usage(pd.read_csv('../input/sample_submission.csv'))
train.info()
train.describe()
train.head(10)
print(mulliken.shape)

print(mulliken.head(10))
print(dipole.shape)

print(dipole.head(10))
print(structure.shape)

print(structure.head(10))
print(magnetic.shape)

print(magnetic.head(5))
print(potential.shape)

print(potential.head(10))
print(scaler.shape)

print(scaler.head(10))
#Check for Missing Values



obs = train.isnull().sum().sort_values(ascending = False)

percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)

pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
def one_hot_encoder(data):

    original_columns = data.columns.tolist()

    categorical_columns = list(filter(lambda c: c in ['object'], data.dtypes))

    new_data = pd.get_dummies(data, columns=categorical_columns)



    new_columns = list(filter(lambda c: c not in original_columns, new_data.columns))

    return new_data, new_columns
#Distribution for Target Variable

plotly.tools.set_credentials_file('roy.gupta','LIKu6GnqVhkB1BaoUuHP') # Please change the credentials in your version



trace1 = go.Histogram(

    x= train['scalar_coupling_constant'],

    opacity=0.75,

    name = "scalar_coupling_constant",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



data = [trace1]

layout = go.Layout(barmode='overlay',

                   title='Distribution of scalar coupling constant'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
import gc

gc.collect()
plotly.tools.set_credentials_file('roy.gupta','LIKu6GnqVhkB1BaoUuHP')

example = structure.loc[structure['molecule_name'] == 'dsgdb9nsd_000001']

trace1 = go.Scatter3d(

    x=example['x'],

    y=example['y'],

    z=example['z'],

    mode='markers',

    marker=dict(

        size=12,

        line=dict(

            color='rgba(217, 217, 217, 0.14)',

            width=0.5

        ),

        opacity=0.8

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = dict(data = data, layout = layout)

iplot(fig)
gc.collect()
# Plot the distribution of mulliken_charges

#Distribution for Target Variable

plotly.tools.set_credentials_file('roy.gupta','LIKu6GnqVhkB1BaoUuHP') # Please change the credentials in your version



trace1 = go.Histogram(

    x= mulliken['mulliken_charge'],

    opacity=0.75,

    name = "scalar_coupling_constant",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



data = [trace1]

layout = go.Layout(barmode='overlay',

                   title='Distribution of Mulliken Charges'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

gc.collect()
# Plot the distribution of Potential Energy

#Distribution for Target Variable

plotly.tools.set_credentials_file('roy.gupta','LIKu6GnqVhkB1BaoUuHP') # Please change the credentials in your version



trace1 = go.Histogram(

    x= potential['potential_energy'],

    opacity=0.75,

    name = "scalar_coupling_constant",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



data = [trace1]

layout = go.Layout(barmode='overlay',

                   title='Potential Energy'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

gc.collect()
scaler.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',

                                                                color='grey',

                                                               figsize=(15, 5),

                                                               title='Count of Coupling Type in Train Set')

plt.show()
def Eval_matrix(data, preds):

    data["prediction"] = preds

    maes = []

    for t in data.type.unique():

        y_true = data[data.type==t].scalar_coupling_constant.values

        y_pred = data[data.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)
smul = mulliken.merge(train)

sns.pairplot(data=smul.sample(500), hue='type', vars=['mulliken_charge','scalar_coupling_constant'])

plt.show()
smg = magnetic.merge(train)

sns.pairplot(data=smg.sample(500), hue='type', vars=['XX','YX','ZX','XY','YY','ZY','XZ','YZ','ZZ','scalar_coupling_constant'])

plt.show()
scc = scaler.merge(train)

sns.pairplot(data=scc.sample(500), hue='type', vars=['fc','sd','pso','dso','scalar_coupling_constant'])

plt.show()
gc.collect()
def map_atom_info(data, atom_idx):

    data = pd.merge(data, structure, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    data = data.drop('atom_index', axis=1)

    data = data.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return data



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
# make categorical variables

atom_map = {'H': 0,

            'C': 1,

            'N': 2}

train['atom_0_cat'] = train['atom_0'].map(atom_map).astype('int')

train['atom_1_cat'] = train['atom_1'].map(atom_map).astype('int')

test['atom_0_cat'] = test['atom_0'].map(atom_map).astype('int')

test['atom_1_cat'] = test['atom_1'].map(atom_map).astype('int')
# One Hot Encode the Type

train_df = pd.concat([train, pd.get_dummies(train['type'])], axis=1)

test_df = pd.concat([test, pd.get_dummies(test['type'])], axis=1)
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')

test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')
train.head(10)