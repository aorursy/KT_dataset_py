import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore",category=DeprecationWarning)

warnings.filterwarnings(action="ignore",category=FutureWarning)
df_train=pd.read_csv('../input/train.csv')

#df_test=pd.read_csv('../input/test.csv')

df_struct=pd.read_csv('../input/structures.csv')
def map_atom_info(df_1,df_2, atom_idx):

    df = pd.merge(df_1, df_2, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)



    return df

for atom_idx in [0,1]:

    df_train = map_atom_info(df_train,df_struct, atom_idx)

    df_train = df_train.rename(columns={'atom': f'atom_{atom_idx}',

                                        'x': f'x_{atom_idx}',

                                        'y': f'y_{atom_idx}',

                                        'z': f'z_{atom_idx}'})
def make_features(df):

    df['dx']=df['x_1']-df['x_0']

    df['dy']=df['y_1']-df['y_0']

    df['dz']=df['z_1']-df['z_0']

    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)

    return df

df_train=make_features(df_train)
#I apologize for my poor coding skill. Please make the better one.

print(df_train.shape)

df_temp=df_train.loc[:,["molecule_name","atom_index_0","atom_index_1","distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()



df_temp_=df_temp.copy()

df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',

                                   'atom_index_1': 'atom_index_0',

                                   'x_0': 'x_1',

                                   'y_0': 'y_1',

                                   'z_0': 'z_1',

                                   'x_1': 'x_0',

                                   'y_1': 'y_0',

                                   'z_1': 'z_0'})

df_temp=pd.concat((df_temp,df_temp_),axis=0)



df_temp["min_distance"]=df_temp.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')

df_temp= df_temp[df_temp["min_distance"]==df_temp["distance"]]



df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)

df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                 'atom_index_1': 'atom_index_closest',

                                 'distance': 'distance_closest',

                                 'x_1': 'x_closest',

                                 'y_1': 'y_closest',

                                 'z_1': 'z_closest'})



print(df_temp.duplicated(subset=['molecule_name', 'atom_index']).value_counts())

#delete duplicated rows (some atom pairs have perfectly same distance)

#This code is added based on Adriano Avelar's comment.

df_temp=df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])



for atom_idx in [0,1]:

    df_train = map_atom_info(df_train,df_temp, atom_idx)

    df_train = df_train.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',

                                        'distance_closest': f'distance_closest_{atom_idx}',

                                        'x_closest': f'x_closest_{atom_idx}',

                                        'y_closest': f'y_closest_{atom_idx}',

                                        'z_closest': f'z_closest_{atom_idx}'})



print(df_train.shape)
df_train.head()
def add_cos_features(df):

    df["distance_0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)

    df["distance_1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["vec_0_x"]=(df['x_0']-df['x_closest_0'])/df["distance_0"]

    df["vec_0_y"]=(df['y_0']-df['y_closest_0'])/df["distance_0"]

    df["vec_0_z"]=(df['z_0']-df['z_closest_0'])/df["distance_0"]

    df["vec_1_x"]=(df['x_1']-df['x_closest_1'])/df["distance_1"]

    df["vec_1_y"]=(df['y_1']-df['y_closest_1'])/df["distance_1"]

    df["vec_1_z"]=(df['z_1']-df['z_closest_1'])/df["distance_1"]

    df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]

    df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]

    df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]

    df["cos_0_1"]=df["vec_0_x"]*df["vec_1_x"]+df["vec_0_y"]*df["vec_1_y"]+df["vec_0_z"]*df["vec_1_z"]

    df["cos_0"]=df["vec_0_x"]*df["vec_x"]+df["vec_0_y"]*df["vec_y"]+df["vec_0_z"]*df["vec_z"]

    df["cos_1"]=df["vec_1_x"]*df["vec_x"]+df["vec_1_y"]*df["vec_y"]+df["vec_1_z"]*df["vec_z"]

    df=df.drop(['vec_0_x','vec_0_y','vec_0_z','vec_1_x','vec_1_y','vec_1_z','vec_x','vec_y','vec_z'], axis=1)

    return df

    

df_train=add_cos_features(df_train)
mol_types=df_train["type"].unique()

df_train_=df_train.iloc[:100000,:].copy()



fig, ax = plt.subplots(8, 1, figsize=(8, 32))

for i, mol_type in enumerate(mol_types):

    ax[i].scatter(df_train_.loc[df_train_['type'] ==mol_type]["cos_0_1"], df_train_.loc[df_train_['type'] == mol_type] ['scalar_coupling_constant'],s=10,alpha=0.3);

    ax[i].set_title(str(mol_type))
mol_types=df_train["type"].unique()

df_train_=df_train.iloc[:100000,:].copy()



fig, ax = plt.subplots(8, 1, figsize=(8, 32))

for i, mol_type in enumerate(mol_types):

    ax[i].scatter(df_train_.loc[df_train_['type'] ==mol_type]["cos_0"], df_train_.loc[df_train_['type'] == mol_type] ['scalar_coupling_constant'],s=10,alpha=0.3);

    ax[i].set_title(str(mol_type))
mol_types=df_train["type"].unique()

df_train_=df_train.iloc[:100000,:].copy()



fig, ax = plt.subplots(8, 1, figsize=(8, 32))

for i, mol_type in enumerate(mol_types):

    ax[i].scatter(df_train_.loc[df_train_['type'] ==mol_type]["cos_1"], df_train_.loc[df_train_['type'] == mol_type] ['scalar_coupling_constant'],s=10,alpha=0.3);

    ax[i].set_title(str(mol_type))