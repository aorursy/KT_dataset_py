# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
dipole_moments_csv=pd.read_csv("../input/dipole_moments.csv")

magnetic_shielding_tensors_csv=pd.read_csv("../input/magnetic_shielding_tensors.csv")

mulliken_charges_csv=pd.read_csv("../input/mulliken_charges.csv")

potential_energy_csv=pd.read_csv("../input/potential_energy.csv")

sample_submission_csv=pd.read_csv("../input/sample_submission.csv")

scalar_coupling_contributions_csv=pd.read_csv("../input/scalar_coupling_contributions.csv")

structures_csv=pd.read_csv("../input/structures.csv")

test_csv=pd.read_csv("../input/test.csv")

train_csv=pd.read_csv("../input/train.csv")



print("dipole_moments.shape:",dipole_moments_csv.shape)

print("magnetic_shielding_tensors.shape:",magnetic_shielding_tensors_csv.shape)

print("mulliken_charges.shape:",mulliken_charges_csv.shape)

print("potential_energy:",potential_energy_csv.shape)

print("sample_submission:",sample_submission_csv.shape)

print("scalar_coupling_contributions.shape:",scalar_coupling_contributions_csv.shape)

print("structures.shape:",structures_csv.shape)

print("test_csv.shape:",test_csv.shape)

print("train_csv.shape:",train_csv.shape)
################## Auxiliar functions



# Memory reduct

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



# Merge train_csv with Mulliken charges



def map_mulliken_charge(database,atom_idx) :

    database = pd.merge(database,mulliken_charges_csv,how = 'left',

                 left_on = ['molecule_name',f'atom_index_{atom_idx}'],

                 right_on = ['molecule_name','atom_index']

                 )

    database = database.rename(columns={'mulliken_charge': f'mulliken_charge_{atom_idx}'}

                  )

    database = database.drop('atom_index',axis = 1)

    database = reduce_mem_usage(database)

    return database



# Calculate number each type of atoms in a molecule

def number_each_atoms_molecule(df,df1,index):

    df['Num_each_atom_in_mol'] = df.groupby(['molecule_name','atom'])['molecule_name'].transform('count')

    df1 = pd.merge(df1, df, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{index}'],

                  right_on = ['molecule_name',  'atom_index'])

    df1= df1.rename(columns={'x':f'x{index}','y':f'y{index}','z':f'z{index}','atom':f'atom{index}'})

    df1 = df1.drop('atom_index',axis=1)

    df1 = reduce_mem_usage(df1)

    

    return df1



    # GENERATION OF TYPES OF COUPLINGS

def database_type(db_type):

    #db_type = db[db['type'] == index]

    

    coupling= db_type[['scalar_coupling_constant']].copy()

    dipole = db_type[['Dipole']].copy()

    potential = db_type[['potential_energy']].copy()

    fermi = db_type[['fc']].copy()

    spin_dipolar = db_type[['sd']].copy()

    paramagnetic_spin = db_type[['pso']].copy()

    diamagnetic_spin = db_type[['dso']].copy()

    mulliken_0 = db_type[['mulliken_charge_0']].copy()

    mulliken_1 = db_type[['mulliken_charge_1']].copy()

    

    db_type = db_type.drop(['type', 'scalar_coupling_constant','Dipole','potential_energy','fc','sd','pso','dso','mulliken_charge_0','mulliken_charge_1'], axis=1)

    db_type = db_type.drop(['id','molecule_name'], axis=1)

    db_type = reduce_mem_usage(db_type)

    

    return [db_type, coupling, dipole, potential, fermi, spin_dipolar,paramagnetic_spin,diamagnetic_spin,mulliken_0,mulliken_1]



# Metrics MAE

def metric_mae(df, preds):

    df["prediction"] = preds

    maes = []

    y_true = df.scalar_coupling_constant.values

    y_pred = df.prediction.values

    mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

    maes.append(mae)

    

    return np.mean(maes)

# Modification of structures_csv



def mod_structures(structures_csv):

    structures_csv.round(5)

    atomic_num = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9}

    atoms = structures_csv['atom'].values

    atomic_number = [atomic_num[x] for x in atoms]

    

    structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

    structures_csv['Atomic_mass'] = atomic_number

    structures_csv['Tot_atoms_molecule']= structures_csv.groupby(['molecule_name'])['molecule_name'].transform('count')



    # Center of gravity. 



    structures_csv['mass_x'] = structures_csv['Atomic_mass'] * structures_csv['x']

    structures_csv['mass_y'] = structures_csv['Atomic_mass'] * structures_csv['y']

    structures_csv['mass_z'] = structures_csv['Atomic_mass'] * structures_csv['z']

    structures_csv['Mol_mass']= structures_csv.groupby(['molecule_name'])['Atomic_mass'].transform('sum')



    structures_csv['sum_mass_x']= structures_csv.groupby(['molecule_name'])['mass_x'].transform('sum')

    structures_csv['sum_mass_y']= structures_csv.groupby(['molecule_name'])['mass_y'].transform('sum')

    structures_csv['sum_mass_z']= structures_csv.groupby(['molecule_name'])['mass_z'].transform('sum')



    structures_csv['XG']= structures_csv['sum_mass_x'] / structures_csv['Mol_mass']

    structures_csv['YG']= structures_csv['sum_mass_y'] / structures_csv['Mol_mass']

    structures_csv['ZG']= structures_csv['sum_mass_z'] / structures_csv['Mol_mass']



    # Total size of molecule



    structures_csv['min_x_mol'] = structures_csv.groupby(['molecule_name'])['x'].transform('min')

    structures_csv['max_x_mol'] = structures_csv.groupby(['molecule_name'])['x'].transform('max')

    structures_csv['Size_mol_x'] = structures_csv['max_x_mol'] - structures_csv['min_x_mol']



    structures_csv['min_y_mol'] = structures_csv.groupby(['molecule_name'])['y'].transform('min')

    structures_csv['max_y_mol'] = structures_csv.groupby(['molecule_name'])['y'].transform('max')

    structures_csv['Size_mol_y'] = structures_csv['max_y_mol'] - structures_csv['min_y_mol']



    structures_csv['min_z_mol'] = structures_csv.groupby(['molecule_name'])['z'].transform('min')

    structures_csv['max_z_mol'] = structures_csv.groupby(['molecule_name'])['z'].transform('max')

    structures_csv['Size_mol_z'] = structures_csv['max_z_mol'] - structures_csv['min_z_mol']



    structures_csv['Size_mol'] = np.sqrt(np.square(structures_csv['Size_mol_x'])+np.square(structures_csv['Size_mol_y'])+np.square(structures_csv['Size_mol_z']))

    structures_csv['Cos_x_size_mol'] = structures_csv['Size_mol_x'] / structures_csv['Size_mol']

    structures_csv['Cos_y_size_mol'] = structures_csv['Size_mol_y'] / structures_csv['Size_mol']

    structures_csv['Cos_z_size_mol'] = structures_csv['Size_mol_z'] / structures_csv['Size_mol']



    structures_csv = structures_csv.drop(['min_x_mol','max_x_mol','min_y_mol','max_y_mol','min_z_mol','max_z_mol'], axis=1)

    structures_csv = structures_csv.drop({'mass_x','mass_y','mass_z','sum_mass_x','sum_mass_y','sum_mass_z','Atomic_mass'}, axis=1)

    structures_csv = reduce_mem_usage(structures_csv)

    return structures_csv





# Creation of superfeatures

def superfeatures(train_csv):

    # distances 

    train_csv['Dist_XG_to_x0'] = train_csv['XG_x']-train_csv['x0']

    train_csv['Dist_YG_to_y0'] = train_csv['YG_x']-train_csv['y0']

    train_csv['Dist_ZG_to_z0'] = train_csv['ZG_x']-train_csv['z0']

    train_csv['Dist_XG_to_x1'] = train_csv['XG_x']-train_csv['x1']

    train_csv['Dist_YG_to_y1'] = train_csv['YG_x']-train_csv['y1']

    train_csv['Dist_ZG_to_z1'] = train_csv['ZG_x']-train_csv['z1']

    

    dx =train_csv['x0']-train_csv['x1']

    dy =train_csv['y0']-train_csv['y1']

    dz =train_csv['z0']-train_csv['z1']

    distances_train = np.sqrt(dx**2 + dy**2 + dz**2)



    train_csv['At_dist']=distances_train

    train_csv['At_dist_x']=dx

    train_csv['At_dist_y']=dy

    train_csv['At_dist_z']=dz

    

    train_csv['Mol_dist_mean']=train_csv.groupby('molecule_name')['At_dist'].transform('mean')

    train_csv['Mol_dist_max']=train_csv.groupby('molecule_name')['At_dist'].transform('max')

    train_csv['Mol_dist_min']=train_csv.groupby('molecule_name')['At_dist'].transform('min')



    train_csv['Num_coupl_mol'] = train_csv.groupby('molecule_name')['molecule_name'].transform('count')

    train_csv['Num_coupl_mol_and_type'] = train_csv.groupby(['molecule_name','type'])['molecule_name'].transform('count')

    

    train_csv['Atom_0_coupl_mol'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    train_csv['Atom_0_coupl_mol_and_type'] = train_csv.groupby(['molecule_name', 'atom_index_0','type'])['id'].transform('count')

    

    train_csv['Atom_1_coupl_mol'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    train_csv['Atom_1_coupl_mol_and_type'] = train_csv.groupby(['molecule_name', 'atom_index_1','type'])['id'].transform('count')



    train_csv['Rel_dist_x'] = train_csv['At_dist_x'] / (train_csv['At_dist'])

    train_csv['Rel_dist_y'] = train_csv['At_dist_y'] / (train_csv['At_dist'])

    train_csv['Rel_dist_z'] = train_csv['At_dist_z'] / (train_csv['At_dist'])

    

    distances_0_CG_train = np.sqrt(np.square(train_csv['Dist_XG_to_x0'])+np.square(train_csv['Dist_YG_to_y0'])+np.square(train_csv['Dist_ZG_to_z0']))

    distances_1_CG_train = np.sqrt(np.square(train_csv['Dist_XG_to_x1'])+np.square(train_csv['Dist_YG_to_y1'])+np.square(train_csv['Dist_ZG_to_z1']))

    train_csv['Dist_CG_x0'] = distances_0_CG_train

    train_csv['Dist_CG_x1'] = distances_1_CG_train



    train_csv['Rel_XG_dist_x0'] = train_csv['Dist_XG_to_x0'] / (train_csv['Dist_CG_x0'] + 1e-5) 

    train_csv['Rel_YG_dist_x0'] = train_csv['Dist_YG_to_y0'] / (train_csv['Dist_CG_x0'] + 1e-5)

    train_csv['Rel_ZG_dist_x0'] = train_csv['Dist_ZG_to_z0'] / (train_csv['Dist_CG_x0'] + 1e-5)



    train_csv['Rel_XG_dist_x1'] = train_csv['Dist_XG_to_x1'] / (train_csv['Dist_CG_x1'] + 1e-5) 

    train_csv['Rel_YG_dist_x1'] = train_csv['Dist_YG_to_y1'] / (train_csv['Dist_CG_x1'] + 1e-5)

    train_csv['Rel_ZG_dist_x1'] = train_csv['Dist_ZG_to_z1'] / (train_csv['Dist_CG_x1'] + 1e-5)



    train_csv['At_0_dist_mean'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('mean')

    train_csv['At_0_dist_max'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('max')

    train_csv['At_0_dist_min'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('min')



    train_csv['At_0_dist_x_max'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_x'].transform('max')

    train_csv['At_0_dist_x_min'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_x'].transform('min')

    train_csv['At_0_dist_y_max'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_y'].transform('max')

    train_csv['At_0_dist_y_min'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_y'].transform('min')

    train_csv['At_0_dist_z_max'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_z'].transform('max')

    train_csv['At_0_dist_z_min'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist_z'].transform('min')

    

    train_csv['At_1_dist_mean'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('mean')

    train_csv['At_1_dist_max'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('max')

    train_csv['At_1_dist_min'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('min')

    

    train_csv['At_1_dist_x_max'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_x'].transform('max')

    train_csv['At_1_dist_x_min'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_x'].transform('min')

    train_csv['At_1_dist_y_max'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_y'].transform('max')

    train_csv['At_1_dist_y_min'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_y'].transform('min')

    train_csv['At_1_dist_z_max'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_z'].transform('max')

    train_csv['At_1_dist_z_min'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist_z'].transform('min')

    

    #cosine

    uve_0_x = train_csv['XG_x']-train_csv['x0']

    uve_0_y = train_csv['YG_x']-train_csv['y0']

    uve_0_z = train_csv['ZG_x']-train_csv['z0']



    uve_1_x = train_csv['XG_x']-train_csv['x1']

    uve_1_y = train_csv['YG_x']-train_csv['y1']

    uve_1_z = train_csv['ZG_x']-train_csv['z1']



    train_csv['cos_G_x0'] = ((uve_0_x * dx)+(uve_0_y * dy)+(uve_0_z * dz))/((abs(distances_train)*abs(distances_0_CG_train))+ 1e-2)

    train_csv['cos_G_x1'] = ((uve_1_x * dx)+(uve_1_y * dy)+(uve_1_z * dz))/((abs(distances_train)*abs(distances_1_CG_train))+ 1e-2)



    dist_xy = np.sqrt(np.square(dx)+np.square(dy))

    dist_xz = np.sqrt(np.square(dx)+np.square(dz))

    dist_yz = np.sqrt(np.square(dy)+np.square(dz))



    uve_0_xy = np.sqrt(np.square(uve_0_x)+np.square(uve_0_y))

    uve_0_xz = np.sqrt(np.square(uve_0_x)+np.square(uve_0_z))

    uve_0_yz = np.sqrt(np.square(uve_0_y)+np.square(uve_0_z))

    uve_1_xy = np.sqrt(np.square(uve_1_x)+np.square(uve_1_y))

    uve_1_xz = np.sqrt(np.square(uve_1_x)+np.square(uve_1_z))

    uve_1_yz = np.sqrt(np.square(uve_1_y)+np.square(uve_1_z))



    train_csv['cos_G_x0_pl_xy'] = ((dx*uve_0_x)+(dy*uve_0_y))/((abs(dist_xy)*abs(uve_0_xy))+1e-2)

    train_csv['cos_G_x0_pl_xz'] = ((dx*uve_0_x)+(dz*uve_0_z))/((abs(dist_xz)*abs(uve_0_xz))+1e-2)

    train_csv['cos_G_x0_pl_yz'] = ((dy*uve_0_y)+(dz*uve_0_z))/((abs(dist_yz)*abs(uve_0_yz))+1e-2)

    train_csv['cos_G_x1_pl_xy'] = ((dx*uve_1_x)+(dy*uve_1_y))/((abs(dist_xy)*abs(uve_1_xy))+1e-2)

    train_csv['cos_G_x1_pl_xz'] = ((dx*uve_1_x)+(dz*uve_1_z))/((abs(dist_xz)*abs(uve_1_xz))+1e-2)

    train_csv['cos_G_x1_pl_yz'] = ((dy*uve_1_y)+(dz*uve_1_z))/((abs(dist_yz)*abs(uve_1_yz))+1e-2)



    #---------------------------------------------------------

    train_csv['Mol_x_mean']=train_csv.groupby('molecule_name')['x0'].transform('mean')

    train_csv['Mol_x_max']=train_csv.groupby('molecule_name')['x0'].transform('max')

    train_csv['Mol_x_min']=train_csv.groupby('molecule_name')['x0'].transform('min')

    train_csv['Mol_y_mean']=train_csv.groupby('molecule_name')['y0'].transform('mean')

    train_csv['Mol_y_max']=train_csv.groupby('molecule_name')['y0'].transform('max')

    train_csv['Mol_y_min']=train_csv.groupby('molecule_name')['y0'].transform('min')

    train_csv['Mol_z_mean']=train_csv.groupby('molecule_name')['z0'].transform('mean')

    train_csv['Mol_z_max']=train_csv.groupby('molecule_name')['z0'].transform('max')

    train_csv['Mol_z_min']=train_csv.groupby('molecule_name')['z0'].transform('min')



    dx0_mean= train_csv['x0']-train_csv['Mol_x_mean']

    dy0_mean= train_csv['y0']-train_csv['Mol_y_mean']

    dz0_mean= train_csv['z0']-train_csv['Mol_z_mean']



    dx0_max= train_csv['x0']-train_csv['Mol_x_max']

    dy0_max= train_csv['y0']-train_csv['Mol_y_max']

    dz0_max= train_csv['z0']-train_csv['Mol_z_max']



    dx0_min= train_csv['x0']-train_csv['Mol_x_min']

    dy0_min= train_csv['y0']-train_csv['Mol_y_min']

    dz0_min= train_csv['z0']-train_csv['Mol_z_min']



    dx1_mean= train_csv['x1']-train_csv['Mol_x_mean']

    dy1_mean= train_csv['y1']-train_csv['Mol_y_mean']

    dz1_mean= train_csv['z1']-train_csv['Mol_z_mean']



    dx1_max= train_csv['x1']-train_csv['Mol_x_max']

    dy1_max= train_csv['y1']-train_csv['Mol_y_max']

    dz1_max= train_csv['z1']-train_csv['Mol_z_max']



    dx1_min= train_csv['x1']-train_csv['Mol_x_min']

    dy1_min= train_csv['y1']-train_csv['Mol_y_min']

    dz1_min= train_csv['z1']-train_csv['Mol_z_min']



    dist_x0_mean_xy = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean))

    dist_x0_mean_xz = np.sqrt(np.square(dx0_mean)+np.square(dz0_mean))

    dist_x0_mean_yz = np.sqrt(np.square(dy0_mean)+np.square(dz0_mean))

    

    train_csv['cos_Mean_x0_pl_xy'] = ((dx*dx0_mean)+(dy*dy0_mean))/((abs(dist_xy)*abs(dist_x0_mean_xy))+1e-2)

    train_csv['cos_Mean_x0_pl_xz'] = ((dx*dx0_mean)+(dz*dz0_mean))/((abs(dist_xz)*abs(dist_x0_mean_xz))+1e-2)

    train_csv['cos_Mean_x0_pl_yz'] = ((dy*dy0_mean)+(dz*dz0_mean))/((abs(dist_yz)*abs(dist_x0_mean_yz))+1e-2)



    dist_x0_max_xy = np.sqrt(np.square(dx0_max)+np.square(dy0_max))

    dist_x0_max_xz = np.sqrt(np.square(dx0_max)+np.square(dz0_max))

    dist_x0_max_yz = np.sqrt(np.square(dy0_max)+np.square(dz0_max))

    

    train_csv['cos_Max_x0_pl_xy'] = ((dx*dx0_max)+(dy*dy0_max))/((abs(dist_xy)*abs(dist_x0_max_xy))+1e-2)

    train_csv['cos_Max_x0_pl_xz'] = ((dx*dx0_max)+(dz*dz0_max))/((abs(dist_xz)*abs(dist_x0_max_xz))+1e-2)

    train_csv['cos_Max_x0_pl_yz'] = ((dy*dy0_max)+(dz*dz0_max))/((abs(dist_yz)*abs(dist_x0_max_yz))+1e-2)



    dist_x0_min_xy = np.sqrt(np.square(dx0_min)+np.square(dy0_min))

    dist_x0_min_xz = np.sqrt(np.square(dx0_min)+np.square(dz0_min))

    dist_x0_min_yz = np.sqrt(np.square(dy0_min)+np.square(dz0_min))

    

    train_csv['cos_Min_x0_pl_xy'] = ((dx*dx0_min)+(dy*dy0_min))/((abs(dist_xy)*abs(dist_x0_min_xy))+1e-2)

    train_csv['cos_Min_x0_pl_xz'] = ((dx*dx0_min)+(dz*dz0_min))/((abs(dist_xz)*abs(dist_x0_min_xz))+1e-2)

    train_csv['cos_Min_x0_pl_yz'] = ((dy*dy0_min)+(dz*dz0_min))/((abs(dist_yz)*abs(dist_x0_min_yz))+1e-2)



    dist_x0_mean = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean)+np.square(dz0_mean))

    dist_x0_max = np.sqrt(np.square(dx0_max)+np.square(dy0_max)+np.square(dz0_max))

    dist_x0_min = np.sqrt(np.square(dx0_min)+np.square(dy0_min)+np.square(dz0_min))

    

    train_csv['cos_Mean_x0'] = ((dx*dx0_mean)+(dy*dy0_mean)+(dz*dz0_mean))/((abs(distances_train)*abs(dist_x0_mean))+1e-2)

    train_csv['cos_Max_x0'] = ((dx*dx0_max)+(dy*dy0_max)+(dz*dz0_max))/((abs(distances_train)*abs(dist_x0_max))+1e-2)

    train_csv['cos_Min_x0'] = ((dx*dx0_min)+(dy*dy0_min)+(dz*dz0_min))/((abs(distances_train)*abs(dist_x0_min))+1e-2)



    dist_x1_mean_xy = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean))

    dist_x1_mean_xz = np.sqrt(np.square(dx1_mean)+np.square(dz1_mean))

    dist_x1_mean_yz = np.sqrt(np.square(dy1_mean)+np.square(dz1_mean))

    

    train_csv['cos_Mean_x1_pl_xy'] = ((dx*dx1_mean)+(dy*dy1_mean))/((abs(dist_xy)*abs(dist_x1_mean_xy))+1e-2)

    train_csv['cos_Mean_x1_pl_xz'] = ((dx*dx1_mean)+(dz*dz1_mean))/((abs(dist_xz)*abs(dist_x1_mean_xz))+1e-2)

    train_csv['cos_Mean_x1_pl_yz'] = ((dy*dy1_mean)+(dz*dz1_mean))/((abs(dist_yz)*abs(dist_x1_mean_yz))+1e-2)



    dist_x1_max_xy = np.sqrt(np.square(dx1_max)+np.square(dy1_max))

    dist_x1_max_xz = np.sqrt(np.square(dx1_max)+np.square(dz1_max))

    dist_x1_max_yz = np.sqrt(np.square(dy1_max)+np.square(dz1_max))

    

    train_csv['cos_Max_x1_pl_xy'] = ((dx*dx1_max)+(dy*dy1_max))/((abs(dist_xy)*abs(dist_x1_max_xy))+1e-2)

    train_csv['cos_Max_x1_pl_xz'] = ((dx*dx1_max)+(dz*dz1_max))/((abs(dist_xz)*abs(dist_x1_max_xz))+1e-2)

    train_csv['cos_Max_x1_pl_yz'] = ((dy*dy1_max)+(dz*dz1_max))/((abs(dist_yz)*abs(dist_x1_max_yz))+1e-2)



    dist_x1_min_xy = np.sqrt(np.square(dx1_min)+np.square(dy1_min))

    dist_x1_min_xz = np.sqrt(np.square(dx1_min)+np.square(dz1_min))

    dist_x1_min_yz = np.sqrt(np.square(dy1_min)+np.square(dz1_min))

    

    train_csv['cos_Min_x1_pl_xy'] = ((dx*dx1_min)+(dy*dy1_min))/((abs(dist_xy)*abs(dist_x1_min_xy))+1e-2)

    train_csv['cos_Min_x1_pl_xz'] = ((dx*dx1_min)+(dz*dz1_min))/((abs(dist_xz)*abs(dist_x1_min_xz))+1e-2)

    train_csv['cos_Min_x1_pl_yz'] = ((dy*dy1_min)+(dz*dz1_min))/((abs(dist_yz)*abs(dist_x1_min_yz))+1e-2)



    dist_x1_mean = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean)+np.square(dz1_mean))

    dist_x1_max = np.sqrt(np.square(dx1_max)+np.square(dy1_max)+np.square(dz1_max))

    dist_x1_min = np.sqrt(np.square(dx1_min)+np.square(dy1_min)+np.square(dz1_min))

    

    train_csv['cos_Mean_x1'] = ((dx*dx1_mean)+(dy*dy1_mean)+(dz*dz1_mean))/((abs(distances_train)*abs(dist_x1_mean))+1e-2)

    train_csv['cos_Max_x1'] = ((dx*dx1_max)+(dy*dy1_max)+(dz*dz1_max))/((abs(distances_train)*abs(dist_x1_max))+1e-2)

    train_csv['cos_Min_x1'] = ((dx*dx1_min)+(dy*dy1_min)+(dz*dz1_min))/((abs(distances_train)*abs(dist_x1_min))+1e-2)



    # Edit columns



    train_csv = train_csv.drop({'x0','x1','y1','z0','z1'}, axis=1)

    train_csv = train_csv.drop({'y0'}, axis=1)

    train_csv = train_csv.drop({'XG_x','YG_x','ZG_x'}, axis=1) 

    #train_csv = train_csv.drop({'atom_index_0','atom_index_1'}, axis=1)

    train_csv= train_csv.drop({'atom0','atom1','XG_y','YG_y','ZG_y','Tot_atoms_molecule_y','Mol_mass_y'}, axis=1)

    train_csv = train_csv.rename(columns={'Tot_atoms_molecule_x':'Tot_atoms_mol','Mol_mass_x':'Mol_mass'})

    train_csv= train_csv.drop({'Size_mol_x_y','Size_mol_y_y','Size_mol_z_y','Size_mol_y'}, axis=1)

    train_csv = train_csv.rename(columns={'Size_mol_x_x':'Size_mol_x','Size_mol_y_x':'Size_mol_y', 'Size_mol_z_x':'Size_mol_z','Size_mol_x':'Size_mol'})

    train_csv= train_csv.drop({'Cos_x_size_mol_y','Cos_y_size_mol_y','Cos_z_size_mol_y'}, axis=1)

    train_csv = train_csv.rename(columns={'Cos_x_size_mol_x':'Cos_x_size_mol','Cos_y_size_mol_x':'Cos_y_size_mol','Cos_z_size_mol_x':'Cos_z_size_mol'})

    train_csv= train_csv.drop({'molecule_index_y'}, axis=1)

    train_csv = train_csv.rename(columns={'molecule_index_x':'molecule_index'})



    train_csv = reduce_mem_usage(train_csv)

    

    return train_csv



# Split train_csv in several dataframes, each type of coupling



def split_train_dev(db,index, split_coef = 0.1, threshold = 0.90):

    

    db_type = db[db['type'] == index]

    train_csv, val_csv = train_test_split(db_type, test_size = split_coef, random_state=42)

    # Threshold for removing correlated variables

    # https://www.kaggle.com/adrianoavelar/gridsearch-for-eachtype-lb-1-0



    # Absolute value correlation matrix

    corr_matrix = train_csv.corr().abs()

    # Getting the upper triangle of correlations

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold

    

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(to_drop)

    for col in to_drop:

        if col == 'fc':

            to_drop.remove('fc')

        if col == 'dso':

            to_drop.remove('dso')

        if col == 'sd':

            to_drop.remove('sd')          

        

    #to_drop.remove('fc') 



    print('There are %d columns to remove.' % (len(to_drop)))

    print('The columns are: ', to_drop )

    

    train_csv = train_csv.drop(to_drop,axis=1)

    val_csv = val_csv.drop(to_drop,axis=1)

    train_csv = reduce_mem_usage(train_csv)

    val_csv = reduce_mem_usage(val_csv)

    

    print('Training shape: ', train_csv.shape)

    print('Validation shape: ', val_csv.shape)



    return train_csv, val_csv , to_drop
# Edit structures_csv

structures_mod_csv = mod_structures(structures_csv)
# Merge train_csv and test_csv with structures_csv



train_csv = number_each_atoms_molecule(structures_mod_csv, train_csv,0)

train_csv = number_each_atoms_molecule(structures_mod_csv, train_csv,1)



test_csv = number_each_atoms_molecule(structures_mod_csv, test_csv,0)

test_csv = number_each_atoms_molecule(structures_mod_csv, test_csv,1)
# Merge train_csv with scalar_coupling contributions

train_csv = pd.merge(train_csv, scalar_coupling_contributions_csv, how = 'left',

                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])



# Merge train_csv with mulliken

train_csv = map_mulliken_charge(train_csv,0)

train_csv = map_mulliken_charge(train_csv,1)
# Creation a lot of features

train_csv = superfeatures(train_csv)

test_csv = superfeatures(test_csv)
# Merge train_csv with dipole moments and potential energy

train_csv = pd.merge(train_csv,dipole_moments_csv,left_on='molecule_name', right_on='molecule_name')

train_csv = pd.merge(train_csv,potential_energy_csv,left_on='molecule_name', right_on='molecule_name')

dipole=np.sqrt(np.square(train_csv['X'])+np.square(train_csv['Y'])+np.square(train_csv['Z']))

train_csv = train_csv.drop({'X','Y','Z'}, axis=1)

train_csv['Dipole']=dipole

train_csv = reduce_mem_usage(train_csv)
train_csv.head()
test_csv.head()
print(train_csv.shape)

print(test_csv.shape)
## DEEP NEURAL NETWORK 



from keras.models import Sequential

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

from keras.layers.normalization import BatchNormalization 

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense, Dropout

from keras.initializers import glorot_uniform

from keras.engine.topology import Layer

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint
 # TRAINING. HYPERPARAMETERS



epochs_1JHN= 80

batch_1JHN = 32



epochs_2JHH= 120

batch_2JHH = 128



epochs_2JHN= 100

batch_2JHN = 128



epochs_3JHN= 200

batch_3JHN = 128



epochs_1JHC= 60 

batch_1JHC = 128



epochs_2JHC= 120

batch_2JHC = 1024



epochs_3JHH= 80

batch_3JHH = 512



epochs_3JHC= 70

batch_3JHC = 1024

#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



# Split train and validation set, and split them in several dataframes, each type of coupling 



train_1JHN, val_1JHN, to_drop = split_train_dev(train_csv,'1JHN', split_coef = 0.05, threshold = 1)

train_1JHN = database_type(train_1JHN)

val_1JHN = database_type(val_1JHN)

test_1JHN = test_csv[test_csv['type'] == '1JHN']

test_1JHN = test_1JHN.drop(to_drop, axis=1)



plt.plot(train_1JHN[4], train_1JHN[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights1.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



train = {'scalar_coupling': train_1JHN[1].values,         

        'dipolar_moment': train_1JHN[2].values,

        'potential_energy': train_1JHN[3].values,

         'fermi_coupling': train_1JHN[4].values,

        'spin_dipolar': train_1JHN[5].values,

        'paramagnetic_spin': train_1JHN[6].values,

        'diamagnetic_spin': train_1JHN[7].values,

        'mulliken_0': train_1JHN[8].values,

        'mulliken_1': train_1JHN[9].values}

validation = {'scalar_coupling': val_1JHN[1].values,

        'dipolar_moment': val_1JHN[2].values,

        'potential_energy': val_1JHN[3].values,

         'fermi_coupling': val_1JHN[4].values,

        'spin_dipolar': val_1JHN[5].values,

        'paramagnetic_spin': val_1JHN[6].values,

        'diamagnetic_spin': val_1JHN[7].values,

        'mulliken_0': val_1JHN[8].values,

        'mulliken_1': val_1JHN[9].values}



def model_coupling_constant_1JHN(X):

    X_input = Input(shape = (X.shape[1],))

 

    xsc = BatchNormalization()(X_input)

    xsc = Dense(256, activation='elu')(xsc) 

    xsc = Dense(512, activation='elu')(xsc)

    xsc = Dense(64, activation='elu')(xsc)

    xsc = Dense(64, activation='elu')(xsc)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(16, activation='elu')(xsc)

    xsc = Dense(4, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model

    

Model_Coupling_1JHN = model_coupling_constant_1JHN(train_1JHN[0])

Model_Coupling_1JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_1JHN = Model_Coupling_1JHN.fit(train_1JHN[0],train, validation_data=(val_1JHN[0],validation),epochs=epochs_1JHN,verbose=1,batch_size = batch_1JHN, callbacks=[best_param])#, early_stopping])
plt.plot(history_1JHN.history['loss'])

plt.plot(history_1JHN.history['val_loss'])

plt.title('loss 1JHN')

plt.ylabel('Loss 1JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_1JHN.load_weights("weights1.best.hdf5")

Model_Coupling_1JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])



model_check_1JHN = Model_Coupling_1JHN.predict(val_1JHN[0])

log_MAE_1JHN = metric_mae(val_1JHN[1], model_check_1JHN)

val_1JHN[1] = val_1JHN[1].drop('prediction',axis=1)



print("log MAE_1JHN=", log_MAE_1JHN)
inter_1JHN = pd.DataFrame(test_1JHN['id']) 

inter_1JHN['type'] = test_1JHN['type']

test_1JHN_bis = test_1JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_1JHN = Model_Coupling_1JHN.predict(test_1JHN_bis)

inter_1JHN['pred_values'] = model_predict_1JHN
# mejorado, 

#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights2.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



# Split train_csv in several dataframes, each type of coupling



train_2JHH, val_2JHH, to_drop = split_train_dev(train_csv,'2JHH', split_coef = 0.1, threshold = 1)

train_2JHH = database_type(train_2JHH)

val_2JHH = database_type(val_2JHH)

test_2JHH = test_csv[test_csv['type'] == '2JHH']

test_2JHH = test_2JHH.drop(to_drop, axis=1)



plt.plot(train_2JHH[4], train_2JHH[1], 'b.')

plt.show()
def model_coupling_constant_2JHH(X):  

    X_input = Input(shape = (X.shape[1],))

                    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHH = model_coupling_constant_2JHH(train_2JHH[0]) # R1

Model_Coupling_2JHH.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHH = Model_Coupling_2JHH.fit(train_2JHH[0],{'scalar_coupling': train_2JHH[1].values,

                            'fermi_coupling': train_2JHH[4].values},validation_data=(val_2JHH[0],{'scalar_coupling': val_2JHH[1].values,

                            'fermi_coupling': val_2JHH[4].values}),epochs=epochs_2JHH,verbose=1,batch_size = batch_2JHH, callbacks=[best_param])#, early_stopping])
plt.plot(history_2JHH.history['loss'])

plt.plot(history_2JHH.history['val_loss'])

plt.title('loss 2JHH')

plt.ylabel('Loss 2JHH')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_2JHH.load_weights("weights2.best.hdf5")

Model_Coupling_2JHH.compile(loss='mae', optimizer='Adam')



model_check_2JHH = Model_Coupling_2JHH.predict(val_2JHH[0])

log_MAE_2JHH = metric_mae(val_2JHH[1], model_check_2JHH)

val_2JHH[1] = val_2JHH[1].drop('prediction',axis=1)



print("log MAE_2JHH=", log_MAE_2JHH)
inter_2JHH = pd.DataFrame(test_2JHH['id']) 

inter_2JHH['type'] = test_2JHH['type']

test_2JHH_bis = test_2JHH.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHH = Model_Coupling_2JHH.predict(test_2JHH_bis)

inter_2JHH['pred_values'] = model_predict_2JHH
# mejorado 



#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights3.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



# Split train and validation set



train_2JHN, val_2JHN, to_drop = split_train_dev(train_csv,'2JHN', split_coef = 0.1, threshold = 1)

train_2JHN = database_type(train_2JHN) # ESTE YA NO ES TAN LINEAL ENTRE 'FC' Y 'COUPLING CONSTANT' ALGUN PUNTO ESTA MUY FUEra

val_2JHN = database_type(val_2JHN)

test_2JHN = test_csv[test_csv['type'] == '2JHN']

test_2JHN = test_2JHN.drop(to_drop, axis=1)



plt.plot(train_2JHN[4], train_2JHN[1], 'b.')

plt.show()
def model_coupling_constant_2JHN(X):

    X_input = Input(shape = (X.shape[1],))

                    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHN = model_coupling_constant_2JHN(train_2JHN[0]) # R1

Model_Coupling_2JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHN = Model_Coupling_2JHN.fit(train_2JHN[0],{'scalar_coupling': train_2JHN[1].values,

                            'fermi_coupling': train_2JHN[4].values},validation_data=(val_2JHN[0],{'scalar_coupling': val_2JHN[1].values,

                            'fermi_coupling': val_2JHN[4].values}),epochs=epochs_2JHN,verbose=1,batch_size = batch_2JHN, callbacks=[best_param])
plt.plot(history_2JHN.history['loss'])

plt.plot(history_2JHN.history['val_loss'])

plt.title('loss 2JHN')

plt.ylabel('Loss 2JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_2JHN.load_weights("weights3.best.hdf5")

Model_Coupling_2JHN.compile(loss='mae', optimizer='Adam')



model_check_2JHN = Model_Coupling_2JHN.predict(val_2JHN[0])

log_MAE_2JHN = metric_mae(val_2JHN[1], model_check_2JHN)

val_2JHN[1] = val_2JHN[1].drop('prediction',axis=1)



print("log MAE_2JHN=", log_MAE_2JHN)
inter_2JHN = pd.DataFrame(test_2JHN['id']) 

inter_2JHN['type'] = test_2JHN['type']

test_2JHN_bis = test_2JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHN = Model_Coupling_2JHN.predict(test_2JHN_bis)

inter_2JHN['pred_values'] = model_predict_2JHN
# Split train and validation set

train_3JHN, val_3JHN, to_drop = split_train_dev(train_csv,'3JHN', split_coef = 0.1, threshold = 1)

train_3JHN = database_type(train_3JHN) # ESTE YA NO ES TAN LINEAL ENTRE 'FC' Y 'COUPLING CONSTANT'

val_3JHN = database_type(val_3JHN)

test_3JHN = test_csv[test_csv['type'] == '3JHN']

test_3JHN = test_3JHN.drop(to_drop, axis=1)



plt.plot(train_3JHN[4], train_3JHN[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights4.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



def model_coupling_constant_3JHN(X):

    X_input = Input(shape = (X.shape[1],))

                    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHN = model_coupling_constant_3JHN(train_3JHN[0]) # R1

Model_Coupling_3JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHN = Model_Coupling_3JHN.fit(train_3JHN[0],{'scalar_coupling': train_3JHN[1].values,

                            'fermi_coupling': train_3JHN[4].values},validation_data=(val_3JHN[0],{'scalar_coupling': val_3JHN[1].values,

                            'fermi_coupling': val_3JHN[4].values}),epochs=epochs_3JHN,verbose=1,batch_size = batch_3JHN, callbacks=[best_param])
plt.plot(history_3JHN.history['loss'])

plt.plot(history_3JHN.history['val_loss'])

plt.title('loss 3JHN')

plt.ylabel('Loss 3JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_3JHN.load_weights("weights4.best.hdf5")

Model_Coupling_3JHN.compile(loss='mae', optimizer='Adam')



model_check_3JHN = Model_Coupling_3JHN.predict(val_3JHN[0])

log_MAE_3JHN = metric_mae(val_3JHN[1], model_check_3JHN)

val_3JHN[1] = val_3JHN[1].drop('prediction',axis=1)



print("log MAE_3JHN=", log_MAE_3JHN)
inter_3JHN = pd.DataFrame(test_3JHN['id']) 

inter_3JHN['type'] = test_3JHN['type']

test_3JHN_bis = test_3JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_3JHN = Model_Coupling_3JHN.predict(test_3JHN_bis)

inter_3JHN['pred_values'] = model_predict_3JHN
# MEJORADA 



# Split train_csv in several dataframes, each type of coupling



train_1JHC, val_1JHC, to_drop = split_train_dev(train_csv,'1JHC', split_coef = 0.1, threshold = 1.0)



train_1JHC = database_type(train_1JHC)

val_1JHC = database_type(val_1JHC)

test_1JHC = test_csv[test_csv['type'] == '1JHC']

test_1JHC = test_1JHC.drop(to_drop, axis=1)



plt.plot(train_1JHC[4], train_1JHC[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights5.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



train = {'scalar_coupling': train_1JHC[1].values,         

        'dipolar_moment': train_1JHC[2].values,

        'potential_energy': train_1JHC[3].values,

         'fermi_coupling': train_1JHC[4].values,

        'spin_dipolar': train_1JHC[5].values,

        'paramagnetic_spin': train_1JHC[6].values,

        'diamagnetic_spin': train_1JHC[7].values,

        'mulliken_0': train_1JHC[8].values,

        'mulliken_1': train_1JHC[9].values}

validation = {'scalar_coupling': val_1JHC[1].values,

        'dipolar_moment': val_1JHC[2].values,

        'potential_energy': val_1JHC[3].values,

         'fermi_coupling': val_1JHC[4].values,

        'spin_dipolar': val_1JHC[5].values,

        'paramagnetic_spin': val_1JHC[6].values,

        'diamagnetic_spin': val_1JHC[7].values,

        'mulliken_0': val_1JHC[8].values,

        'mulliken_1': val_1JHC[9].values}



def model_coupling_constant_1JHC(X):

    X_input = Input(shape = (X.shape[1],))

 

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(32, activation='elu')(xsc) #esta no

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_1JHC = model_coupling_constant_1JHC(train_1JHC[0])

Model_Coupling_1JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_1JHC = Model_Coupling_1JHC.fit(train_1JHC[0],train, validation_data=(val_1JHC[0],validation),epochs=epochs_1JHC,verbose=1,batch_size = batch_1JHC, callbacks=[best_param])
plt.plot(history_1JHC.history['loss'])

plt.plot(history_1JHC.history['val_loss'])

plt.title('loss 1JHC')

plt.ylabel('Loss 1JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')

Model_Coupling_1JHC.load_weights("weights5.best.hdf5")

Model_Coupling_1JHC.compile(loss='mae', optimizer='Adam')



model_check_1JHC = Model_Coupling_1JHC.predict(val_1JHC[0])

log_MAE_1JHC = metric_mae(val_1JHC[1], model_check_1JHC)

val_1JHC[1] = val_1JHC[1].drop('prediction',axis=1)



print("log MAE_1JHC=", log_MAE_1JHC)
inter_1JHC = pd.DataFrame(test_1JHC['id']) 

inter_1JHC['type'] = test_1JHC['type']

test_1JHC_bis = test_1JHC.drop(['id','molecule_name','type'], axis=1)



model_predict_1JHC = Model_Coupling_1JHC.predict(test_1JHC_bis)

inter_1JHC['pred_values'] = model_predict_1JHC
# Split train and validation set



train_2JHC, val_2JHC, to_drop = split_train_dev(train_csv,'2JHC', split_coef = 0.1, threshold = 1.0)

train_2JHC = database_type(train_2JHC) #  ESTE YA NO ES TAN LINEAL ENTRE 'FC' Y 'COUPLING CONSTANT'

val_2JHC = database_type(val_2JHC)

test_2JHC = test_csv[test_csv['type'] == '2JHC']

test_2JHC = test_2JHC.drop(to_drop, axis=1)



plt.plot(train_2JHC[4], train_2JHC[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights6.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



def model_coupling_constant_2JHC(X):



    X_input = Input(shape = (X.shape[1],))

                    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(32, activation='elu')(xsc) #esta no

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHC = model_coupling_constant_2JHC(train_2JHC[0]) # R1

Model_Coupling_2JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHC = Model_Coupling_2JHC.fit(train_2JHC[0],{'scalar_coupling': train_2JHC[1].values,

                            'fermi_coupling': train_2JHC[4].values},validation_data=(val_2JHC[0],{'scalar_coupling': val_2JHC[1].values,

                            'fermi_coupling': val_2JHC[4].values}),epochs=epochs_2JHC,verbose=1,batch_size = batch_2JHC, callbacks=[best_param])
plt.plot(history_2JHC.history['loss'])

plt.plot(history_2JHC.history['val_loss'])

plt.title('loss 2JHC')

plt.ylabel('Loss 2JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_2JHC.load_weights("weights6.best.hdf5")

Model_Coupling_2JHC.compile(loss='mae', optimizer='Adam')



model_check_2JHC = Model_Coupling_2JHC.predict(val_2JHC[0])

log_MAE_2JHC = metric_mae(val_2JHC[1], model_check_2JHC)

val_2JHC[1] = val_2JHC[1].drop('prediction',axis=1)



print("log MAE_2JHC=", log_MAE_2JHC)
inter_2JHC = pd.DataFrame(test_2JHC['id']) 

inter_2JHC['type'] = test_2JHC['type']

test_2JHC_bis = test_2JHC.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHC = Model_Coupling_2JHC.predict(test_2JHC_bis)

inter_2JHC['pred_values'] = model_predict_2JHC
# Mejorada,



# Split train_csv in several dataframes, each type of coupling



train_3JHH, val_3JHH, to_drop = split_train_dev(train_csv,'3JHH', split_coef = 0.1, threshold = 1.0)

train_3JHH = database_type(train_3JHH) # ESTE YA NO ES TAN LINEAL ENTRE 'FC' Y 'COUPLING CONSTANT'

val_3JHH = database_type(val_3JHH)

test_3JHH = test_csv[test_csv['type'] == '3JHH']

test_3JHH = test_3JHH.drop(to_drop, axis=1)



plt.plot(train_3JHH[4], train_3JHH[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights7.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



def model_coupling_constant_3JHH(X):



    X_input = Input(shape = (X.shape[1],))

                    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(512, activation='elu')(xfc) 

    #xfc = Dropout(0.1)(xfc) 

    xfc = Dense(256, activation='elu')(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHH = model_coupling_constant_3JHH(train_3JHH[0]) # R1

Model_Coupling_3JHH.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHH = Model_Coupling_3JHH.fit(train_3JHH[0],{'scalar_coupling': train_3JHH[1].values,

                            'fermi_coupling': train_3JHH[4].values},validation_data=(val_3JHH[0],{'scalar_coupling': val_3JHH[1].values,

                            'fermi_coupling': val_3JHH[4].values}),epochs=epochs_3JHH,verbose=1,batch_size = batch_3JHH, callbacks=[best_param])
plt.plot(history_3JHH.history['loss'])

plt.plot(history_3JHH.history['val_loss'])

plt.title('loss 3JHH')

plt.ylabel('Loss 3JHH')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_3JHH.load_weights("weights7.best.hdf5")

Model_Coupling_3JHH.compile(loss='mae', optimizer='Adam')



model_check_3JHH = Model_Coupling_3JHH.predict(val_3JHH[0])

log_MAE_3JHH = metric_mae(val_3JHH[1], model_check_3JHH)

val_3JHH[1] = val_3JHH[1].drop('prediction',axis=1)



print("log MAE_3JHH=", log_MAE_3JHH)
inter_3JHH = pd.DataFrame(test_3JHH['id']) 

inter_3JHH['type'] = test_3JHH['type']

test_3JHH_bis = test_3JHH.drop(['id','molecule_name','type'], axis=1)



model_predict_3JHH = Model_Coupling_3JHH.predict(test_3JHH_bis)

inter_3JHH['pred_values'] = model_predict_3JHH
# Mejorada,



# Split train_csv in several dataframes, each type of coupling



train_3JHC, val_3JHC, to_drop = split_train_dev(train_csv,'3JHC', split_coef = 0.1, threshold = 1.0)

train_3JHC = database_type(train_3JHC) # ESTE YA NO ES TAN LINEAL ENTRE 'FC' Y 'COUPLING CONSTANT'

val_3JHC = database_type(val_3JHC)

test_3JHC = test_csv[test_csv['type'] == '3JHC']

test_3JHC = test_3JHC.drop(to_drop, axis=1)



plt.plot(train_3JHC[4], train_3JHC[1], 'b.')

plt.show()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights8.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



def model_coupling_constant_3JHC(X):



    X_input = Input(shape = (X.shape[1],))

                    

    #concat1 = concatenate([concat, x2_output])    

    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

    xfc = Dropout(0.1)(xfc) #0,1

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dropout(0.1)(xfc) 

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHC = model_coupling_constant_3JHC(train_3JHC[0]) # R1

Model_Coupling_3JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHC = Model_Coupling_3JHC.fit(train_3JHC[0],{'scalar_coupling': train_3JHC[1].values,

                            'fermi_coupling': train_3JHC[4].values},validation_data=(val_3JHC[0],{'scalar_coupling': val_3JHC[1].values,

                            'fermi_coupling': val_3JHC[4].values}),epochs=epochs_3JHC,verbose=1,batch_size = batch_3JHC, callbacks=[best_param])
plt.plot(history_3JHC.history['loss'])

plt.plot(history_3JHC.history['val_loss'])

plt.title('loss 3JHC')

plt.ylabel('Loss 3JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
Model_Coupling_3JHC.load_weights("weights8.best.hdf5")

Model_Coupling_3JHC.compile(loss='mae', optimizer='Adam')



model_check_3JHC = Model_Coupling_3JHC.predict(val_3JHC[0])

log_MAE_3JHC = metric_mae(val_3JHC[1], model_check_3JHC)

val_3JHC[1] = val_3JHC[1].drop('prediction',axis=1)



print("log MAE_3JHC=", log_MAE_3JHC)
inter_3JHC = pd.DataFrame(test_3JHC['id']) 

inter_3JHC['type'] = test_3JHC['type']

test_3JHC_bis = test_3JHC.drop(['id','molecule_name','type'], axis=1)



model_predict_3JHC = Model_Coupling_3JHC.predict(test_3JHC_bis)

inter_3JHC['pred_values'] = model_predict_3JHC
# Concatenate all predictions' model

pred = pd.concat([inter_1JHN, inter_1JHC, inter_2JHN, inter_3JHN, inter_2JHC, inter_2JHH, inter_3JHH, inter_3JHC])
pred = pred.sort_values(by=['id'], ascending=[True])

pred = pred['pred_values']
 # SUBMISSION:

predictions = sample_submission_csv.copy()

predictions['scalar_coupling_constant'] = pred

predictions.to_csv('submission_MOLECULAR.csv', index=False)
# CHECKING THE MODEL WITH MAE METRICS



print("log MAE_1JHN=", log_MAE_1JHN)

print("log MAE_2JHH=", log_MAE_2JHH)

print("log MAE_2JHN=", log_MAE_2JHN)

print("log MAE_3JHN=", log_MAE_3JHN)

print("log MAE_1JHC=", log_MAE_1JHC)

print("log MAE_2JHC=", log_MAE_2JHC)

print("log MAE_3JHH=", log_MAE_3JHH)

print("log MAE_3JHC=", log_MAE_3JHC)
score = (log_MAE_1JHN+log_MAE_2JHH+log_MAE_2JHN+log_MAE_3JHN+log_MAE_1JHC+log_MAE_2JHC+log_MAE_3JHH+log_MAE_3JHC)/8

print('SCORE:', score)
#############################################################
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#train_csv_1JHC = scaler.fit_transform(train_1JHC[0])

#train_csv_1JHC = pd.DataFrame(train_csv_1JHC)



#val_csv_1JHC = scaler.fit_transform(val_1JHC[0])

#val_csv_1JHC = pd.DataFrame(val_csv_1JHC)

#train_csv_1JHC.columns = train_1JHC[0].columns

#val_csv_1JHC.columns = val_1JHC[0].columns

def add_center(df):

    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))

    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))

    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))



def add_distance_to_center(df):

    df['d_c'] = (((df['x_c'] - df['x'])**np.float32(2) +

                  (df['y_c'] - df['y'])**np.float32(2) +

                  (df['z_c'] - df['z'])**np.float32(2))**np.float32(0.5))



def add_distance_between(df, suffix1, suffix2):

    df[f'd_{suffix1}_{suffix2}'] = ((

        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +

        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 

        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)

    )**np.float32(0.5))