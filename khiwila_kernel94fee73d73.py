# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
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
# Modification of structures_csv



#from tqdm import tqdm_notebook as tqdm

atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor



fudge_factor = 0.05

atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}





electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}

atomic_num = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9}



atoms = structures_csv['atom'].values

atoms_en = [electronegativity[x] for x in atoms]

atoms_rad = [atomic_radius[x] for x in atoms]

atomic_number = [atomic_num[x] for x in atoms]



structures_csv['Atomic_EN'] = atoms_en

structures_csv['Atomic_radius'] = atoms_rad

structures_csv['Atomic_mass'] = atomic_number



structures_csv['Tot_atoms_molecule']= structures_csv.groupby(['molecule_name'])['molecule_name'].transform('count')



# Center of gravity. Begining



structures_csv['mass_x'] = structures_csv['Atomic_mass'] * structures_csv['x']

structures_csv['mass_y'] = structures_csv['Atomic_mass'] * structures_csv['y']

structures_csv['mass_z'] = structures_csv['Atomic_mass'] * structures_csv['z']

structures_csv['Molecular_mass']= structures_csv.groupby(['molecule_name'])['Atomic_mass'].transform('sum')



structures_csv['sum_mass_x']= structures_csv.groupby(['molecule_name'])['mass_x'].transform('sum')

structures_csv['sum_mass_y']= structures_csv.groupby(['molecule_name'])['mass_y'].transform('sum')

structures_csv['sum_mass_z']= structures_csv.groupby(['molecule_name'])['mass_z'].transform('sum')



structures_csv['XG']= structures_csv['sum_mass_x'] / structures_csv['Molecular_mass']

structures_csv['YG']= structures_csv['sum_mass_y'] / structures_csv['Molecular_mass']

structures_csv['ZG']= structures_csv['sum_mass_z'] / structures_csv['Molecular_mass']



# Center of gravity. Begining



structures_csv= structures_csv.drop({'mass_x','mass_y','mass_z','sum_mass_x','sum_mass_y','sum_mass_z'}, axis=1)



structures_csv['Dist_XG_to_x']= np.abs(structures_csv['XG']-structures_csv['x'])

structures_csv['Dist_YG_to_y']= np.abs(structures_csv['YG']-structures_csv['y'])

structures_csv['Dist_ZG_to_z']= np.abs(structures_csv['ZG']-structures_csv['z'])



structures_csv = pd.concat([structures_csv,pd.get_dummies(structures_csv['atom'],prefix='Atom_type')], axis=1)



# Merge train_csv with scalar_coupling contributions

train_csv = pd.merge(train_csv, scalar_coupling_contributions_csv, how = 'left',

                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])



# Merge train_csv with dipole moments and potential energy

train_csv = pd.merge(train_csv,dipole_moments_csv,left_on='molecule_name', right_on='molecule_name')

train_csv = pd.merge(train_csv,potential_energy_csv,left_on='molecule_name', right_on='molecule_name')

dipole=np.sqrt(np.square(train_csv['X'])+np.square(train_csv['Y'])+np.square(train_csv['Z']))

train_csv['Dipole']=dipole
# Merge train_csv with Mulliken charges



def map_mulliken_charge(database,atom_idx) :

    database = pd.merge(database,mulliken_charges_csv,how = 'left',

                 left_on = ['molecule_name',f'atom_index_{atom_idx}'],

                 right_on = ['molecule_name','atom_index']

                 )

    database = database.rename(columns={'mulliken_charge': f'mulliken_charge_{atom_idx}'}

                  )

    database = database.drop('atom_index',axis = 1)

    return database
train_csv = map_mulliken_charge(train_csv,0)

train_csv = map_mulliken_charge(train_csv,1)
# Calculate number each type of atoms in a molecule

def number_each_atoms_molecule(df,df1,index):

    df['number_each_at_mol'] = df.groupby(['molecule_name','atom'])['molecule_name'].transform('count')

    df1 = pd.merge(df1, df, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{index}'],

                  right_on = ['molecule_name',  'atom_index'])

    df1=df1.rename(columns={'x':f'x{index}','y':f'y{index}','z':f'z{index}','atom':f'atom{index}'})



    return df1
# Merge train_csv and test_csv with structures_csv



train_csv = number_each_atoms_molecule(structures_csv, train_csv,0)

train_csv = number_each_atoms_molecule(structures_csv, train_csv,1)

train_csv = train_csv.rename(columns={'x_x':'x0','y_x':'y0','z_x':'z0','x_y':'x1','y_y':'y1','z_y':'z1','atom_x':'atom_0','atom_y':'atom_1','number_each_at_mol_x':'number_each_at_mol_0','number_each_at_mol_y':'number_each_at_mol_1'})



test_csv = number_each_atoms_molecule(structures_csv, test_csv,0)

test_csv = number_each_atoms_molecule(structures_csv, test_csv,1)

test_csv = test_csv.rename(columns={'x_x':'x0','y_x':'y0','z_x':'z0','x_y':'x1','y_y':'y1','z_y':'z1','atom_x':'atom_0','atom_y':'atom_1','number_each_at_mol_x':'number_each_at_mol_0','number_each_at_mol_y':'number_each_at_mol_1'})



train_csv = reduce_mem_usage(train_csv)

test_csv = reduce_mem_usage(test_csv)
######### Operations with train_csv and test_csv ########################################



# distances train

distances_train=np.sqrt(np.square(train_csv['x0']-train_csv['x1'])+np.square(train_csv['y0']-train_csv['y1'])+np.square(train_csv['z0']-train_csv['z1']))

distances_train_x =np.abs(train_csv['x0']-train_csv['x1'])

distances_train_y =np.abs(train_csv['y0']-train_csv['y1'])

distances_train_z =np.abs(train_csv['z0']-train_csv['z1'])





train_csv['At_dist']=distances_train

train_csv['At_dist_x']=distances_train_x

train_csv['At_dist_y']=distances_train_y

train_csv['At_dist_z']=distances_train_z



train_csv['Molec_dist_mean']=train_csv.groupby('molecule_name')['At_dist'].transform('mean')

train_csv['Molec_dist_max']=train_csv.groupby('molecule_name')['At_dist'].transform('max')

train_csv['Molec_dist_min']=train_csv.groupby('molecule_name')['At_dist'].transform('min')

#train_csv['Molec_dist_std']=train_csv.groupby('molecule_name')['At_dist'].transform('std')



train_csv['Molec_type_dist_mean']=train_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('mean')

train_csv['Molec_type_dist_max']=train_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('max')

train_csv['Molec_type_dist_min']=train_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('min')



train_csv['Num_couplings']=train_csv.groupby('molecule_name')['molecule_name'].transform('count')

train_csv['Atom_0_couples_count'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

train_csv['Atom_1_couples_count'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')



train_csv['Relative_dist_x']= train_csv['At_dist_x'] / (train_csv['At_dist'] + 1e-5)

train_csv['Relative_dist_y']= train_csv['At_dist_y'] / (train_csv['At_dist'] + 1e-5)

train_csv['Relative_dist_z']= train_csv['At_dist_z'] / (train_csv['At_dist'] + 1e-5)



distances_0_CG_train= np.sqrt(np.square(train_csv['Dist_XG_to_x_x'])+np.square(train_csv['Dist_YG_to_y_x'])+np.square(train_csv['Dist_ZG_to_z_x']))

distances_1_CG_train= np.sqrt(np.square(train_csv['Dist_XG_to_x_y'])+np.square(train_csv['Dist_YG_to_y_y'])+np.square(train_csv['Dist_ZG_to_z_y']))

train_csv['Dist_CG_x0']= distances_0_CG_train

train_csv['Dist_CG_x1']= distances_1_CG_train





train_csv['Relative_XG_dist_x0']= train_csv['Dist_XG_to_x_x'] / (train_csv['Dist_CG_x0'] + 1e-5) 

train_csv['Relative_YG_dist_x0']= train_csv['Dist_YG_to_y_x'] / (train_csv['Dist_CG_x0'] + 1e-5)

train_csv['Relative_ZG_dist_x0']= train_csv['Dist_ZG_to_z_x'] / (train_csv['Dist_CG_x0'] + 1e-5)



train_csv['Relative_XG_dist_x1']= train_csv['Dist_XG_to_x_y'] / (train_csv['Dist_CG_x1'] + 1e-5) 

train_csv['Relative_YG_dist_x1']= train_csv['Dist_YG_to_y_y'] / (train_csv['Dist_CG_x1'] + 1e-5)

train_csv['Relative_ZG_dist_x1']= train_csv['Dist_ZG_to_z_y'] / (train_csv['Dist_CG_x1'] + 1e-5)



train_csv['Molecule_atom_index_0_dist_mean'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('mean')

train_csv['Molecule_atom_index_0_dist_max'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('max')

train_csv['Molecule_atom_index_0_dist_min'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('min')

#train_csv['Molecule_atom_index_0_dist_std'] = train_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('std') #### ojito con este



train_csv['Molecule_atom_index_1_dist_mean'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('mean')

train_csv['Molecule_atom_index_1_dist_max'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('max')

train_csv['Molecule_atom_index_1_dist_min'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('min')

#train_csv['Molecule_atom_index_1_dist_std'] = train_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('std') #### ojito con este



#train_csv['links'] = train_csv['type'].apply(lambda x: x[0])

#train_csv = pd.concat([train_csv,pd.get_dummies(train_csv['links'],prefix='Number_links')], axis=1)

#train_csv = pd.concat([train_csv,pd.get_dummies(train_csv['type'])], axis=1)

train_csv = pd.concat([train_csv,pd.get_dummies(train_csv['atom0'],prefix='Coupling_atom_0_type')], axis=1)

train_csv = pd.concat([train_csv,pd.get_dummies(train_csv['atom1'],prefix='Coupling_atom_1_type')], axis=1)



train_csv = train_csv.rename(columns={'Atomic_EN_x':'Atomic_EN_x0'})

train_csv = train_csv.rename(columns={'Atomic_EN_y':'Atomic_EN_x1'}) 

train_csv = train_csv.rename(columns={'Atomic_radius_x':'Atomic_radius_x0'})

train_csv = train_csv.rename(columns={'Atomic_radius_y':'Atomic_radius_x1'})

train_csv = train_csv.rename(columns={'Atomic_mass_x':'Atomic_mass_x0'})

train_csv = train_csv.rename(columns={'Atomic_mass_y':'Atomic_mass_x1'})



train_csv = train_csv.rename(columns={'Tot_atoms_molecule_x':'Tot_atoms_molecule'})

train_csv = train_csv.rename(columns={'Atom_type_C_x':'Atom_type_C_0'})  

train_csv = train_csv.rename(columns={'Atom_type_F_x':'Atom_type_F_0'})

train_csv = train_csv.rename(columns={'Atom_type_H_x':'Atom_type_H_0'}) 

train_csv = train_csv.rename(columns={'Atom_type_N_x':'Atom_type_N_0'}) 

train_csv = train_csv.rename(columns={'Atom_type_O_x':'Atom_type_O_0'}) 

train_csv = train_csv.rename(columns={'Atom_type_C_y':'Atom_type_C_1'})  

train_csv = train_csv.rename(columns={'Atom_type_F_y':'Atom_type_F_1'})

train_csv = train_csv.rename(columns={'Atom_type_H_y':'Atom_type_H_1'}) 

train_csv = train_csv.rename(columns={'Atom_type_N_y':'Atom_type_N_1'}) 

train_csv = train_csv.rename(columns={'Atom_type_O_y':'Atom_type_O_1'}) 

train_csv = train_csv.rename(columns={'Molecular_mass_x':'Molecular_mass'})

#cosine

dx= train_csv['x0']-train_csv['x1']

dy= train_csv['y0']-train_csv['y1']

dz= train_csv['z0']-train_csv['z1']



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



train_csv['cos_G_x0_plane_xy'] = ((dx*uve_0_x)+(dy*uve_0_y))/((abs(dist_xy)*abs(uve_0_xy))+1e-2)

train_csv['cos_G_x0_plane_xz'] = ((dx*uve_0_x)+(dz*uve_0_z))/((abs(dist_xz)*abs(uve_0_xz))+1e-2)

train_csv['cos_G_x0_plane_yz'] = ((dy*uve_0_y)+(dz*uve_0_z))/((abs(dist_yz)*abs(uve_0_yz))+1e-2)

train_csv['cos_G_x1_plane_xy'] = ((dx*uve_1_x)+(dy*uve_1_y))/((abs(dist_xy)*abs(uve_1_xy))+1e-2)

train_csv['cos_G_x1_plane_xz'] = ((dx*uve_1_x)+(dz*uve_1_z))/((abs(dist_xz)*abs(uve_1_xz))+1e-2)

train_csv['cos_G_x1_plane_yz'] = ((dy*uve_1_y)+(dz*uve_1_z))/((abs(dist_yz)*abs(uve_1_yz))+1e-2)







#---------------------------------------------------------

train_csv['Molec_x_mean']=train_csv.groupby('molecule_name')['x0'].transform('mean')

train_csv['Molec_x_max']=train_csv.groupby('molecule_name')['x0'].transform('max')

train_csv['Molec_x_min']=train_csv.groupby('molecule_name')['x0'].transform('min')

train_csv['Molec_y_mean']=train_csv.groupby('molecule_name')['y0'].transform('mean')

train_csv['Molec_y_max']=train_csv.groupby('molecule_name')['y0'].transform('max')

train_csv['Molec_y_min']=train_csv.groupby('molecule_name')['y0'].transform('min')

train_csv['Molec_z_mean']=train_csv.groupby('molecule_name')['z0'].transform('mean')

train_csv['Molec_z_max']=train_csv.groupby('molecule_name')['z0'].transform('max')

train_csv['Molec_z_min']=train_csv.groupby('molecule_name')['z0'].transform('min')



dx0_mean= train_csv['x0']-train_csv['Molec_x_mean']

dy0_mean= train_csv['y0']-train_csv['Molec_y_mean']

dz0_mean= train_csv['z0']-train_csv['Molec_z_mean']



dx0_max= train_csv['x0']-train_csv['Molec_x_max']

dy0_max= train_csv['y0']-train_csv['Molec_y_max']

dz0_max= train_csv['z0']-train_csv['Molec_z_max']



dx0_min= train_csv['x0']-train_csv['Molec_x_min']

dy0_min= train_csv['y0']-train_csv['Molec_y_min']

dz0_min= train_csv['z0']-train_csv['Molec_z_min']



dx1_mean= train_csv['x1']-train_csv['Molec_x_mean']

dy1_mean= train_csv['y1']-train_csv['Molec_y_mean']

dz1_mean= train_csv['z1']-train_csv['Molec_z_mean']



dx1_max= train_csv['x1']-train_csv['Molec_x_max']

dy1_max= train_csv['y1']-train_csv['Molec_y_max']

dz1_max= train_csv['z1']-train_csv['Molec_z_max']



dx1_min= train_csv['x1']-train_csv['Molec_x_min']

dy1_min= train_csv['y1']-train_csv['Molec_y_min']

dz1_min= train_csv['z1']-train_csv['Molec_z_min']



dist_x0_mean_xy = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean))

dist_x0_mean_xz = np.sqrt(np.square(dx0_mean)+np.square(dz0_mean))

dist_x0_mean_yz = np.sqrt(np.square(dy0_mean)+np.square(dz0_mean))

train_csv['cos_Mean_x0_plane_xy'] = ((dx*dx0_mean)+(dy*dy0_mean))/((abs(dist_xy)*abs(dist_x0_mean_xy))+1e-2)

train_csv['cos_Mean_x0_plane_xz'] = ((dx*dx0_mean)+(dz*dz0_mean))/((abs(dist_xz)*abs(dist_x0_mean_xz))+1e-2)

train_csv['cos_Mean_x0_plane_yz'] = ((dy*dy0_mean)+(dz*dz0_mean))/((abs(dist_yz)*abs(dist_x0_mean_yz))+1e-2)



dist_x0_max_xy = np.sqrt(np.square(dx0_max)+np.square(dy0_max))

dist_x0_max_xz = np.sqrt(np.square(dx0_max)+np.square(dz0_max))

dist_x0_max_yz = np.sqrt(np.square(dy0_max)+np.square(dz0_max))

train_csv['cos_Max_x0_plane_xy'] = ((dx*dx0_max)+(dy*dy0_max))/((abs(dist_xy)*abs(dist_x0_max_xy))+1e-2)

train_csv['cos_Max_x0_plane_xz'] = ((dx*dx0_max)+(dz*dz0_max))/((abs(dist_xz)*abs(dist_x0_max_xz))+1e-2)

train_csv['cos_Max_x0_plane_yz'] = ((dy*dy0_max)+(dz*dz0_max))/((abs(dist_yz)*abs(dist_x0_max_yz))+1e-2)



dist_x0_min_xy = np.sqrt(np.square(dx0_min)+np.square(dy0_min))

dist_x0_min_xz = np.sqrt(np.square(dx0_min)+np.square(dz0_min))

dist_x0_min_yz = np.sqrt(np.square(dy0_min)+np.square(dz0_min))

train_csv['cos_Min_x0_plane_xy'] = ((dx*dx0_min)+(dy*dy0_min))/((abs(dist_xy)*abs(dist_x0_min_xy))+1e-2)

train_csv['cos_Min_x0_plane_xz'] = ((dx*dx0_min)+(dz*dz0_min))/((abs(dist_xz)*abs(dist_x0_min_xz))+1e-2)

train_csv['cos_Min_x0_plane_yz'] = ((dy*dy0_min)+(dz*dz0_min))/((abs(dist_yz)*abs(dist_x0_min_yz))+1e-2)



dist_x0_mean = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean)+np.square(dz0_mean))

dist_x0_max = np.sqrt(np.square(dx0_max)+np.square(dy0_max)+np.square(dz0_max))

dist_x0_min = np.sqrt(np.square(dx0_min)+np.square(dy0_min)+np.square(dz0_min))

train_csv['cos_Mean_x0'] = ((dx*dx0_mean)+(dy*dy0_mean)+(dy*dy0_mean))/((abs(distances_train)*abs(dist_x0_mean))+1e-2)

train_csv['cos_Max_x0'] = ((dx*dx0_max)+(dy*dy0_max)+(dy*dy0_max))/((abs(distances_train)*abs(dist_x0_max))+1e-2)

train_csv['cos_Min_x0'] = ((dx*dx0_min)+(dy*dy0_min)+(dy*dy0_min))/((abs(distances_train)*abs(dist_x0_min))+1e-2)



dist_x1_mean_xy = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean))

dist_x1_mean_xz = np.sqrt(np.square(dx1_mean)+np.square(dz1_mean))

dist_x1_mean_yz = np.sqrt(np.square(dy1_mean)+np.square(dz1_mean))

train_csv['cos_Mean_x1_plane_xy'] = ((dx*dx1_mean)+(dy*dy1_mean))/((abs(dist_xy)*abs(dist_x1_mean_xy))+1e-2)

train_csv['cos_Mean_x1_plane_xz'] = ((dx*dx1_mean)+(dz*dz1_mean))/((abs(dist_xz)*abs(dist_x1_mean_xz))+1e-2)

train_csv['cos_Mean_x1_plane_yz'] = ((dy*dy1_mean)+(dz*dz1_mean))/((abs(dist_yz)*abs(dist_x1_mean_yz))+1e-2)



dist_x1_max_xy = np.sqrt(np.square(dx1_max)+np.square(dy1_max))

dist_x1_max_xz = np.sqrt(np.square(dx1_max)+np.square(dz1_max))

dist_x1_max_yz = np.sqrt(np.square(dy1_max)+np.square(dz1_max))

train_csv['cos_Max_x1_plane_xy'] = ((dx*dx1_max)+(dy*dy1_max))/((abs(dist_xy)*abs(dist_x1_max_xy))+1e-2)

train_csv['cos_Max_x1_plane_xz'] = ((dx*dx1_max)+(dz*dz1_max))/((abs(dist_xz)*abs(dist_x1_max_xz))+1e-2)

train_csv['cos_Max_x1_plane_yz'] = ((dy*dy1_max)+(dz*dz1_max))/((abs(dist_yz)*abs(dist_x1_max_yz))+1e-2)



dist_x1_min_xy = np.sqrt(np.square(dx1_min)+np.square(dy1_min))

dist_x1_min_xz = np.sqrt(np.square(dx1_min)+np.square(dz1_min))

dist_x1_min_yz = np.sqrt(np.square(dy1_min)+np.square(dz1_min))

train_csv['cos_Min_x1_plane_xy'] = ((dx*dx1_min)+(dy*dy1_min))/((abs(dist_xy)*abs(dist_x1_min_xy))+1e-2)

train_csv['cos_Min_x1_plane_xz'] = ((dx*dx1_min)+(dz*dz1_min))/((abs(dist_xz)*abs(dist_x1_min_xz))+1e-2)

train_csv['cos_Min_x1_plane_yz'] = ((dy*dy1_min)+(dz*dz1_min))/((abs(dist_yz)*abs(dist_x1_min_yz))+1e-2)



dist_x1_mean = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean)+np.square(dz1_mean))

dist_x1_max = np.sqrt(np.square(dx1_max)+np.square(dy1_max)+np.square(dz1_max))

dist_x1_min = np.sqrt(np.square(dx1_min)+np.square(dy1_min)+np.square(dz1_min))

train_csv['cos_Mean_x1'] = ((dx*dx1_mean)+(dy*dy1_mean)+(dz*dz1_mean))/((abs(distances_train)*abs(dist_x1_mean))+1e-2)

train_csv['cos_Max_x1'] = ((dx*dx1_max)+(dy*dy1_max)+(dz*dz1_max))/((abs(distances_train)*abs(dist_x1_max))+1e-2)

train_csv['cos_Min_x1'] = ((dx*dx1_min)+(dy*dy1_min)+(dz*dz1_min))/((abs(distances_train)*abs(dist_x1_min))+1e-2)



#-----------------------



#  Test

distances_test=np.sqrt(np.square(test_csv['x0']-test_csv['x1'])+np.square(test_csv['y0']-test_csv['y1'])+np.square(test_csv['z0']-test_csv['z1']))

distances_test_x =np.abs(test_csv['x0']-test_csv['x1'])

distances_test_y =np.abs(test_csv['y0']-test_csv['y1'])

distances_test_z =np.abs(test_csv['z0']-test_csv['z1'])



test_csv['At_dist']=distances_test

test_csv['At_dist_x']=distances_test_x

test_csv['At_dist_y']=distances_test_y

test_csv['At_dist_z']=distances_test_z



test_csv['Molec_dist_mean']=test_csv.groupby('molecule_name')['At_dist'].transform('mean')

test_csv['Molec_dist_max']=test_csv.groupby('molecule_name')['At_dist'].transform('max')

test_csv['Molec_dist_min']=test_csv.groupby('molecule_name')['At_dist'].transform('min')

#test_csv['Molec_dist_std']=test_csv.groupby('molecule_name')['At_dist'].transform('std')



test_csv['Molec_type_dist_mean']=test_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('mean')

test_csv['Molec_type_dist_max']=test_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('max')

test_csv['Molec_type_dist_min']=test_csv.groupby(['molecule_name', 'type'])['At_dist'].transform('min')



test_csv['Num_couplings']=test_csv.groupby('molecule_name')['molecule_name'].transform('count')

test_csv['Atom_0_couples_count'] = test_csv.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

test_csv['Atom_1_couples_count'] = test_csv.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')



test_csv['Relative_dist_x']= test_csv['At_dist_x'] / (test_csv['At_dist'] + 1e-5)

test_csv['Relative_dist_y']= test_csv['At_dist_y'] / (test_csv['At_dist'] + 1e-5)

test_csv['Relative_dist_z']= test_csv['At_dist_z'] / (test_csv['At_dist'] + 1e-5)



distances_0_CG_test=np.sqrt(np.square(test_csv['Dist_XG_to_x_x'])+np.square(test_csv['Dist_YG_to_y_x'])+np.square(test_csv['Dist_ZG_to_z_x']))

distances_1_CG_test=np.sqrt(np.square(test_csv['Dist_XG_to_x_y'])+np.square(test_csv['Dist_YG_to_y_y'])+np.square(test_csv['Dist_ZG_to_z_y']))

test_csv['Dist_CG_x0']= distances_0_CG_test

test_csv['Dist_CG_x1']= distances_1_CG_test



test_csv['Relative_XG_dist_x0']= test_csv['Dist_XG_to_x_x'] / (test_csv['Dist_CG_x0'] + 1e-5)

test_csv['Relative_YG_dist_x0']= test_csv['Dist_YG_to_y_x'] / (test_csv['Dist_CG_x0'] + 1e-5)

test_csv['Relative_ZG_dist_x0']= test_csv['Dist_ZG_to_z_x'] / (test_csv['Dist_CG_x0'] + 1e-5)



test_csv['Relative_XG_dist_x1']= test_csv['Dist_XG_to_x_y'] / (test_csv['Dist_CG_x1'] + 1e-5)

test_csv['Relative_YG_dist_x1']= test_csv['Dist_YG_to_y_y'] / (test_csv['Dist_CG_x1'] + 1e-5)

test_csv['Relative_ZG_dist_x1']= test_csv['Dist_ZG_to_z_y'] / (test_csv['Dist_CG_x1'] + 1e-5)



test_csv['Molecule_atom_index_0_dist_mean'] = test_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('mean')

test_csv['Molecule_atom_index_0_dist_max'] = test_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('max')

test_csv['Molecule_atom_index_0_dist_min'] = test_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('min')

#test_csv['Molecule_atom_index_0_dist_std'] = test_csv.groupby(['molecule_name', 'atom_index_0'])['At_dist'].transform('std')



test_csv['Molecule_atom_index_1_dist_mean'] = test_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('mean')

test_csv['Molecule_atom_index_1_dist_max'] = test_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('max')

test_csv['Molecule_atom_index_1_dist_min'] = test_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('min')

#test_csv['Molecule_atom_index_1_dist_std'] = test_csv.groupby(['molecule_name', 'atom_index_1'])['At_dist'].transform('std')



#test_csv['links'] = test_csv['type'].apply(lambda x: x[0])

#test_csv = pd.concat([test_csv,pd.get_dummies(test_csv['links'],prefix='Number_links')], axis=1)

#test_csv = pd.concat([test_csv,pd.get_dummies(test_csv['type'])], axis=1)

test_csv = pd.concat([test_csv,pd.get_dummies(test_csv['atom0'],prefix='Coupling_atom_0_type')], axis=1)

test_csv = pd.concat([test_csv,pd.get_dummies(test_csv['atom1'],prefix='Coupling_atom_1_type')], axis=1)



test_csv = test_csv.rename(columns={'Atomic_EN_x':'Atomic_EN_x0'})

test_csv = test_csv.rename(columns={'Atomic_EN_y':'Atomic_EN_x1'}) 

test_csv = test_csv.rename(columns={'Atomic_radius_x':'Atomic_radius_x0'})

test_csv = test_csv.rename(columns={'Atomic_radius_y':'Atomic_radius_x1'})

test_csv = test_csv.rename(columns={'Atomic_mass_x':'Atomic_mass_x0'})

test_csv = test_csv.rename(columns={'Atomic_mass_y':'Atomic_mass_x1'})



test_csv = test_csv.rename(columns={'Tot_atoms_molecule_x':'Tot_atoms_molecule'})

test_csv = test_csv.rename(columns={'Atom_type_C_x':'Atom_type_C_0'})  

test_csv = test_csv.rename(columns={'Atom_type_F_x':'Atom_type_F_0'})

test_csv = test_csv.rename(columns={'Atom_type_H_x':'Atom_type_H_0'}) 

test_csv = test_csv.rename(columns={'Atom_type_N_x':'Atom_type_N_0'}) 

test_csv = test_csv.rename(columns={'Atom_type_O_x':'Atom_type_O_0'}) 

test_csv = test_csv.rename(columns={'Atom_type_C_y':'Atom_type_C_1'})  

test_csv = test_csv.rename(columns={'Atom_type_F_y':'Atom_type_F_1'})

test_csv = test_csv.rename(columns={'Atom_type_H_y':'Atom_type_H_1'}) 

test_csv = test_csv.rename(columns={'Atom_type_N_y':'Atom_type_N_1'}) 

test_csv = test_csv.rename(columns={'Atom_type_O_y':'Atom_type_O_1'})

test_csv = test_csv.rename(columns={'Molecular_mass_x':'Molecular_mass'})

#cosine

tdx= test_csv['x0']-test_csv['x1']

tdy= test_csv['y0']-test_csv['y1']

tdz= test_csv['z0']-test_csv['z1']



tuve_0_x = test_csv['XG_x']-test_csv['x0']

tuve_0_y = test_csv['YG_x']-test_csv['y0']

tuve_0_z = test_csv['ZG_x']-test_csv['z0']



tuve_1_x = test_csv['XG_x']-test_csv['x1']

tuve_1_y = test_csv['YG_x']-test_csv['y1']

tuve_1_z = test_csv['ZG_x']-test_csv['z1']



test_csv['cos_G_x0'] = ((tuve_0_x * tdx)+(tuve_0_y * tdy)+(tuve_0_z * tdz))/((abs(distances_test)*abs(distances_0_CG_test))+ 1e-2)

test_csv['cos_G_x1'] = ((tuve_1_x * tdx)+(tuve_1_y * tdy)+(tuve_1_z * tdz))/((abs(distances_test)*abs(distances_1_CG_test))+ 1e-2)



tdist_xy = np.sqrt(np.square(tdx)+np.square(tdy))

tdist_xz = np.sqrt(np.square(tdx)+np.square(tdz))

tdist_yz = np.sqrt(np.square(tdy)+np.square(tdz))



tuve_0_xy = np.sqrt(np.square(tuve_0_x)+np.square(tuve_0_y))

tuve_0_xz = np.sqrt(np.square(tuve_0_x)+np.square(tuve_0_z))

tuve_0_yz = np.sqrt(np.square(tuve_0_y)+np.square(tuve_0_z))

tuve_1_xy = np.sqrt(np.square(tuve_1_x)+np.square(tuve_1_y))

tuve_1_xz = np.sqrt(np.square(tuve_1_x)+np.square(tuve_1_z))

tuve_1_yz = np.sqrt(np.square(tuve_1_y)+np.square(tuve_1_z))



test_csv['cos_G_x0_plane_xy'] = ((tdx*tuve_0_x)+(tdy*tuve_0_y))/((abs(tdist_xy)*abs(tuve_0_xy))+1e-2)

test_csv['cos_G_x0_plane_xz'] = ((tdx*tuve_0_x)+(tdz*tuve_0_z))/((abs(tdist_xz)*abs(tuve_0_xz))+1e-2)

test_csv['cos_G_x0_plane_yz'] = ((tdy*tuve_0_y)+(tdz*tuve_0_z))/((abs(tdist_yz)*abs(tuve_0_yz))+1e-2)

test_csv['cos_G_x1_plane_xy'] = ((tdx*tuve_1_x)+(tdy*tuve_1_y))/((abs(tdist_xy)*abs(tuve_1_xy))+1e-2)

test_csv['cos_G_x1_plane_xz'] = ((tdx*tuve_1_x)+(tdz*tuve_1_z))/((abs(tdist_xz)*abs(tuve_1_xz))+1e-2)

test_csv['cos_G_x1_plane_yz'] = ((tdy*tuve_1_y)+(tdz*tuve_1_z))/((abs(tdist_yz)*abs(tuve_1_yz))+1e-2)



#---------------------------------------------------------

test_csv['Molec_x_mean']=test_csv.groupby('molecule_name')['x0'].transform('mean')

test_csv['Molec_x_max']=test_csv.groupby('molecule_name')['x0'].transform('max')

test_csv['Molec_x_min']=test_csv.groupby('molecule_name')['x0'].transform('min')

test_csv['Molec_y_mean']=test_csv.groupby('molecule_name')['y0'].transform('mean')

test_csv['Molec_y_max']=test_csv.groupby('molecule_name')['y0'].transform('max')

test_csv['Molec_y_min']=test_csv.groupby('molecule_name')['y0'].transform('min')

test_csv['Molec_z_mean']=test_csv.groupby('molecule_name')['z0'].transform('mean')

test_csv['Molec_z_max']=test_csv.groupby('molecule_name')['z0'].transform('max')

test_csv['Molec_z_min']=test_csv.groupby('molecule_name')['z0'].transform('min')



dx0_mean= test_csv['x0']-test_csv['Molec_x_mean']

dy0_mean= test_csv['y0']-test_csv['Molec_y_mean']

dz0_mean= test_csv['z0']-test_csv['Molec_z_mean']



dx0_max= test_csv['x0']-test_csv['Molec_x_max']

dy0_max= test_csv['y0']-test_csv['Molec_y_max']

dz0_max= test_csv['z0']-test_csv['Molec_z_max']



dx0_min= test_csv['x0']-test_csv['Molec_x_min']

dy0_min= test_csv['y0']-test_csv['Molec_y_min']

dz0_min= test_csv['z0']-test_csv['Molec_z_min']



dx1_mean= test_csv['x1']-test_csv['Molec_x_mean']

dy1_mean= test_csv['y1']-test_csv['Molec_y_mean']

dz1_mean= test_csv['z1']-test_csv['Molec_z_mean']



dx1_max= test_csv['x1']-test_csv['Molec_x_max']

dy1_max= test_csv['y1']-test_csv['Molec_y_max']

dz1_max= test_csv['z1']-test_csv['Molec_z_max']



dx1_min= test_csv['x1']-test_csv['Molec_x_min']

dy1_min= test_csv['y1']-test_csv['Molec_y_min']

dz1_min= test_csv['z1']-test_csv['Molec_z_min']



dist_x0_mean_xy = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean))

dist_x0_mean_xz = np.sqrt(np.square(dx0_mean)+np.square(dz0_mean))

dist_x0_mean_yz = np.sqrt(np.square(dy0_mean)+np.square(dz0_mean))

test_csv['cos_Mean_x0_plane_xy'] = ((tdx*dx0_mean)+(tdy*dy0_mean))/((abs(tdist_xy)*abs(dist_x0_mean_xy))+1e-2)

test_csv['cos_Mean_x0_plane_xz'] = ((tdx*dx0_mean)+(tdz*dz0_mean))/((abs(tdist_xz)*abs(dist_x0_mean_xz))+1e-2)

test_csv['cos_Mean_x0_plane_yz'] = ((tdy*dy0_mean)+(tdz*dz0_mean))/((abs(tdist_yz)*abs(dist_x0_mean_yz))+1e-2)



dist_x0_max_xy = np.sqrt(np.square(dx0_max)+np.square(dy0_max))

dist_x0_max_xz = np.sqrt(np.square(dx0_max)+np.square(dz0_max))

dist_x0_max_yz = np.sqrt(np.square(dy0_max)+np.square(dz0_max))

test_csv['cos_Max_x0_plane_xy'] = ((tdx*dx0_max)+(tdy*dy0_max))/((abs(tdist_xy)*abs(dist_x0_max_xy))+1e-2)

test_csv['cos_Max_x0_plane_xz'] = ((tdx*dx0_max)+(tdz*dz0_max))/((abs(tdist_xz)*abs(dist_x0_max_xz))+1e-2)

test_csv['cos_Max_x0_plane_yz'] = ((tdy*dy0_max)+(tdz*dz0_max))/((abs(tdist_yz)*abs(dist_x0_max_yz))+1e-2)



dist_x0_min_xy = np.sqrt(np.square(dx0_min)+np.square(dy0_min))

dist_x0_min_xz = np.sqrt(np.square(dx0_min)+np.square(dz0_min))

dist_x0_min_yz = np.sqrt(np.square(dy0_min)+np.square(dz0_min))

test_csv['cos_Min_x0_plane_xy'] = ((tdx*dx0_min)+(tdy*dy0_min))/((abs(tdist_xy)*abs(dist_x0_min_xy))+1e-2)

test_csv['cos_Min_x0_plane_xz'] = ((tdx*dx0_min)+(tdz*dz0_min))/((abs(tdist_xz)*abs(dist_x0_min_xz))+1e-2)

test_csv['cos_Min_x0_plane_yz'] = ((tdy*dy0_min)+(tdz*dz0_min))/((abs(tdist_yz)*abs(dist_x0_min_yz))+1e-2)



dist_x0_mean = np.sqrt(np.square(dx0_mean)+np.square(dy0_mean)+np.square(dz0_mean))

dist_x0_max = np.sqrt(np.square(dx0_max)+np.square(dy0_max)+np.square(dz0_max))

dist_x0_min = np.sqrt(np.square(dx0_min)+np.square(dy0_min)+np.square(dz0_min))

test_csv['cos_Mean_x0'] = ((tdx*dx0_mean)+(tdy*dy0_mean)+(tdz*dz0_mean))/((abs(distances_test)*abs(dist_x0_mean))+1e-2)

test_csv['cos_Max_x0'] = ((tdx*dx0_max)+(tdy*dy0_max)+(tdz*dz0_max))/((abs(distances_test)*abs(dist_x0_max))+1e-2)

test_csv['cos_Min_x0'] = ((tdx*dx0_min)+(tdy*dy0_min)+(tdz*dz0_min))/((abs(distances_test)*abs(dist_x0_min))+1e-2)



dist_x1_mean_xy = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean))

dist_x1_mean_xz = np.sqrt(np.square(dx1_mean)+np.square(dz1_mean))

dist_x1_mean_yz = np.sqrt(np.square(dy1_mean)+np.square(dz1_mean))

test_csv['cos_Mean_x1_plane_xy'] = ((tdx*dx1_mean)+(tdy*dy1_mean))/((abs(tdist_xy)*abs(dist_x1_mean_xy))+1e-2)

test_csv['cos_Mean_x1_plane_xz'] = ((tdx*dx1_mean)+(tdz*dz1_mean))/((abs(tdist_xz)*abs(dist_x1_mean_xz))+1e-2)

test_csv['cos_Mean_x1_plane_yz'] = ((tdy*dy1_mean)+(tdz*dz1_mean))/((abs(tdist_yz)*abs(dist_x1_mean_yz))+1e-2)



dist_x1_max_xy = np.sqrt(np.square(dx1_max)+np.square(dy1_max))

dist_x1_max_xz = np.sqrt(np.square(dx1_max)+np.square(dz1_max))

dist_x1_max_yz = np.sqrt(np.square(dy1_max)+np.square(dz1_max))

test_csv['cos_Max_x1_plane_xy'] = ((tdx*dx1_max)+(tdy*dy1_max))/((abs(tdist_xy)*abs(dist_x1_max_xy))+1e-2)

test_csv['cos_Max_x1_plane_xz'] = ((tdx*dx1_max)+(tdz*dz1_max))/((abs(tdist_xz)*abs(dist_x1_max_xz))+1e-2)

test_csv['cos_Max_x1_plane_yz'] = ((tdy*dy1_max)+(tdz*dz1_max))/((abs(tdist_yz)*abs(dist_x1_max_yz))+1e-2)



dist_x1_min_xy = np.sqrt(np.square(dx1_min)+np.square(dy1_min))

dist_x1_min_xz = np.sqrt(np.square(dx1_min)+np.square(dz1_min))

dist_x1_min_yz = np.sqrt(np.square(dy1_min)+np.square(dz1_min))

test_csv['cos_Min_x1_plane_xy'] = ((tdx*dx1_min)+(tdy*dy1_min))/((abs(tdist_xy)*abs(dist_x1_min_xy))+1e-2)

test_csv['cos_Min_x1_plane_xz'] = ((tdx*dx1_min)+(tdz*dz1_min))/((abs(tdist_xz)*abs(dist_x1_min_xz))+1e-2)

test_csv['cos_Min_x1_plane_yz'] = ((tdy*dy1_min)+(tdz*dz1_min))/((abs(tdist_yz)*abs(dist_x1_min_yz))+1e-2)



dist_x1_mean = np.sqrt(np.square(dx1_mean)+np.square(dy1_mean)+np.square(dz1_mean))

dist_x1_max = np.sqrt(np.square(dx1_max)+np.square(dy1_max)+np.square(dz1_max))

dist_x1_min = np.sqrt(np.square(dx1_min)+np.square(dy1_min)+np.square(dz1_min))

test_csv['cos_Mean_x1'] = ((tdx*dx1_mean)+(tdy*dy1_mean)+(tdz*dz1_mean))/((abs(distances_test)*abs(dist_x1_mean))+1e-2)

test_csv['cos_Max_x1'] = ((tdx*dx1_max)+(tdy*dy1_max)+(tdz*dz1_max))/((abs(distances_test)*abs(dist_x1_max))+1e-2)

test_csv['cos_Min_x1'] = ((tdx*dx1_min)+(tdy*dy1_min)+(tdz*dz1_min))/((abs(distances_test)*abs(dist_x1_min))+1e-2)



#-----------------------



# Drop columns

train_csv = train_csv.drop({'X','Y','Z'}, axis=1)



train_csv = train_csv.drop(['atom_index_x','atom_index_y'], axis=1)

train_csv = train_csv.drop(['XG_y','YG_y','ZG_y'], axis=1)

train_csv = train_csv.drop(['XG_x','YG_x','ZG_x'], axis=1)

train_csv = train_csv.drop({'Molecular_mass_y'}, axis=1)

train_csv = train_csv.drop(['atom0','atom1'], axis=1)

#train_csv = train_csv.drop(['links'], axis=1)

#train_csv = train_csv.drop(['id','molecule_name'], axis=1)

train_csv = train_csv.drop({'Tot_atoms_molecule_y'}, axis=1)

train_csv = train_csv.drop({'x0','x1','y1','z0','z1'}, axis=1)

train_csv = train_csv.drop({'y0'}, axis=1)

train_csv = train_csv.drop({'Molec_x_mean','Molec_x_max','Molec_x_min'}, axis=1)

train_csv = train_csv.drop({'Molec_y_mean','Molec_y_max','Molec_y_min'}, axis=1)

train_csv = train_csv.drop({'Molec_z_mean','Molec_z_max','Molec_z_min'}, axis=1)

#------------------------------------------------------------------------------

test_csv = test_csv.drop(['atom_index_x','atom_index_y'], axis=1)

test_csv = test_csv.drop(['XG_y','YG_y','ZG_y'], axis=1)

test_csv = test_csv.drop(['XG_x','YG_x','ZG_x'], axis=1)

test_csv = test_csv.drop(['Molecular_mass_y'], axis=1)

test_csv = test_csv.drop(['atom0','atom1'], axis=1)

#test_csv = test_csv.drop(['links'], axis=1)

#test_csv = test_csv.drop(['id','molecule_name'], axis=1)

test_csv = test_csv.drop({'Tot_atoms_molecule_y'}, axis=1)

test_csv = test_csv.drop({'x0','x1','y1','z0','z1'}, axis=1)

test_csv = test_csv.drop({'y0'}, axis=1)

test_csv = test_csv.drop({'Molec_x_mean','Molec_x_max','Molec_x_min'}, axis=1)

test_csv = test_csv.drop({'Molec_y_mean','Molec_y_max','Molec_y_min'}, axis=1)

test_csv = test_csv.drop({'Molec_z_mean','Molec_z_max','Molec_z_min'}, axis=1)
train_csv = reduce_mem_usage(train_csv)

test_csv = reduce_mem_usage(test_csv)
train_csv.head()
test_csv.head()
# Intermediate databases:

train_csv = train_csv.drop(['id','molecule_name'], axis=1)

train_csv = train_csv.drop(['Atomic_EN_x0','Atomic_radius_x0','Atomic_mass_x0','Atom_type_C_0','Atom_type_F_0','Atom_type_H_0','Atom_type_N_0','Atom_type_O_0','Coupling_atom_0_type_H'], axis=1)

train_csv = train_csv.drop(['Atomic_EN_x1','Atomic_radius_x1','Atomic_mass_x1','Atom_type_C_1','Atom_type_F_1','Atom_type_H_1','Atom_type_N_1','Atom_type_O_1'], axis=1)

train_csv = train_csv.drop(['Coupling_atom_1_type_C','Coupling_atom_1_type_H','Coupling_atom_1_type_N'], axis=1)



test_csv = test_csv.drop(['Atomic_EN_x0','Atomic_radius_x0','Atomic_mass_x0','Atom_type_C_0','Atom_type_F_0','Atom_type_H_0','Atom_type_N_0','Atom_type_O_0','Coupling_atom_0_type_H'], axis=1)

test_csv = test_csv.drop(['Atomic_EN_x1','Atomic_radius_x1','Atomic_mass_x1','Atom_type_C_1','Atom_type_F_1','Atom_type_H_1','Atom_type_N_1','Atom_type_O_1'], axis=1)

test_csv = test_csv.drop(['Coupling_atom_1_type_C','Coupling_atom_1_type_H','Coupling_atom_1_type_N'], axis=1)

# Threshold for removing correlated variables

# https://www.kaggle.com/adrianoavelar/gridsearch-for-eachtype-lb-1-0



threshold = 0.95



# Absolute value correlation matrix





corr_matrix = train_csv.corr().abs()



# Getting the upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

print('The columns are: ', to_drop )





train_csv = train_csv.drop(to_drop[1:],axis=1)

test_csv = test_csv.drop(to_drop[1:],axis=1)

train_csv = reduce_mem_usage(train_csv)

test_csv = reduce_mem_usage(test_csv)



print('Training shape: ', train_csv.shape)

print('Testing shape: ', test_csv.shape)
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

from keras.layers import LeakyReLU

from sklearn import metrics

from keras import regularizers
def database_type(db,index):

    db_type = db[db['type'] == index]

    

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

    db_type = reduce_mem_usage(db_type)

    

    return [db_type, coupling, dipole, potential, fermi, spin_dipolar,paramagnetic_spin,diamagnetic_spin,mulliken_0,mulliken_1]
def metric_mae(df, preds):

    df["prediction"] = preds

    maes = []

    y_true = df.scalar_coupling_constant.values

    y_pred = df.prediction.values

    mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

    maes.append(mae)

    

    return np.mean(maes)
# Split train_csv in several dataframes, each type of coupling

train_csv, val_csv = train_test_split(train_csv, test_size=0.10, random_state=111)



train_1JHN = database_type(train_csv,'1JHN')

train_1JHC = database_type(train_csv,'1JHC')

train_2JHH = database_type(train_csv,'2JHH')

train_2JHN = database_type(train_csv,'2JHN')

train_2JHC = database_type(train_csv,'2JHC')

train_3JHH = database_type(train_csv,'3JHH')

train_3JHC = database_type(train_csv,'3JHC')

train_3JHN = database_type(train_csv,'3JHN')



val_1JHN = database_type(val_csv,'1JHN')

val_1JHC = database_type(val_csv,'1JHC')

val_2JHH = database_type(val_csv,'2JHH')

val_2JHN = database_type(val_csv,'2JHN')

val_2JHC = database_type(val_csv,'2JHC')

val_3JHH = database_type(val_csv,'3JHH')

val_3JHC = database_type(val_csv,'3JHC')

val_3JHN = database_type(val_csv,'3JHN')



test_1JHN = test_csv[test_csv['type'] == '1JHN']

test_1JHC = test_csv[test_csv['type'] == '1JHC']

test_2JHH = test_csv[test_csv['type'] == '2JHH']

test_2JHN = test_csv[test_csv['type'] == '2JHN']

test_2JHC = test_csv[test_csv['type'] == '2JHC']

test_3JHH = test_csv[test_csv['type'] == '3JHH']

test_3JHC = test_csv[test_csv['type'] == '3JHC']

test_3JHN = test_csv[test_csv['type'] == '3JHN']
 # TRAINING:
# HYPERPARAMETERS

epochs_1JHN= 100

batch_1JHN = 32



epochs_2JHH= 50

batch_2JHH = 128



epochs_2JHN= 50

batch_2JHN = 128



epochs_3JHN= 50

batch_3JHN = 128



epochs_1JHC= 50

batch_1JHC = 128



epochs_2JHC= 50

batch_2JHC = 1024



epochs_3JHH= 50

batch_3JHH = 512



epochs_3JHC= 50

batch_3JHC = 1024

#### 1) 

def model_coupling_constant_1JHN(X):

    X_input = Input(shape = (X.shape[1],))

                    

  

    #concat1 = concatenate([concat, x2_output])    

    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(256, activation='elu')(xfc)

    xfc = Dense(128, activation='elu')(xfc)

    xfc = Dense(64, activation='elu')(xfc)

    xfc = Dense(32, activation='elu')(xfc)

    xfc = Dense(16, activation='elu')(xfc)

    xfc = Dense(8, activation='elu')(xfc)

    x3_output  = Dense(1, activation='linear', name = 'fermi_coupling')(xfc)

    

    concat = concatenate([X_input, x3_output])

    

    # Scalar Coupling Constant

    

    xsc = BatchNormalization()(concat)

    xsc = Dense(256, activation='elu')(xsc) 

    xsc = Dense(128, activation='elu')(xsc)

    xsc = Dense(64, activation='elu')(xsc)

    xsc = Dense(32, activation='elu')(xsc)

    xsc = Dense(16, activation='elu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_1JHN = model_coupling_constant_1JHN(train_1JHN[0]) # R1

Model_Coupling_1JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_1JHN = Model_Coupling_1JHN.fit(train_1JHN[0],{'scalar_coupling': train_1JHN[1].values,

                            'fermi_coupling': train_1JHN[4].values},validation_data=(val_1JHN[0],{'scalar_coupling': val_1JHN[1].values,

                            'fermi_coupling': val_1JHN[4].values}),epochs=epochs_1JHN,verbose=1,batch_size = batch_1JHN)
plt.plot(history_1JHN.history['loss'])

plt.plot(history_1JHN.history['val_loss'])

plt.title('loss 1JHN')

plt.ylabel('Loss 1JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 1JHN



model_check_1JHN = Model_Coupling_1JHN.predict(val_1JHN[0])

log_MAE_1JHN = metric_mae(val_1JHN[1], model_check_1JHN)

val_1JHN[1] = val_1JHN[1].drop('prediction',axis=1)



print("log MAE_1JHN=", log_MAE_1JHN)
inter_1JHN = pd.DataFrame(test_1JHN['id']) 

inter_1JHN['type'] = test_1JHN['type']

test_1JHN_bis = test_1JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_1JHN = Model_Coupling_1JHN.predict(test_1JHN_bis)

inter_1JHN['pred_values'] = model_predict_1JHN
#### 2) 

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHH = model_coupling_constant_2JHH(train_2JHH[0]) # R1

Model_Coupling_2JHH.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHH = Model_Coupling_2JHH.fit(train_2JHH[0],{'scalar_coupling': train_2JHH[1].values,

                            'fermi_coupling': train_2JHH[4].values},validation_data=(val_2JHH[0],{'scalar_coupling': val_2JHH[1].values,

                            'fermi_coupling': val_2JHH[4].values}),epochs=epochs_2JHH,verbose=1,batch_size = batch_2JHH)
plt.plot(history_2JHH.history['loss'])

plt.plot(history_2JHH.history['val_loss'])

plt.title('loss 2JHH')

plt.ylabel('Loss 2JHH')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 2JHH



model_check_2JHH = Model_Coupling_2JHH.predict(val_2JHH[0])

log_MAE_2JHH = metric_mae(val_2JHH[1], model_check_2JHH)

val_2JHH[1] = val_2JHH[1].drop('prediction',axis=1)



print("log MAE_2JHH=", log_MAE_2JHH)
inter_2JHH = pd.DataFrame(test_2JHH['id']) 

inter_2JHH['type'] = test_2JHH['type']

test_2JHH_bis = test_2JHH.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHH = Model_Coupling_2JHH.predict(test_2JHH_bis)

inter_2JHH['pred_values'] = model_predict_2JHH
#### 3) 

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHN = model_coupling_constant_2JHN(train_2JHN[0]) # R1

Model_Coupling_2JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHN = Model_Coupling_2JHN.fit(train_2JHN[0],{'scalar_coupling': train_2JHN[1].values,

                            'fermi_coupling': train_2JHN[4].values},validation_data=(val_2JHN[0],{'scalar_coupling': val_2JHN[1].values,

                            'fermi_coupling': val_2JHN[4].values}),epochs=epochs_2JHN,verbose=1,batch_size = batch_2JHN)
plt.plot(history_2JHN.history['loss'])

plt.plot(history_2JHN.history['val_loss'])

plt.title('loss 2JHN')

plt.ylabel('Loss 2JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 2JHN



model_check_2JHN = Model_Coupling_2JHN.predict(val_2JHN[0])

log_MAE_2JHN = metric_mae(val_2JHN[1], model_check_2JHN)

val_2JHN[1] = val_2JHN[1].drop('prediction',axis=1)



print("log MAE_2JHN=", log_MAE_2JHN)
inter_2JHN = pd.DataFrame(test_2JHN['id']) 

inter_2JHN['type'] = test_2JHN['type']

test_2JHN_bis = test_2JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHN = Model_Coupling_2JHN.predict(test_2JHN_bis)

inter_2JHN['pred_values'] = model_predict_2JHN
#### 4)

def model_coupling_constant_3JHN(X):

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHN = model_coupling_constant_3JHN(train_3JHN[0]) # R1

Model_Coupling_3JHN.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHN = Model_Coupling_3JHN.fit(train_3JHN[0],{'scalar_coupling': train_3JHN[1].values,

                            'fermi_coupling': train_3JHN[4].values},validation_data=(val_3JHN[0],{'scalar_coupling': val_3JHN[1].values,

                            'fermi_coupling': val_3JHN[4].values}),epochs=epochs_3JHN,verbose=1,batch_size = batch_3JHN)
plt.plot(history_3JHN.history['loss'])

plt.plot(history_3JHN.history['val_loss'])

plt.title('loss 3JHN')

plt.ylabel('Loss 3JHN')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 3JHN



model_check_3JHN = Model_Coupling_3JHN.predict(val_3JHN[0])

log_MAE_3JHN = metric_mae(val_3JHN[1], model_check_3JHN)

val_3JHN[1] = val_3JHN[1].drop('prediction',axis=1)



print("log MAE_3JHN=", log_MAE_3JHN)
inter_3JHN = pd.DataFrame(test_3JHN['id']) 

inter_3JHN['type'] = test_3JHN['type']

test_3JHN_bis = test_3JHN.drop(['id','molecule_name','type'], axis=1)



model_predict_3JHN = Model_Coupling_3JHN.predict(test_3JHN_bis)

inter_3JHN['pred_values'] = model_predict_3JHN
################ B) 1JHC , 2JHC, 3JHH , 3JHC  #############################################
#### 5)  

def model_coupling_constant_1JHC(X):

    X_input = Input(shape = (X.shape[1],))

    act= LeakyReLU(alpha=0.1)

    dr=0.4

    

    X_input = Input(shape = (X.shape[1],))   

    

    #concat1 = concatenate([concat, x2_output])    

    

    # Fermi coupling

    xfc = BatchNormalization()(X_input)

    xfc = Dense(512, activation='elu')(xfc) 

    xfc = Dense(1024, activation='elu')(xfc)

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_1JHC = model_coupling_constant_1JHC(train_1JHC[0]) # R1

Model_Coupling_1JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_1JHC = Model_Coupling_1JHC.fit(train_1JHC[0],{'scalar_coupling': train_1JHC[1].values,

                            'fermi_coupling': train_1JHC[4].values},validation_data=(val_1JHC[0],{'scalar_coupling': val_1JHC[1].values,

                            'fermi_coupling': val_1JHC[4].values}),epochs=epochs_1JHC,verbose=1,batch_size = batch_1JHC)
plt.plot(history_1JHC.history['loss'])

plt.plot(history_1JHC.history['val_loss'])

plt.title('loss 1JHC')

plt.ylabel('Loss 1JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 1JHC



model_check_1JHC = Model_Coupling_1JHC.predict(val_1JHC[0])

log_MAE_1JHC = metric_mae(val_1JHC[1], model_check_1JHC)

val_1JHC[1] = val_1JHC[1].drop('prediction',axis=1)



print("log MAE_1JHC=", log_MAE_1JHC)
inter_1JHC = pd.DataFrame(test_1JHC['id']) 

inter_1JHC['type'] = test_1JHC['type']

test_1JHC_bis = test_1JHC.drop(['id','molecule_name','type'], axis=1)



model_predict_1JHC = Model_Coupling_1JHC.predict(test_1JHC_bis)

inter_1JHC['pred_values'] = model_predict_1JHC
## 6)

def model_coupling_constant_2JHC(X):

    X_input = Input(shape = (X.shape[1],))

    act= LeakyReLU(alpha=0.1)

    dr=0.4

    

    X_input = Input(shape = (X.shape[1],))

                    

  

    #concat1 = concatenate([concat, x2_output])    

    

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_2JHC = model_coupling_constant_2JHC(train_2JHC[0]) # R1

Model_Coupling_2JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_2JHC = Model_Coupling_2JHC.fit(train_2JHC[0],{'scalar_coupling': train_2JHC[1].values,

                            'fermi_coupling': train_2JHC[4].values},validation_data=(val_2JHC[0],{'scalar_coupling': val_2JHC[1].values,

                            'fermi_coupling': val_2JHC[4].values}),epochs=epochs_2JHC,verbose=1,batch_size = batch_2JHC)
plt.plot(history_2JHC.history['loss'])

plt.plot(history_2JHC.history['val_loss'])

plt.title('loss 2JHC')

plt.ylabel('Loss 2JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 2JHC



model_check_2JHC = Model_Coupling_2JHC.predict(val_2JHC[0])

log_MAE_2JHC = metric_mae(val_2JHC[1], model_check_2JHC)

val_2JHC[1] = val_2JHC[1].drop('prediction',axis=1)



print("log MAE_2JHC=", log_MAE_2JHC)
inter_2JHC = pd.DataFrame(test_2JHC['id']) 

inter_2JHC['type'] = test_2JHC['type']

test_2JHC_bis = test_2JHC.drop(['id','molecule_name','type'], axis=1)



model_predict_2JHC = Model_Coupling_2JHC.predict(test_2JHC_bis)

inter_2JHC['pred_values'] = model_predict_2JHC
### 7)

def model_coupling_constant_3JHH(X):



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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='relu')(xsc)

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='relu')(xsc)

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHH = model_coupling_constant_3JHH(train_3JHH[0]) # R1

Model_Coupling_3JHH.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHH = Model_Coupling_3JHH.fit(train_3JHH[0],{'scalar_coupling': train_3JHH[1].values,

                            'fermi_coupling': train_3JHH[4].values},validation_data=(val_3JHH[0],{'scalar_coupling': val_3JHH[1].values,

                            'fermi_coupling': val_3JHH[4].values}),epochs=epochs_3JHH,verbose=1,batch_size = batch_3JHH)
plt.plot(history_3JHH.history['loss'])

plt.plot(history_3JHH.history['val_loss'])

plt.title('loss 3JHH')

plt.ylabel('Loss 3JHH')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 3JHH



model_check_3JHH = Model_Coupling_3JHH.predict(val_3JHH[0])

log_MAE_3JHH = metric_mae(val_3JHH[1], model_check_3JHH)

val_3JHH[1] = val_3JHH[1].drop('prediction',axis=1)



print("log MAE_3JHH=", log_MAE_3JHH)
inter_3JHH = pd.DataFrame(test_3JHH['id']) 

inter_3JHH['type'] = test_3JHH['type']

test_3JHH_bis = test_3JHH.drop(['id','molecule_name','type'], axis=1)



model_predict_3JHH = Model_Coupling_3JHH.predict(test_3JHH_bis)

inter_3JHH['pred_values'] = model_predict_3JHH
### 8)

def model_coupling_constant_3JHC(X):



    X_input = Input(shape = (X.shape[1],))

    act= LeakyReLU(alpha=0.1)

    dr=0.4

    

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

    #xsc = Dense(256, activation='relu')(xsc) 

    #xsc = Dense(128, activation='elu')(xsc) #esta

    #xsc = Dense(64, activation='relu')(xsc)

    #xsc = Dense(32, activation='relu')(xsc)

    #xsc = Dense(16, activation='elu')(xsc) #esta

    xsc = Dense(8, activation='elu')(xsc)

    x9_output  = Dense(1, activation='linear', name = 'scalar_coupling')(xsc)



    model= Model(inputs = [X_input] , outputs = [x9_output])



    model.summary()

    return model



Model_Coupling_3JHC = model_coupling_constant_3JHC(train_3JHC[0]) # R1

Model_Coupling_3JHC.compile(loss='mae', optimizer='Adam')#, metrics=['accuracy'])

history_3JHC = Model_Coupling_3JHC.fit(train_3JHC[0],{'scalar_coupling': train_3JHC[1].values,

                            'fermi_coupling': train_3JHC[4].values},validation_data=(val_3JHC[0],{'scalar_coupling': val_3JHC[1].values,

                            'fermi_coupling': val_3JHC[4].values}),epochs=epochs_3JHC,verbose=1,batch_size = batch_3JHC)
plt.plot(history_3JHC.history['loss'])

plt.plot(history_3JHC.history['val_loss'])

plt.title('loss 3JHC')

plt.ylabel('Loss 3JHC')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
### 3JHC



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