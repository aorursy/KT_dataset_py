import numpy as np

import pandas as pd

from operator import itemgetter

from scipy.sparse.csgraph import shortest_path

from scipy.stats import kurtosis, skew

import gc

from itertools import combinations, permutations

import time

import gc

import warnings

import copy





pd.options.display.precision = 15

np.set_printoptions(suppress=True)

warnings.filterwarnings("ignore")
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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def get_dihedrals(dihedral_paths, vec_mat):

    

    x0, x1, x2, x3 = dihedral_paths.T

    

    b0 = -1.0 * vec_mat[x1, x0]

    b1 = vec_mat[x2, x1]

    b2 = vec_mat[x3, x2]



    b0xb1 = np.cross(b0, b1)

    b1xb2 = np.cross(b1, b2)



    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)



    y = np.sum(b0xb1_x_b1xb2*b1, axis=1)/np.linalg.norm(b1, axis=1)

    x = np.sum(b0xb1*b1xb2, axis=1)

    

    grad = np.arctan2(y, x)

    return grad



def get_path_matrix(Pr, n_atoms, dist_mat):

    

    path_matrix = [0]*(n_atoms**2)

    path_length_matrix = [0]*(n_atoms**2)

    path_distance = [0]*(n_atoms**2)

    for i in range(n_atoms):

        for j in range(n_atoms):                 

            path = [j]

            k = j

            while Pr[i, k] != -9999:

                path.append(Pr[i, k])

                k = Pr[i, k]

            path = np.array(path[::-1])

            path_matrix[i*n_atoms+j] = path

            path_len = len(path)

            path_length_matrix[i*n_atoms+j] = path_len

            path_distance[i*n_atoms+j] = np.sum([dist_mat[path[i], path[i+1]] for i in range(path_len-1)])

    

    return np.array(path_length_matrix).reshape(n_atoms, n_atoms), np.array(path_matrix).reshape(n_atoms, n_atoms), np.array(path_distance).reshape(n_atoms, n_atoms)

    

def get_fast_dist_matrix(xyz, df_start_indexes, struct_start_indexes, molecule_id):

    struct_start_index, struct_end_index = struct_start_indexes[molecule_id], struct_start_indexes[molecule_id+1]

    locs = xyz[struct_start_index:struct_end_index]    

    num_atoms = struct_end_index - struct_start_index

    loc_tile = np.tile(locs.T, (num_atoms,1,1))

    dist_mat = np.sqrt(((loc_tile - loc_tile.T)**2).sum(axis=1))

    vec_mat = loc_tile.T - loc_tile 

    vec_mat = np.transpose(vec_mat, axes=(1,2,0)).T

    

    df_start_index, df_end_index = df_start_indexes[molecule_id], df_start_indexes[molecule_id+1]

    atom_index = atom_indexes[df_start_index:df_end_index]    

    return dist_mat, vec_mat, atom_index, locs



def get_projection(u,v):

    #v must be an unit-vector

    return np.sum(u*v, axis=1).reshape(-1,1)*v



def calculate_force_resultant(unit_vec, force_mat, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat):

    

    forces = unit_vec*force_mat.reshape(n_atoms,n_atoms,1)

    

    #forces_resultant_per_axis has shape: [[X, Y, Z resultant_force on atom_num_0], [X, Y, Z resultant_force on atom_num_1], [...], [X, Y, Z resultant_force on atom_num_n]]

    forces_resultant_per_axis = np.sum(forces, axis=1)

    

    u1 = forces_resultant_per_axis[atom_index[:,0]]

    u2 = forces_resultant_per_axis[atom_index[:,1]]

    v_X = main_X_unit_vec.reshape(n_combinations,3)

    proj_X_1 = get_projection(u1,v_X)

    proj_X_2 = get_projection(u2,v_X)

    proj_X = proj_X_1 + proj_X_2

    angle_X = np.sum(proj_X*v_X, axis=1)/(np.linalg.norm(proj_X, axis=1))

    X_resultant = (np.linalg.norm(proj_X, axis = 1)*angle_X).reshape(-1,1)

    X_resultant = np.nan_to_num(X_resultant)

    

    v_Y = main_Y_unit_vec.reshape(n_combinations,3)

    proj_Y_1 = get_projection(u1,v_Y)

    proj_Y_2 = get_projection(u2,v_Y)

    proj_Y_1_norm = np.linalg.norm(proj_Y_1, axis=1)

    proj_Y_2_norm = np.linalg.norm(proj_Y_2, axis=1)

    angle_Y = np.sum(proj_Y_1*proj_Y_2, axis=1)/(proj_Y_1_norm*proj_Y_2_norm)

    angle_Y = np.nan_to_num(angle_Y)

    force_Y_momentum = (proj_Y_1_norm - proj_Y_2_norm*angle_Y)

    

    v_Z = main_Z_unit_vec.reshape(n_combinations,3)

    proj_Z_1 = get_projection(u1,v_Z)

    proj_Z_2 = get_projection(u2,v_Z)

    proj_Z_1_norm = np.linalg.norm(proj_Z_1, axis=1)

    proj_Z_2_norm = np.linalg.norm(proj_Z_2, axis=1)

    angle_Z = np.sum(proj_Z_1*proj_Z_2, axis=1)/(proj_Z_1_norm*proj_Z_2_norm)

    angle_Z = np.nan_to_num(angle_Z)

    force_Z_momentum = (proj_Z_1_norm - proj_Z_2_norm*angle_Z)

    

    momentum_Y = (force_Y_momentum*dist_mat[atom_index[:,1], atom_index[:,0]]).reshape(n_combinations, 1)

    momentum_Z = (force_Z_momentum*dist_mat[atom_index[:,1], atom_index[:,0]]).reshape(n_combinations, 1)

    momentum = np.sqrt(momentum_Y**2 + momentum_Z**2)

    

    Y_Z_resultant = np.linalg.norm(proj_Y_1 + proj_Y_2 + proj_Z_1 + proj_Z_2, axis=1).reshape(n_combinations, 1)

    Y_Z_resultant_0 = np.linalg.norm(proj_Y_1 + proj_Z_1, axis=1).reshape(n_combinations, 1)

    Y_Z_resultant_1 = np.linalg.norm(proj_Y_2 + proj_Z_2, axis=1).reshape(n_combinations, 1)

    Y_resultant =  np.linalg.norm(proj_Y_1 + proj_Y_2, axis=1).reshape(n_combinations, 1)

    Z_resultant =  np.linalg.norm(proj_Z_1 + proj_Z_2, axis=1).reshape(n_combinations, 1)

    proportion_between_X_and_Y_Z = 1 - Y_Z_resultant/(Y_Z_resultant+np.abs(X_resultant))

    proportion_between_X_and_Y_Z = np.nan_to_num(proportion_between_X_and_Y_Z)

    

    return X_resultant, Y_Z_resultant, proportion_between_X_and_Y_Z, momentum, Y_Z_resultant_0, Y_Z_resultant_1, Y_resultant, Z_resultant



def f_cutoff(data, Rc):

    

    fc = 0.5*(np.cos(np.pi*data/Rc) + 1)

    fc[data > Rc] = 0

    

    return fc



def fill_array(array, n_combinations, expected_length, fill):

    

    if expected_length > array.shape[1]:

        array = np.concatenate([array, np.zeros((n_combinations,expected_length - array.shape[1])) + fill], axis=1)

    

    return array



def get_G2(Rij, eta, Rs, Rc):

     return (np.exp(-eta*(Rij-Rs)**2) * f_cutoff(Rij, Rc)).sum(axis=-1)
#path to where the data competition is

original_data_folder = '../input/champs-scalar-coupling'



#'train' or 'test' depending on the dataset that you want to build features for

df_type = 'train' #or 'test'



#verbose

verbose = True



#number o nearest atoms to consider when calculating features.

#More atom = more features

#After the 8th atom, it starts to become irrelevant to the model

max_n_nearest = 11



#it will be a lot faster to generate features for all mol_types at once. 

#Just do one at a time if you have less than 16MB of RAM

mol_types = ['1JHN'] 



#Number of CPU cores for parallel processing

number_of_cores = 10
################

#Get all the data

################



df = pd.read_csv(f'{original_data_folder}/{df_type}.csv')

df = df[df['type'].isin(mol_types)] #selecte only the rows with desired coupling types

df = reduce_mem_usage(df, verbose=True)



unique_mol = df['molecule_name'].unique()



def get_data(data_path, verbose):

    

    data = pd.read_csv(data_path)

    data = data.loc[data['molecule_name'].isin(unique_mol)]

    data = reduce_mem_usage(data, verbose=True)

    

    return data



df_bonds = get_data(data_path = f'../input/predicting-molecular-properties-bonds/{df_type}_bonds.csv', verbose = verbose)

df_structures = get_data(data_path = '../input/champs-scalar-coupling/structures.csv', verbose = verbose)

df_structures_eigen = get_data(data_path = '../input/qm7-coulomb-matrix-eigenvalue-features/struct_eigen.csv', verbose = verbose)

if df_type == 'test':

    df_mulliken = get_data(data_path = '../input/predicting-mulliken-charges-with-acsf-descriptors/mulliken_charges_test_set.csv', verbose = verbose)

else:

    df_mulliken = get_data(data_path = '../input/champs-scalar-coupling/mulliken_charges.csv', verbose = verbose)

          



################

#Get indexes per molecule, so we can work with each molecule at a time.

#start_indexes = index where the molecule start

#end_indexes = index where the molecule end

################



xyz = df_structures[['x','y','z']].values

atom_indexes = df[['atom_index_0', 'atom_index_1']].values



struct_end_indexes = df_structures.groupby('molecule_name').size().cumsum()

df_end_indexes = df[['molecule_name', 'atom_index_0', 'atom_index_1']].groupby('molecule_name').size().cumsum()

df_bonds_end_indexes = df_bonds[['molecule_name', 'atom_index_0', 'atom_index_1']].groupby('molecule_name').size().cumsum()



struct_start_indexes = np.zeros(len(struct_end_indexes) + 1, 'int')

struct_start_indexes[1:] = struct_end_indexes

struct_start_indexes[0] = 0



df_start_indexes = np.zeros(len(df_end_indexes) + 1, 'int')

df_start_indexes[1:] = df_end_indexes

df_start_indexes[0] = 0



df_bonds_start_indexes = np.zeros(len(df_bonds_end_indexes) + 1, 'int')

df_bonds_start_indexes[1:] = df_bonds_end_indexes

df_bonds_start_indexes[0] = 0



################

#Data about atoms that we use in our calculations. The variable radius depends on the bonds, 

#while atom_radius is an acceptable average.

#EN is the eletronegativity

#Enconde is the atomic number

################



atom_variable_radius = {'H': np.array([0.32, np.NaN, np.NaN]),

                'C': np.array([.75, .67, .60]),

                'O': np.array([.63, .57, .54]),

                'N': np.array([.71, .60, .54]),

                'F': np.array([.64, .59, .53])}



atoms_radius = {'H':0.38, 'C':0.77, 'N':0.75,'O':0.73, 'F':0.71}

atoms_EN = {'H':2.2, 'C':2.55, 'N':3.04,'O':3.44, 'F':3.98}

atoms_encoding = {0:0, '0':0, 'H':1, 'C':6, 'N':7,'O':8, 'F':9}
def get_mol_feats(molecule_id):

    

    ##########

    #get some initial data

    #########

    

    dist_mat, vec_mat, atom_index, locs = get_fast_dist_matrix(xyz, df_start_indexes, struct_start_indexes, molecule_id)  

    

    #Uncomment if you want to augument the data by changing atom_0 and atom_1 positions

    #atom_index = np.array([[i,j] for j, i in atom_index])

    

    #n_nearest is the value of how many atoms you wanna analyze and it is defined by:

    #mininum value between the chosen max_n_nearest and how many atoms are in this molecule

    n_nearest = min(dist_mat.shape[1], max_n_nearest)

    

    #n_combinations is how many couplings are being calculated in this molecule

    n_combinations = len(atom_index)

    

    #Number of atoms in the molecule

    n_atoms = len(dist_mat)

    

    #unit vec between atoms

    unit_vec = np.nan_to_num(vec_mat/np.linalg.norm(vec_mat, axis=2).reshape(-1, n_atoms, 1))

    

    #n_nearest atoms to atom_0 and atom_1 ordered by their distance to the center of atom_0 and atom_1

    ordered_smallest_dist_indexes = np.argpartition(dist_mat, range(n_nearest))[:,1:n_nearest+1]

    

    

    struct_start_index, struct_end_index = struct_start_indexes[molecule_id], struct_start_indexes[molecule_id+1]

    df_start_index, df_end_index = df_start_indexes[molecule_id], df_start_indexes[molecule_id+1]

    df_bonds_start_index, df_bonds_end_index = df_bonds_start_indexes[molecule_id], df_bonds_start_indexes[molecule_id+1]

    

    #taken from the structures_eigen df

    eigen_cols = [ 'connectedness', 'coulomb_mean', 'eigv_max',

                   'eigv_min', 'fiedler_eig', 'sv_0', 'sv_1', 'sv_2',

                   'sv_3', 'sv_4', 'sv_min']

    

    eigen_info = df_structures_eigen.iloc[struct_start_index:struct_end_index]

    eigen_info = eigen_info[eigen_cols].to_numpy()

    

    atom_names = np.array(df_structures[struct_start_index:struct_end_index]['atom'])

    mol_name = df_structures.iloc[struct_start_index, 0]

    atom_code = np.array(itemgetter(*atom_names)(atoms_encoding))

    

    ##########

    #get distances from all n_nearest atoms to atom_0, atom_1 and other 2 closest atoms

    #########

    

    locs0 = locs[atom_index[:,0]]

    locs1 = locs[atom_index[:,1]]

    

    mid_point = ((locs0 + locs1)/2).reshape(-1, 1, 3)

    mid_point_distances = locs-mid_point

    mid_point_distances = np.sqrt(np.sum(mid_point_distances**2, axis=-1))

    ordered_smallest_dist_indexes = np.argpartition(mid_point_distances, range(n_nearest))[:,:n_nearest]

    

    closest_atoms = []

    for i, row in enumerate(ordered_smallest_dist_indexes):

        atoms_0_and_1_index = atom_index[i,:]

        nearest_ordered_atoms_excluding_0_and_1 = row[np.logical_and(row != atom_index[i,0], row != atom_index[i,1])][:n_nearest-2]

        closest_atoms.append(np.concatenate([atoms_0_and_1_index, nearest_ordered_atoms_excluding_0_and_1]))



    closest_atoms = np.array(closest_atoms)

    

    n_combs_atom_pairs = []

    for c in range(n_combinations):

        atom_pairs = []

        for i in range(1, len(closest_atoms[0])):

            for vi in range(min(4, i)):

                atom_pairs.append([closest_atoms[c][i], closest_atoms[c][vi]])

        

        n_combs_atom_pairs.append(atom_pairs)

        

    n_combs_atom_pairs = np.array(n_combs_atom_pairs)     

            

    distances_atom_pairs = dist_mat[n_combs_atom_pairs[:,:,0], n_combs_atom_pairs[:,:,1]]       

    

    expected_number_of_pair_combinations = 6 + 4*(max_n_nearest-4)

    distances_atom_pairs = fill_array(distances_atom_pairs, n_combinations, expected_length = expected_number_of_pair_combinations, fill = 0)

    

    ##########

    #Calculate G4 angle descriptors

    #########

    

    

    all_Rij_Rik =  np.array(list(permutations(list(range(n_atoms)), 3)))

    

    

    Rij = dist_mat[all_Rij_Rik[:,0], all_Rij_Rik[:,1]].reshape(n_atoms, -1)

    Rik = dist_mat[all_Rij_Rik[:,0], all_Rij_Rik[:,2]].reshape(n_atoms, -1)

    Rjk = dist_mat[all_Rij_Rik[:,2], all_Rij_Rik[:,1]].reshape(n_atoms, -1)

    all_ij_unit_vec = unit_vec[all_Rij_Rik[:,1], all_Rij_Rik[:,0]].reshape(n_atoms, -1, 3)

    all_ik_unit_vec = unit_vec[all_Rij_Rik[:,2], all_Rij_Rik[:,0]].reshape(n_atoms, -1, 3)

    cosijk = np.sum(all_ij_unit_vec * all_ik_unit_vec, axis=-1)

    



    G4 = []

    Rc = 10

    

    for eta, ksi, lamb, Rs in [[0.01, 4,  1, 2], [1, 4,  -1, 6], [0.1, 4,  -1, 6], [0.5, 2,  1, 6]]:

        G4_values = (1+lamb*cosijk)**ksi

        G4_values *= np.exp(-eta*(Rij**2 + Rik**2 + Rjk**2))

        G4_values = G4_values * f_cutoff(Rij, Rc) * f_cutoff(Rjk, Rc) * f_cutoff(Rik, Rc)

        G4_values = (2**(1-ksi))*np.sum(G4_values, axis=-1)

        G4.append(G4_values)

        

    G4_selected = []



    for row_G4 in G4:

        G4_selected.append(fill_array(row_G4[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0))

    

    G4_selected = np.concatenate(G4_selected, axis=-1)



    ##########

    #get some bond based properties, like std, mean, kurtosis, etc

    #########

    

    bond_info = df_bonds.iloc[df_bonds_start_index:df_bonds_end_index, [1, 2, 3, 4]].to_numpy() # 'atom_index_0', 'atom_index_1', 'n_bonds', 'L2dist'

    

    bonds_dist_mean = np.zeros(n_atoms)

    bonds_dist_std = np.zeros(n_atoms)

    bonds_dist_kurt = np.zeros(n_atoms)

    bonds_dist_skew = np.zeros(n_atoms)

    atom_cov_radius = np.zeros(n_atoms)

    n_bonds = np.zeros(n_atoms)

    

    bond_matrix = np.zeros((n_atoms, n_atoms))

    

    column_atoms_0 = bond_info[:, 0].astype('int')

    column_atoms_1 = bond_info[:, 1].astype('int')

    

    bond_matrix[column_atoms_0, column_atoms_1] = 1

    bond_matrix[column_atoms_1, column_atoms_0] = 1

    is_atom_bond = bond_matrix[n_combs_atom_pairs[:,:,0], n_combs_atom_pairs[:,:,1]]   

    

    for i in range(n_atoms):

        atom_name = atom_names[i]

        atom_column_0 = bond_info[:, 0] == i

        atom_column_1 = bond_info[:, 1] == i

        

        bond_info_atom_i = bond_info[np.logical_or(atom_column_0, atom_column_1)]

        bonds_lengths = bond_info_atom_i[:,3]

        bonds_dist_mean[i] = np.mean(bonds_lengths)

        bonds_dist_std[i] = np.std(bonds_lengths)

        bonds_dist_kurt[i] = kurtosis(bonds_lengths)

        bonds_dist_skew[i] = skew(bonds_lengths)

        

        highest_order_bond = max(bond_info_atom_i[:,2])

        n_bonds[i] = int(np.sum(atom_column_0)+ np.sum(atom_column_1))

        atom_cov_radius[i] = atom_variable_radius[atom_name][int(highest_order_bond-1)]

        



    bonds_dist_mean = fill_array(bonds_dist_mean[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0)    

    bonds_dist_std = fill_array(bonds_dist_std[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0)    

    bonds_dist_kurt = fill_array(bonds_dist_kurt[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0)    

    bonds_dist_skew = fill_array(bonds_dist_skew[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0)    

    n_bonds = fill_array(n_bonds[closest_atoms],n_combinations, expected_length = max_n_nearest, fill = 0)    

    

    expected_number_of_pair_combinations = 6 + 4*(max_n_nearest-4)

    is_atom_bond = fill_array(is_atom_bond, n_combinations, expected_length = expected_number_of_pair_combinations, fill = 0)

    

    ##########

    #Calculate dihedral angles and graph path distances between atoms

    #########

    

    atom_radius = np.array(itemgetter(*list(range(n_atoms)))(atom_cov_radius))

    atom_radius = np.tile(atom_radius, n_atoms).reshape(dist_mat.shape)

    atom_radius_mult = atom_radius*atom_radius.T

    atom_radius_sum = atom_radius + atom_radius.T

    distance_minus_radius = dist_mat-atom_radius_sum

    distance_minus_radius[distance_minus_radius<0] = 0



    #GET DIHEDRALS AND PATHS

    graph_matrix = np.zeros(distance_minus_radius.shape)

    bonds_index = distance_minus_radius<=0.3

    graph_matrix[bonds_index] = dist_mat[bonds_index]

    

    D, Pr = shortest_path(graph_matrix, directed=False, method='FW', return_predecessors=True)

    

    path_length_matrix, path_matrix, path_distance_matrix = get_path_matrix(Pr, n_atoms, dist_mat)

    

    dihedral_indices = np.where(path_length_matrix==4)

    dihedral_matrix = np.zeros((n_atoms, n_atoms)) - 11

    dihedral_paths = []

    dihedral = []

    if len(dihedral_indices[0]) > 0:

        dihedral_paths = np.concatenate(path_matrix[dihedral_indices[0], dihedral_indices[1]]).reshape(-1, 4)

        dihedral = get_dihedrals(dihedral_paths, vec_mat)

        

        #We use -11 to set apart atoms without dihedrals

        #We use -10 to se apart cos(dihedrals) of atoms without dihedrals

        dihedral_matrix[dihedral_indices[0], dihedral_indices[1]] = dihedral

        

        error = 0.001 #is used to compare floats

        dihedral_main = dihedral_matrix[atom_index[:,0], atom_index[:,1]]

        cos_dihedral_main = np.cos(dihedral_main)

        cos_dihedral_main[(np.abs(cos_dihedral_main - np.cos(-11))) < error] = -10

        

        cos_2_times_dihedral_main = np.cos(2*dihedral_main)

        cos_2_times_dihedral_main[(np.abs(cos_2_times_dihedral_main - np.cos(-2*11))) < error] = -10

        

    else:

        dihedral_main = np.zeros((n_combinations, 1)) - 11

        cos_dihedral_main = np.zeros((n_combinations, 1)) - 10

        cos_2_times_dihedral_main = np.zeros((n_combinations, 1)) - 10

    

    cos_dihedral_main = cos_dihedral_main.reshape(n_combinations,1)

    cos_2_times_dihedral_main = cos_2_times_dihedral_main.reshape(n_combinations,1)

  

    

    ##########

    #Calculate forces like coulomb, yukawa, harmonic, van der waals, and others

    #########

    



    atoms_mulliken_charges = np.array(df_mulliken[struct_start_index:struct_end_index]['mulliken_charge'])

    

    atoms_mulliken = np.tile(atoms_mulliken_charges, n_atoms).reshape(dist_mat.shape)

    atoms_mulliken_mult = atoms_mulliken*atoms_mulliken.T

    

    coulomb = atoms_mulliken_mult/dist_mat

    coulomb[np.eye(n_atoms)==1] = 0

    

    yukawa = np.exp(-dist_mat)/dist_mat

    yukawa[np.eye(n_atoms)==1] = 0

    

    harmonic_distance = atom_radius_mult/atom_radius_sum

    harmonic_distance[np.eye(n_atoms)==1] = 0

    

    vander_numerator = harmonic_distance

    vander_denominator = 6*(distance_minus_radius)**2

    vander_denominator[vander_denominator < 1] = 1 #avoid division by 0

    vander = vander_numerator/vander_denominator

    vander[np.eye(n_atoms)==1] = 0



    inv_dist_1 = 1/dist_mat

    inv_dist_1[np.eye(n_atoms)==1] = 0

    

    inv_dist_2 = 1/dist_mat**2

    inv_dist_2[np.eye(n_atoms)==1] = 0

    

    inv_dist_3 = 1/dist_mat**3

    inv_dist_3[np.eye(n_atoms)==1] = 0

    

    atom_EN = np.array(itemgetter(*atom_names)(atoms_EN))

    atom_EN = np.tile(atom_EN, n_atoms).reshape(dist_mat.shape)

    atom_EN_sum = atom_EN + atom_EN.T

    

    inv_EN = 1/(dist_mat*0.5*atom_EN_sum)**2

    inv_EN[np.eye(n_atoms)==1] = 0

    

    yukawa_resultant = np.linalg.norm(np.sum(unit_vec*yukawa.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    coulomb_resultant = np.linalg.norm(np.sum(unit_vec*coulomb.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    harmonic_resultant = np.linalg.norm(np.sum(unit_vec*harmonic_distance.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    vander_resultant = np.linalg.norm(np.sum(unit_vec*vander.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    inv_EN_resultant = np.linalg.norm(np.sum(unit_vec*inv_EN.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    inv_dist_2_resultant = np.linalg.norm(np.sum(unit_vec*inv_dist_2.reshape(n_atoms,n_atoms,1), axis=1), axis=-1)

    

    

    yukawa_resultant = fill_array(yukawa_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    coulomb_resultant = fill_array(coulomb_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    harmonic_resultant = fill_array(harmonic_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    vander_resultant = fill_array(vander_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    inv_EN_resultant = fill_array(inv_EN_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    inv_dist_2_resultant = fill_array(inv_dist_2_resultant[closest_atoms], n_combinations, expected_length = max_n_nearest, fill = 0)

    

    ####################

        

    #Every pair of atom_0 and atom_1 form a plane with the center(0, 0, 0) of the atom.

    #To get the plane we need cross(a,b) resulting in coeficients c1, c2, c3 and a point, in this case, we will use (0,0,0)

    #The final place equation is c1(x -0) + c2(y - 0) + c3(z -0) = 0, which is just the cross product.

    planes = np.cross(locs[atom_index[:,0]], locs[atom_index[:,1]])

    

    #lets get the unit orthogonal vec of this place by diving by the norm and reshape it for the next step

    main_Z_unit_vec = ( planes/np.linalg.norm(planes,axis=1).reshape(-1,1) ).reshape(-1, 1, 3)

    main_X_unit_vec = unit_vec[atom_index[:,1], atom_index[:,0]].reshape(-1, 1, 3)

    main_Y_unit_vec = np.cross(main_Z_unit_vec, main_X_unit_vec)



    ####################    

    

    #calculate resultant forces

    yukawa_parallel, yukawa_orthogonal, yukawa_proportion_between_X_and_Y_Z, yukawa_momentum, yukawa_Y_Z_resultant_0, yukawa_Y_Z_resultant_1, yukawa_Y_resultant, yukawa_Z_resultant = calculate_force_resultant(unit_vec, yukawa, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    inv_EN_parallel, inv_EN_orthogonal, inv_EN_proportion_between_X_and_Y_Z, inv_EN_momentum, inv_EN_Y_Z_resultant_0, inv_EN_Y_Z_resultant_1, inv_EN_Y_resultant, inv_EN_Z_resultant = calculate_force_resultant(unit_vec, inv_EN, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    coulomb_parallel, coulomb_orthogonal, coulomb_proportion_between_X_and_Y_Z, coulomb_momentum, coulomb_Y_Z_resultant_0, coulomb_Y_Z_resultant_1, coulomb_Y_resultant, coulomb_Z_resultant = calculate_force_resultant(unit_vec, coulomb, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    harmonic_parallel, harmonic_orthogonal, harmonic_proportion_between_X_and_Y_Z, harmonic_momentum, harmonic_Y_Z_resultant_0, harmonic_Y_Z_resultant_1, harmonic_Y_resultant, harmonic_Z_resultant  = calculate_force_resultant(unit_vec, harmonic_distance, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    vander_parallel, vander_orthogonal, vander_proportion_between_X_and_Y_Z, vander_momentum, vander_Y_Z_resultant_0, vander_Y_Z_resultant_1, vander_Y_resultant, vander_Z_resultant  = calculate_force_resultant(unit_vec, vander, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    dist_1_parallel, dist_1_orthogonal, dist_1_proportion_between_X_and_Y_Z, dist_1_momentum, dist_1_Y_Z_resultant_0, dist_1_Y_Z_resultant_1, dist_1_Y_resultant, dist_1_Z_resultant  = calculate_force_resultant(unit_vec, dist_mat, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    inv_dist_1_parallel, inv_dist_1_orthogonal, inv_dist_1_proportion_between_X_and_Y_Z, inv_dist_1_momentum, inv_dist_1_Y_Z_resultant_0, inv_dist_1_Y_Z_resultant_1, inv_dist_1_Y_resultant, inv_dist_1_Z_resultant  = calculate_force_resultant(unit_vec, inv_dist_1, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    inv_dist_2_parallel, inv_dist_2_orthogonal, inv_dist_2_proportion_between_X_and_Y_Z, inv_dist_2_momentum, inv_dist_2_Y_Z_resultant_0, inv_dist_2_Y_Z_resultant_1, inv_dist_2_Y_resultant, inv_dist_2_Z_resultant  = calculate_force_resultant(unit_vec, inv_dist_2, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

    inv_dist_3_parallel, inv_dist_3_orthogonal, inv_dist_3_proportion_between_X_and_Y_Z, inv_dist_3_momentum, inv_dist_3_Y_Z_resultant_0, inv_dist_3_Y_Z_resultant_1, inv_dist_3_Y_resultant, inv_dist_3_Z_resultant  = calculate_force_resultant(unit_vec, inv_dist_3, n_atoms, atom_index, n_combinations, main_X_unit_vec, main_Y_unit_vec, main_Z_unit_vec, dist_mat)

      

    #Some final feactures based on angle, mulliken charge, and if there is a bond between atoms



    unit_nearest_0 = unit_vec[closest_atoms[:, 2:],atom_index[:,0].reshape(n_combinations, 1)]

    unit_nearest_1 = unit_vec[closest_atoms[:, 2:],atom_index[:,1].reshape(n_combinations, 1)]

    angle_between_0_1_nearest_to_center = np.sum(unit_nearest_0*unit_nearest_1, axis=2)

    

    angle_between_0_1_nearest_to_center = fill_array(angle_between_0_1_nearest_to_center, n_combinations, expected_length = max_n_nearest, fill = 0)

    

    mulliken_charges = atoms_mulliken_charges[closest_atoms].reshape(n_combinations, -1)

    mulliken_charges = fill_array(mulliken_charges, n_combinations, expected_length = max_n_nearest, fill = 0)

   

    atom_code = atom_code[closest_atoms].reshape(n_combinations, -1)

    atom_code = fill_array(atom_code, n_combinations, expected_length = max_n_nearest, fill = 0)

    

    

    ##########

    #Put it all together:

    #########

    

    one_column_data= np.concatenate([np.tile(np.array([sum(atom_names == 'C')]), n_combinations).reshape(n_combinations,1),

                                     np.tile(np.array([sum(atom_names == 'H')]), n_combinations).reshape(n_combinations,1),

                                     np.tile(np.array([sum(atom_names == 'F')]), n_combinations).reshape(n_combinations,1),

                                     np.tile(np.array([sum(atom_names == 'N')]), n_combinations).reshape(n_combinations,1),

                                     np.tile(np.array([sum(atom_names == 'O')]), n_combinations).reshape(n_combinations,1),

                                     

                                     cos_dihedral_main ,

                                     cos_2_times_dihedral_main ,



                                       yukawa_Y_Z_resultant_0, 

                                       yukawa_Y_Z_resultant_1 ,

                                       yukawa_Y_resultant ,

                                       yukawa_Z_resultant ,

                                       yukawa_momentum,

                                       

                                       inv_EN_Y_Z_resultant_0, 

                                       inv_EN_Y_Z_resultant_1 ,

                                       inv_EN_Y_resultant ,

                                       inv_EN_Z_resultant ,

                                       inv_EN_momentum,

                                       

                                       vander_Y_Z_resultant_0, 

                                       vander_Y_Z_resultant_1 ,

                                       vander_Y_resultant ,

                                       vander_Z_resultant ,

                                       vander_momentum,

                                       

                                       coulomb_Y_Z_resultant_0, 

                                       coulomb_Y_Z_resultant_1 ,

                                       coulomb_Y_resultant ,

                                       coulomb_Z_resultant ,

                                       coulomb_momentum,

                                       

                                       yukawa_parallel ,

                                       coulomb_parallel ,

                                       harmonic_parallel ,

                                       vander_parallel ,

                                       dist_1_parallel ,

                                       inv_dist_1_parallel ,

                                       inv_dist_2_parallel ,

                                       inv_dist_3_parallel ,

                                       

                                       yukawa_orthogonal ,

                                       coulomb_orthogonal ,

                                       harmonic_orthogonal, 

                                       vander_orthogonal ,

                                       dist_1_orthogonal,

                                       inv_dist_1_orthogonal,

                                       inv_dist_2_orthogonal,

                                       inv_dist_3_orthogonal,

                                       

                                       yukawa_proportion_between_X_and_Y_Z, 

                                       coulomb_proportion_between_X_and_Y_Z, 

                                       harmonic_proportion_between_X_and_Y_Z, 

                                       vander_proportion_between_X_and_Y_Z ,

                                       dist_1_proportion_between_X_and_Y_Z,

                                       inv_dist_1_proportion_between_X_and_Y_Z,

                                       inv_dist_2_proportion_between_X_and_Y_Z,

                                       inv_dist_3_proportion_between_X_and_Y_Z,

                                       

                                       bonds_dist_kurt[:,0].reshape(n_combinations,1),

                                       bonds_dist_kurt[:,1].reshape(n_combinations,1),

                                       bonds_dist_kurt[:,2].reshape(n_combinations,1),

                                       bonds_dist_skew[:,0].reshape(n_combinations,1),

                                       bonds_dist_skew[:,1].reshape(n_combinations,1),

                                       bonds_dist_skew[:,2].reshape(n_combinations,1)], axis=1) 

    

    multiple_columns_data = [ distances_atom_pairs,

                             is_atom_bond,

                             n_bonds,

                             G4_selected,

                             bonds_dist_mean,

                             bonds_dist_std,

                             yukawa_resultant,

                             coulomb_resultant,

                             inv_EN_resultant,

                             angle_between_0_1_nearest_to_center,

                             mulliken_charges,

                             atom_code,

                             eigen_info[closest_atoms[:,0]],

                             eigen_info[closest_atoms[:,1]],

                             eigen_info[closest_atoms[:,2]]]

    

    results = np.concatenate([one_column_data, np.concatenate(multiple_columns_data, axis=1)], axis=1).astype('float32')

    

    return results
original_max_n_nearest = copy.copy(max_n_nearest)



molecule_id = 0

max_n_nearest = 11

features = get_mol_feats(molecule_id)

print(features.shape)
pd.DataFrame(features)
max_n_nearest = 8

molecule_id = 0

features = get_mol_feats(molecule_id)

print(features.shape)
max_n_nearest = original_max_n_nearest



import multiprocessing

import traceback

p = multiprocessing.Pool(number_of_cores)



try:

    t1 = time.time()

    result = p.map(get_mol_feats, list(range(df_structures.molecule_name.nunique())))

    t2 = time.time()

except Exception as e:

    print('Caught exception in worker thread (x = d):')

    traceback.print_exc()

    print()

    raise e

    

p.terminate()

p.join



print(f'Calculating features took {(t2 - t1)/60:0.2f} minutes')
del df_structures_eigen, df_bonds_end_indexes, df_bonds_start_indexes, xyz, unique_mol, df_start_indexes, df_end_indexes, df_bonds, df, df_structures, df_mulliken, atom_indexes



import gc

gc.collect()



#Let's create the names of the columns of our soon to be DataFrame



#column names for distance atom pairs

distances_atom_pairs = []

is_atom_bond = []

for i in range(1, max_n_nearest):

    for vi in range(min(4, i)):

        distances_atom_pairs.append(f'{i}_{vi}_distances_atom_pairs')

        is_atom_bond.append(f'{i}_{vi}_atom_bond')





eigen_0 = [col + '_0' for col in [ 'connectedness', 'coulomb_mean', 'eigv_max', 'eigv_min', 'fiedler_eig', 'sv_0', 'sv_1', 'sv_2','sv_3', 'sv_4', 'sv_min']]

eigen_1 = [col + '_1' for col in [ 'connectedness', 'coulomb_mean', 'eigv_max', 'eigv_min', 'fiedler_eig', 'sv_0', 'sv_1', 'sv_2','sv_3', 'sv_4', 'sv_min']]

eigen_2 = [col + '_2' for col in [ 'connectedness', 'coulomb_mean', 'eigv_max', 'eigv_min', 'fiedler_eig', 'sv_0', 'sv_1', 'sv_2','sv_3', 'sv_4', 'sv_min']]

n_neighbours = [f'{i}_n_neighbours' for i in range(1,max_n_nearest + 1)]

G4_selected_1 = [f'{i}_G4_1_selected' for i in range(1,max_n_nearest + 1)]

G4_selected_2 = [f'{i}_G4_2_selected' for i in range(1,max_n_nearest + 1)]

G4_selected_3 = [f'{i}_G4_3_selected' for i in range(1,max_n_nearest + 1)]

G4_selected_4 = [f'{i}_G4_5_selected' for i in range(1,max_n_nearest + 1)]

G4_selected_5 = [f'{i}_G4_6_selected' for i in range(1,max_n_nearest + 1)]

G4_selected_6 = [f'{i}_G4_7_selected' for i in range(1,max_n_nearest + 1)]

bonds_dist_mean = [f'{i}_bonds_dist_mean' for i in range(1,max_n_nearest + 1)]

bonds_dist_std = [f'{i}_bonds_dist_std' for i in range(1,max_n_nearest + 1)]

yukawa_resultant = [f'{i}_yukawa_resultant' for i in range(1,max_n_nearest + 1)]

coulomb_resultant = [f'{i}_coulomb_resultant' for i in range(1,max_n_nearest + 1)]

harmonic_resultant = [f'{i}_harmonic_resultant' for i in range(1,max_n_nearest + 1)]

vander_resultant = [f'{i}_vander_resultant' for i in range(1,max_n_nearest + 1)]

inv_EN_resultant = [f'{i}_inv_EN_resultant' for i in range(1,max_n_nearest + 1)]

inv_dist_2_resultant = [f'{i}_inv_dist_2_resultant' for i in range(1,max_n_nearest + 1)]

angle_between_0_1_nearest_to_center = [f'{i}_angle_between_0_1_nearest_to_center' for i in range(1,max_n_nearest + 1)]

mulliken_charges = [f'{i}_mulliken_charges' for i in range(1,max_n_nearest + 1)]

atom_code = [f'{i}_atom_code' for i in range(1,max_n_nearest + 1)]

             

single_columns = ['N_C',

                  'N_H',

                  'N_F',

                  'N_N',

                  'N_O',

                  'cos_dihedral_main',

                  'cos_2_times_dihedral_main',

                  

                   'yukawa_Y_Z_resultant_0', 

                   'yukawa_Y_Z_resultant_1' ,

                   'yukawa_Y_resultant',

                   'yukawa_Z_resultant' ,

                   'yukawa_momentum',

                   

                   'inv_EN_Y_Z_resultant_0', 

                   'inv_EN_Y_Z_resultant_1' ,

                   'inv_EN_Y_resultant',

                   'inv_EN_Z_resultant' ,

                   'inv_EN_momentum',

                   

                   'vander_Y_Z_resultant_0', 

                   'vander_Y_Z_resultant_1' ,

                   'vander_Y_resultant',

                   'vander_Z_resultant' ,

                   'vander_momentum',

                   

                   'coulomb_Y_Z_resultant_0', 

                   'coulomb_Y_Z_resultant_1' ,

                   'coulomb_Y_resultant',

                   'coulomb_Z_resultant' ,

                   'coulomb_momentum',

                   

                   'yukawa_parallel',

                   'coulomb_parallel',

                   'harmonic_parallel',

                   'vander_parallel',

                   'dist_1_parallel',

                   'inv_dist_1_parallel',

                   'inv_dist_2_parallel',

                   'inv_dist_3_parallel',

                   

                   'yukawa_orthogonal',

                   'coulomb_orthogonal',

                   'harmonic_orthogonal', 

                   'vander_orthogonal',

                   'dist_1_orthogonal',

                   'inv_dist_1_orthogonal',

                   'inv_dist_2_orthogonal',

                   'inv_dist_3_orthogonal',

                   

                   'yukawa_proportion_between_X_and_Y_Z', 

                   'coulomb_proportion_between_X_and_Y_Z', 

                   'harmonic_proportion_between_X_and_Y_Z', 

                   'vander_proportion_between_X_and_Y_Z' ,

                   'dist_1_proportion_between_X_and_Y_Z',

                   'inv_dist_1_proportion_between_X_and_Y_Z',

                   'inv_dist_2_proportion_between_X_and_Y_Z',

                   'inv_dist_3_proportion_between_X_and_Y_Z',

                   

                   '0_bonds_dist_kurt',

                   '1_bonds_dist_kurt',

                   '2_bonds_dist_kurt',

                   '0_bonds_dist_skew',

                   '1_bonds_dist_skew',

                   '2_bonds_dist_skew']

                             





mul_columns = [  distances_atom_pairs,

                   n_neighbours,

                 is_atom_bond,

                 G4_selected_1,

                 G4_selected_2,

                 G4_selected_3,

                 G4_selected_4,

                 bonds_dist_mean,

                 bonds_dist_std,

                 yukawa_resultant,

                 coulomb_resultant,

                 inv_EN_resultant,

                 angle_between_0_1_nearest_to_center,

                 mulliken_charges,

                 atom_code,

                 eigen_0,

                 eigen_1,

                 eigen_2]





all_col = np.sum([single_columns, np.sum(mul_columns)])





dtype_dict = dict()

for col in all_col:

    

    dtype_dict[col] = 'float32'

    

    if 'connectedness' in col:

        dtype_dict[col] = 'int8'

    if 'atom_code' in col:

        dtype_dict[col] = 'int8'

    if col.startswith('N_'):

        dtype_dict[col] = 'int8'
data_concat = np.concatenate(result, axis=0)

del result

gc.collect()



features_df = pd.DataFrame(data_concat, columns = dtype_dict).astype(dtype_dict)

del data_concat

gc.collect()



df = pd.read_csv(f'{original_data_folder}/{df_type}.csv')

df = df[df['type'].isin(mol_types)] #selecte only the rows with desired coupling types

df = reduce_mem_usage(df, verbose=True)

df.reset_index(drop = True, inplace=True)



df = features_df.join(df)

del features_df

gc.collect()



for t in mol_types:

    df_only_selected_type = df[df['type'] == t]

    df_only_selected_type.to_csv(f'{df_type}_{t}.csv')

    del df_only_selected_type
df.head(50)
print(list(df.columns))
import matplotlib.pyplot as plt

%matplotlib inline



df_1JHN = df[df['type'] == '1JHN']

selected_cols = ['yukawa_Y_Z_resultant_0', 'yukawa_Y_Z_resultant_1', 'yukawa_Y_resultant',

                 'yukawa_Z_resultant', 'yukawa_momentum', 'inv_EN_Y_Z_resultant_0', 'inv_EN_Y_Z_resultant_1',

                 'inv_EN_Y_resultant', 'inv_EN_Z_resultant', 'inv_EN_momentum', 'vander_Y_Z_resultant_0',

                 'vander_Y_Z_resultant_1', 'vander_Y_resultant', 'vander_Z_resultant', 'vander_momentum',

                 'coulomb_Y_Z_resultant_0', 'coulomb_Y_Z_resultant_1', 'coulomb_Y_resultant',

                 'coulomb_Z_resultant', 'coulomb_momentum', 'yukawa_parallel', 'coulomb_parallel', 

                 'harmonic_parallel', 'vander_parallel', 'dist_1_parallel', 'inv_dist_1_parallel',

                 'inv_dist_2_parallel', 'inv_dist_3_parallel', 'yukawa_orthogonal', 'coulomb_orthogonal',

                 'harmonic_orthogonal', 'vander_orthogonal', 'dist_1_orthogonal', 'inv_dist_1_orthogonal',

                 'inv_dist_2_orthogonal', 'inv_dist_3_orthogonal', 'yukawa_proportion_between_X_and_Y_Z',

                 'coulomb_proportion_between_X_and_Y_Z', 'harmonic_proportion_between_X_and_Y_Z',

                 'vander_proportion_between_X_and_Y_Z', 'dist_1_proportion_between_X_and_Y_Z',

                 'inv_dist_1_proportion_between_X_and_Y_Z', 'inv_dist_2_proportion_between_X_and_Y_Z',

                 'inv_dist_3_proportion_between_X_and_Y_Z', '1_yukawa_resultant', '2_yukawa_resultant',

                '1_coulomb_resultant', '2_coulomb_resultant', '1_inv_EN_resultant', '2_inv_EN_resultant',

                 '2_G4_3_selected']



for col in selected_cols:

    print(col)

    fig = plt.figure()

    plt.subplots()

    plt.scatter(df_1JHN.loc[0::20,'scalar_coupling_constant'], df_1JHN.loc[0::20,col], s=0.2)

    plt.title(f'{col} VS coupling constant for 1JHN')

    plt.ylabel(col)

    plt.xlabel('scalar_coupling_constant')

    plt.show()
df_type = 'test' #or 'train'



################

#Get all the data

################



df = pd.read_csv(f'{original_data_folder}/{df_type}.csv')

df = df[df['type'].isin(mol_types)] #selecte only the rows with desired coupling types

df = reduce_mem_usage(df, verbose=True)



unique_mol = df['molecule_name'].unique()



def get_data(data_path, verbose):

    

    data = pd.read_csv(data_path)

    data = data.loc[data['molecule_name'].isin(unique_mol)]

    data = reduce_mem_usage(data, verbose=True)

    

    return data



df_bonds = get_data(data_path = f'../input/predicting-molecular-properties-bonds/{df_type}_bonds.csv', verbose = verbose)

df_structures = get_data(data_path = '../input/champs-scalar-coupling/structures.csv', verbose = verbose)

df_structures_eigen = get_data(data_path = '../input/qm7-coulomb-matrix-eigenvalue-features/struct_eigen.csv', verbose = verbose)

if df_type == 'test':

    df_mulliken = get_data(data_path = '../input/predicting-mulliken-charges-with-acsf-descriptors/mulliken_charges_test_set.csv', verbose = verbose)

else:

    df_mulliken = get_data(data_path = '../input/champs-scalar-coupling/mulliken_charges.csv', verbose = verbose)

          



################

#Get indexes per molecule, so we can work with each molecule at a time.

#start_indexes = index where the molecule start

#end_indexes = index where the molecule end

################



xyz = df_structures[['x','y','z']].values

atom_indexes = df[['atom_index_0', 'atom_index_1']].values



struct_end_indexes = df_structures.groupby('molecule_name').size().cumsum()

df_end_indexes = df[['molecule_name', 'atom_index_0', 'atom_index_1']].groupby('molecule_name').size().cumsum()

df_bonds_end_indexes = df_bonds[['molecule_name', 'atom_index_0', 'atom_index_1']].groupby('molecule_name').size().cumsum()



struct_start_indexes = np.zeros(len(struct_end_indexes) + 1, 'int')

struct_start_indexes[1:] = struct_end_indexes

struct_start_indexes[0] = 0



df_start_indexes = np.zeros(len(df_end_indexes) + 1, 'int')

df_start_indexes[1:] = df_end_indexes

df_start_indexes[0] = 0



df_bonds_start_indexes = np.zeros(len(df_bonds_end_indexes) + 1, 'int')

df_bonds_start_indexes[1:] = df_bonds_end_indexes

df_bonds_start_indexes[0] = 0



################

#run in parallel

################



p = multiprocessing.Pool(number_of_cores)



try:

    t1 = time.time()

    result = p.map(get_mol_feats, list(range(df_structures.molecule_name.nunique())))

    t2 = time.time()

except Exception as e:

    print('Caught exception in worker thread (x = d):')

    traceback.print_exc()

    print()

    raise e

    

p.terminate()

p.join



print(f'Calculating features took {(t2 - t1)/60:0.2f} minutes')



################

#save to df

################



del df_structures_eigen, df_bonds_end_indexes, df_bonds_start_indexes, xyz, unique_mol, df_start_indexes, df_end_indexes, df_bonds, df, df_structures, df_mulliken, atom_indexes



import gc

gc.collect()



data_concat = np.concatenate(result, axis=0)

del result

gc.collect()



features_df = pd.DataFrame(data_concat, columns = dtype_dict).astype(dtype_dict)

del data_concat

gc.collect()



df = pd.read_csv(f'{original_data_folder}/{df_type}.csv')

df = df[df['type'].isin(mol_types)] #selecte only the rows with desired coupling types

df = reduce_mem_usage(df, verbose=True)

df.reset_index(drop = True, inplace=True)



df = features_df.join(df)

del features_df

gc.collect()



for t in mol_types:

    df_only_selected_type = df[df['type'] == t]

    df_only_selected_type.to_csv(f'{df_type}_{t}.csv')

    del df_only_selected_type
sub = pd.read_csv('../input/preds-on-oof-and-test/final_model_submission.csv')

sub.to_csv('final_sub.csv', index=False)