# Libraries Import

import pandas as pd

import numpy as np



# Opening training dataset

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_str = pd.read_csv("../input/structures.csv")



# Infos

df_train.info()
df_test.head()
df_train.head()
df_str.head()
# Credits for: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_memory(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props
# Reduce memory dataframe train

df_train = reduce_memory(df_train)
# Reduce Memory test

df_test = reduce_memory(df_test)
# Reduce memory data structures

df_str = reduce_memory(df_str)
%%time

def merge_datasets(df,df_str):

    '''

    Operation:

        To improve performance when joining large dataframes, using the left merge helps

    because I can keep my main dataframe information and add information from my dataframes

    more simply and computationally efficient.

    

    Input:

        Receives two merge dataframes - 1 being either the training or the test and the other being

    the dataframe that contains the information about the atom structures.

    

    Exit:

        Returns the dataframe containing the columns with information of each atom, and inform which was

    the atom in that molecule.

    '''

    # Parte 1

    df = df.rename(columns = {"atom_index_0":"atom_index"})

    df = pd.merge(df,df_str,on = ["molecule_name","atom_index"], how = "left") 

    df = df.rename(columns = {"atom_index":"atom_index_0","x":"x_0","y":"y_0","z":"z_0"})

    # Parte 2

    df = df.rename(columns = {"atom_index_1":"atom_index"})

    df = pd.merge(df,df_str,on = ["molecule_name","atom_index"], how = "left")

    df = df.rename(columns = {"atom_index":"atom_index_1","x":"x_1","y":"y_1","z":"z_1"})

    # Return dataframe mesclado

    

    df = reduce_memory(df)

    return df
# Auxiliar

df_test_aux = df_test.copy()

# Merge

df_train = merge_datasets(df_train,df_str)

df_test_aux = merge_datasets(df_test_aux,df_str)

df_train.head()
def create_features(df):

    ## Auxiliary Information

    # Atoms Information

    m_atomica = {"H":1.0079,"C":12.0107,"N":14.0067,"O":15.9994,"F":18.9984}

    n_atomico = {"H":1,"C":6,"N":7,"O":8,"F":9}

    eletro_atomico = {"H":2.2,"C":2.55,"N":3.04,"O":3.44,"F":3.98}

    

    

    ## Basic Calculations

    # Euclidean distance calculation

    df["dist"] = np.sqrt((df["x_1"]-df["x_0"])**2 +

                                       (df["y_1"]-df["y_0"])**2 +

                                       (df["z_1"]-df["z_0"])**2)



    # Angle Calculation

    df["dihedral_angle"] = np.abs(df["x_0"]*df["x_1"] + df["y_0"]*df["y_1"] + df["z_0"]*df["z_1"])/(

                                  np.sqrt(df["x_0"]**2 + df["y_0"]**2 + df["z_0"]**2) * 

                                  np.sqrt(df["x_1"]**2 + df["y_1"]**2 + df["z_1"]**2))

    # Midpoint 

    df["dx"] = ((df["x_0"] - df["x_1"])**2)

    df["dy"] = ((df["y_0"] - df["y_1"])**2)

    df["dz"] = ((df["z_0"] - df["z_1"])**2)



    # Creating atomic information columns

    df["atom_mass_1"] = df["atom_y"].replace(m_atomica)

    df["n_atom_1"] = df["atom_y"].replace(n_atomico)

    df["eletro_atom_1"] = df["atom_y"].replace(eletro_atomico)

    df["n_type"] = [int(x[0]) for x in df["type"].values]

    

    

    ## Merging with other columns

    # Creating miscellaneous columns with this information

    # Coupling Amount per Molecule

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    # Coupling distance average per molecule

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    # Minimum coupling distance per molecule

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    # Maximum coupling distance per molecule

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    # Amount of coupling per molecule and atoms

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    # Mean, Max, Min and Std of the diehdro angle of the molecules

    df['molecule_dihedral_angle_mean'] = df.groupby('molecule_name')['dihedral_angle'].transform('mean')

    df['molecule_dihedral_angle_max'] = df.groupby('molecule_name')['dihedral_angle'].transform('max')

    df['molecule_dihedral_angle_min'] = df.groupby('molecule_name')['dihedral_angle'].transform('min')

    df['molecule_dihedral_angle_std'] = df.groupby('molecule_name')['dihedral_angle'].transform('std')

    #- Atributos Atom_index_0

    # Dist

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    # dihedral_angle

    df[f'molecule_atom_index_0_dihedral_angle_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dihedral_angle'].transform('mean')

    df[f'molecule_atom_index_0_dihedral_angle_mean_diff'] = df[f'molecule_atom_index_0_dihedral_angle_mean'] - df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_mean_div'] = df[f'molecule_atom_index_0_dihedral_angle_mean'] / df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dihedral_angle'].transform('max')

    df[f'molecule_atom_index_0_dihedral_angle_max_diff'] = df[f'molecule_atom_index_0_dihedral_angle_max'] - df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_max_div'] = df[f'molecule_atom_index_0_dihedral_angle_max'] / df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dihedral_angle'].transform('min')

    df[f'molecule_atom_index_0_dihedral_angle_min_diff'] = df[f'molecule_atom_index_0_dihedral_angle_min'] - df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_min_div'] = df[f'molecule_atom_index_0_dihedral_angle_min'] / df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dihedral_angle'].transform('std')

    df[f'molecule_atom_index_0_dihedral_angle_std_diff'] = df[f'molecule_atom_index_0_dihedral_angle_std'] - df['dihedral_angle']

    df[f'molecule_atom_index_0_dihedral_angle_std_div'] = df[f'molecule_atom_index_0_dihedral_angle_std'] / df['dihedral_angle']

    #- Atributos Atom_index_1

    ## Dist

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_y'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_y'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_y'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    ## dihedral_angle

    df[f'molecule_atom_index_1_dihedral_angle_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dihedral_angle'].transform('mean')

    df[f'molecule_atom_index_1_dihedral_angle_mean_diff'] = df[f'molecule_atom_index_1_dihedral_angle_mean'] - df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_mean_div'] = df[f'molecule_atom_index_1_dihedral_angle_mean'] / df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dihedral_angle'].transform('max')

    df[f'molecule_atom_index_1_dihedral_angle_max_diff'] = df[f'molecule_atom_index_1_dihedral_angle_max'] - df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_max_div'] = df[f'molecule_atom_index_1_dihedral_angle_max'] / df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dihedral_angle'].transform('min')

    df[f'molecule_atom_index_1_dihedral_angle_min_diff'] = df[f'molecule_atom_index_1_dihedral_angle_min'] - df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_min_div'] = df[f'molecule_atom_index_1_dihedral_angle_min'] / df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dihedral_angle'].transform('std')

    df[f'molecule_atom_index_1_dihedral_angle_std_diff'] = df[f'molecule_atom_index_1_dihedral_angle_std'] - df['dihedral_angle']

    df[f'molecule_atom_index_1_dihedral_angle_std_div'] = df[f'molecule_atom_index_1_dihedral_angle_std'] / df['dihedral_angle']

    df[f'molecule_atom_1_dihedral_angle_mean'] = df.groupby(['molecule_name', 'atom_y'])['dihedral_angle'].transform('mean')

    df[f'molecule_atom_1_dihedral_angle_min'] = df.groupby(['molecule_name', 'atom_y'])['dihedral_angle'].transform('min')

    df[f'molecule_atom_1_dihedral_angle_min_diff'] = df[f'molecule_atom_1_dihedral_angle_min'] - df['dihedral_angle']

    df[f'molecule_atom_1_dihedral_angle_min_div'] = df[f'molecule_atom_1_dihedral_angle_min'] / df['dihedral_angle']

    df[f'molecule_atom_1_dihedral_angle_std'] = df.groupby(['molecule_name', 'atom_y'])['dihedral_angle'].transform('std')

    df[f'molecule_atom_1_dihedral_angle_std_diff'] = df[f'molecule_atom_1_dihedral_angle_std'] - df['dihedral_angle']

    # Eletronegatividade

    #df[f'molecule_atom_index_10_eletro_mean'] = df.groupby(['molecule_name', 'atom_index_1','atom_index_0'])['eletro_atom_1'].transform('mean')

    # Reduce Memory

    #df = reduce_memory(df)

    

    return df
# Applying Feature Engineering

# Train

df_train = create_features(df_train)

# Test

df_test_aux = create_features(df_test_aux)
#df_train.corr()["scalar_coupling_constant"]
# Feature Selection

# Correlation with output variable 

cor = df_train.corr()

cor_target = abs(cor["scalar_coupling_constant"]) 



#Selecting Highly Correlated Features 

relevant_features_test = cor_target[(cor_target > 0.1) & (cor_target.index != "scalar_coupling_constant")] 

relevant_features_train = cor_target[(cor_target > 0.1)]
#del df_train, df_test_aux

df_train = df_train[relevant_features_train.index]

df_test_aux = df_test_aux[relevant_features_test.index]
df_train.info()
# Libraries Import

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler


# X and Y

# Train

X = df_train.drop("scalar_coupling_constant", axis = 1)

Y = df_train["scalar_coupling_constant"]

# Test

X_test = df_test_aux.copy()

# Validation

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=23)
# Catboost



#categorical = ["type","atom_y"]



cb_model = CatBoostRegressor(iterations=100,

                             learning_rate=0.6,

                             max_depth = 9,

                             eval_metric='MAE',

                             random_seed = 23,

                             bagging_temperature = 0.4,

                             od_type='Iter',

                             metric_period = 100,

                             od_wait=200,

                             nan_mode = "Min")



cb_model.fit(x_train, y_train,

             eval_set=(x_test,y_test),

             use_best_model=True,

             plot=False)
fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': X.columns})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

#plt.savefig('catboost_feature_importance.png')
# Predict

y_prev_model = cb_model.predict(X_test)
submission = pd.DataFrame({"id":df_test["id"].values,"scalar_coupling_constant":y_prev_model})
submission.to_csv("submission.csv", index = False)
submission.head()