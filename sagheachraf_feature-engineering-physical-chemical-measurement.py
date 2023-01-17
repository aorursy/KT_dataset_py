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
import scipy as sp #collection of functions for scientific computing and advance mathematics

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax





# warnings mute

import warnings

warnings.simplefilter('ignore')
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
train = import_data("../input/train.csv")

test =import_data("../input/test.csv")

sub = import_data("../input/sample_submission.csv")

structures = import_data("../input/structures.csv")

#dipole = import_data("../input/dipole_moments.csv")

#potential = import_data("../input/potential_energy.csv")

#mulliken = import_data("../input/mulliken_charges.csv")

contributions = import_data("../input/scalar_coupling_contributions.csv")

#This dataframe to be used for merging with all chemichal features ( Electronegativity , radius , isotope ..)

structures_notreal = structures


# Importing gc module 

import gc 

  

# Returns the number of 

# objects it has collected 

# and deallocated 

collected = gc.collect() 

  

# Prints Garbage collector  

# as 0 object 

print("Garbage collector: collected", 

          "%d objects." % collected) 
#electronegativity pauling

electronegativity = {'H':2.2, 'C':2.55, 'N':3.04 , 'O':3.44, 'F':3.98 }

#Type de charge 

charge ={'H':0, 'C':1, 'N':1 , 'O':3.44, 'F':3.98 }

#etat

etat ={'H':0, 'C':1, 'N':0, 'O':3.44, 'F':3.98 }

#Masse kg/m^3

masse = {'H':76, 'C':3513, 'N':1026, 'O':3.44, 'F':3.98 }

#volume cm^3/mole

volume = {'H':13.26, 'C':3.42, 'N':13.65, 'O':3.44, 'F':3.98 }

#Rayon atomique (mesuré)

rayon_am = {'H':25, 'C':70, 'N':65, 'O':3.44, 'F':3.98 }

#Rayon atomique (calculé)

rayon_ac = {'H':53, 'C':67, 'N':56 ,'O':0.73, 'F':0.71}

#Rayon covalent

rayon_c = {'H':38, 'C':77, 'N':75, 'O':3.44, 'F':3.98 }

#Rayon ionique

rayon_i = {'H':-3, 'C':4, 'N':-3, 'O':3.44, 'F':3.98 }

#Rayon de Van der Waals

rayon_vdw = {'H':120, 'C':170, 'N':155, 'O':3.44, 'F':3.98 }

#Point de fusion

fusion = {'H':-259.1, 'C':3546.9, 'N':-209.9, 'O':3.44, 'F':3.98 }

#seuil d'ébulution minimal (celcius)

ebulution_min = {'H':-252.9, 'C':4826.9, 'N':-195.8, 'O':3.44, 'F':3.98 }

#Enthalpie de fusion ΔHf (kj/mol)

enthalpie_fusion = {'H':0.12, 'C':105, 'N':0.72, 'O':3.44, 'F':3.98 }

#Enthalpie de vaporisation ΔHv

enthalpie_vaporisation = {'H':0.46, 'C':710.9, 'N':5.58, 'O':3.44, 'F':3.98 }

#Capacité thermique

capacite_thermique = {'H':14.3, 'C':0.71, 'N':1.04, 'O':3.44, 'F':3.98 }

#Conductivité thermique

conductivite_thermique = {'H':0.18, 'C':990, 'N':0.03, 'O':3.44, 'F':3.98 }





#Nb isotopes

isotopes = {'H':3, 'C':12, 'N':12, 'O':3.44, 'F':3.98 }

#Isotopes emeteurs

isotopes_emeteurs = {'H':0, 'C':3, 'N':3, 'O':3.44, 'F':3.98 }

dico_chemical_elements = {'electronegativity':electronegativity ,

                         'charge':charge,

                          'etat':etat,

                          'masse':masse,

                          'volume':volume,

                          'rayon_am':rayon_am,

                          'rayon_ac':rayon_ac,

                          'rayon_c':rayon_c,

                          'rayon_i':rayon_i,

                          'rayon_vdw':rayon_vdw,

                          'fusion':fusion,

                          'ebulution_min':ebulution_min,

                          'enthalpie_fusion':enthalpie_fusion,

                          'enthalpie_vaporisation':enthalpie_vaporisation,

                          'capacite_thermique':capacite_thermique,

                          'conductivite_thermique':conductivite_thermique,

                          'isotopes':isotopes,

                          'isotopes_emeteurs':isotopes_emeteurs

                         }
def dico_todf(list_dicos, df):

    for k,v in list_dicos.items():

        df[k] = df['atom'].apply(lambda x : v[x])

    return df 



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df
def dihedral_angle(data): 

        

    vals = np.array(data[:, 3:6], dtype=np.float64)

    mol_names = np.array(data[:, 0], dtype=np.str)

 

    result = np.zeros((data.shape[0], 2), dtype=object)

    # use every 4 rows to compute the dihedral angle

    for idx in range(0, vals.shape[0] - 4, 4):



        a0 = vals[idx]

        a1 = vals[idx + 1]

        a2 = vals[idx + 2]

        a3 = vals[idx + 3]

        

        b0 = a0 - a1

        b1 = a2 - a1

        b2 = a3 - a2

        

        # normalize b1 so that it does not influence magnitude of vector

        # rejections that come next

        b1 /= np.linalg.norm(b1)

    

        # vector rejections

        # v = projection of b0 onto plane perpendicular to b1

        #   = b0 minus component that aligns with b1

        # w = projection of b2 onto plane perpendicular to b1

        #   = b2 minus component that aligns with b1



        v = b0 - np.dot(b0, b1) * b1

        w = b2 - np.dot(b2, b1) * b1



        # angle between v and w in a plane is the torsion angle

        # v and w may not be normalized but that's fine since tan is y/x

        x = np.dot(v, w)

        y = np.dot(np.cross(b1, v), w)

       

        # We want all 4 first rows for every molecule to have the same value

        # (in order to have the same length as the dataframe)

        result[idx:idx + 4] = [mol_names[idx], np.degrees(np.arctan2(y, x))]

        

    return result
from datetime import datetime

startTime = datetime.now()

dihedral = dihedral_angle(structures[structures.groupby('molecule_name')['atom_index'].transform('count').ge(4)].groupby('molecule_name').head(4).values)

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))
themap = {k:v for k, v in dihedral if k}

# Add diehral and cos diehral angle to features 

structures['dihedral'] = structures['molecule_name'].map(themap)

structures['cosdihedral'] = structures['dihedral'].map(np.cos)
atoms = structures['atom'].values

atoms_en = [electronegativity[x] for x in (atoms)]

atoms_rad = [rayon_ac[x] for x in (atoms)]



structures['EN'] = atoms_en

structures['rad'] = atoms_rad



#Add bonds to features

i_atom = structures['atom_index'].values

p = structures[['x', 'y', 'z']].values

p_compare = p

m = structures['molecule_name'].values

m_compare = m

r = structures['rad'].values

r_compare = r



source_row = np.arange(len(structures))

max_atoms = 28



bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)

bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)





for i in (range(max_atoms-1)):

    p_compare = np.roll(p_compare, -1, axis=0)

    m_compare = np.roll(m_compare, -1, axis=0)

    r_compare = np.roll(r_compare, -1, axis=0)

    

    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?

    dists = np.linalg.norm(p - p_compare, axis=1) * mask

    r_bond = r + r_compare

    

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    

    source_row = source_row

    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

    

    source_atom = i_atom

    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

    

    bonds[(source_row, target_atom)] = bond

    bonds[(target_row, source_atom)] = bond

    bond_dists[(source_row, target_atom)] = dists

    bond_dists[(target_row, source_atom)] = dists



bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row

bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col

bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row

bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col





bonds_numeric = [[i for i,x in enumerate(row) if x] for row in (bonds)]

bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate((bond_dists))]

bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

n_bonds = [len(x) for x in bonds_numeric]



bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }

bond_df = pd.DataFrame(bond_data)

structures = structures.join(bond_df)
df_struct_aux= structures.groupby(['molecule_name'])['atom'].agg([('bonds_distc',lambda x : len(set(x.tolist())))]).reset_index()

structures=pd.merge(structures,df_struct_aux)
structures=dico_todf(dico_chemical_elements,structures)
structures
structures_aux_xbary=structures.groupby(['molecule_name'])[['x','y','z']].agg({'avg':np.average}).reset_index()

structures_aux_xbary_=pd.DataFrame(structures_aux_xbary.get_values())

structures_aux_xbary_.columns=['molecule_name','x_bar','y_bar','z_bar']

structures_bary = pd.merge(structures,structures_aux_xbary_)

del structures_bary['atom']

structures=structures_bary
train = map_atom_info(train, 0)

train = map_atom_info(train, 1)

test = map_atom_info(test, 0)

test = map_atom_info(test, 1)

del train['x_bar_y']

del train['y_bar_y']

del train['z_bar_y']

del test['x_bar_y']

del test['y_bar_y']

del test['z_bar_y']

train.columns
%env JOBLIB_TEMP_FOLDER=/tmp
train.columns
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2
train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])

train['type_1'] = train['type'].apply(lambda x: x[1:])

test['type_1'] = test['type'].apply(lambda x: x[1:])
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')

test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')



train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')

test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')



train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')

test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')



train[f'molecule_type_dist_mean'] = train.groupby(['molecule_name', 'type'])['dist'].transform('mean')

test[f'molecule_type_dist_mean'] = test.groupby(['molecule_name', 'type'])['dist'].transform('mean')
train['dist_barycentre_x0']=np.abs(train['x_bar_x'] - train['x_0'])

train['dist_barycentre_y0']=np.abs(train['y_bar_x'] - train['y_0'])

train['dist_barycentre_z0']=np.abs(train['z_bar_x'] - train['z_0'])

train['dist_barycentre_x1']=np.abs(train['x_bar_x'] - train['x_1'])

train['dist_barycentre_y1']=np.abs(train['y_bar_x'] - train['y_1'])

train['dist_barycentre_z1']=np.abs(train['z_bar_x'] - train['z_1'])

test['dist_barycentre_x0']=np.abs(test['x_bar_x'] - test['x_0'])

test['dist_barycentre_y0']=np.abs(test['y_bar_x'] - test['y_0'])

test['dist_barycentre_z0']=np.abs(test['z_bar_x'] - test['z_0'])

test['dist_barycentre_x1']=np.abs(test['x_bar_x'] - test['x_1'])

test['dist_barycentre_y1']=np.abs(test['y_bar_x'] - test['y_1'])

test['dist_barycentre_z1']=np.abs(test['z_bar_x'] - test['z_1'])





train['squared_measure_0'] =  np.sqrt(

    np.array(   

    np.power(train['dist_barycentre_x0'],2) +

    

    np.power(train['dist_barycentre_y0'],2) +

    

    np.power(train['dist_barycentre_z0'],2) 

     ,dtype=float   

    ))



train['squared_measure_1'] =   np.sqrt(

    np.array(

    np.power(train['dist_barycentre_x1'],2) +

    

    np.power(train['dist_barycentre_y1'],2) +

    

    np.power(train['dist_barycentre_z1'],2) 

,dtype=float   

    ))



test['squared_measure_0'] =  np.sqrt(

    np.array(

    np.power(test['dist_barycentre_x0'],2) +

    

    np.power(test['dist_barycentre_y0'],2) +

    

    np.power(test['dist_barycentre_z0'],2) 

,dtype=float   

    ))



test['squared_measure_1'] =   np.sqrt(

    np.array(

    np.power(test['dist_barycentre_x1'],2) +

    

    np.power(test['dist_barycentre_y1'],2) +

    

    np.power(test['dist_barycentre_z1'],2) 

     ,dtype=float   

    )  )
train.columns
from sklearn import preprocessing

for f in [ 'type_0', 'type_1', 'type']:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train[f].values) + list(test[f].values))

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))

#collect residual garbage

gc.collect()

import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product



import altair as alt

from altair.vega import v5

from IPython.display import HTML



# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

def prepare_altair():

    """

    Helper function to prepare altair for working.

    """



    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

    noext = "?noext"

    

    paths = {

        'vega': vega_url + noext,

        'vega-lib': vega_lib_url + noext,

        'vega-lite': vega_lite_url + noext,

        'vega-embed': vega_embed_url + noext

    }

    

    workaround = f"""    requirejs.config({{

        baseUrl: 'https://cdn.jsdelivr.net/npm/',

        paths: {paths}

    }});

    """

    

    return workaround

    



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

           



@add_autoincrement

def render(chart, id="vega-chart"):

    """

    Helper function to plot altair visualizations.

    """

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )

    



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

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float64).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    



@jit

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()

    



def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    :params: verbose - parameters for gradient boosting models

    :params: early_stopping_rounds - parameters for gradient boosting models

    :params: n_estimators - parameters for gradient boosting models

    

    """

    columns = X.columns if columns is None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'sklearn_scoring_function': metrics.mean_absolute_error},

                    'group_mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'scoring_function': group_mean_log_mae},

                    'mse': {'lgb_metric_name': 'mse',

                        'catboost_metric_name': 'MSE',

                        'sklearn_scoring_function': metrics.mean_squared_error}

                    }



    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X))

    

    # averaged predictions on train data

    prediction = np.zeros(len(X_test))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        if eval_metric != 'group_mae':

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        else:

            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict

    





def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of classification models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    :params: verbose - parameters for gradient boosting models

    :params: early_stopping_rounds - parameters for gradient boosting models

    :params: n_estimators - parameters for gradient boosting models

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict



# setting up altair

workaround = prepare_altair()

HTML("".join((

    "<script>",

    workaround,

    "</script>",

)))
train[['x_bar_x', 'y_bar_x', 'z_bar_x', 'dist_barycentre_x0', 'dist_barycentre_y0', 'dist_barycentre_z0','dist_barycentre_x1', 'dist_barycentre_y1', 'dist_barycentre_z1']] = train[['x_bar_x', 

'y_bar_x', 'z_bar_x', 'dist_barycentre_x0', 'dist_barycentre_y0', 'dist_barycentre_z0', 'dist_barycentre_x1', 'dist_barycentre_y1', 'dist_barycentre_z1']].apply(pd.to_numeric)

test[['x_bar_x', 'y_bar_x', 'z_bar_x', 'dist_barycentre_x0', 'dist_barycentre_y0', 'dist_barycentre_z0','dist_barycentre_x1', 'dist_barycentre_y1', 'dist_barycentre_z1']] = test[['x_bar_x', 

'y_bar_x', 'z_bar_x', 'dist_barycentre_x0', 'dist_barycentre_y0', 'dist_barycentre_z0', 'dist_barycentre_x1', 'dist_barycentre_y1', 'dist_barycentre_z1']].apply(pd.to_numeric)
train.columns
X = train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)

y = train['scalar_coupling_constant']

X_test = test.drop(['id', 'molecule_name'], axis=1)

X.columns
del train

del structures

#del test

del structures_notreal

del structures_bary

del structures_aux_xbary

del structures_aux_xbary_

del train_p_0

del train_p_1

del test_p_0

del test_p_1

gc.collect()
y.head()
X1=reduce_mem_usage(X)

#y1=reduce_mem_usage(y)

X_test1 =reduce_mem_usage(X_test)
X1.columns
del X

del X_test

gc.collect()
from sklearn.model_selection import KFold

n_fold = 3

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': 15,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0,

          

         }

result_dict_lgb = train_model_regression(X=X1, X_test=X_test1, y=y, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,

                                                      verbose=1000, early_stopping_rounds=200, n_estimators=10000)
submission=pd.DataFrame({'id':test["id"], 'scalar_coupling_constant':result_dict_lgb['prediction']})

submission.to_csv('submission.csv',index=False)