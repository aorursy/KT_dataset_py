import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')

test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

sub = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')

structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')

train_sub_charge=pd.read_csv('../input/champs-scalar-coupling/mulliken_charges.csv')

train_sub_tensor=pd.read_csv('../input/champs-scalar-coupling/magnetic_shielding_tensors.csv')
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



print(train.shape, test.shape, structures.shape)

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

structures = reduce_mem_usage(structures)

train_sub_charge = reduce_mem_usage(train_sub_charge)

train_sub_tensor = reduce_mem_usage(train_sub_tensor)

print(train.shape, test.shape, structures.shape)
print(f'There are {train.shape[0]} rows in train data.')

print(f'There are {test.shape[0]} rows in test data.')



print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")

print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")

print(f"There are {structures['atom'].nunique()} unique atoms.")

print(f"There are {train['type'].nunique()} unique types.")
#train_=train
#add atom coords

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
''' 

Map atom info from the structures.csv into the train/test files

'''

import psutil

import os



def map_atom_info(df_1,df_2, atom_idx):

    print('Mapping...', df_1.shape, df_2.shape, atom_idx)

    

    df = pd.merge(df_1, df_2.drop_duplicates(subset=['molecule_name', 'atom_index']), how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)



    return df



def show_ram_usage():

    py = psutil.Process(os.getpid())

    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))



show_ram_usage()



for atom_idx in [0,1]:

    train = map_atom_info(train,structures, atom_idx)

    train = map_atom_info(train,train_sub_charge, atom_idx)

    train = map_atom_info(train,train_sub_tensor, atom_idx)

    train = train.rename(columns={'atom': f'atom_{atom_idx}',

                                        'x': f'x_{atom_idx}',

                                        'y': f'y_{atom_idx}',

                                        'z': f'z_{atom_idx}',

                                        'mulliken_charge': f'charge_{atom_idx}',

                                        'XX': f'XX_{atom_idx}',

                                        'YX': f'YX_{atom_idx}',

                                        'ZX': f'ZX_{atom_idx}',

                                        'XY': f'XY_{atom_idx}',

                                        'YY': f'YY_{atom_idx}',

                                        'ZY': f'ZY_{atom_idx}',

                                        'XZ': f'XZ_{atom_idx}',

                                        'YZ': f'YZ_{atom_idx}',

                                        'ZZ': f'ZZ_{atom_idx}',})



    test = map_atom_info(test,structures, atom_idx)

    test = test.rename(columns={'atom': f'atom_{atom_idx}',

                                'x': f'x_{atom_idx}',

                                'y': f'y_{atom_idx}',

                                'z': f'z_{atom_idx}'})

    #add some features

    

    structures['c_x']=structures.groupby('molecule_name')['x'].transform('mean')

    structures['c_y']=structures.groupby('molecule_name')['y'].transform('mean')

    structures['c_z']=structures.groupby('molecule_name')['z'].transform('mean')

    structures['atom_n']=structures.groupby('molecule_name')['atom_index'].transform('max')

    

    show_ram_usage()

    print(train.shape, test.shape)
#create distance feature

def make_features(df):

    df['dx']=df['x_1']-df['x_0']

    df['dy']=df['y_1']-df['y_0']

    df['dz']=df['z_1']-df['z_0']

    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)

    return df



train_=make_features(train)

test=make_features(test)

test_prediction=np.zeros(len(test))

show_ram_usage()

print(train_.shape, test.shape)
#create more complex feature

def get_dist(df):

    df_temp=df.loc[:,["molecule_name","atom_index_0","atom_index_1","distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()

    df_temp_=df_temp.copy()

    df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',

                                       'atom_index_1': 'atom_index_0',

                                       'x_0': 'x_1',

                                       'y_0': 'y_1',

                                       'z_0': 'z_1',

                                       'x_1': 'x_0',

                                       'y_1': 'y_0',

                                       'z_1': 'z_0'})

    df_temp_all=pd.concat((df_temp,df_temp_),axis=0)



    df_temp_all["min_distance"]=df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')

    df_temp_all["max_distance"]=df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max')

    

    df_temp= df_temp_all[df_temp_all["min_distance"]==df_temp_all["distance"]].copy()

    df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                         'atom_index_1': 'atom_index_closest',

                                         'distance': 'distance_closest',

                                         'x_1': 'x_closest',

                                         'y_1': 'y_closest',

                                         'z_1': 'z_closest'})

    

    for atom_idx in [0,1]:

        df = map_atom_info(df,df_temp, atom_idx)

        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',

                                        'distance_closest': f'distance_closest_{atom_idx}',

                                        'x_closest': f'x_closest_{atom_idx}',

                                        'y_closest': f'y_closest_{atom_idx}',

                                        'z_closest': f'z_closest_{atom_idx}'})

        

    df_temp= df_temp_all[df_temp_all["max_distance"]==df_temp_all["distance"]].copy()

    df_temp=df_temp.drop(['x_0','y_0','z_0','max_distance'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                         'atom_index_1': 'atom_index_farthest',

                                         'distance': 'distance_farthest',

                                         'x_1': 'x_farthest',

                                         'y_1': 'y_farthest',

                                         'z_1': 'z_farthest'})

        

    for atom_idx in [0,1]:

        df = map_atom_info(df,df_temp, atom_idx)

        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',

                                        'distance_farthest': f'distance_farthest_{atom_idx}',

                                        'x_farthest': f'x_farthest_{atom_idx}',

                                        'y_farthest': f'y_farthest_{atom_idx}',

                                        'z_farthest': f'z_farthest_{atom_idx}'})

    return df
#create more complex feature

test=(get_dist(test)) 

train=(get_dist(train))



print(train.shape, test.shape)

show_ram_usage()
#create cosinus distance

def add_features(df):

    df["distance_center0"]=((df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)

    df["distance_center1"]=((df['x_1']-df['c_x'])**2+(df['y_1']-df['c_y'])**2+(df['z_1']-df['c_z'])**2)**(1/2)

    df["distance_c0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)

    df["distance_c1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["distance_f0"]=((df['x_0']-df['x_farthest_0'])**2+(df['y_0']-df['y_farthest_0'])**2+(df['z_0']-df['z_farthest_0'])**2)**(1/2)

    df["distance_f1"]=((df['x_1']-df['x_farthest_1'])**2+(df['y_1']-df['y_farthest_1'])**2+(df['z_1']-df['z_farthest_1'])**2)**(1/2)

    df["vec_center0_x"]=(df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)

    df["vec_center0_y"]=(df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)

    df["vec_center0_z"]=(df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)

    df["vec_center1_x"]=(df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)

    df["vec_center1_y"]=(df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)

    df["vec_center1_z"]=(df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)

    df["vec_c0_x"]=(df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c0_y"]=(df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c0_z"]=(df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c1_x"]=(df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_c1_y"]=(df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_c1_z"]=(df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_f0_x"]=(df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f0_y"]=(df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f0_z"]=(df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f1_x"]=(df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_f1_y"]=(df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_f1_z"]=(df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]

    df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]

    df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]

    df["cos_c0_c1"]=df["vec_c0_x"]*df["vec_c1_x"]+df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]

    df["cos_f0_f1"]=df["vec_f0_x"]*df["vec_f1_x"]+df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]

    df["cos_center0_center1"]=df["vec_center0_x"]*df["vec_center1_x"]+df["vec_center0_y"]*df["vec_center1_y"]+df["vec_center0_z"]*df["vec_center1_z"]

    df["cos_c0"]=df["vec_c0_x"]*df["vec_x"]+df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]

    df["cos_c1"]=df["vec_c1_x"]*df["vec_x"]+df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]

    df["cos_f0"]=df["vec_f0_x"]*df["vec_x"]+df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]

    df["cos_f1"]=df["vec_f1_x"]*df["vec_x"]+df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]

    df["cos_center0"]=df["vec_center0_x"]*df["vec_x"]+df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]

    df["cos_center1"]=df["vec_center1_x"]*df["vec_x"]+df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]

    df=df.drop(['vec_c0_x','vec_c0_y','vec_c0_z','vec_c1_x','vec_c1_y','vec_c1_z',

                'vec_f0_x','vec_f0_y','vec_f0_z','vec_f1_x','vec_f1_y','vec_f1_z',

                'vec_center0_x','vec_center0_y','vec_center0_z','vec_center1_x','vec_center1_y','vec_center1_z',

                'vec_x','vec_y','vec_z'], axis=1)

    return df
#create cosinus distance

train=add_features(train)

test=add_features(test)

print(train.shape, test.shape)

show_ram_usage()
train.head()
#quantitative distribution

quantitative = [f for f in train_.columns if train_.dtypes[f] != 'object']

quantitative.remove('scalar_coupling_constant')

quantitative.remove('id')

#qualitative ditribution

qualitative = [f for f in train_.columns if train_.dtypes[f] == 'object']

qualitative.remove('molecule_name')

for c in qualitative:

    train_[c] = train_[c].astype('category')
#print qualitative features

qualitative
train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])

train['type_1'] = train['type'].apply(lambda x: x[1:])

test['type_1'] = test['type'].apply(lambda x: x[1:])
print(train.columns.values)
#scatter plot cos_c0/scalar_coupling_constant

var = 'cos_c0'

for ctype in train['type'].unique() :

    train_plot  = train.loc[train['type']==ctype][:3000]

    data = pd.concat([train_plot['scalar_coupling_constant'], train_plot[var]], axis=1)

    data.plot.scatter(x=var, y='scalar_coupling_constant');

    plt.title(f'{ctype}', fontsize=18)
#histogram and normal probability plot

from scipy import stats

from scipy.stats import norm

var = 'cos_f1'

train_plot  = train[:10000]

for ctype in train_.type.unique():

    plt.figure()

    plt.title(f'{ctype}', fontsize=18)

    sns.distplot((train_plot.loc[train['type']==ctype])[var], fit=norm);

    plt.figure()

    res = stats.probplot((train_plot.loc[train['type']==ctype])[var], plot=plt)

#correlation matrix with absolute value

corrmat = train[:10000].corr()

corrmat = corrmat.abs()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#scalar_coupling_constant correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'scalar_coupling_constant')['scalar_coupling_constant'].index

cm = np.corrcoef(train[:10000][cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#LabelEncoder : string labels to integers

from sklearn.preprocessing import LabelEncoder

for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:

    lbl = LabelEncoder()

    lbl.fit(list(train[f].values) )

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))
#scatter plot totalbsmtsf/saleprice

var = 'distance'

for ctype in range(0,7):

    train_plot  = train_.loc[train['type']==ctype][:3000]

    data = pd.concat([train_plot['scalar_coupling_constant'], train_plot[var]], axis=1)

    data.plot.scatter(x=var, y='scalar_coupling_constant');
test_ = train[4200000:]

train_ = train[:4200000]
X = train_.drop(['id', 'molecule_name', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1'], axis=1)

y = train_['scalar_coupling_constant']

X_test = test.drop(['id', 'molecule_name', 'atom_index_0', 'atom_index_1'], axis=1)

X_test_ = test_.drop(['id', 'molecule_name','scalar_coupling_constant', 'atom_index_0', 'atom_index_1'], axis=1) #subset of training to test model

y_verif = test_['scalar_coupling_constant'] #verifying data to test model accuracy
print(X_test.shape , X_test_.shape)
print(X_test_.columns.values)
from keras import callbacks

# Set to True if we want to train from scratch.  False will reuse saved models as a starting point.

retrain = True

model_name_rd = ('../keras-neural-net-for-champs/molecule_model.hdf5')

model_name_wrt = ('/kaggle/working/molecule_model.hdf5')



es = callbacks.EarlyStopping(monitor='loss', min_delta=0.1, patience=8,verbose=1, mode='auto', restore_best_weights=True)

# Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced

rlr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,patience=7, min_lr=1e-6, min_delta=0.1, mode='auto', verbose=1)

# Save the best value of the model for future use

sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='loss', save_best_only=True, period=1)
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential

from sklearn.metrics import accuracy_score

from keras.layers import Dense, Input, Activation

from keras.layers import BatchNormalization,Add,Dropout

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU

import eli5

from eli5.sklearn import PermutationImportance



def baseline_model():

    model = Sequential()

    model.add(Dense(256, input_dim=31, activation='relu',kernel_initializer='normal'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.4))

    model.add(Dense(1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.2))

    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.2))

    model.add(Dense(64))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    if not retrain:

        my_model = load_model(model_name_wrt)

        return my_model

    return model



import time

start_time = time.time()



my_model = KerasRegressor(build_fn=baseline_model, batch_size=3000, epochs=400, verbose=True,  callbacks=[es, rlr, sv_mod])

my_model.fit(np.array(X[['type','distance','distance_center0','distance_center1','distance_c0','distance_c1','distance_f0','distance_f1', 'type_0', 'type_1',

                         'cos_f0', 'cos_f1',  'cos_c0_c1', 'cos_f0_f1','cos_c0', 'cos_c1', "x_0","y_0","z_0","x_1","y_1","z_1","c_x","c_y","c_z",

                    'x_closest_0','y_closest_0','z_closest_0','x_closest_1','y_closest_1','z_closest_1']]),y)



print("--- %s seconds ---" % (time.time() - start_time))



#perm = PermutationImportance(my_model, random_state=1).fit(np.array(X),y)

#eli5.show_weights(perm, feature_names = X.columns.tolist())
test_y = my_model.predict(np.array(X_test_[['type','distance','distance_center0','distance_center1','distance_c0','distance_c1','distance_f0','distance_f1', 'type_0', 'type_1',

                         'cos_f0', 'cos_f1',  'cos_c0_c1', 'cos_f0_f1','cos_c0', 'cos_c1', "x_0","y_0","z_0","x_1","y_1","z_1","c_x","c_y","c_z",

                    'x_closest_0','y_closest_0','z_closest_0','x_closest_1','y_closest_1','z_closest_1']]))
#metric for this competition

from sklearn import metrics

def metric(df, preds):

    df["prediction"] = preds

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)



#metric(test_, y_verif) #gets -inf

metric(test_, test_y)
test_y = my_model.predict(np.array(X_test[['type','distance','distance_center0','distance_center1','distance_c0','distance_c1','distance_f0','distance_f1', 'type_0', 'type_1',

                         'cos_f0', 'cos_f1',  'cos_c0_c1', 'cos_f0_f1','cos_c0', 'cos_c1', "x_0","y_0","z_0","x_1","y_1","z_1","c_x","c_y","c_z",

                    'x_closest_0','y_closest_0','z_closest_0','x_closest_1','y_closest_1','z_closest_1']]))

sub['scalar_coupling_constant'] = test_y

sub.to_csv('submission.csv', index=False)

sub.head()