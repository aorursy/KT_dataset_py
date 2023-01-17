% matplotlib inline

import matplotlib.pyplot as plt

import os

import pandas as pd
import numpy as np
import seaborn as sns
test = pd.read_csv('../input/nomad2018-predict-transparent-conductors/test.csv')
test_id = test.id

train = pd.read_csv('../input/nomad2018-predict-transparent-conductors/train.csv')
# get coordinate information

ga_cols = []
al_cols = []
o_cols = []
in_cols = []

import warnings
warnings.filterwarnings("ignore")

for i in range(6):
    ga_cols.append("Ga_"+str(i))

for i in range(6):
    al_cols.append("Al_"+str(i))

for i in range(6):
    o_cols.append("O_"+str(i))

for i in range(6):
    in_cols.append("In_"+str(i))



ga_df= pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns= o_cols)
in_df = pd.DataFrame(columns=in_cols)

def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)



for i in train.id.values:
    fn = "../input/nomad2018-predict-transparent-conductors/train/{}/geometry.xyz".format(i)
    train_xyz, train_lat = get_xyz_data(fn)
    
    ga_list = []
    al_list = []
    o_list = []
    in_list = []
    
    for li in train_xyz:
        try:
            if li[1] == "Ga":
                ga_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "Al":
                al_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "In":
                in_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "O":
                o_list.append(li[0])
        except:
            pass
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
       
    except:
        temp_ga = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
#         print i
    except:
        temp_al = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
#         print i
    except:
        temp_o = [0,0,0,0,0,0]
#         print i
    
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
#         print i
    except:
        temp_in = [0,0,0,0,0,0]
#         print i

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])
    
    

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])    
    in_df = pd.concat([in_df,temp_in])
    
ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

train = pd.merge(train,ga_df,on = ["id"],how = "left")
train = pd.merge(train,al_df,on = ["id"],how = "left")
train = pd.merge(train,o_df,on = ["id"],how = "left")
train = pd.merge(train,in_df,on = ["id"],how = "left")

ga_df= pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns= o_cols)
in_df = pd.DataFrame(columns=in_cols)

for i in test.id.values:
    fn = "../input/nomad2018-predict-transparent-conductors/test/{}/geometry.xyz".format(i)
    test_xyz, test_lat = get_xyz_data(fn)
    
    ga_list = []
    al_list = []
    o_list = []
    in_list = []
    
    for li in test_xyz:
        try:
            if li[1] == "Ga":
                ga_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "Al":
                al_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "In":
                in_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "O":
                o_list.append(li[0])
        except:
            pass
    
#     ga_list = [item for sublist in ga_list for item in sublist]
#     al_list = [item for sublist in al_list for item in sublist]
#     o_list = [item for sublist in o_list for item in sublist]
   
    
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
       
    except:
        temp_ga = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
#         print i
    except:
        temp_al = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
#         print i
    except:
        temp_o = [0,0,0,0,0,0]
#         print i
    
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
#         print i
    except:
        temp_in = [0,0,0,0,0,0]
#         print i

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])
    
    

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])    
    in_df = pd.concat([in_df,temp_in])
    

ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

test = pd.merge(test,ga_df,on = ["id"],how = "left")
test = pd.merge(test,al_df,on = ["id"],how = "left")
test = pd.merge(test,o_df,on = ["id"],how = "left")
test = pd.merge(test,in_df,on = ["id"],how = "left")
train.rename(columns={
    'spacegroup' : 'sg',
    'number_of_total_atoms' : 'Natoms',
    'percent_atom_al' : 'x_Al',
    'percent_atom_ga' : 'x_Ga',
    'percent_atom_in' : 'x_In',
    'lattice_vector_1_ang' : 'a',
    'lattice_vector_2_ang' : 'b',
    'lattice_vector_3_ang' : 'c',
    'lattice_angle_alpha_degree' : 'alpha',
    'lattice_angle_beta_degree' : 'beta',
    'lattice_angle_gamma_degree' : 'gamma',
    'formation_energy_ev_natom' : 'E',
    'bandgap_energy_ev' : 'Eg'}, inplace=True)

test.rename(columns={
    'spacegroup' : 'sg',
    'number_of_total_atoms' : 'Natoms',
    'percent_atom_al' : 'x_Al',
    'percent_atom_ga' : 'x_Ga',
    'percent_atom_in' : 'x_In',
    'lattice_vector_1_ang' : 'a',
    'lattice_vector_2_ang' : 'b',
    'lattice_vector_3_ang' : 'c',
    'lattice_angle_alpha_degree' : 'alpha',
    'lattice_angle_beta_degree' : 'beta',
    'lattice_angle_gamma_degree' : 'gamma',
}, inplace=True)


target = [
    'E',
    'Eg']

all_data = pd.concat((train, test))
def get_prop_list(path_to_element_data):
    """
    Args:
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        list of elemental properties (str) which have corresponding .csv files
    """
    return [f[:-4] for f in os.listdir(path_to_element_data)]

# folder which contains element data
path_to_element_data = '../input/elemental-properties/'
# get list of properties which have data files
properties = get_prop_list(path_to_element_data)
print(sorted(properties))
def get_prop(prop, path_to_element_data):
    """
    Args:
        prop (str) - name of elemental property
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        dictionary of {element (str) : property value (float)}
    """
    fin = os.path.join(path_to_element_data, prop+'.csv')
    with open(fin) as f:
        all_els = {line.split(',')[0] : float(line.split(',')[1][:-1]) for line in f}
        my_els = ['Al', 'Ga', 'In']
        return {el : all_els[el] for el in all_els if el in my_els}

# make nested dictionary which maps {property (str) : {element (str) : property value (float)}}
prop_dict = {prop : get_prop(prop, path_to_element_data) for prop in properties}
print('The mass of aluminum is %.2f amu' % prop_dict['mass']['Al'])
# average each property using the composition

def avg_prop(x_Al, x_Ga, x_In, prop):
    """
    Args:
        x_Al (float or DataFrame series) - concentration of Al
        x_Ga (float or DataFrame series) - concentration of Ga
        x_In (float or DataFrame series) - concentration of In
        prop (str) - name of elemental property
    Returns:
        average property for the compound (float or DataFrame series), 
        weighted by the elemental concentrations
    """
    els = ['Al', 'Ga', 'In']
    concentration_dict = dict(zip(els, [x_Al, x_Ga, x_In]))
    return np.sum(prop_dict[prop][el] * concentration_dict[el] for el in els)

# add averaged properties to DataFrame
for prop in properties:
    all_data['_'.join(['avg', prop])] = avg_prop(all_data['x_Al'], 
                                                 all_data['x_Ga'],
                                                 all_data['x_In'],
                                                 prop)
# calculate the volume of the structure

def get_vol(a, b, c, alpha, beta, gamma):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    return a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)

# convert lattice angles from degrees to radians for volume calculation
lattice_angles = ['alpha', 'beta', 'gamma']
for lang in lattice_angles:
    all_data['_'.join([lang, 'r'])] = np.pi * all_data[lang] / 180
    
# compute the cell volumes 
all_data['vol'] = get_vol(all_data['a'], all_data['b'], all_data['c'],
                          all_data['alpha_r'], all_data['beta_r'], all_data['gamma_r'])

# calculate the atomic density
all_data['atomic_density'] = all_data['Natoms'] / all_data['vol']  
all_data.head()
for col in ['x_Al', 'x_Ga', 'x_In', 'a', 'b', 'c', 'vol', 'atomic_density']:
    for x in all_data.sg.unique():
        sns.distplot(all_data[all_data['sg'] == x][col])
    plt.title(col)
    plt.show()
# make new features using averages of the following columns by sg group
avg_cols = ['x_Al','x_Ga','x_In','a','b','c','avg_rs_max','avg_electronegativity',
            'avg_rp_max','avg_LUMO','avg_IP','avg_rd_max','avg_EA','avg_HOMO',
            'avg_mass','vol','atomic_density']


for col in avg_cols:
    new_col = col + "_avg"
    all_data[new_col] = np.nan
    for group in all_data['sg'].unique():
        all_data.loc[(all_data['sg'] == group), new_col] = all_data[(all_data['sg'] == group)][col].mean()

print('Number of Null Values: {}'.format(pd.isnull(all_data[avg_cols]).sum().sum()))
# Handle the values with categorical variables using one hot encoding
# This will create a much more sparse set of variables

all_data[['sg', 'Natoms']] = all_data[['sg', 'Natoms']].astype(str)
all_data = pd.get_dummies(all_data)
#  both of the target variables are skewed a bit

for col in ['E', 'Eg']:
    sns.distplot((train[col]))
    plt.title(col)
    plt.show()
all_data.columns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# features to use
features = ['x_Al', 'x_Ga', 'x_In', 'a', 'b', 'c', 'alpha', 'beta',
            'gamma', 'vol', 'atomic_density', 'x_Al_avg','x_Ga_avg', 'x_In_avg', 'a_avg',
            'b_avg', 'c_avg', 'vol_avg', 'atomic_density_avg', 'pca_abc', 'pca_AlGaInDensity',
            'O_0_0','O_1_0', 'O_2_0', 'O_3_0', 'O_4_0', 'O_5_0', 'Al_0_0', 'Al_1_0', 'Al_2_0', 'Al_3_0', 'Al_4_0', 'Al_5_0', 'Ga_0_0',
            'Ga_1_0', 'Ga_2_0', 'Ga_3_0', 'Ga_4_0', 'Ga_5_0', 'In_0_0', 'In_1_0',
            'In_2_0', 'In_3_0', 'In_4_0', 'In_5_0',]

# two different vectors for pca
vector1 = all_data[['a', 'b', 'c']].values
vector2 = all_data[['x_Al', 'x_Ga', 'x_In', 'atomic_density_avg']].values

# use pca to add new features
pca = PCA()
pca.fit(vector1)
all_data['pca_abc'] = pca.transform(vector1)[:,0]

pca = PCA()
pca.fit(vector2)
all_data['pca_AlGaInDensity'] = pca.transform(vector2)[:,0]

# scaling the data. Linear models tend to like more normally distributed
# I tried training on non-scaled, with slightly worse results
scale = StandardScaler()
scaled = scale.fit(all_data[features]).transform(all_data[features])

X_scale = scaled[:train.shape[0]]
X_scaled_test = scaled[train.shape[0]:]

X_tr = all_data[:train.shape[0]][features].values
X_te = all_data[train.shape[0]:][features].values

y1 = np.log1p(train['E'])
y2 = np.log1p(train['Eg'])

y12 = np.column_stack((y1, y2))

X_tr.shape, y1.shape, y2.shape, y12.shape, X_scaled_test.shape
# performance matric
def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    
#     h, y = np.expm1(h), np.expm1(y)
    
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
import keras
import tensorflow as tf
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization, Input
def get_model(shape):
    """ Returns a model of specific shape
    """
    X_input = Input(shape=(shape,))
    
    X = Dense(64, activation='relu')(X_input)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(64, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(64, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(64, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(64, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dense(2, activation='linear')(X)

    
    return Model(inputs=X_input, outputs=X)

# computes RMSLE from tensorflow
def rmsle_K(y, y0):
    return K.sqrt(K.mean(K.square(tf.log1p(tf.expm1(y)) - tf.log1p(tf.expm1(y0)))))

def compile_model(shape, lr=0.001):
    model = get_model(shape)
    optimizer = Adam(lr=lr, decay=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=[rmsle_K])
    return model
# uncomment below to train the model.

# out = []
# for idx, label in enumerate([0.002, 0.0022, 0.0024, 0.0026]):
#     model = compile_model(X_scale.shape[1], lr=label)
#     evl = model.fit(x=X_scale, y=y12, epochs=600, batch_size=16, verbose=2)
#     out.append(evl.history.get('rmsle_K'))
#     print(str(label) + " rmsle: {}".format(np.mean(out[idx][180:])))
# Will assess the model using cross validation.
# This will take a long time depending on your hardwear
def assess_nn(X, y, params):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    kfold = KFold(n_splits=2, shuffle=True)
    out = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = compile_model(X_scale.shape[1], lr=0.003)
        model.fit(x=X_scale, y=y12, epochs=600, batch_size=16, verbose=0)
        h =  model.predict(X_test)
        e = rmsle(np.expm1(h), np.expm1(y_test))
        print(e)
        out.append(e)
    final.append(np.array(out).mean())
                      
    print('y1 best iteration: {}'.format(np.mean(best_iter[0])))
    print('y2 best iteration: {}'.format(np.mean(best_iter[1])))
    return(np.array(final).mean(), np.array(final).std())
# uncomment to assess model

# model = access(X_scale, y12, params)
# print(model)
# if you run the access function above, use the plotting below to plot the last 40 rows of each kfold

# for x in out:
#     plt.plot(x[-40:])
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
# uses baysian optimization to find model parameters

# model = GradientBoostingRegressor(
#     loss='ls',
#     learning_rate = 0.0035,
#     max_depth=23,
#     n_estimators=30275,
#     max_features=9,
#     min_samples_leaf=22,
#     min_samples_split=15,
#     min_weight_fraction_leaf=0.0102470171519909
# )

# search_params = {
#     "n_estimators": Integer(1000, 4000),
#     'max_depth': Integer(2, 40),
#     'min_samples_split': Integer(2, 15),
#     'min_samples_leaf': Integer(2, 50),
#     'min_weight_fraction_leaf': Real(0., .5),
#     'max_features': Integer(2, 13)
# }

# opt = BayesSearchCV(model, search_params, n_iter=50, n_jobs=8, )
# opt.fit(X_scale, y2)
# opt.best_params_
# run different model for different Target Variables

grad_1 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=7,
                n_estimators=1120,
                max_features=7,
                min_samples_leaf=43,
                min_samples_split=14,
                min_weight_fraction_leaf=0.01556)

grad_2 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=6,
                n_estimators=3275,
                max_features=2,
                min_samples_leaf=2,
                min_samples_split=2,
                min_weight_fraction_leaf=0.08012)

def assess_grad(X, y_list, model_list):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_list[idx]
            model.fit(X_train, y_train)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print(e)
            out.append(e)
        final.append(np.array(out).mean())
                      
    return(np.array(final).mean(), np.array(final).std())
model = assess_grad(X_tr, [y1, y2], [grad_1, grad_2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
import lightgbm as lgb
from lightgbm import Dataset
# find useful parameters for model

# model = lgb.LGBMRegressor(
#                 objective= 'regression',
#                 boosting_type= 'gbdt',
#                 learning_rate= 0.002,
#                 num_boost_round = 2000,
#                 num_threads=1,
#                 bagging_fraction=0.50173,
#                 bagging_freq= 14,
#                 feature_fraction= 0.62509,
#                 lambda_l2= 0.0086298,
# #                 max_depth=10,
#                 num_leaves=196
#             )

# search_params = {
#     'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': {'rmse', 'rmsle'},
#         'max_depth': Integer(20, 100),
#         'num_leaves': Integer(100, 200),
#         'learning_rate': 0.010,
#         'feature_fraction': Real(0.5, 1.),
#         'bagging_fraction': Real(0.5, 1),
#         'bagging_freq': Integer(5, 15),
#         'num_threads': -1,
#         'lambda_l2': Real(.00001, 0.01, 'log_normal'),
#             'lambda_l1': Real(.00001, 0.01, 'log_normal'),
#     'num_iterations': Integer(1000, 4000)
# 

# opt = BayesSearchCV(model, search_params, n_iter=50, n_jobs=-1, verbose=0)
# opt.fit(X_scale, y1)
# opt.best_params_
# Run accessment using parameters found below

lgb_1 = lgb.LGBMRegressor(
                objective= 'regression',
                boosting_type= 'gbdt',
                learning_rate= 0.002,
                n_estimators = 2000,
                num_threads=3,
                bagging_fraction=0.56369,
                bagging_freq= 14,
                feature_fraction= 0.88868,
                lambda_l2= 0.0091689,
                max_depth=20,
                )

lgb_2 = lgb.LGBMRegressor(
                objective= 'regression',
                boosting_type= 'gbdt',
                learning_rate= 0.002,
                n_estimators = 2838,
                num_threads=3,
                bagging_fraction=0.50173,
                bagging_freq= 14,
                feature_fraction= 0.62509,
                lambda_l2= 0.0086298,
                max_depth=20,
                )

# This access uses the built in early stopping functions that make light gmb so great.

def assess_light(X, y_list, model_list):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_list[idx]
            model.fit(X_train, y_train)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())

    return(np.array(final).mean(), np.array(final).std())
model = assess_light(X_scale, [y1, y2], [lgb_1, lgb_2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))

def assess_early_stop_light(X, y_list, params):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
            model = lgb.train(params[idx],
                            lgb_train,
                            num_boost_round=1000000,
                            valid_sets=[lgb_valid],
                            early_stopping_rounds=100,
                            verbose_eval=0) 
            best_iter[idx].append(model.best_iteration)
            h =  model.predict(X_test, num_iteration=model.best_iteration)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())
                      
    print('y1 best iteration: {}'.format(np.mean(best_iter[0])))
    print('y2 best iteration: {}'.format(np.mean(best_iter[1])))
    return(np.array(final).mean(), np.array(final).std())

params1 = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse', 'rmsle'},
            'max_depth': 10,
            'learning_rate': 0.010,
            'feature_fraction': 0.8632,
            'bagging_fraction': 0.8759,
            'bagging_freq': 4,
            'verbose': 0,
            'verbose_eval':0,
            'num_threads': 3,
            'lambda_l2': 0.0005597442104287973,
            'lambda_l1': 0.00015997163552092318 
        }

params2 = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse', 'rmsle'},
            'learning_rate': 0.010,
            'verbose': 0,
            'verbose_eval':0,
            'num_threads': 3,
            'bagging_fraction': 0.9311539021934098,
             'bagging_freq': 15,
             'feature_fraction': 0.9989744117209727,
             'lambda_l1': 8.337666829263869e-05,
             'lambda_l2': 0.005541689229153562,
             'max_depth': 19
        }

model = assess_early_stop_light(X_tr, [y1, y2], [params1, params2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
from xgboost import XGBRegressor
# find useful parameters for model

# model = XGBRegressor(
#     silent=True,
#     learning_rate= 0.0050,
#     n_jobs=1
# )

# search_params1 = {

#     'n_estimators': Integer(1804, 1806),
#     'gamma': Real(0, 100),
#     'max_depth' : Integer(15, 100),
#     'min_child_weight': Integer(10, 1000),
#     'max_delta_step': Integer(1, 100),
#     'subsample': Real(0, 1),
#     'colsample_bytree': Real(0.0001, 1),
#     'colsample_bylevel': Real(0.0001, 1),
#     'reg_lambda': Real(0.000000001, 1.0),
# }

# xgb_params_1 = {
#     'learning_rate':0.005,
#     'n_jobs':-1,
#     'n_estimators': 1804,
#     'gamma': 0.0,
#     'subsample': 0.749700,
#     'colsample_bytree': 1.0,
#     'colsample_bylevel': 0.2790166932949295,
#     'max_delta_step': 1,
#     'max_depth': int(15),
#     'min_child_weight': 30,
#     'reg_lambda': 1e-09,
#     'silent': True,
#     'n_jobs': 8}

# search_params2 = {

#     'n_estimators': Integer(2383, 2400),
# #     'gamma': Real(0, 100),
#     'max_depth' : Integer(15, 100),
#     'min_child_weight': Integer(10, 100),
#     'max_delta_step': Integer(1, 200),
#     'subsample': Real(0, 1),
#     'colsample_bytree': Real(0.01, 1),
#     'colsample_bylevel': Real(0.01, 1),
#     'reg_lambda': Real(0.0000001, 0.5),
    
# }

# xgb_params_2 = {
#     'learning_rate':0.005,
#     'n_jobs':-1,
#     'n_estimators': 2383,
#     'colsample_bylevel': 0.982522,
#     'colsample_bytree': 1.0,
#     'gamma': 0.0,
#     'max_depth': 15,
#     'min_child_weight': 63,
#     'colsample_bytree': 0.4254153401195336,
#     'max_delta_step': 65,
#     'reg_lambda': 0.031165789070644215,
#     'subsample': 0.6831707073621087,
#     'silent':True,
#     'n_jobs':8,
# }

# for params, y, i in zip([search_params1, search_params2], [y1, y2], ['1', '2']):
#     opt = BayesSearchCV(model, params, n_iter=100, n_jobs=-1, verbose=0)
#     opt.fit(X_tr, y)
#     print('y' + i, opt.best_params_)
xgb_1 = XGBRegressor(
    learning_rate=0.005,
    n_jobs=3,
    n_estimators= 1804,
    gamma= 0.0,
    subsample= 0.222159,
    colsample_bytree= 0.5359,
    colsample_bylevel= 0.19958,
    max_delta_step= 64,
    max_depth=28,
    min_child_weight= 10,
    reg_lambda=0.33038,
    silent= True,
)

xgb_2 = XGBRegressor(
    learning_rate=0.005,
    n_jobs=3,
    n_estimators= 2386,
    gamma= 0.0,
    subsample= 0.90919,
    colsample_bytree= 0.59049,
    colsample_bylevel= 0.59404,
    max_delta_step= 99,
    max_depth=58,
    min_child_weight= 85,
    reg_lambda= 0.031165789070644215,
    silent= True,
)
def assess_xgb(X, y_list, model_num):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_num[idx]
            model.fit(X_train, y_train)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())
    return(np.array(final).mean(), np.array(final).std())
model = assess_xgb(X_scale, [y1, y2], [xgb_1, xgb_2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
from xgboost import DMatrix
import xgboost as xgb
from sklearn.model_selection import train_test_split

xgb_param_1 = {
    'learning_rate': 0.005,
    'n_jobs': 3,
    'gamma': 0.0,
    'subsample': 0.222159,
    'colsample_bytree': 0.5359,
    'colsample_bylevel': 0.19958,
    'max_delta_step': 64,
    'max_depth': 28,
    'min_child_weight': 10,
    'reg_lambda': 0.33038,
    'silent':  True
}
    
xgb_param_2 = {
    'learning_rate': 0.005,
    'n_jobs': 3,
    'gamma': 0.0,
    'subsample': 0.90919,
    'colsample_bytree': 0.59049,
    'colsample_bylevel': 0.59404,
    'max_delta_step': 99,
    'max_depth': 58,
    'min_child_weight': 85,
    'reg_lambda': 0.031165789070644215,
    'silent': True
}

def assess_early_cv_stop_xgb(X, y_list, param_list):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_iter, X_test = X[train_index], X[test_index]
            y_iter, y_test = y[train_index], y[test_index]
            X_train, X_valid, y_train, y_valid = train_test_split(X_iter, y_iter, test_size=0.3)
            
            xgb_train = DMatrix(X_train, label=y_train)
            xgb_valid = DMatrix(X_valid, y_valid)
            xgb_test = DMatrix(X_test, y_test)
            watchlist = [ (xgb_train,'train'), (xgb_valid, 'valid') ]
            model = xgb.train(param_list[idx], xgb_train, 1000000,
                              watchlist, early_stopping_rounds=20, verbose_eval=False)
            
            h =  model.predict(xgb_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())
    return(np.array(final).mean(), np.array(final).std())
model = assess_early_cv_stop_xgb(X_tr, [y1, y2], [xgb_param_1, xgb_param_2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
from catboost import CatBoostRegressor
# I found these parameterw worked for both y variables
cat_1 = CatBoostRegressor(iterations=2300,
                          learning_rate=0.020,
                          depth=5,
                          loss_function='RMSE',
                          eval_metric='RMSE',
                          od_type='Iter',
                          od_wait=50,
                         )

def assess_cat(X, y_list, model_num):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_num[idx]
            model.fit(X_train, y_train, verbose=False)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())
    return(np.array(final).mean(), np.array(final).std())
model = assess_cat(X_tr, [y1, y2], [cat_1, cat_1])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
from sklearn.model_selection import train_test_split

catboost_cv = CatBoostRegressor(iterations=1200,
                            learning_rate=0.03,
                            depth=5,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50)
    
def assess_cv_catboost(X, y_list):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            # splitting the data up into train, test, and valid sets
            X_iter, X_test = X[train_index], X[test_index]
            y_iter, y_test = y[train_index], y[test_index]
            X_train, X_valid, y_train, y_valid = train_test_split(X_iter, y_iter, test_size=0.3)
            model =  catboost_cv
            model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      use_best_model=True,
                      verbose=False)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print('RMSLE: {}'.format(e))
            out.append(e)
        final.append(np.array(out).mean())
    return(np.array(final).mean(), np.array(final).std())
model = assess_cv_catboost(X_tr, [y1, y2])
print("Model RMSLE: {}, std: {}".format(model[0], model[1]))
from sklearn.linear_model import  LinearRegression
from sklearn.base import clone

class StackingAveragedModels():
    def __init__(self, base_models, meta_model, n_folds=15):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                try:
                    instance.fit(X[train_index], y[train_index], verbose=False)
                except:
                    instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
# stacking with early stopping for all models

from sklearn.linear_model import  LinearRegression
from sklearn.base import clone
import sklearn
import xgboost
import catboost
import lightgbm

class StackingStoppingAveragedModels():
    def __init__(self, base_models, meta_model, n_folds=10):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):

                 # all models support early stopping in one way and need a valid set for it
                X_train, X_valid, y_train, y_valid = train_test_split(X[train_index],
                                                                      y[train_index],
                                                                      test_size=0.3)
                instance = clone(model)
                
                if type(instance) == sklearn.ensemble.gradient_boosting.GradientBoostingRegressor:
                    instance.min_impurity_decrease = 0.01
                    
                    # use all data since grad boosting's "early stopping" is a parameter on leafs
                    instance.fit(X[train_index], y[train_index])
                    
                elif type(instance) == xgboost.sklearn.XGBRegressor:
                    instance.n_estimators = 10000000
                    instance.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="rmse",
                                 eval_set=[(X_valid, y_valid)], verbose=False)
                    
                elif type(instance) == catboost.core.CatBoostRegressor:
                        instance.fit(X_train, y_train, eval_set=[X_valid, y_valid],
                                     use_best_model=True, verbose=False)
                        
                elif type(instance) == lightgbm.sklearn.LGBMRegressor:
                        instance.n_estimators = 10000000
                        instance.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                                     early_stopping_rounds=100, verbose=False)
                    
                self.base_models_[i].append(instance)
                
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
# stacking for y1
cat_1 = CatBoostRegressor(iterations=1400,
                            learning_rate=0.03,
                            depth=5,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            od_type='Iter',
                            od_wait=50)
xgb_1 = XGBRegressor(
    learning_rate=0.005,
    n_jobs=3,
    n_estimators= 1804,
    gamma= 0.0,
    subsample= 0.222159,
    colsample_bytree= 0.5359,
    colsample_bylevel= 0.19958,
    max_delta_step= 64,
    max_depth=28,
    min_child_weight= 10,
    reg_lambda=0.33038,
    silent= True)

lgb_1 = lgb.LGBMRegressor(
                objective= 'regression',
                boosting_type= 'gbdt',
                learning_rate= 0.002,
                n_estimators = 2000,
                num_threads=3,
                bagging_fraction=0.56369,
                bagging_freq= 14,
                feature_fraction= 0.88868,
                lambda_l2= 0.0091689,
                max_depth=96,
                )

grad_1 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=7,
                n_estimators=1120,
                max_features=7,
                min_samples_leaf=43,
                min_samples_split=14,
                min_weight_fraction_leaf=0.01556)
linear = LinearRegression()
cat_2 = CatBoostRegressor(iterations=1200,
                            learning_rate=0.03,
                            depth=5,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            od_type='Iter',
                            od_wait=50)
lgb_2 = lgb.LGBMRegressor(
                objective= 'regression',
                boosting_type= 'gbdt',
                learning_rate= 0.002,
                n_estimators = 2838,
                num_threads=3,
                bagging_fraction=0.50173,
                bagging_freq= 14,
                feature_fraction= 0.62509,
                lambda_l2= 0.0086298,
                max_depth=20
                )

grad_2 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=6,
                n_estimators=3275,
                max_features=2,
                min_samples_leaf=2,
                min_samples_split=2,
                min_weight_fraction_leaf=0.08012)

xgb_2 = XGBRegressor(
    learning_rate=0.005,
    n_jobs=3,
    n_estimators= 2386,
    gamma= 0.0,
    subsample= 0.90919,
    colsample_bytree= 0.59049,
    colsample_bylevel= 0.59404,
    max_delta_step= 99,
    max_depth=58,
    min_child_weight= 85,
    reg_lambda= 0.031165789070644215,
    silent= True,
)
# without early stopping
stacked_1 = StackingAveragedModels(base_models=[xgb_1, lgb_1, grad_1, cat_1], meta_model=linear)
stack_1 = stacked_1.fit(X_scale, y1)

stacked_2 = StackingAveragedModels(base_models=[xgb_2, lgb_2, grad_2, cat_1], meta_model=linear)
stack_2 = stacked_2.fit(X_scale, y2)
# with early stopping
# uncomment to run this

# stacked_stop_1 = StackingStoppingAveragedModels(base_models=[xgb_1, lgb_1, grad_1, cat_1], meta_model=linear)
# stack_1 = stacked_stop_1.fit(X_scale, y1)

# stacked_stop_2 = StackingStoppingAveragedModels(base_models=[xgb_2, lgb_2, grad_2, cat_1], meta_model=linear)
# stack_2 = stacked_stop_2.fit(X_scale, y2)
all_data
one = stack_1.predict(X_scaled_test)
two = stack_2.predict(X_scaled_test)
_id = test['id']

submit = pd.DataFrame({'id':_id,
                       'formation_energy_ev_natom':np.expm1(one).flatten(),
                       'bandgap_energy_ev':np.expm1(two).flatten()})
submit.head()
submit.to_csv('submission.csv', index=False)