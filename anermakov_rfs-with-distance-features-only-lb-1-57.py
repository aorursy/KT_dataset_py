EXP_NUMBER = 63
import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')

df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

df_struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
df_train.shape, df_test.shape, df_struct.shape
from sklearn.preprocessing import LabelEncoder
type_enc = LabelEncoder()



df_train['type'] = type_enc.fit_transform(df_train['type'])

df_test['type'] = type_enc.transform(df_test['type'])
atom_enc = LabelEncoder()



df_struct['atom'] = atom_enc.fit_transform(df_struct['atom'])
df_train = df_train.merge(df_struct, 

                          left_on=['molecule_name','atom_index_0'], 

                          right_on=['molecule_name','atom_index'])
df_train = df_train.merge(df_struct, 

                          left_on=['molecule_name','atom_index_1'], 

                          right_on=['molecule_name','atom_index'],

                          suffixes=('_0', '_1'))
df_train.columns
n_neighbors = 5



neigh_da_cols = [f'da_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) for typ in range(5)]

neigh_d0_cols = [f'd0_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) for typ in range(5)]

neigh_d1_cols = [f'd1_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) for typ in range(5)]

neigh_cosa_cols = [f'cosine_da_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) 

                   for typ in range(5)]

neigh_cos0_cols = [f'cosine_d0_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) 

                   for typ in range(5)]

neigh_cos1_cols = [f'cosine_d1_typ{typ}_sep2_neighbor{i}' for i in range(1, n_neighbors+1) 

                   for typ in range(5)]

neigh_all_cols = neigh_d0_cols + neigh_d1_cols # + neigh_da_cols

#                   neigh_cosa_cols

#                   neigh_cos0_cols

#                   neigh_cos1_cols

neigh_cols = ['id'] + neigh_all_cols
def load_frames(files, columns, parent_folder='../input/dist-feats'):

    df_neigh = pd.DataFrame()

    for name in files:

        print(f'Loading {name} ...')

        dfn = pd.read_csv(f'{parent_folder}/{name}', usecols=columns)

        df_neigh = pd.concat([df_neigh, dfn])

        del(dfn)

    return df_neigh
df_train_neigh_files = [

'top-5_atoms-[\'sep2\']_count-1164536_44c3ddd5-0dd4-42b7-acac-d0a0e7bd444e.csv',

'top-5_atoms-[\'sep2\']_count-1164536_7e41a7b8-d1ed-4d11-be5e-5166d347a8a2.csv',

'top-5_atoms-[\'sep2\']_count-1164536_e2e300a0-55c4-453a-b40d-b62e4cf7b336.csv',

'top-5_atoms-[\'sep2\']_count-1164539_42ba6898-a223-4a78-8157-b1f015c09806.csv',

]



df_neigh = load_frames(df_train_neigh_files, neigh_cols)
df_train = df_train.merge(df_neigh, on='id')
df_train.shape
n_neighbors = 5



neigh_dm_cols_2 = [f'dm_med2_neighbor{i}' for i in range(1, n_neighbors+1)]

neigh_d0_cols_2 = [f'd0_med2_neighbor{i}' for i in range(1, n_neighbors+1)]

neigh_d1_cols_2 = [f'd1_med2_neighbor{i}' for i in range(1, n_neighbors+1)]

neigh_cos0_cols_2 = [f'cos0_med2_neighbor{i}' for i in range(1, n_neighbors+1)]

neigh_all_cols_2 = neigh_d0_cols_2 + neigh_d1_cols_2 # + neigh_dm_cols_2 + neigh_cos0_cols_2

neigh_cols_2 = ['id'] + neigh_all_cols_2
df_train_neigh_files = [

'top-5_atoms-[\'med2\']_count-1164536_0a918683-2fe3-43a1-b239-c900712c1433.csv',

'top-5_atoms-[\'med2\']_count-1164536_b845e873-855d-424b-a7f3-50d89824b0ea.csv',

'top-5_atoms-[\'med2\']_count-1164536_df446883-d0af-486c-877b-f8d834c0b00f.csv',

'top-5_atoms-[\'med2\']_count-1164539_de6e2745-1a72-490c-ac29-69ad07c56910.csv',

]



df_neigh = load_frames(df_train_neigh_files, neigh_cols_2)
df_train = df_train.merge(df_neigh, on='id')
df_train.shape
df_train.set_index('id', inplace=True)

df_train.sort_index(inplace=True)
df_train.columns
df_train['atom_0'].value_counts(), df_train['atom_1'].value_counts()
df_train = df_train.drop(['atom_0'], axis=1)
df_test = df_test.merge(df_struct, 

                        left_on=['molecule_name','atom_index_0'], 

                        right_on=['molecule_name','atom_index'],

                        sort=False)
df_test = df_test.merge(df_struct, 

                        left_on=['molecule_name','atom_index_1'], 

                        right_on=['molecule_name','atom_index'],

                        suffixes=('_0', '_1'),

                        sort=False)
df_test_neigh_files = [

'top-5_atoms-[\'sep2\']_count-626385_7623bd50-41b8-4c44-b40a-e37837700460.csv',

'top-5_atoms-[\'sep2\']_count-626385_878ae0ca-67c7-491e-b580-fbdc384c2477.csv',

'top-5_atoms-[\'sep2\']_count-626385_e4f75a2d-302c-476b-b60c-c5624925ef01.csv',

'top-5_atoms-[\'sep2\']_count-626387_4792052c-d539-4e62-8e37-c4ecf81ef669.csv',

]



df_neigh = load_frames(df_test_neigh_files, neigh_cols)
df_test = df_test.merge(df_neigh, on='id')
df_test.shape
df_test_neigh_files = [

'top-5_atoms-[\'med2\']_count-626385_0072225a-03f6-42bf-b1ed-b81be41c6d14.csv',

'top-5_atoms-[\'med2\']_count-626385_7ed7d305-6c6a-4d3c-a6e8-f8e274fd461f.csv',

'top-5_atoms-[\'med2\']_count-626385_b4f9dd09-7c61-42a7-9534-85a31496d1a5.csv',

'top-5_atoms-[\'med2\']_count-626387_a9d566a5-1248-4341-abec-b71a87bbe21d.csv',

]



df_neigh = load_frames(df_test_neigh_files, neigh_cols_2)
df_test = df_test.merge(df_neigh, on='id')
df_test.shape
df_test.set_index('id', inplace=True)

df_test.sort_index(inplace=True)
df_test.columns
df_test['atom_0'].value_counts(), df_test['atom_1'].value_counts()
df_test = df_test.drop(['atom_0'], axis=1)
del(df_struct)

del(df_neigh)
def calc_polar(df, suffix_0='_0', suffix_1='_1', angles=True):

    '''Calculate polar coords

    '''

    dx = df['x'+suffix_0] - df['x'+suffix_1]

    dy = df['y'+suffix_0] - df['y'+suffix_1]

    dz = df['z'+suffix_0] - df['z'+suffix_1]

    if angles:

        dx2 = np.power(dx, 2)

        dy2 = np.power(dy, 2)

        dz2 = np.power(dz, 2)

        d = np.sqrt(dx2 + dy2 + dz2)

        az = np.arctan2(dy, dx)

        el = np.arctan2(dz, np.sqrt(dx2 + dy2))

        return d, az, el

    else:

        return np.sqrt(dx**2 + dy**2 + dz**2)
def feat_eng(df):

    d, _, _ = calc_polar(df)

    df_dummy = pd.get_dummies(df[['atom_1']], columns=['atom_1'])

    atom_1_cols = [x for x in df_dummy.columns if 'atom_1' in x]

    d_cols = [f'd_{x[7:]}' for x in atom_1_cols]

    df[d_cols] = df_dummy[atom_1_cols].multiply(d, axis='index')
feat_eng(df_train)

feat_eng(df_test)
import matplotlib.pyplot as plt
%%time

corr = df_train.corr()
plt.figure(figsize=(10,10))

plt.matshow(corr, fignum=1, cmap='coolwarm')

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns)

plt.colorbar()

plt.show()
del(corr)
target = 'scalar_coupling_constant'
d_cols = [x for x in df_train.columns if x.startswith('d_') and len(x) == 3]

feat_cols = d_cols + neigh_all_cols + neigh_all_cols_2
train_types = df_train['type'].values

train_types.shape
X_train = df_train[feat_cols].values

X_train.shape
y_train = df_train[target].values
test_types = df_test['type'].values

test_types.shape
X_test = df_test[feat_cols].values

X_test.shape
del(df_train)

del(df_test)
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp





def split(X, y, types, random_state=1, p_value=0.9, test_size=0.2, verbose=True):

    X_train, X_test, y_train, y_test, train_types, test_types = None, None, None, None, None, None



    while True:

        X_train, X_test, y_train, y_test, train_types, test_types = train_test_split(

            X, y, types, 

            test_size=test_size, 

            stratify=types, 

            random_state=random_state

        )

        if p_value is None:

            return X_train, X_test, y_train, y_test, train_types, test_types

        st1 = ks_2samp(y, y_train)

        st2 = ks_2samp(y, y_test)

        st3 = ks_2samp(y_train, y_test)

        if verbose:

            print('RS =', random_state)

            print(st1)

            print(st2)

            print(st3)

        if st1.pvalue > p_value and st2.pvalue > p_value and st3.pvalue > p_value:

            return X_train, X_test, y_train, y_test, train_types, test_types

        del(X_train)

        del(X_test)

        del(y_train)

        del(y_test)

        del(train_types)

        del(test_types)

        random_state += 1
X_train_part, X_valid, y_train_part, y_valid, train_part_types, valid_types = split(X_train, y_train, train_types, 

                                                                                    random_state=42, p_value=0.0)
X_valid.shape, y_valid.shape, valid_types.shape
X_train_small, _, y_train_small, _, train_small_types, _ = split(X_train, y_train, train_types, test_size=0.90) 
X_train_small.shape, y_train_small.shape
X_train.shape, X_train_part.shape, X_valid.shape
# %%time

# from sklearn.ensemble import RandomForestRegressor

# from boruta import BorutaPy



# reg = RandomForestRegressor(n_estimators=12, n_jobs=-1, random_state=42)

# fs = BorutaPy(reg, max_iter=10, n_estimators=12, verbose=2, random_state=42)



# fs.fit(X_train_small, y_train_small)  



# feat_cols_dropped = sorted([feat_cols[i] for i in range(len(feat_cols)) if not fs.support_[i]])

# print('Feats dropped:', len(feat_cols_dropped))



# feat_cols_selected = sorted([feat_cols[i] for i in range(len(feat_cols)) if fs.support_[i]])

# print('Feats selected:', len(feat_cols_selected))



# feat_cols_selected



feat_cols_selected = [

 'd0_med2_neighbor1',

 'd0_med2_neighbor2',

 'd0_med2_neighbor3',

 'd0_med2_neighbor4',

 'd0_med2_neighbor5',

 'd0_typ0_sep2_neighbor1',

 'd0_typ0_sep2_neighbor2',

 'd0_typ0_sep2_neighbor3',

 'd0_typ0_sep2_neighbor4',

 'd0_typ0_sep2_neighbor5',

 'd0_typ2_sep2_neighbor1',

 'd0_typ2_sep2_neighbor2',

 'd0_typ2_sep2_neighbor3',

 'd0_typ2_sep2_neighbor4',

 'd0_typ2_sep2_neighbor5',

 'd0_typ3_sep2_neighbor1',

 'd0_typ4_sep2_neighbor1',

 'd1_med2_neighbor1',

 'd1_med2_neighbor2',

 'd1_med2_neighbor3',

 'd1_med2_neighbor4',

 'd1_med2_neighbor5',

 'd1_typ0_sep2_neighbor1',

 'd1_typ0_sep2_neighbor2',

 'd1_typ0_sep2_neighbor3',

 'd1_typ0_sep2_neighbor4',

 'd1_typ0_sep2_neighbor5',

 'd1_typ2_sep2_neighbor1',

 'd1_typ2_sep2_neighbor2',

 'd1_typ2_sep2_neighbor3',

 'd1_typ2_sep2_neighbor4',

 'd1_typ2_sep2_neighbor5',

 'd1_typ3_sep2_neighbor1',

 'd1_typ4_sep2_neighbor1',

 'd_0',

 'd_2',

 'd_3'

]
X_train.shape
# X_train = fs.transform(X_train)

# X_train_part = fs.transform(X_train_part)

# X_valid = fs.transform(X_valid)

# X_train_small = fs.transform(X_train_small)

# X_test = fs.transform(X_test)



feat_cols_selected_indices = [idx for idx, name in enumerate(feat_cols) if name in feat_cols_selected]

X_train = X_train[:, feat_cols_selected_indices]

X_train_part = X_train_part[:, feat_cols_selected_indices]

X_valid = X_valid[:, feat_cols_selected_indices]

X_train_small = X_train_small[:, feat_cols_selected_indices]

X_test = X_test[:, feat_cols_selected_indices]
X_train.shape
from sklearn.ensemble import ExtraTreesRegressor
reg = ExtraTreesRegressor(n_estimators=10, n_jobs=-1, random_state=42, verbose=2)
%%time

reg.fit(X_train_part, y_train_part)
y_pred = reg.predict(X_valid)
def metric(y, y_hat, types, type_filter=None, verbose=True):

    res = 0

    uniq_types = np.unique(types)

    if verbose:

        print('typ|   cnt(%)   |    sum     |  log')

        cnt_total = types.shape[0]

    for typ in uniq_types:

        if type_filter is not None and type_filter != typ:

            continue

        idx = np.where(types == typ)[0]

        cnt = idx.shape[0]

        res_typ = np.sum(np.abs(y[idx] - y_hat[idx])) / cnt

        assert cnt == y[idx].shape[0] == y_hat[idx].shape[0], 'inconsistent idx'

        if verbose:

            print(f'{typ:3}|{cnt:8}({1+cnt*100//cnt_total:02d})|{res_typ:12.9}|{np.log(res_typ)}')

        res += np.log(res_typ)

    uniq_types_cnt = uniq_types.shape[0] if type_filter is None else 1

    res /= uniq_types_cnt   

    if verbose:

        print(f'Result: {res} for {uniq_types_cnt} types')

    return res
metric(y_valid, y_pred, valid_types)
from skopt.space import Real, Integer, Categorical
# The list of hyper-parameters we want to optimize. For each one we define the bounds,

# the corresponding scikit-learn parameter name, as well as how to sample values

# from that dimension (`'log-uniform'` for the learning rate)

space = [

    Integer(10, 100, name='n_estimators'),

#     Integer(2, 100, name='min_samples_split'),  # always max

#     Integer(1, 100, name='min_samples_leaf'),   # always min

#     Categorical([True, False], name='bootstrap'),  # problem with plotting results

    Real(0.1, 1.0, prior='uniform', name='max_features'),

#     Integer(25, 125, name='max_depth'),  # always max

]
n_estimators_by_type = {

    0: 32, 1: 160, 2: 28, 3: 100, 4: 160, 5: 28, 6: 32, 7: 160

}
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from lightgbm import LGBMRegressor

from tqdm import tqdm_notebook





def get_estimator(estimator_model='rf', n_estimators=10, verbose=False):

    if estimator_model == 'rf':

        cls = RandomForestRegressor

    elif estimator_model == 'et':

        cls = ExtraTreesRegressor

    elif estimator_model == 'lgbm':

        cls = LGBMRegressor

    else:

        return None

    print('Selected estimator class:', cls)

    return cls(n_estimators=n_estimators, n_jobs=-1, random_state=42, verbose=verbose)
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from skopt.space import Real, Integer, Categorical

from skopt.utils import use_named_args

from skopt import gp_minimize

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook





def skopt_hyper(space, n_estimators_by_type, 

                X_train, y_train, train_types, 

                X_valid, y_valid, valid_types,

                estimator_model='rf', n_calls=30, 

                random_state=42, verbose=True):

    

    reg = get_estimator(estimator_model, verbose=0)  # False

    if reg is None:

        return None



    res_opts = {}

    for typ in tqdm_notebook(np.unique(train_types)):



        # This decorator allows your objective function to receive the parameters as

        # keyword arguments. This is particularly convenient when you want to set scikit-learn

        # estimator parameters

        @use_named_args(space)

        def objective(**params):

            reg.set_params(**params)

            mask = train_types == typ

    #         scores = cross_val_score(reg, X_train[mask], y_train[mask], cv=3, n_jobs=-1,

    #                                  scoring="neg_mean_absolute_error")

    #         return -np.mean(scores)

            reg.fit(X_train[mask], y_train[mask])

            y_valid_pred = reg.predict(X_valid)

            score = metric(y_valid, y_valid_pred, valid_types, typ, verbose=verbose)

            return score



        if n_estimators_by_type is not None:

            space[0] = Integer(10, n_estimators_by_type[typ], name='n_estimators')

        if verbose:

            print(f'--- Type {typ}, space {space} ---')

            

        res_opt = gp_minimize(objective, space, n_calls=n_calls, 

                              verbose=True, random_state=random_state)

        res_opts[typ] = res_opt

        if verbose:

            print(f'--- optimal params: {res_opt.x} ---')



    from skopt.plots import plot_convergence, plot_evaluations, plot_objective

    for typ in np.unique(train_types):

        if verbose:

            print('Type', typ)

        res_opt = res_opts[typ]

        plot_convergence(res_opt)

        plot_evaluations(res_opt)

        plot_objective(res_opt)

        plt.show()

       

    return {k: v.x for k, v in res_opts.items()}
%time

# res_opts = skopt_hyper(space, n_estimators_by_type, 

#                        X_train_small, y_train_small, train_small_types,

#                        X_valid, y_valid, valid_types, 

#                        estimator_model='et', verbose=True)

res_opts = {

 0: [32, 0.9356368713812235],

 1: [160, 0.9226985331078089],

 2: [28, 0.978079768528465],

 3: [100, 1.0],

 4: [160, 1.0],

 5: [28, 0.9688476690205474],

 6: [32, 1.0],

 7: [160, 0.9976351777608269]

}
res_opts
def train_and_predict_sep(X_train, y_train, X_train_part, y_train_part, X_valid, y_valid, 

                          train_types, train_part_types, valid_types, 

                          X_test=None, test_types=None,

                          estimator_model='rf', cnt_estimators=None, res_opts=None, params=None):

    assert X_train.shape[0] == y_train.shape[0] == train_types.shape[0], 'Inconsistent train - X and y'

    assert X_train_part.shape[0] == y_train_part.shape[0] == train_part_types.shape[0], 'Inconsistent train_part - X and y'

    assert X_valid.shape[0] == y_valid.shape[0] == valid_types.shape[0], 'Inconsistent valid - X and y'

    y_pred_valid = np.empty(X_valid.shape[0])

    y_pred_test = None

    if X_test is not None and test_types is not None:

        assert X_test.shape[0] == test_types.shape[0], 'Inconsistent train - X and y'

        y_pred_test = np.empty(X_test.shape[0])

    typ_cnt = np.unique(train_types).shape[0]

    if cnt_estimators is None:

        cnt_estimators = [10] * typ_cnt

    assert len(cnt_estimators) == typ_cnt, 'Inconsistent estimators count'

    for n_estimators, typ in tqdm_notebook(zip(cnt_estimators, range(typ_cnt)), total=len(cnt_estimators)):

        print('+++++ Fitting model for typ', typ)

        mask = train_part_types == typ

        reg = get_estimator(estimator_model, cnt_estimators, verbose=2)

        if reg is None:

            return None

        if params is not None:

            reg.set_params(**params[typ])

        elif res_opts is not None:

            res_opt = res_opts[typ]

            reg_params = {

                'n_estimators':      res_opt[0],

                'max_features':      res_opt[1],

            }

            reg.set_params(**reg_params)

        print(reg)    

        reg.fit(X_train_part[mask], y_train_part[mask])

        print('+++++ Predicting with model for typ', typ)

        mask = valid_types == typ

        y_pred_valid[mask] = reg.predict(X_valid[mask])

        metric(y_valid, y_pred_valid, valid_types, typ)

        if X_test is not None and test_types is not None:

            print('+++++ Fitting model for typ', typ)

            mask = train_types == typ

            reg.fit(X_train[mask], y_train[mask])

            print('+++++ Predicting with model for typ', typ)

            mask = test_types == typ

            y_pred_test[mask] = reg.predict(X_test[mask])

    print('+++++ Overall validation report')

    metric(y_valid, y_pred_valid, valid_types)

    return y_pred_test, y_pred_valid
cnt_estimators = None
%%time

y_pred, _ = train_and_predict_sep(X_train, y_train, X_train_part, y_train_part, X_valid, y_valid, 

                                  train_types, train_part_types, valid_types, 

                                  X_test=X_test, test_types=test_types,

                                  estimator_model='et', cnt_estimators=cnt_estimators, res_opts=res_opts)
plt.hist(y_pred, bins=100)

pass
df_subm = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv', index_col='id')
df_subm[target] = y_pred
df_subm.head()
fn_csv = f'./{EXP_NUMBER}.subm.csv'
df_subm.to_csv(fn_csv)