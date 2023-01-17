%matplotlib inline



import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import matplotlib

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import xgboost as xgb



from IPython.display import display

from collections import defaultdict
dtrain = pd.read_csv('../input/train-and-test-csv/train.csv')

dtest = pd.read_csv('../input/train-and-test-csv/test.csv')
def display_all(data):

    with pd.option_context('display.max_rows', 1000):

        with pd.option_context('display.max_columns', 1000):

            return display(data)
print(dtrain.shape)

print(dtest.shape)
#Target

print(dtrain.target.value_counts(normalize=True))
display_all(dtrain.head())
display_all(dtest.head())
dtrain.replace(to_replace=-1, value=np.nan, inplace=True)

dtest.replace(to_replace=-1, value=np.nan, inplace=True)
pred_columns = dtrain.columns[2:].values
### Let's do the column dtype conversions

pred_columns = dtrain.columns[2:]

bin_cols = [c for c in pred_columns if 'bin' in c]

cat_cols = [c for c in pred_columns if 'cat' in c]

num_cols= [c for c in pred_columns if c not in bin_cols and c not in cat_cols]
for c in pred_columns:

    if 'cat' in c or 'bin' in c:

        print(f'Column: {c.upper()}')

        print('Train Summary')

        print(f'Cardinality {len(dtrain[c].unique())}')

        print(dtrain[c].value_counts(dropna=False))

        print('Test Summary')

        print(f'Cardinality {len(dtest[c].unique())}')

        print(dtest[c].value_counts(dropna=False))

        print()

        print()
# For fast iteration we will be using a subsample of data

def subsample(data, ratio=0.5):

    subsample, _ = train_test_split(data, test_size =ratio, stratify=dtrain.target)

    return subsample
subsample = subsample(dtrain)
# If feature is not binary include in correlation matrix

corr_cols = [c for c in num_cols if len(dtrain[c].unique()) > 2]

plt.imshow((subsample[corr_cols].corr()), cmap='hot', interpolation='nearest')

plt.show()
corr_mat = np.array(subsample[corr_cols].corr())



i_ix = np.where((corr_mat > 0.4) | (corr_mat < -0.4))[0]

j_ix = np.where((corr_mat > 0.4) | (corr_mat < -0.4))[1]

for i, j in zip(i_ix, j_ix):

    if i != j:

        print(f'Corr between {corr_cols[i]} and {corr_cols[j]}: {corr_mat[i, j]}')
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)
shuffleSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state= 10)
subsample.reset_index(drop=True, inplace=True)
def xgb_feedback(params, nrounds, prc_subsample):

    val_scores = []

    for train_ix, val_ix in shuffleSplit.split(prc_subsample, prc_subsample.target):

        X_train, y_train = prc_subsample.loc[train_ix].drop(['id', 'target'], axis=1), prc_subsample.target.loc[train_ix]

        X_val, y_val = prc_subsample.loc[val_ix].drop(['id', 'target'], axis=1), prc_subsample.target.loc[val_ix]



        #create dmatrix

        dtrain = xgb.DMatrix(data=X_train, label=y_train, missing= np.nan)

        dval = xgb.DMatrix(data=X_val, label=y_val, missing= np.nan)



        #train

        model = xgb.train(params, dtrain, num_boost_round=nrounds)

        preds = model.predict(dval)

        score = gini_normalized(y_val, preds)

        val_scores.append(score)

    return val_scores, model
def score_summary(scores):return f'Mean: {np.mean(scores)} Std: {np.std(scores)}'
iter_performances = defaultdict()

def add_to_iter(scores, name):

    iter_performances[name]= scores
def plot_model(model):

    matplotlib.rcParams['figure.figsize'] = [10, 7]

    ax = xgb.plot_importance(model)
#first process is just to put those NA -1s back

def prc1(data):

    data = data.fillna(-1)

    return data
prc_subsample = prc1(subsample)



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



init_benchmark = score_summary(val_scores)



# So this will be our initial benchmark

print(init_benchmark)



add_to_iter(init_benchmark, "benchmark")
matplotlib.rcParams['figure.figsize'] = [10, 7]

ax = xgb.plot_importance(model)
subsample.ps_car_13.hist(bins=30)
np.log(subsample.ps_car_13).hist(bins=30)
round(subsample.ps_car_13**2 * 48400).hist(bins =100)
def prc2(data):

    data = data.copy()

    #log transform ps_car_13

    data['log_ps_car_13'] = np.log(data["ps_car_13"])

    #create ps_car_13 feature from the kernel link

    #thanks to @raddar

    data['f1_ps_car_13'] = round(subsample["ps_car_13"]**2 * 48400)

    #also log this

    data['log_f1_ps_car_13'] = np.log(round(subsample["ps_car_13"]**2 * 48400))

    return data
prc_subsample = prc2(subsample)



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



iter1_benchmark = score_summary(val_scores)



# So this will be our initial benchmark

print(iter1_benchmark)



add_to_iter(iter1_benchmark, "iter1")
prc_subsample = prc2(prc1(subsample))



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



iter1_nas_benchmark = score_summary(val_scores)



# So this will be our initial benchmark

print(iter1_nas_benchmark)



add_to_iter(iter1_nas_benchmark, "iter1_na_as_neg1")
iter_performances
def prc3(data):

    data= data.copy()

    data["number_of_nan"] = data.isnull().sum(axis=1)

    return data
prc_subsample = prc3(subsample)



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



iter2 = score_summary(val_scores)



# So this will be our initial benchmark

print(iter2)



add_to_iter(iter2, "iter2")
iter_performances
plot_model(model)
# thanks to Pascal Nagel's kernel

def recon(reg):

    if np.isnan(reg):

        return reg

    else:

        integer = int(np.round((40*reg)**2)) # gives 2060 for our example

        for f in range(28):

            if (integer - f) % 27 == 0:

                F = f

        M = (integer - F)//27

        return F, M

# Using the above example to test

ps_reg_03_example = 1.1321312468057179

print("Federative Unit (F): ", recon(ps_reg_03_example)[0])

print("Municipality (M): ", recon(ps_reg_03_example)[1])
def prc4(data):

    data = data.copy()

    data["f1_ps_car_15"] = 1 / np.exp(data["ps_car_15"])

    data["f2_ps_car_15"] = (data["ps_car_15"])**2 

    data['ps_reg_F'] = data['ps_reg_03'].apply(lambda x: recon(x) if np.isnan(x) else recon(x)[0])

    data['ps_reg_M'] = data['ps_reg_03'].apply(lambda x: recon(x) if np.isnan(x) else recon(x)[1])

    return data
prc_subsample = prc4(subsample)



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



iter3 = score_summary(val_scores)



# So this will be our initial benchmark

print(iter3)



add_to_iter(iter3, "iter3")
iter_performances
plot_model(model)
sum(subsample[["ps_ind_06_bin","ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]].sum(axis =1) > 2)
def prc5(data):

    data = data.copy()

    arr = np.array(data[["ps_ind_06_bin","ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]])

    data["ps_ind_bin_6789"] = arr.dot(np.array([6, 7, 8, 9]))

    return data
prc_subsample = prc5(subsample)



params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



iter4 = score_summary(val_scores)



# So this will be our initial benchmark

print(iter4)



add_to_iter(iter4, "iter4")
iter_performances
plot_model(model)
prc_subsample = prc5(prc4(prc3(prc2(subsample))))
params = {"objective":"binary:logistic", "max_depth":1}

val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)



mid_score = score_summary(val_scores)



# So this will be our initial benchmark

print(mid_score)



add_to_iter(mid_score, "mid_way_score")
iter_performances
plot_model(model)
def midway_prc(data):return prc5(prc4(prc3(prc2(data))))