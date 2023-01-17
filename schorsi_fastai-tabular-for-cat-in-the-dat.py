import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai.tabular import *
dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test_dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

out = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

dat.head(3)
cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 

             'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

cont_names = []
dep_var = ['target']

procs = [FillMissing, Categorify, Normalize]
FillMissing.FillStrategy='MEAN'



PATH = Path('/kaggle/input/cat-in-the-dat-ii/')

test = TabularList.from_df(test_dat, path=PATH, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(dat, path=PATH,

                            cat_names=cat_names, 

                            cont_names=cont_names,

                            procs=procs)

                           .split_by_idx(valid_idx = range(len(dat)-50000, len(dat)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())
data.show_batch(rows=3)
learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, FBeta(average='weighted')], ps=0.15)

Model_Path = Path('/kaggle/working/cat-in-dat/')

learn.model_dir = Model_Path
learn.lr_find()

learn.recorder.plot()
learn.fit(1, lr=1e-2)
learn.save('vinilla')
row = dat.iloc[9]

q = learn.predict(row)

row = dat.iloc[0]

v = learn.predict(row)

print('Positive Case:')

print(q)

print(float(q[2][1]),'\n\n')

print('Negaitive Case:')

print(v)

print(float(v[2][1]))
preds = learn.get_preds(ds_type=DatasetType.Test)[0][:,1].numpy()
submission_1 = pd.DataFrame({'id': out.index, 'target': preds})

submission_1.to_csv('/kaggle/working/cat-in-dat/submission_1.csv', header=True, index=False)
submission_1.describe()
# Resetting everything for round two

learn.destroy

dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test_dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')
train_dat = dat.copy()

def int_it(num, x):

    try:

        return int(num, x)

    except:

        return np.nan



train_dat['nom_5'] = pd.Series([int_it(x,16) for x in train_dat.nom_5], index=train_dat.index)

test_dat['nom_5'] = pd.Series([int_it(x,16) for x in test_dat.nom_5], index=test_dat.index)



train_dat['nom_6'] = pd.Series([int_it(x,16) for x in train_dat.nom_6], index=train_dat.index)

test_dat['nom_6'] = pd.Series([int_it(x,16) for x in test_dat.nom_6], index=test_dat.index)



train_dat['nom_7'] = pd.Series([int_it(x,16) for x in train_dat.nom_7], index=train_dat.index)

test_dat['nom_7'] = pd.Series([int_it(x,16) for x in test_dat.nom_7], index=test_dat.index)



train_dat['nom_8'] = pd.Series([int_it(x,16) for x in train_dat.nom_8], index=train_dat.index)

test_dat['nom_8'] = pd.Series([int_it(x,16) for x in test_dat.nom_8], index=test_dat.index)



train_dat['nom_9'] = pd.Series([int_it(x,16) for x in train_dat.nom_9], index=train_dat.index)

test_dat['nom_9'] = pd.Series([int_it(x,16) for x in test_dat.nom_9], index=test_dat.index)



train_dat['ord_1'] = train_dat['ord_1'].fillna(0)

ord_1_map = {0 : np.nan, 'Novice': 1, 'Contributor': 2,'Expert': 3 , 'Master' : 4, 'Grandmaster': 5}

train_dat['ord_1'] = pd.Series([ord_1_map[x] for x in train_dat.ord_1], index=train_dat.index)

test_dat['ord_1'] = test_dat['ord_1'].fillna(0)

test_dat['ord_1'] = pd.Series([ord_1_map[x] for x in test_dat.ord_1], index=test_dat.index)



train_dat['ord_2'] = train_dat['ord_2'].fillna(0)

ord_2_map = {0 : np.nan, 'Freezing': 1, 'Cold': 2,'Warm': 3 , 'Hot' : 4, 'Boiling Hot': 5, 'Lava Hot' : 6}

train_dat['ord_2'] = pd.Series([ord_2_map[x] for x in train_dat.ord_2], index=train_dat.index)

test_dat['ord_2'] = test_dat['ord_2'].fillna(0)

test_dat['ord_2'] = pd.Series([ord_2_map[x] for x in test_dat.ord_2], index=test_dat.index)



train_dat['ord_3'] = train_dat['ord_3'].fillna(0)

ord_3_map = {0:np.nan,'a':1, 'b':2,'c':3,'d':4, 'e': 5, 'f' : 6, 'g' : 7, 'h' : 8, 'i' : 9, 'j' : 10,

             'k' : 11, 'l' : 12, 'm' : 13, 'n' : 14, 'o' : 15, 'p' : 16, 'q' : 17, 'r' : 18, 's' : 19,

             't' : 20, 'u' : 21, 'v' : 22, 'w': 23, 'x' : 24, 'y' : 25, 'z' : 26}

train_dat['ord_3'] = pd.Series([ord_3_map[x] for x in train_dat.ord_3], index=train_dat.index)

test_dat['ord_3'] = test_dat['ord_3'].fillna(0)

test_dat['ord_3'] = pd.Series([ord_3_map[x] for x in test_dat.ord_3], index=test_dat.index)



train_dat['ord_4'] = train_dat['ord_4'].fillna(0)

ord_4_map = {0:np.nan,'A':1, 'B':2,'C':3,'D':4, 'E': 5, 'F' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'J' : 10,

             'K' : 11, 'L' : 12, 'M' : 13, 'N' : 14, 'O' : 15, 'P' : 16, 'Q' : 17, 'R' : 18, 'S' : 19,

             'T' : 20, 'U' : 21, 'V' : 22, 'W': 23, 'X' : 24, 'Y' : 25, 'Z' : 26}

train_dat['ord_4'] = pd.Series([ord_4_map[x] for x in train_dat.ord_4], index=train_dat.index)

test_dat['ord_4'] = test_dat['ord_4'].fillna(0)

test_dat['ord_4'] = pd.Series([ord_4_map[x] for x in test_dat.ord_4], index=test_dat.index)



def ord_alph(val):

    val = val.lower()

    if val == 'a':

        return 1

    if val == 'b':

        return 2

    if val == 'c':

        return 3

    if val == 'd':

        return 4

    if val == 'e':

        return 5

    if val == 'f':

        return 6

    if val == 'g':

        return 7

    if val == 'h':

        return 8

    if val == 'i':

        return 9

    if val == 'j':

        return 10

    if val == 'k':

        return 11

    if val == 'l':

        return 12

    if val == 'm':

        return 13

    if val == 'n':

        return 14

    if val == 'o':

        return 15

    if val == 'p':

        return 16

    if val == 'q':

        return 17

    if val == 'r':

        return 18

    if val == 's':

        return 19

    if val == 't':

        return 20

    if val == 'u':

        return 21

    if val == 'v':

        return 22

    if val == 'w':

        return 23

    if val == 'x':

        return 24

    if val == 'y':

        return 25

    if val == 'z':

        return 26

    else:

        return 0



def ord_five_b(val):

    try:

        if val[0] == val[0].lower():

            val_0 = ord_alph(val[0]) + 26

        else:

            val_0 = ord_alph(val[0])

        if val[1] == val[1].lower():

            val_1 = ord_alph(val[1]) + 26

        else:

            val_1 = ord_alph(val[1])

        val_1 = val_1/100

        return(val_0 + val_1)

    except:

        return np.nan



train_dat['ord_5'] = pd.Series([ord_five_b(x) for x in train_dat.ord_5], index=train_dat.index)

test_dat['ord_5'] = pd.Series([ord_five_b(x) for x in test_dat.ord_5], index=test_dat.index)



dat = train_dat.copy()

# use these definitions when running this cell

cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 

             'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

cont_names = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
PATH = Path('/kaggle/input/cat-in-the-dat-ii/')

test = TabularList.from_df(test_dat, path=PATH, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(dat, path=PATH,

                            cat_names=cat_names, 

                            cont_names=cont_names,

                            procs=procs)

                           .split_by_idx(valid_idx = range(len(dat)-50000, len(dat)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())

learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, FBeta(average='weighted')], ps=0.15)

Model_Path = Path('/kaggle/working/cat-in-dat/')

learn.model_dir = Model_Path
learn.lr_find()

learn.recorder.plot()
learn.fit(1, lr=1e-2)
learn.save('label_encoded')
preds = learn.get_preds(ds_type=DatasetType.Test)[0][:,1].numpy()

submission_2 = pd.DataFrame({'id': out.index, 'target': preds})

submission_2.to_csv('/kaggle/working/cat-in-dat/submission_2.csv', header=True, index=False)

submission_2.describe()
# Resetting everything for round three

learn.destroy

dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test_dat = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')
train_dat = dat.copy()



train_dat['bin_3'] = train_dat['bin_3'].fillna(0) ## fills in the missing values for a column

bin_3_map = {0 : np.nan, 'F': 1, 'T': 2} ## this dictionary will be used to transform the data to distinct integers

train_dat['bin_3'] = pd.Series([bin_3_map[x] for x in train_dat.bin_3], index=train_dat.index) ## This replaces the original values

test_dat['bin_3'] = test_dat['bin_3'].fillna(0)

test_dat['bin_3'] = pd.Series([bin_3_map[x] for x in test_dat.bin_3], index=test_dat.index)



train_dat['bin_4'] = train_dat['bin_4'].fillna(0)

bin_4_map = {0 : np.nan, 'N': 1, 'Y': 2}

train_dat['bin_4'] = pd.Series([bin_4_map[x] for x in train_dat.bin_4], index=train_dat.index)

test_dat['bin_4'] = test_dat['bin_4'].fillna(0)

test_dat['bin_4'] = pd.Series([bin_4_map[x] for x in test_dat.bin_4], index=test_dat.index)



train_dat['nom_0'] = train_dat['nom_0'].fillna(0)

nom_0_map = {0 : np.nan, 'Red': 1, 'Blue': 2, 'Green': 3}

train_dat['nom_0'] = pd.Series([nom_0_map[x] for x in train_dat.nom_0], index=train_dat.index)

test_dat['nom_0'] = test_dat['nom_0'].fillna(0)

test_dat['nom_0'] = pd.Series([nom_0_map[x] for x in test_dat.nom_0], index=test_dat.index)



train_dat['nom_1'] = train_dat['nom_1'].fillna(0)

nom_1_map = {0 : np.nan, 'Circle': 1, 'Triangle': 2,'Square': 3 , 'Trapezoid' : 4, 'Star': 5, 'Polygon': 6}

train_dat['nom_1'] = pd.Series([nom_1_map[x] for x in train_dat.nom_1], index=train_dat.index)



test_dat['nom_1'] = test_dat['nom_1'].fillna(0)

test_dat['nom_1'] = pd.Series([nom_1_map[x] for x in test_dat.nom_1], index=test_dat.index)



train_dat['nom_2'] = train_dat['nom_2'].fillna(0)

nom_2_map = {0 : np.nan, 'Hamster': 1, 'Axolotl': 2,'Lion': 3 , 'Dog' : 4, 'Cat': 5, 'Snake': 6}

train_dat['nom_2'] = pd.Series([nom_2_map[x] for x in train_dat.nom_2], index=train_dat.index)

test_dat['nom_2'] = test_dat['nom_2'].fillna(0)

test_dat['nom_2'] = pd.Series([nom_2_map[x] for x in test_dat.nom_2], index=test_dat.index)



train_dat['nom_3'] = train_dat['nom_3'].fillna(0)

nom_3_map = {0 : np.nan, 'Finland': 1, 'Russia': 2,'Costa Rica': 3 , 'India' : 4, 'China': 5, 'Canada': 6}

train_dat['nom_3'] = pd.Series([nom_3_map[x] for x in train_dat.nom_3], index=train_dat.index)

test_dat['nom_3'] = test_dat['nom_3'].fillna(0)

test_dat['nom_3'] = pd.Series([nom_3_map[x] for x in test_dat.nom_3], index=test_dat.index)



train_dat['nom_4'] = train_dat['nom_4'].fillna(0)

nom_4_map = {0 : np.nan, 'Piano': 1, 'Bassoon': 2,'Theremin': 3 , 'Oboe' : 4}

train_dat['nom_4'] = pd.Series([nom_4_map[x] for x in train_dat.nom_4], index=train_dat.index)

test_dat['nom_4'] = test_dat['nom_4'].fillna(0)

test_dat['nom_4'] = pd.Series([nom_4_map[x] for x in test_dat.nom_4], index=test_dat.index)



def int_it(num, x):

    try:

        return int(num, x)

    except:

        return np.nan



train_dat['nom_5'] = pd.Series([int_it(x,16) for x in train_dat.nom_5], index=train_dat.index)

test_dat['nom_5'] = pd.Series([int_it(x,16) for x in test_dat.nom_5], index=test_dat.index)



train_dat['nom_6'] = pd.Series([int_it(x,16) for x in train_dat.nom_6], index=train_dat.index)

test_dat['nom_6'] = pd.Series([int_it(x,16) for x in test_dat.nom_6], index=test_dat.index)



train_dat['nom_7'] = pd.Series([int_it(x,16) for x in train_dat.nom_7], index=train_dat.index)

test_dat['nom_7'] = pd.Series([int_it(x,16) for x in test_dat.nom_7], index=test_dat.index)



train_dat['nom_8'] = pd.Series([int_it(x,16) for x in train_dat.nom_8], index=train_dat.index)

test_dat['nom_8'] = pd.Series([int_it(x,16) for x in test_dat.nom_8], index=test_dat.index)



train_dat['nom_9'] = pd.Series([int_it(x,16) for x in train_dat.nom_9], index=train_dat.index)

test_dat['nom_9'] = pd.Series([int_it(x,16) for x in test_dat.nom_9], index=test_dat.index)



train_dat['ord_1'] = train_dat['ord_1'].fillna(0)

ord_1_map = {0 : np.nan, 'Novice': 1, 'Contributor': 2,'Expert': 3 , 'Master' : 4, 'Grandmaster': 5}

train_dat['ord_1'] = pd.Series([ord_1_map[x] for x in train_dat.ord_1], index=train_dat.index)

test_dat['ord_1'] = test_dat['ord_1'].fillna(0)

test_dat['ord_1'] = pd.Series([ord_1_map[x] for x in test_dat.ord_1], index=test_dat.index)



train_dat['ord_2'] = train_dat['ord_2'].fillna(0)

ord_2_map = {0 : np.nan, 'Freezing': 1, 'Cold': 2,'Warm': 3 , 'Hot' : 4, 'Boiling Hot': 5, 'Lava Hot' : 6}

train_dat['ord_2'] = pd.Series([ord_2_map[x] for x in train_dat.ord_2], index=train_dat.index)

test_dat['ord_2'] = test_dat['ord_2'].fillna(0)

test_dat['ord_2'] = pd.Series([ord_2_map[x] for x in test_dat.ord_2], index=test_dat.index)



train_dat['ord_3'] = train_dat['ord_3'].fillna(0)

ord_3_map = {0:np.nan,'a':1, 'b':2,'c':3,'d':4, 'e': 5, 'f' : 6, 'g' : 7, 'h' : 8, 'i' : 9, 'j' : 10,

             'k' : 11, 'l' : 12, 'm' : 13, 'n' : 14, 'o' : 15, 'p' : 16, 'q' : 17, 'r' : 18, 's' : 19,

             't' : 20, 'u' : 21, 'v' : 22, 'w': 23, 'x' : 24, 'y' : 25, 'z' : 26}

train_dat['ord_3'] = pd.Series([ord_3_map[x] for x in train_dat.ord_3], index=train_dat.index)

test_dat['ord_3'] = test_dat['ord_3'].fillna(0)

test_dat['ord_3'] = pd.Series([ord_3_map[x] for x in test_dat.ord_3], index=test_dat.index)



train_dat['ord_4'] = train_dat['ord_4'].fillna(0)

ord_4_map = {0:np.nan,'A':1, 'B':2,'C':3,'D':4, 'E': 5, 'F' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'J' : 10,

             'K' : 11, 'L' : 12, 'M' : 13, 'N' : 14, 'O' : 15, 'P' : 16, 'Q' : 17, 'R' : 18, 'S' : 19,

             'T' : 20, 'U' : 21, 'V' : 22, 'W': 23, 'X' : 24, 'Y' : 25, 'Z' : 26}

train_dat['ord_4'] = pd.Series([ord_4_map[x] for x in train_dat.ord_4], index=train_dat.index)

test_dat['ord_4'] = test_dat['ord_4'].fillna(0)

test_dat['ord_4'] = pd.Series([ord_4_map[x] for x in test_dat.ord_4], index=test_dat.index)



def ord_alph(val):

    val = val.lower()

    if val == 'a':

        return 1

    if val == 'b':

        return 2

    if val == 'c':

        return 3

    if val == 'd':

        return 4

    if val == 'e':

        return 5

    if val == 'f':

        return 6

    if val == 'g':

        return 7

    if val == 'h':

        return 8

    if val == 'i':

        return 9

    if val == 'j':

        return 10

    if val == 'k':

        return 11

    if val == 'l':

        return 12

    if val == 'm':

        return 13

    if val == 'n':

        return 14

    if val == 'o':

        return 15

    if val == 'p':

        return 16

    if val == 'q':

        return 17

    if val == 'r':

        return 18

    if val == 's':

        return 19

    if val == 't':

        return 20

    if val == 'u':

        return 21

    if val == 'v':

        return 22

    if val == 'w':

        return 23

    if val == 'x':

        return 24

    if val == 'y':

        return 25

    if val == 'z':

        return 26

    else:

        return 0



def ord_five_b(val):

    try:

        if val[0] == val[0].lower():

            val_0 = ord_alph(val[0]) + 26

        else:

            val_0 = ord_alph(val[0])

        if val[1] == val[1].lower():

            val_1 = ord_alph(val[1]) + 26

        else:

            val_1 = ord_alph(val[1])

        val_1 = val_1/100

        return(val_0 + val_1)

    except:

        return np.nan



train_dat['ord_5'] = pd.Series([ord_five_b(x) for x in train_dat.ord_5], index=train_dat.index)

test_dat['ord_5'] = pd.Series([ord_five_b(x) for x in test_dat.ord_5], index=test_dat.index)



dat = train_dat.copy()

# use these definitions when running this cell

cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 

             'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

cont_names = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
from sklearn.linear_model import LinearRegression

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

td = dat.copy()

test_dat_copy = test_dat.copy()

features = list(td.columns)

features.remove('target')

X = td[features]

X_plus = pd.merge(X, test_dat_copy, how='outer')

iterimp = IterativeImputer(estimator=LinearRegression(),max_iter=300, add_indicator=False,tol=2e-4, random_state=0)#add_indicator=True,

iterimp.fit(X_plus)

dat[features] = iterimp.transform(dat[features])#[features]

test_dat[features] = iterimp.transform(test_dat[features])
PATH = Path('/kaggle/input/cat-in-the-dat-ii/')

test = TabularList.from_df(test_dat, path=PATH, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(dat, path=PATH,

                            cat_names=cat_names, 

                            cont_names=cont_names,

                            procs=procs)

                           .split_by_idx(valid_idx = range(len(dat)-50000, len(dat)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())

learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, FBeta(average='weighted')], ps=0.15)

Model_Path = Path('/kaggle/working/cat-in-dat/')

learn.model_dir = Model_Path
learn.lr_find()

learn.recorder.plot()
learn.fit(1, lr=1e-2)
learn.save('Iteritive Imputation')
preds = learn.get_preds(ds_type=DatasetType.Test)[0][:,1].numpy()

submission_3 = pd.DataFrame({'id': out.index, 'target': preds})

submission_3.to_csv('/kaggle/working/cat-in-dat/submission_3.csv', header=True, index=False)

submission_3.describe()
final = out.copy()
#need to test this section when time permits

final['target_1'] = pd.Series([x for x in submission_1.target], index=out.index)

final['target_2'] = pd.Series([x for x in submission_2.target], index=out.index)

final['target_3'] = pd.Series([x for x in submission_3.target], index=out.index)
final = final.drop('target', axis=1)
final.describe()
final['target'] = final.mean(axis=1)

final = final.drop('target_1', axis=1)

final = final.drop('target_2', axis=1)

final = final.drop('target_3', axis=1)

final.index = out.index
final.to_csv('/kaggle/working/final.csv')