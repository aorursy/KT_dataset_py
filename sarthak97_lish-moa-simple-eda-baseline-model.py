import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss

from category_encoders import CountEncoder

from sklearn.model_selection import KFold

from xgboost import XGBClassifier

from sklearn.multioutput import MultiOutputClassifier

import random



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
ROOT = '../input/lish-moa'

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# training data

train = pd.read_csv(f'{ROOT}/train_features.csv')



# training data targets

target = pd.read_csv(f'{ROOT}/train_targets_scored.csv')



# testing data

test = pd.read_csv(f'{ROOT}/test_features.csv')
train.head()
target.head()
print("No. of rows in training set : {}".format(train.shape[0]))

print('No. of columns in training set : {}'.format(train.shape[1]))

print('No. of rows in target set : {}'.format(target.shape[0]))

print('No. of columns in target set : {}'.format(target.shape[1]))
cols = train.columns

g_cols = [x for x in cols if x.startswith('g-')]

c_cols = [x for x in cols if x.startswith('c-')]

print(f"There are {train.shape[1]} columns in training and test set out of which :- ")

print(f"There are {len(g_cols)} columns starting with 'g-' i.e. gene expression features.")

print(f"There are {len(c_cols)} columns starting with 'c-' i.e. cell viability features.")

print("'sig_id', 'cp_type', 'cp_time', 'cp_dose' account for the remaining 4 columns.")
labels = ['g-','c-','sig_id','cp_type', 'cp_time', 'cp_dose']

values = [len(g_cols), len(c_cols), 1, 1, 1, 1]



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])



fig.update_layout(title_text="Distribution of columns in train and test features.")

fig.show()
# Unique values for features

print("There are {} unique values in 'sig_id' column. So, there are no duplicate rows".format(train.sig_id.nunique()))

print("There are {} unique values in 'cp_type' column having values : {}".format(train.cp_type.nunique(), train.cp_type.unique()))

print("There are {} unique values in 'cp_time' column having values : {}".format(train.cp_time.nunique(), train.cp_time.unique()))

print("There are {} unique values in 'cp_dose' column having values : {}".format(train.cp_dose.nunique(), train.cp_dose.unique()))
trt_cp_count = train.cp_type.value_counts()[0]

ctl_vehicle_count = train.cp_type.value_counts()[1]

print(f"In 'cp_type' feature there are {trt_cp_count} records with value 'trt_cp' and {ctl_vehicle_count} records " 

      + "with value 'ctl_vehicle'")
x=train.cp_type.unique()

y_train=[trt_cp_count,ctl_vehicle_count]

y_test=[test.cp_type.value_counts()[0], test.cp_type.value_counts()[1]]



fig = go.Figure(data=[

    go.Bar(name='test', x=x, y=y_test),

    go.Bar(name='train', x=x, y=y_train)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
print("There are {} distinct values for 'cp_time' : {}".format(train.cp_time.nunique(), train.cp_time.unique()))

print('No. of records where cp_time=24 : {}'.format(train[train.cp_time == 24].shape[0]))

print('No. of records where cp_time=48 : {}'.format(train[train.cp_time == 48].shape[0]))

print('No. of records where cp_time=72 : {}'.format(train[train.cp_time == 72].shape[0]))
x=train.cp_time.unique()

y_train = [train[train.cp_time == 24].shape[0],train[train.cp_time == 72].shape[0],train[train.cp_time == 48].shape[0]]

y_test=[test[test.cp_time == 24].shape[0],test[test.cp_time == 72].shape[0],test[test.cp_time == 48].shape[0]]

fig = go.Figure(data=[

    go.Bar(name='test', x=x, y=y_test),

    go.Bar(name='train', x=x, y=y_train)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
d1_count = train.cp_dose.value_counts()[0]

d2_count = train.cp_dose.value_counts()[1]

print(f"In 'cp_dose' feature there are {d1_count} records with value 'D1' and {d2_count} records " 

      + "with value 'D2'")
x=train.cp_dose.unique()

y_train=[d1_count,d2_count]

y_test=[test.cp_dose.value_counts()[0], test.cp_dose.value_counts()[1]]



fig = go.Figure(data=[

    go.Bar(name='test', x=x, y=y_test),

    go.Bar(name='train', x=x, y=y_train)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
# distributions of few random g- features

random_g_cols = random.sample(g_cols, 10)



fig = make_subplots(rows=5, cols=2, subplot_titles=random_g_cols)



fig.add_trace(go.Histogram(x=train[random_g_cols[0]], name=random_g_cols[0]), row=1, col=1)

fig.add_trace(go.Histogram(x=train[random_g_cols[1]], name=random_g_cols[1]), row=1, col=2)

fig.add_trace(go.Histogram(x=train[random_g_cols[2]], name=random_g_cols[2]), row=2, col=1)

fig.add_trace(go.Histogram(x=train[random_g_cols[3]], name=random_g_cols[3]), row=2, col=2)

fig.add_trace(go.Histogram(x=train[random_g_cols[4]], name=random_g_cols[4]), row=3, col=1)

fig.add_trace(go.Histogram(x=train[random_g_cols[5]], name=random_g_cols[5]), row=3, col=2)

fig.add_trace(go.Histogram(x=train[random_g_cols[6]], name=random_g_cols[6]), row=4, col=1)

fig.add_trace(go.Histogram(x=train[random_g_cols[7]], name=random_g_cols[7]), row=4, col=2)

fig.add_trace(go.Histogram(x=train[random_g_cols[8]], name=random_g_cols[8]), row=5, col=1)

fig.add_trace(go.Histogram(x=train[random_g_cols[9]], name=random_g_cols[9]), row=5, col=2)



fig.update_layout(

    title_text='Distribution of a few random Gene Expression features',

    height = 1200,

    width=675

)



fig.show()
# distributions of few random c- features

random_c_cols = random.sample(c_cols, 10)



fig = make_subplots(rows=5, cols=2, subplot_titles=random_c_cols)



fig.add_trace(go.Histogram(x=train[random_c_cols[0]], name=random_c_cols[0]), row=1, col=1)

fig.add_trace(go.Histogram(x=train[random_c_cols[1]], name=random_c_cols[1]), row=1, col=2)

fig.add_trace(go.Histogram(x=train[random_c_cols[2]], name=random_c_cols[2]), row=2, col=1)

fig.add_trace(go.Histogram(x=train[random_c_cols[3]], name=random_c_cols[3]), row=2, col=2)

fig.add_trace(go.Histogram(x=train[random_c_cols[4]], name=random_c_cols[4]), row=3, col=1)

fig.add_trace(go.Histogram(x=train[random_c_cols[5]], name=random_c_cols[5]), row=3, col=2)

fig.add_trace(go.Histogram(x=train[random_c_cols[6]], name=random_c_cols[6]), row=4, col=1)

fig.add_trace(go.Histogram(x=train[random_c_cols[7]], name=random_c_cols[7]), row=4, col=2)

fig.add_trace(go.Histogram(x=train[random_c_cols[8]], name=random_c_cols[8]), row=5, col=1)

fig.add_trace(go.Histogram(x=train[random_c_cols[9]], name=random_c_cols[9]), row=5, col=2)



fig.update_layout(

    title_text='Distribution of a few random Cell Viability features',

    height = 1200,

    width=675

)



fig.show()
trt_cp_24 = train[(train.cp_type == 'trt_cp') & (train.cp_time == 24)].shape[0]

trt_cp_48 = train[(train.cp_type == 'trt_cp') & (train.cp_time == 48)].shape[0]

trt_cp_72 = train[(train.cp_type == 'trt_cp') & (train.cp_time == 72)].shape[0]

ctl_veh_24 = train[(train.cp_type == 'ctl_vehicle') & (train.cp_time == 24)].shape[0]

ctl_veh_48 = train[(train.cp_type == 'ctl_vehicle') & (train.cp_time == 48)].shape[0]

ctl_veh_72 = train[(train.cp_type == 'ctl_vehicle') & (train.cp_time == 72)].shape[0]



rel_df = pd.DataFrame([['trt_cp', trt_cp_24, trt_cp_48, trt_cp_72],

                       ['ctl_vehicle', ctl_veh_24, ctl_veh_48, ctl_veh_72]], 

                      columns = ['cp_type', '24', '48', '72'])



fig = px.bar(rel_df, x="cp_type", y=["24", "48", "72"], title="Cp_type V/S Cp_time")

fig.show()
trt_cp_d1 = train[(train.cp_type == 'trt_cp') & (train.cp_dose == 'D1')].shape[0]

trt_cp_d2 = train[(train.cp_type == 'trt_cp') & (train.cp_dose == 'D2')].shape[0]

ctl_veh_d1 = train[(train.cp_type == 'ctl_vehicle') & (train.cp_dose == 'D1')].shape[0]

ctl_veh_d2 = train[(train.cp_type == 'ctl_vehicle') & (train.cp_dose == 'D2')].shape[0]



rel_df = pd.DataFrame([['trt_cp', trt_cp_d1, trt_cp_d2],

                       ['ctl_vehicle', ctl_veh_d1, ctl_veh_d2]], 

                      columns = ['cp_type', 'D1', 'D2'])



fig = px.bar(rel_df, x="cp_type", y=['D1', 'D2'], title="Cp_type V/S Cp_dose")

fig.show()
d1_24 = train[(train.cp_dose == 'D1') & (train.cp_time == 24)].shape[0]

d1_48 = train[(train.cp_dose == 'D1') & (train.cp_time == 48)].shape[0]

d1_72 = train[(train.cp_dose == 'D1') & (train.cp_time == 72)].shape[0]

d2_24 = train[(train.cp_dose == 'D2') & (train.cp_time == 24)].shape[0]

d2_48 = train[(train.cp_dose == 'D2') & (train.cp_time == 48)].shape[0]

d2_72 = train[(train.cp_dose == 'D2') & (train.cp_time == 72)].shape[0]



rel_df = pd.DataFrame([['D1', d1_24, d1_48, d1_72],

                       ['D2', d2_24, d2_48, d2_72]], 

                      columns = ['cp_dose', '24', '48', '72'])



fig = px.bar(rel_df, x="cp_dose", y=["24", "48", "72"], title="Cp_dose V/S Cp_time")

fig.show()
# Correlation b/w random 40 g- features



g_df = train[random.sample(g_cols, 40)]

f = plt.figure(figsize=(19, 15))

plt.matshow(g_df.corr(), fignum=f.number)

plt.xticks(range(g_df.shape[1]), g_df.columns, fontsize=14, rotation=50)

plt.yticks(range(g_df.shape[1]), g_df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
# Correlation b/w random 40 c- features



c_df = train[random.sample(c_cols, 40)]

f = plt.figure(figsize=(19, 15))

plt.matshow(c_df.corr(), fignum=f.number)

plt.xticks(range(c_df.shape[1]), c_df.columns, fontsize=14, rotation=50)

plt.yticks(range(c_df.shape[1]), c_df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
cols = ['cp_time'] + g_cols + c_cols

all_columns = []

for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:

            all_columns.append(cols[i])

            all_columns.append(cols[j])



all_columns = list(set(all_columns))

print('Number of columns: ', len(all_columns))
all_cols_df = train[all_columns]

f = plt.figure(figsize=(19, 15))

plt.matshow(all_cols_df.corr(), fignum=f.number)

plt.xticks(range(all_cols_df.shape[1]), all_cols_df.columns, fontsize=14, rotation=50)

plt.yticks(range(all_cols_df.shape[1]), all_cols_df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
target.head()
print("No. of rows : {}".format(target.shape[0]))

print("No. of columns : {}".format(target.shape[1]))
target_copy = target.copy()
# Checking columns for all same values



same_value_cols = []

colwise_sum = ['colwise_sum']

moa_df = target_copy.iloc[:, 1:]

for col in moa_df.columns:

    colwise_sum.append(moa_df[col].sum())

    if moa_df[col].sum() == 0:

        same_value_cols.append(col)

        

print(f"There are {len(same_value_cols)} column(s) with all values same.")



# Append the colwise_sum list as last row to our target dataframe. We will use this row later.

target_copy.loc[len(target_copy)] = colwise_sum
# Checking rows for all same values



moa_df = target_copy.iloc[:-1,1:]

rowwise_sum = moa_df.sum(axis=1)

rowsum_zero_idx = []

for i, sum in enumerate(rowwise_sum):

    if sum == 0:

        rowsum_zero_idx.append(i)



print(f"There are {len(rowsum_zero_idx)} drug samples having all same values i.e all zero MoA labels.")



# Append this rowwise sum to target dataframe. We will use this column later.

target_copy['rowwise_sum'] = rowwise_sum
# counting the number of non-zeros

print("Total number of elements in target set : {}".format(target.shape[0] * target.shape[1]))

print("No. of non-zero elements in target set : {}".format(np.count_nonzero(target)))
target_copy = target_copy.sort_values('rowwise_sum', ascending=False)

temp_df = target_copy.iloc[:50,:]

fig = px.bar(temp_df, x='sig_id', y='rowwise_sum')

fig.show()
# Number of activations in targets for every sample
# Initialize variables

SEED = 42

NFOLDS = 5

np.random.seed(SEED)

ROOT = '../input/lish-moa/'
train = pd.read_csv(ROOT + 'train_features.csv')

targets = pd.read_csv(ROOT + 'train_targets_scored.csv')



test = pd.read_csv(ROOT + 'test_features.csv')

sub = pd.read_csv(ROOT + 'sample_submission.csv')



# drop id col

X = train.iloc[:,1:].to_numpy()

X_test = test.iloc[:,1:].to_numpy()

y = targets.iloc[:,1:].to_numpy() 
# Build the ML Pipeline



classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))



clf = Pipeline([('encode', CountEncoder(cols=[0, 2])),

                ('classify', classifier)

               ])
# set parameters for Pipeline classifier



params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
oof_preds = np.zeros(y.shape)

test_preds = np.zeros((test.shape[0], y.shape[1]))

oof_losses = []

kf = KFold(n_splits=NFOLDS)

for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):

    print('Starting fold: ', fn)

    X_train, X_val = X[trn_idx], X[val_idx]

    y_train, y_val = y[trn_idx], y[val_idx]

    

    # drop where cp_type==ctl_vehicle (baseline)

    ctl_mask = X_train[:,0]=='ctl_vehicle'

    X_train = X_train[~ctl_mask,:]

    y_train = y_train[~ctl_mask]

    

    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val) # list of preds per class

    val_preds = np.array(val_preds)[:,:,1].T # take the positive class

    oof_preds[val_idx] = val_preds

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds += preds / NFOLDS

    

print(oof_losses)

print('Mean OOF loss across folds', np.mean(oof_losses))

print('STD OOF loss across folds', np.std(oof_losses))
# set control train preds to 0

control_mask = train['cp_type']=='ctl_vehicle'

oof_preds[control_mask] = 0



print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# set control test preds to 0

control_mask = test['cp_type']=='ctl_vehicle'

test_preds[control_mask] = 0
# create the submission file

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)