import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('fivethirtyeight')



type_colors = sns.color_palette("hls", 16)
# Read the Dataset



data_train=pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

data_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

target_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

# First Five rows



data_train.head()
# Shape of the training data



data_train.shape
# Check for Null



data_train.isna().sum()
# Check if id is unique



data_train.sig_id.nunique()
print('There are  {:} rows in training data.'.format(len(data_train)))
# Describe the training Dataset



data_train.describe()
def plot_fn(df, feature):



    ## Create a chart

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    ## create a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
# c-90 cell

plot_fn(data_train, 'c-90')
# c-0 cell

plot_fn(data_train, 'c-0')
# c-93 cell

plot_fn(data_train, 'c-93')
# g-90 gene

plot_fn(data_train, 'g-90')
# g-0 gene

plot_fn(data_train, 'g-0')
# g-93 gene

plot_fn(data_train, 'g-93')
cp_plot = data_train.cp_type.value_counts()

ax = cp_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise Contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(cp_plot.index, cp_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
cp_dose_plot = data_train.cp_dose.value_counts()

ax = cp_dose_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise Contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(cp_dose_plot.index, cp_dose_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
## Target Scored Multi-Label data

target_scored.head()
## Target non-Scored Multi-Label data

target_nonscored.head()
target_scored.sum()[1:].sort_values()
## Skewness in target_scored



target_scored.skew().sort_values()
target_nonscored.sum()[1:].sort_values()  # remove the first column id(sig_id) and sort the values
## Skewness in target_nonscored



target_nonscored.skew().sort_values()
data_train.skew().sort_values()
def sig_fn(data):

    e = np.exp(1)

    y = 1/(1+e**(-data))

    return y
## Get all numerical columns and create new dataset

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeri_train = data_train.select_dtypes(include=numerics)
#numeri_train.head()
## Apply the sigmoid function on training data

sig_data = sig_fn(numeri_train)



## Find the Skewness

#sig_data.skew()

sig_data.skew().sort_values()
sig_data.head()
# First Five rows

data_test.head()
# Shape of the training data



data_test.shape
data_test.skew().sort_values()
## Get numerical columns on test data

numeri_test = data_test.select_dtypes(include=numerics)



## Apply the sigmoid function on test data

sig_data_test = sig_fn(numeri_test)



## Find the Skewness

#sig_data_test.skew()

sig_data_test.skew().sort_values()
data_train=data_train[list(data_test)]

all_data=pd.concat((data_train, data_test))

print(data_train.shape, data_test.shape, all_data.shape)
## Apply Dummies



all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_dose'], prefix='cp_dose', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_time'], prefix='cp_time', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_type'], prefix='cp_type', dtype=float)],axis=1)

all_data = all_data.drop(['cp_dose', 'cp_time', 'cp_type'], axis=1)
## After Dummies



## Create a copy of data

full_data = all_data.copy()

all_data.head()
## Check Skewness on whole dataset (training and test)

all_data.skew().sort_values()
## Get numerical columns from whole dataset

numeri_all = all_data.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_all = sig_fn(numeri_all)



## Find the Skewness



sig_data_all.skew().sort_values()
sig_data_all.head()
def normalize_fn(data):

    upper = data.max()

    lower = data.min()

    y = (data - lower)/(upper-lower)

    return y

data_normalized = normalize_fn(sig_data_all)



data_normalized.skew().sort_values()
data_normalized.head()
# data_log = np.log(numeri_all + 1)

# data_log_normalized = normalize_fn(data_log)

# data_log_normalized.describe()





# Divide by Zero Occurs
# def sig_inf_fn(data):

#     e = np.exp(1)

#     y = 2/(1+e**(-data))

#     return y







## Apply the  function

#sig_data_al = sig_inf_fn(numeri_all)



## Find the Skewness

#sig_data_al.skew().sort_values()

## Plot the Skewness values to check the value range

plt.plot(sig_data_all.skew())
all_data = all_data.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_all = all_data.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_all = sig_fn(numeri_all)



## Find the Skewness



#sig_data_all.skew().sort_values()



plt.plot(sig_data_all.skew())
Xtrain=all_data[:len(data_train)]

Xtest=all_data[len(data_train):]

plt.plot(Xtrain.skew())
## Separate the data

Xtrain=full_data[:len(data_train)]

Xtest=full_data[len(data_train):]
## Get numerical columns from training dataset

numerical_train = Xtrain.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_train = sig_fn(numerical_train)



## Find the Skewness



plt.plot(sig_data_train.skew())
## Get numerical columns from test dataset

numerical_test = Xtest.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_test = sig_fn(numerical_test)



## Find the Skewness



plt.plot(sig_data_test.skew())
final_train = Xtrain.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_final_train = final_train.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_final_train = sig_fn(numeri_final_train)



## Find the Skewness



#sig_data_final_train.skew().sort_values()



plt.plot(sig_data_final_train.skew())

#sig_data_final_train.head()
final_test = Xtest.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_final_test = final_test.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_final_test = sig_fn(numeri_final_test)



## Find the Skewness



#sig_data_final_test.skew().sort_values()



plt.plot(sig_data_final_test.skew())

#sig_data_final_test.head()
numeri_target_score = target_scored.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_target_score = sig_fn(numeri_target_score)



## Plot the Skewness



#plt.plot(sig_target_score.skew())

sig_target_score.skew()
sig_target_score.head()
# Check percentage

sig_target_score['5-alpha_reductase_inhibitor'].value_counts(normalize=True)



target_scored['11-beta-hsd1_inhibitor'].value_counts(normalize=True)
# Choose 1st target column and build a model

Ytrain=target_scored['11-beta-hsd1_inhibitor']
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

import scipy

from sklearn.linear_model import LogisticRegression

import optuna

from sklearn.metrics import log_loss, make_scorer





ftwo_scorer = make_scorer(log_loss)

ftwo_scorer
#kf=StratifiedKFold(n_splits=10)

kf = KFold(n_splits=10)
Xtrain=sig_data_final_train

Xtest=sig_data_final_test

from imblearn.over_sampling import SMOTE



oversample = SMOTE()

X, y = oversample.fit_resample(Xtrain, Ytrain)
y.value_counts(normalize=True)
print(list(target_scored.columns))
target_scored.atm_kinase_inhibitor.value_counts()
target_scored['5-alpha_reductase_inhibitor'].value_counts()
Ytrain = target_scored['atm_kinase_inhibitor']

from imblearn.over_sampling import SMOTE



oversample = SMOTE()

X, y = oversample.fit_resample(Xtrain, Ytrain)
y.shape
Ytrain = target_scored['atm_kinase_inhibitor']

from imblearn.over_sampling import SMOTE



oversample = SMOTE()

X, y = oversample.fit_resample(Xtrain, Ytrain)
y.value_counts()
Ytrain = target_scored['atm_kinase_inhibitor']

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline



oversample = SMOTE(sampling_strategy=0.1)

undersample = RandomUnderSampler(sampling_strategy=0.5)



steps = [('o', oversample), ('u', undersample)]

pipeline = Pipeline(steps=steps)





X, y = pipeline.fit_resample(Xtrain, Ytrain)
y.value_counts()
y.shape
target_scored['atm_kinase_inhibitor'].skew()
y.skew()
## Save clean train and test dataset

#sig_data_final_train.to_csv('train_clean', index=False)

sig_data_final_test.to_csv('test_clean', index=False)
