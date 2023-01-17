# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
init_train_df = pd.read_csv('../input/nslkdd/kdd_train.csv')
init_test_df = pd.read_csv('../input/nslkdd/kdd_test.csv')
init_train_df.head()
init_test_df.head()
init_train_df.isnull().sum()
init_train_df.isnull().sum()
init_train_df.info()
init_train_df.nunique()
init_test_df.info()
init_test_df.nunique()
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=init_train_df, 
            order=init_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=init_test_df, 
            order=init_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()
def plot_hist(df, cols, title):
    grid = gridspec.GridSpec(10, 2, wspace=0.5, hspace=0.5) 
    fig = plt.figure(figsize=(15,25)) 
    
    for n, col in enumerate(df[cols]):         
        ax = plt.subplot(grid[n]) 

        ax.hist(df[col], bins=20) 
        #ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{col} distribution', fontsize=15) 
    
    fig.suptitle(title, fontsize=20)
    grid.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()
hist_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    
plot_hist(init_train_df, hist_cols, 'Distributions of Integer Features in Training Set')

hist_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    
plot_hist(init_test_df, hist_cols, 'Distributions of Integer Features in Testing Set')
rate_cols = [ 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(init_train_df, rate_cols, 'Distributions of Rate Features in Training Set')
rate_cols = [ 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(init_test_df, rate_cols, 'Distributions of Rate Features in Testing Set')
random_state = 42
 
proc_train_df = init_train_df.copy()                                                                      # create a copy of our initial train set to use as our preproccessed train set.
proc_test_df = init_test_df.copy()                                                                        # create a copy of our initial test set to use as our preproccessed test set.

proc_train_normal_slice = proc_train_df[proc_train_df['labels']=='normal'].copy()                         # get the slice of our train set with all normal observations
proc_train_neptune_slice = proc_train_df[proc_train_df['labels']=='neptune'].copy()                       # get the slice of our train set with all neptune observations

proc_test_normal_slice = proc_test_df[proc_test_df['labels']=='normal'].copy()                            # get the slice of our test set with all normal observations
proc_test_neptune_slice = proc_test_df[proc_test_df['labels']=='neptune'].copy()                          # get the slice of our test set with all neptune observations

proc_train_normal_sampled = proc_train_normal_slice.sample(n=5000, random_state=random_state)             # downsample train set normal slice to 5000 oberservations
proc_train_neptune_sampled = proc_train_neptune_slice.sample(n=5000, random_state=random_state)           # downsample train set neptune slice to 5000 oberservations

proc_test_normal_sampled = proc_test_normal_slice.sample(n=1000, random_state=random_state)               # downsample test set normal slice to 1000 oberservations
proc_test_neptune_sampled = proc_test_neptune_slice.sample(n=1000, random_state=random_state)             # downsample test set neptune slice to 5000 oberservations

proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='normal'].index, inplace=True)              # drop initial train normal slice
proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='neptune'].index, inplace=True)             # drop initial train neptune slice

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='normal'].index, inplace=True)                 # drop initial test normal slice
proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='neptune'].index, inplace=True)                # drop initial test neptune slice

proc_train_df = pd.concat([proc_train_df, proc_train_normal_sampled, proc_train_neptune_sampled], axis=0) # add sampled train normal and neptune slices back to train set
proc_test_df = pd.concat([proc_test_df, proc_test_normal_sampled, proc_test_neptune_sampled], axis=0)     # add sampled test normal and neptune slices back to test set
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()
keep_labels = ['normal', 'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop', 'warezclient']

proc_train_df['labels'] = proc_train_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
proc_test_df['labels'] = proc_test_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()
from sklearn.model_selection import train_test_split

seed_random = 718

proc_test_other_slice = proc_test_df[proc_test_df['labels']=='other'].copy()

proc_train_other_sampled, proc_test_other_sampled = train_test_split(proc_test_other_slice, test_size=0.2, random_state=seed_random)

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='other'].index, inplace=True)

proc_train_df = pd.concat([proc_train_df, proc_train_other_sampled], axis=0)
proc_test_df = pd.concat([proc_test_df, proc_test_other_sampled], axis=0)
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()
norm_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

for col in norm_cols:
    proc_train_df[col] = np.log(proc_train_df[col]+1e-6)
    proc_test_df[col] = np.log(proc_test_df[col]+1e-6)
    
plot_hist(proc_train_df, norm_cols, 'Distributions in Processed Training Set')
plot_hist(proc_test_df, norm_cols, 'Distributions in Processed Testing Set')
proc_train_df['train']=1                                                                       # add train feature with value 1 to our training set
proc_test_df['train']=0                                                                        # add train feature with value 0 to our testing set

joined_df = pd.concat([proc_train_df, proc_test_df])                                           # join the two sets
 
protocol_dummies = pd.get_dummies(joined_df['protocol_type'], prefix='protocol_type')          # get one-hot encoded features for protocol_type feature
service_dummies = pd.get_dummies(joined_df['service'], prefix='service')                       # get one-hot encoded features for service feature
flag_dummies = pd.get_dummies(joined_df['flag'], prefix='flag')                                # get one-hot encoded features for flag feature

joined_df = pd.concat([joined_df, protocol_dummies, service_dummies, flag_dummies], axis=1)    # join one-hot encoded features to joined dataframe

proc_train_df = joined_df[joined_df['train']==1]                                               # split train set from joined, using the train feature
proc_test_df = joined_df[joined_df['train']==0]                                                # split test set from joined, using the train feature

drop_cols = ['train', 'protocol_type', 'service', 'flag']                                      # columns to drop

proc_train_df.drop(drop_cols, axis=1, inplace=True)                                            # drop original columns from training set
proc_test_df.drop(drop_cols, axis=1, inplace=True)                                             # drop original columns from testing set
proc_train_df.head()
proc_test_df.head()
y_buffer = proc_train_df['labels'].copy()
x_buffer = proc_train_df.drop(['labels'], axis=1)

y_test = proc_test_df['labels'].copy()
x_test = proc_test_df.drop(['labels'], axis=1)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

seed_random = 315

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_buffer)

x_train, x_val, y_train, y_val = train_test_split(x_buffer, y_buffer, test_size=0.3, random_state=seed_random)
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.layers import Dense, Dropout

input_size = len(x_train.columns)

deep_model = Sequential()
deep_model.add(Dense(256, input_dim=input_size, activation='softplus'))
#deep_model.add(Dropout(0.2))
deep_model.add(Dense(128, activation='relu'))
deep_model.add(Dense(64, activation='relu'))
deep_model.add(Dense(32, activation='relu'))
#deep_model.add(Dense(18, activation='softplus'))
deep_model.add(Dense(11, activation='softmax'))

deep_model.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),
                   metrics=['accuracy'])
y_train_econded = label_encoder.transform(y_train)
y_val_econded = label_encoder.transform(y_val)
y_test_econded = label_encoder.transform(y_test)

y_train_dummy = np_utils.to_categorical(y_train_econded)
y_val_dummy = np_utils.to_categorical(y_val_econded)
y_test_dummy = np_utils.to_categorical(y_test_econded)
deep_model.fit(x_train, y_train_dummy, 
               epochs=50, 
               batch_size=2500,
               validation_data=(x_val, y_val_dummy))
deep_val_pred = deep_model.predict_classes(x_val)
deep_val_pred_decoded = label_encoder.inverse_transform(deep_val_pred)

deep_test_pred = deep_model.predict_classes(x_test)
deep_test_pred_decoded = label_encoder.inverse_transform(deep_test_pred)
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer 

# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(14,14)
    #y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
plot_cm(y_val, deep_val_pred_decoded, 'Confusion matrix for predictions on the validation set')
f1_score(y_val, deep_val_pred_decoded, average = 'macro')
plot_cm(y_test, deep_test_pred_decoded, 'Confusion matrix for predictions on the testing set')
f1_score(y_test, deep_test_pred_decoded, average = 'macro')