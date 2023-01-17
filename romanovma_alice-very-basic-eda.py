import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
PATH_TO_DATA = ('../../data')
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')
# Switch time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')
y = train_df['target']

# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]
from datetime import timezone

# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add time features
full_new_feat['month'] = full_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
full_new_feat['day'] = full_df['time1'].apply(lambda ts: ts.day)
full_new_feat['dow'] = full_df['time1'].apply(lambda x: x.date().weekday())
full_new_feat['hour'] = full_df['time1'].apply(lambda ts: ts.hour)
full_new_feat['minute'] = full_df['time1'].apply(lambda ts: ts.minute)
full_new_feat['second'] = full_df['time1'].apply(lambda ts: ts.second)
full_new_feat['count'] = full_df[times].apply(lambda x: x.nunique(), axis=1)
def plot_features(feature):
    tmp = full_new_feat.ix[:idx_split, feature].to_frame()
    tmp['target'] = y
    stm_vs_target = tmp.groupby(feature)['target'].sum()

    # Plot the table
    # print(stm_vs_target)

    # Plot the graph
    x_axis = stm_vs_target.index
    y_axis = stm_vs_target.values
    fig=plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(111)
    line1 = ax1.plot(y_axis,'ro',label='line1')
    plt.xticks(range(len(y_axis)), x_axis)
    ax1.set_ylabel('y values',fontsize=12)
    lines = line1
    labels = [l.get_label() for l in lines]
    ax1.set_xlabel('YYYYMM',fontsize=14)
    ax1.set_ylabel('Number of Sessions',fontsize=14)
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.suptitle(u'Alice', y=1.0, fontsize=17)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.4)
    plt.show() 

    stm_vs_target = tmp.groupby(feature)['target'].count()

    # Plot the graph
    x_axis = stm_vs_target.index
    y_axis = stm_vs_target.values
    fig=plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(111)
    line1 = ax1.plot(y_axis,'ro',label='line1')
    plt.xticks(range(len(y_axis)), x_axis)
    ax1.set_ylabel('y values',fontsize=12)
    lines = line1
    labels = [l.get_label() for l in lines]
    ax1.set_xlabel('YYYYMM',fontsize=14)
    ax1.set_ylabel('Number of Sessions',fontsize=14)
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.suptitle(u'Others', y=1.0, fontsize=17)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.4)
    plt.show() 
plot_features('month')
full_new_feat['month1'] = full_new_feat['month'].apply(lambda x: 1 if (x <= 201309) else 0)
full_new_feat['month2'] = full_new_feat['month'].apply(lambda x: 1 if (x > 201309) & (x <= 201401) else 0)
full_new_feat['month3'] = full_new_feat['month'].apply(lambda x: 1 if (x > 201401) else 0)
full_new_feat.drop(['month'], axis=1, inplace=True)
plot_features('hour')