# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc

#%%reload_extreload_  autoreload
#%autoreload 2
%matplotlib inline
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
application_train=pd.read_csv('../input/application_train.csv')
print(application_train.info())

gc.collect()
import matplotlib.ticker as mtick

percent_coverage=application_train.notnull().sum().sort_values(ascending=False)/application_train.shape[0]#
fig,ax=plt.subplots(3,figsize=(20,12))
for i in range(3):
    if (i+1 )*45>percent_coverage.shape[0]:
        ilim=percent_coverage.shape[0]
    else:
        ilim=(i+1 )*45
    percent_coverage.iloc[i*45:ilim].to_frame().plot(kind='bar',ax=ax[i])
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    vals = ax[0].get_yticks()
    ax[i].set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    #ax[i].set_xticks(rotation=75)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=80)
plt.subplots_adjust(hspace = 1)

gc.collect()
data_types=application_train.dtypes.to_frame().reset_index().rename(columns={'index':'column_name',0:'data_type'})
data_types['data_type'].apply(str)
pivot_data_types=data_types.pivot(columns='data_type')
pivot_data_types.columns=['_'.join([x[0],str(x[1])]) for x in pivot_data_types.columns.values]
temp=pd.concat([pivot_data_types[col].sort_values().reset_index(drop=True) for col in pivot_data_types], axis=1, ignore_index=True)
temp=temp.rename(columns={y[0]:y[1] for y in zip(temp.columns,pivot_data_types.columns)})
temp=temp.dropna(how='all')
temp    
fig,ax=plt.subplots(1,1)
tar_counts=application_train.TARGET.value_counts()
tar_counts.plot.pie()
patches, text, _ = plt.pie(tar_counts, autopct='%.2f')
ax.legend(patches, labels=tar_counts.index, loc='best')
plt.suptitle('# Target in Population')
object_cols=temp.column_name_object.loc[temp.column_name_object.notnull()]
for i in object_cols:
    cat_counts=application_train[i].fillna('Missing').value_counts()
    fig,ax=plt.subplots(1,3,figsize=(12,7))
    patches, text, _ = ax[0].pie(cat_counts, autopct='%.2f')
    ax[0].legend(patches, labels=cat_counts.index, loc='best')
    ax[0].set_title('% of Category in Whole Population')
    temp2=application_train[[i,'TARGET']].fillna('Missing').groupby([i,'TARGET']).size().unstack(i).fillna(0)[cat_counts.index]
    temp2.plot(kind='bar', stacked=True,ax=ax[1])
    ax[1].set_title('# of Category in Target Population')
    temp2_per=temp2.div(temp2.sum(axis=1),axis=0)
    temp2_per.plot(kind='bar', stacked=True,ax=ax[2])
    ax[2].set_title('% of Category in Target Population')
    ax[2].legend_.remove()
    for t in temp2_per.index:
        cumsum=0
        for c in temp2_per.columns:
            if temp2_per.loc[t,c]>0.02:
                ax[2].text(x=t,y=cumsum+temp2_per.loc[t,c]/2,s=round(temp2_per.loc[t,c]*100,2))
                cumsum=cumsum+temp2_per.loc[t,c]

    ax[0].legend_.remove()
    plt.suptitle(str(i))
int_cols=temp.column_name_int64.loc[temp.column_name_int64.notnull()]
int_cols_unique_vals=application_train[int_cols].apply(lambda x:x.nunique(),axis=0)
int_cols_unique_vals
int_cat_cols=int_cols_unique_vals.loc[int_cols_unique_vals<10].drop('TARGET')
for i in int_cat_cols.index:
    cat_counts=application_train[i].fillna('Missing').value_counts()
    fig,ax=plt.subplots(1,3,figsize=(12,7))
    patches, text, _ = ax[0].pie(cat_counts, autopct='%.2f')
    ax[0].legend(patches, labels=cat_counts.index, loc='best')
    ax[0].set_title('% of Category in Whole Population')
    temp2=application_train[[i,'TARGET']].fillna('Missing').groupby([i,'TARGET']).size().unstack(i).fillna(0)[cat_counts.index]
    temp2.plot(kind='bar', stacked=True,ax=ax[1])
    ax[1].set_title('# of Category in Target Population')
    temp2_per=temp2.div(temp2.sum(axis=1),axis=0)
    temp2_per.plot(kind='bar', stacked=True,ax=ax[2])
    ax[2].set_title('% of Category in Target Population')
    ax[2].legend_.remove()
    for t in temp2_per.index:
        cumsum=0
        for c in temp2_per.columns:
            if temp2_per.loc[t,c]>0.02:
                ax[2].text(x=t,y=cumsum+temp2_per.loc[t,c]/2,s=round(temp2_per.loc[t,c]*100,2))
                cumsum=cumsum+temp2_per.loc[t,c]

    ax[0].legend_.remove()
    plt.suptitle(str(i))
import seaborn as sns
import matplotlib

#col=temp.column_name_float64[i]

for col in temp.column_name_float64.dropna():
#fig,ax=plt.subplots(3,2,figsize=(20,7), sharex=True)
    plt.subplots()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    #ax5 = plt.subplot2grid((3, 3), (2, 1))
    #plt.subplots(sharex=True)
    sns.distplot(application_train[col].dropna(),ax=ax1)
    ax1.set_title('Prob. Dist. Full Population')
    sns.distplot(application_train.loc[application_train.TARGET==0,col].dropna(),ax=ax2,color='green')
    ax2.set_title('Prob. Dist. TARGET=0 Population')

    sns.distplot(application_train.loc[application_train.TARGET==1,col].dropna(),ax=ax3,color='red')
    ax3.set_title('Prob. Dist. TARGET=1 Population')

    #plt.subplot(3,2,(2,4,6))
    application_train[['TARGET',col]].boxplot(by='TARGET',ax=ax4)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 6)
    plt.subplots_adjust(hspace=.7)
    plt.subplots_adjust(wspace=.5)
    ax4.set_title('Box Plot')
    plt.suptitle(col)

list(application_train)
