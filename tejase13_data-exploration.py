# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading the dataset
application = pd.read_csv("../input/application_train.csv")
application.head()
target_col = application['TARGET'].value_counts()
x = target_col.index.tolist()
y = target_col.values
total_observations = y.sum()

'''Matplotlib code starts here'''
fig = plt.figure()
ax = fig.add_subplot(111)
#or use this: fig, ax = plt.subplots()
bar_plot = ax.bar(x, y, width=0.5, color='gr')
ax.set_xticks(x)
ax.set_xlabel('Target variable values')
ax.set_xticklabels(['Repay Loan','Default Loan'], rotation=0, fontsize=15)

ax.set_ylim(ymin=0, ymax=300000)
ax.set_ylabel('Count of target variable')
ax.set_yticks(np.arange(0, 325000, 25000))

#The commented code below will convert y axis into percentage
# formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
# ax.yaxis.set_major_formatter(formatter)

for rect in bar_plot:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
            '%.2f' % ((height/total_observations)*100) + "%", ha='center', va='bottom', fontsize=15)

plt.title('Distribution of target variable')
plt.show()
corr_vars = application.corr()['TARGET'].sort_values()
positive_corr = corr_vars[corr_vars > 0].sort_values(ascending=False).drop('TARGET')[:10]
negative_corr = corr_vars[corr_vars < 0].sort_values(ascending=True)[:10]
positive_corr
negative_corr
cols_with_missing_values = len(application.isnull().sum()[application.isnull().sum() > 0])
total_cols = application.shape[1] - 1
print("{} columns out of total {} columns have missing values".format(cols_with_missing_values, total_cols))
missing_cols_prcnt = application.isnull().sum()/application.shape[0] * 100
high_missing_values = missing_cols_prcnt[missing_cols_prcnt > 50]
high_missing_values_index = high_missing_values.index.tolist()
print("{} columns have more than 50% missing values".format(len(high_missing_values_index)))
print("Top 10 columns with highest proportion of missing values:")
high_missing_values.sort_values(ascending=False)[:10]
correlations = application.corr()['TARGET'].sort_values()
corr_missing_cols = correlations.reindex(high_missing_values_index).sort_values()
missing_cols_to_be_dropped = corr_missing_cols.index.difference(corr_missing_cols[(corr_missing_cols > 0.02) | (corr_missing_cols < -0.02)].index).tolist()
print("Number of columns that are to be dropped: {}".format(len(missing_cols_to_be_dropped)))
pd.options.display.float_format = '{:.2f}'.format #Used to avoid scientific notation
application[["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"]].describe()
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(application["AMT_CREDIT"], color = 'blue', ax = ax1)
sns.boxplot(x=application["AMT_CREDIT"], ax = ax2)
plt.show()
import matplotlib.gridspec as gridspec
G = gridspec.GridSpec(4, 4)
G.update(wspace=0.25, hspace=0.5)
plt.figure(figsize = (20,20))
axes_l = []
numeric_cols = ["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"]
row_index = 0
col_index = 0
axes_count = 0
for i, col in enumerate(numeric_cols):
    #Plotting distribution plot 
    row_index = i
    axes_l.append(plt.subplot(G[row_index, col_index]))
    if application[col].isnull().sum() == 0:
        sns.distplot(application[col], color = 'blue', ax = axes_l[axes_count])
    else:
        sns.distplot(application[col].dropna(), color = 'blue', ax = axes_l[axes_count])
    plt.title('Distribution plot:'+col)
              
    axes_count+=1
    col_index+=1
    #Plotting boxplot
    axes_l.append(plt.subplot(G[row_index, col_index]))
    if application[col].isnull().sum() == 0:
        sns.boxplot(application[col], ax = axes_l[axes_count])
    else:
        sns.boxplot(application[col].dropna(), ax = axes_l[axes_count])
    plt.title('Boxplot:'+col)    
              
    axes_count += 1
    col_index = 0

# axes_l
plt.show()
application[['AMT_GOODS_PRICE', 'AMT_CREDIT']].corr()
application[['AMT_GOODS_PRICE', 'AMT_CREDIT']].isnull().sum()
plt.scatter(x = application['AMT_GOODS_PRICE'], y = application['AMT_CREDIT'], alpha=0.5)
plt.show()
application["AGE"] = application["DAYS_BIRTH"].abs()/365
application["AGE"].describe()
#Plotting age distribution
fig, ax = plt.subplots()
sns.distplot(application["AGE"], color = 'blue', bins=20, kde=False, norm_hist=False)
plt.show()
sns.kdeplot(application[application["TARGET"]==1]["AGE"], color = 'blue', label = 'target == 1')
sns.kdeplot(application[application["TARGET"]==0]["AGE"], color = 'red', label = 'target == 0')
plt.xlabel('Age')
plt.ylabel('Density')
plt.plot()
days_cols = []
days_cols = [col for col in application.columns if col.find("DAYS")!=-1]
application[days_cols].describe()
application[application["DAYS_EMPLOYED"] > 0]["DAYS_EMPLOYED"].shape
#Creating the binary column and setting the value = 1 wherevr the value of days_employed will be 365243
application["DAYS_EMPLOYED_ANOMALY"] = 0
anomalous_indices = application[application["DAYS_EMPLOYED"] > 0]["DAYS_EMPLOYED"].index
application.loc[anomalous_indices, "DAYS_EMPLOYED_ANOMALY"] = 1
#Replacing anomalies with 0
application["DAYS_EMPLOYED"].replace(365243, 0, inplace=True)
application[["DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOMALY"]][:15]
application[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].describe()
#The code below was obtained from the kaggle kernel mentioned above
#The code can also be implemented in a way similar to what was described in AMT columns

plt.figure(figsize = (12, 12))
# iterate through the new features
for i, feature in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(application.loc[application['TARGET'] == 0, feature], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(application.loc[application['TARGET'] == 1, feature], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % feature)
    plt.xlabel('%s' % feature); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
print("The following numbers are indicative of missing values in %")
application[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum()/application.shape[0] * 100
#One line of code to plot bar plot
application["CODE_GENDER"].value_counts().plot.bar()
gen_typ = application["CODE_GENDER"].value_counts()
gen_typ_vals = gen_typ.values
gen_typ_idx =  gen_typ.index.tolist()
print(gen_typ_vals, gen_typ_idx)
plt.bar(gen_typ_idx, gen_typ_vals)
repay = []
default = []
gender = application["CODE_GENDER"].unique()
for g in gender:
    default.append(application[(application["CODE_GENDER"]==g) & (application["TARGET"]==1)].shape[0]/application[(application["CODE_GENDER"]==g)].shape[0] * 100)
    repay.append(application[(application["CODE_GENDER"]==g) & (application["TARGET"]==0)].shape[0]/application[(application["CODE_GENDER"]==g)].shape[0] * 100)

fig, ax = plt.subplots(figsize=(10,5))
pos = list(range(len(gender)))
width = 0.33
plt.bar(pos, repay, width, color='g')
plt.bar([p+width for p in pos], default, width, color='r')
plt.legend(["Repay", "Default"])

#X-axis manipulations
ax.set_xticks([p+0.5*width for p in pos])
ax.set_xticklabels(list(gender))
ax.set_xlabel('Gender')

#Y-axis manipulations
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x/100) for x in vals])
ax.set_ylabel('% of repaid/defaulted loans')

plt.show()
repay = []
default = []
gender = application["CODE_GENDER"].unique()
for g in gender:
    default.append(application[(application["CODE_GENDER"]==g) & (application["TARGET"]==1)].shape[0]/application[(application["CODE_GENDER"]==g)].shape[0] * 100)
    repay.append(application[(application["CODE_GENDER"]==g) & (application["TARGET"]==0)].shape[0]/application[(application["CODE_GENDER"]==g)].shape[0] * 100)

fig, ax = plt.subplots(figsize=(10,5))
pos = list(range(len(gender)))
width = 0.5
plt.bar(pos, repay, width=width, color='g', label='Repay')
plt.bar(pos, default, width=width, bottom=repay, color='r', label='Default')
plt.legend(["Repay", "Default"])

ax.set_xticks([p for p in pos])
ax.set_xticklabels(list(gender))
ax.set_xlabel('Gender')

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x/100) for x in vals])
plt.show()
repay = []
default = []
inc_typ = application["NAME_INCOME_TYPE"].unique()
for g in inc_typ:
    default.append(application[(application["NAME_INCOME_TYPE"]==g) & (application["TARGET"]==1)].shape[0]/application[(application["NAME_INCOME_TYPE"]==g)].shape[0] * 100)
    repay.append(application[(application["NAME_INCOME_TYPE"]==g) & (application["TARGET"]==0)].shape[0]/application[(application["NAME_INCOME_TYPE"]==g)].shape[0] * 100)
    
fig,ax = plt.subplots(figsize=(15,5))
width = 0.7
pos = list(range(len(repay)))
plt.bar(pos, repay, width=width, color='g')
plt.bar(pos, default, width=width, bottom=repay, color='r', label='Default')
ax.set_xticks([p for p in pos])
ax.set_xticklabels(inc_typ)
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
inc_type = application["NAME_INCOME_TYPE"].value_counts()
inc_type.plot.bar(color='b')
inc_type


