# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import itertools

import scipy.stats as ss



# graphics

import seaborn as sns

#plotly library

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

%matplotlib inline



import matplotlib.pyplot as plt

import ipywidgets # interactive views to not clutter the notebook



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
print(f"The dataset has {train_df.shape[0]} rows and {train_df.shape[1]} columns")
pd.set_option('display.max_columns', None) # To display all columns / rows

pd.set_option('display.max_rows', None)

train_df.sample(10)
train_df.info()
const_feat=[]



for feat in train_df.columns:

    if train_df[feat].nunique()==1:

        const_feat.append(feat)

        

print(f"There are {len(const_feat)} constant columns. These are: {const_feat}")
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 

                                    gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(train_df["SalePrice"], ax=ax_box)

sns.distplot(train_df["SalePrice"], ax=ax_hist)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)
feats = list(train_df.drop(["Id", "SalePrice"], axis=1).columns)



fig = plt.figure(figsize=(30,200))



for idx, feat in enumerate(feats):

    fig.add_subplot(40, 2,idx+1)

    if train_df[feat].dtype == "object":

        sns.scatterplot(train_df[feat], train_df["SalePrice"])

    else:

        sns.regplot(x=feat, y="SalePrice", data=train_df)

    plt.xlabel(feat)

    plt.ylabel("SalePrice")

plt.tight_layout()

plt.show()
# Countplots of categorical features



categorical = train_df.select_dtypes(include="object").columns

num_categorical= len(categorical)



print(f"There are {num_categorical} categorical features to plot.")



if isinstance(num_categorical**(0.5),int)==True:

    num_x, num_y = num_categorical**(0.5)



else:

    num_y = 2

    num_x = math.ceil(num_categorical/2)





fig, ax = plt.subplots(num_x, num_y, figsize=(20, 100))

#plt.rcParams["xtick.labelsize"] = 9

for variable, subplot in zip(categorical, ax.flatten()):

       

    sns.countplot(train_df[variable].fillna("MISSING"), ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

    subplot.set_xlabel(variable)

    subplot.set_ylabel("Count of Obs.")

plt.tight_layout() 
# Displots of numericals



numerical = list(train_df.select_dtypes(exclude="object").columns)



num_numerical = len(numerical)



print(f"There are {num_numerical} numerical features to plot.")



if isinstance(num_numerical**(0.5),int)==True:

    num_x, num_y = num_categorical**(0.5)



else:

    num_y = 2

    num_x = math.ceil(num_numerical/2)





fig, ax = plt.subplots(num_x, num_y, figsize=(20, 100))

#plt.rcParams["xtick.labelsize"] = 9

for variable, subplot in zip(numerical, ax.flatten()):

    try:

        sns.distplot(train_df[variable].fillna(-9), ax=subplot)

    except RuntimeError as re:

        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

            sns.distplot(train_df[variable].fillna(-9), kde_kws={'bw': 0.1})

        else:

            raise re

      

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

    subplot.set_xlabel(variable)

    subplot.set_ylabel("Distri of Obs.")

plt.tight_layout() 
@ipywidgets.interact

def plot(Cat_Data_Feature = train_df.select_dtypes(include=["object"]).columns):

        sns.countplot(train_df.fillna("missing")[Cat_Data_Feature])

    

         

@ipywidgets.interact

def plot(Num_Data_Feature = train_df.select_dtypes(exclude=["object"]).columns):

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 

                                    gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(train_df[Num_Data_Feature], ax=ax_box)

    try:

        sns.distplot(train_df[Num_Data_Feature].fillna(-9), ax=ax_hist)

    except RuntimeError as re:

        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

            sns.distplot(train_df[Num_Data_Feature].fillna(-9), kde_kws={'bw': 0.1})

        else:

            raise re



    ax_box.set(yticks=[])

    sns.despine(ax=ax_hist)

    sns.despine(ax=ax_box, left=True)

    
def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
#don't want to use all cols

cols_to_check = list(train_df.columns)

cols_to_check.remove("Id")
df = train_df.fillna(-9) #missings need to be recoded - otherwise algo fails

corrM = np.zeros((len(cols_to_check),len(cols_to_check)))



corr_list = []

# there's probably a nice pandas way to do this

for col1, col2 in itertools.combinations(cols_to_check, 2):

    idx1, idx2 = cols_to_check.index(col1), cols_to_check.index(col2)

    corrM[idx1, idx2] = cramers_v(df[col1], df[col2])

    corrM[idx2, idx1] = corrM[idx1, idx2]

    corr_list.append([col1,col2,corrM[idx1, idx2]]) # save output as list

    

corr = pd.DataFrame(corrM, index=cols_to_check, columns=cols_to_check)

fig, ax = plt.subplots(figsize=(60, 60))

ax = sns.heatmap(corr, annot=True, ax=ax, cmap="RdBu_r"); ax.set_title("Cramer V Correlation between Variables");
sorted(corr_list, key=lambda x: x[2], reverse=True)
print(f"There are {len([item for item in corr_list if item[2] >= 0.7])} that have an association higher than 0.7")
corr_lg_07 = [item for item in corr_list if item[2]>=0.7]

corr_lg_07
fig = plt.figure(figsize=(20,30))

for idx, [feat1, feat2, asso] in enumerate(corr_lg_07):

    fig.add_subplot(20,4,idx+1)

    sns.scatterplot(train_df[feat1], train_df[feat2])

    plt.xlabel(feat1)

    plt.ylabel(feat2)

    ax.set_title(asso)

plt.tight_layout()

plt.show()
num_feats = list(train_df.select_dtypes(exclude=["object"]).drop("Id", axis = 1).columns)



fig = plt.figure(figsize=(60,60))

matrix = np.triu(train_df[num_feats].corr())

sns.heatmap(train_df[num_feats].corr(), annot = True, mask = matrix, square=True, cbar_kws= {'orientation': 'horizontal', "shrink": 0.5}, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')

plt.tight_layout()
train_df[['MSSubClass', 'LotFrontage']].corr().iloc[0,1]
num_corr_list = []

for (feat1, feat2) in itertools.combinations(num_feats, 2):

    num_corr_list.append([feat1, feat2, train_df[[feat1, feat2]].corr().iloc[0,1]])
sorted(num_corr_list, key=lambda x: x[2], reverse=True)
num_corr_gt_05 = [item for item in num_corr_list if abs(item[2]) >=0.5]
len(num_corr_gt_05)
fig = plt.figure(figsize=(30,100))



for idx, [feat1, feat2, _] in enumerate(num_corr_gt_05):

    fig.add_subplot(20, 2,idx+1)

    sns.scatterplot(train_df[feat1], train_df[feat2])

    #sns.regplot(x=feat1, y=feat2, data=train_df)

    plt.xlabel(feat1)

    plt.ylabel(feat2)

#plt.tight_layout()

plt.show()