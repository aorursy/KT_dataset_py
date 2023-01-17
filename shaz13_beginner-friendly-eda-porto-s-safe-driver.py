# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# Get information on names of columns

# Only taking till 5 index,

# as names might populate the code output and is not intuitive for this simple EDA

train.columns[:5]
# Get those dtypes of those named columns

train.dtypes[:5] 
df_dtypes = pd.DataFrame({'Feature': train.columns , 'Data Type': train.dtypes.values})
df_dtypes.head(15)
## Fixing -1 with NaN values

train_v1 = train.replace(-1, np.NaN)

test_v1 = test.replace(-1, np.NaN)
plt.figure(figsize=(18,7))

sns.heatmap(train_v1.head(100).isnull() == True, cmap='viridis')
have_null_df = pd.DataFrame(train_v1.isnull().any(), columns=['Have Null?']).reset_index()
have_null_df[have_null_df['Have Null?'] == True]['index']
train_v1.dropna(inplace=True)
plt.figure(figsize=(16,11))

sns.heatmap(train_v1.head(100).corr(), cmap='viridis')
binary_feat = [c for c in train_v1.columns if c.endswith("bin")]

categorical_feat = [c for c in train_v1.columns if c.endswith("cat")]
plt.figure(figsize=(17,20))

for i, c in enumerate(binary_feat):

    ax = plt.subplot(6,3,i+1)

    sns.countplot(train_v1[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)
plt.figure(figsize=(17,20))

for i, c in enumerate(categorical_feat):

    ax = plt.subplot(6,3,i+1)

    sns.countplot(train_v1[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)
plt.figure(figsize=(17,6))

sns.countplot(train_v1['ps_car_11_cat'])
print ("There are {} unique values for ps_car_11_cat" .format(train_v1['ps_car_11_cat'].nunique()))
train_v1['ps_car_11_cat'].value_counts().head(10)
continuous_feat= [i for i in train_v1.columns if 

                    ((i not in binary_feat) and (i not in categorical_feat) and (i not in ["target", "id"]))]
train_v1[continuous_feat].head(5)
ind_feat = [c for c in continuous_feat if c.startswith("ps_ind")]

reg_feat = [c for c in continuous_feat if c.startswith("ps_reg")]

car_feat = [c for c in continuous_feat if c.startswith("ps_car")]

calc_feat = [c for c in continuous_feat if c.startswith("ps_calc")]

target = ['target']
plt.figure(figsize=(17,11))

sns.heatmap(train_v1[ind_feat+ calc_feat + car_feat + reg_feat + target].corr(), cmap= plt.cm.inferno)
plt.figure(figsize=(17,11))

sns.heatmap(train_v1[ind_feat+ car_feat + reg_feat + target].corr(), cmap= 'viridis', annot=True)