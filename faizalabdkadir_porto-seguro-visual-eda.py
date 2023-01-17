import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Any results you write to the current directory are saved as output.
train.head()
#check for null value

train.isnull().any().any()
Counter(train.dtypes.values)
train2= (train.isnull().sum() / len(train)) * 100 

misval = train2.drop(train2[train2 == 0].index).sort_values(ascending=False)[:30]

missing = pd.DataFrame({'Missing %' :misval})

missing.head(10)
train_copy = train.replace(-1, np.NaN)
train_copy= (train_copy.isnull().sum() / len(train_copy)) * 100 

train_copy = train_copy.drop(train_copy[train_copy == 0].index).sort_values(ascending=False)[:30]

missing = pd.DataFrame({'Missing %' :train_copy})

missing.head(10)
f,ax=plt.subplots(1,2,figsize=(18,9))

train['target'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%', fontsize =18, ax=ax[0],shadow=True)

ax[0].set_title('Target')

ax[0].set_ylabel('')

sns.countplot('target',data=train, ax=ax[1])

ax[1].set_title('Target')

plt.show()
# grouped by type of data

train_float = train.select_dtypes(include=['float64'])

train_int = train.select_dtypes(include=['int64'])
#Group by type of features

bin_col = [col for col in train.columns if '_bin' in col] #binary

cat_col = [col for col in train.columns if '_cat' in col] #categorical



# Numeric Features

num_col = [x for x in train.columns if x[-3:] not in ['bin', 'cat']]



#Group by individual, region, car and calculated fields

ind_col = [col for col in train.columns if '_ind_' in col] #individual

car_col = [col for col in train.columns if '_car_' in col] #car

reg_col = [col for col in train.columns if '_reg_' in col] #region

calc_col = [col for col in train.columns if '_calc_' in col] #calculation



zero_list = []

one_list = []

for col in bin_col:

    zero_list.append((train[col]==0).sum())

    one_list.append((train[col]==1).sum())
print('\nFeature - Float')

for x in train_float:

    unique_cat=len(train_float[x].unique())

    print("Feature '{x}' has {unique_cat} unique categories ".format(x=x, unique_cat=unique_cat))
train['ps_reg_01_range']=pd.qcut(train_float['ps_reg_01'],3,labels=['1','2','3'])

train['ps_reg_02_range']=pd.qcut(train_float['ps_reg_02'],4,labels=['1','2','3','4'])

train['ps_reg_03_range']=pd.qcut(train_float['ps_reg_03'],4,labels=['1','2','3','4'])

train['ps_car_12_range']=pd.qcut(train_float['ps_car_12'],5,labels=['1','2','3','4','5'])

train['ps_car_13_range']=pd.qcut(train_float['ps_car_13'],10,labels=['1','2','3','4','5','6','7','8','9','10'])

train['ps_car_14_range']=pd.qcut(train_float['ps_car_14'],2, labels=['1','2'])

train['ps_car_15_range']=pd.qcut(train_float['ps_car_15'],3, labels=['1','2','3'])

train['ps_calc_01_range']=pd.qcut(train_float['ps_calc_01'],8, labels=['1','2','3','4','5','6','7','8'])

train['ps_calc_02_range']=pd.qcut(train_float['ps_calc_02'],4,labels=['1','2','3','4'])

train['ps_calc_03_range']=pd.qcut(train_float['ps_calc_03'],4,labels=['1','2','3','4'])
### Corralation matrix heatmap

cor_matrix = train[num_col].corr().round(2)

# Plotting heatmap 

fig = plt.figure(figsize=(15,15));

sns.heatmap(cor_matrix, annot=True, center=0, cmap = sns.diverging_palette(250, 10, as_cmap=True), ax=plt.subplot(111));

plt.show()
f,ax=plt.subplots(9,4,figsize=(20,55))

sns.countplot('ps_ind_01',data=train,ax=ax[0,0])

ax[0,0].set_title('Count ps_ind_01')

sns.factorplot('ps_ind_01','target',data=train,ax=ax[0,1])

ax[0,1].set_title('Target vs ps_ind_01')

plt.close(2)

sns.countplot('ps_ind_02_cat',data=train,ax=ax[0,2])

ax[0,2].set_title('Count ps_ind_02_cat')

sns.factorplot('ps_ind_02_cat','target',data=train,ax=ax[0,3])

ax[0,3].set_title('Target vs ps_ind_02_cat')

plt.close(2)

sns.countplot('ps_ind_03',data=train,ax=ax[1,0])

ax[1,0].set_title('Count ps_ind_03')

sns.factorplot('ps_ind_03','target',data=train,ax=ax[1,1])

ax[1,1].set_title('Target vs ps_ind_03')

plt.close(2)

sns.countplot('ps_ind_04_cat',data=train,ax=ax[1,2])

ax[1,2].set_title('Count ps_ind_04_cat')

sns.factorplot('ps_ind_04_cat','target',data=train,ax=ax[1,3])

ax[1,3].set_title('Target vs ps_ind_04_cat')

plt.close(2)

sns.countplot('ps_ind_05_cat',data=train,ax=ax[2,0])

ax[2,0].set_title('Count ps_ind_05_cat')

sns.factorplot('ps_ind_05_cat','target',data=train,ax=ax[2,1])

ax[2,1].set_title('Target vs ps_ind_05_cat')

plt.close(2)

sns.countplot('ps_ind_06_bin',data=train,ax=ax[2,2])

ax[2,2].set_title('Count ps_ind_06_bin')

sns.factorplot('ps_ind_06_bin','target',data=train,ax=ax[2,3])

ax[2,3].set_title('Target vs ps_ind_06_bin')

plt.close(2)

sns.countplot('ps_ind_07_bin',data=train,ax=ax[3,0])

ax[3,0].set_title('Count ps_ind_07_bin')

sns.factorplot('ps_ind_07_bin','target',data=train,ax=ax[3,1])

ax[3,1].set_title('Target vs ps_ind_07_bin')

plt.close(2)

sns.countplot('ps_ind_08_bin',data=train,ax=ax[3,2])

ax[3,2].set_title('Count ps_ind_08_bin')

sns.factorplot('ps_ind_08_bin','target',data=train,ax=ax[3,3])

ax[3,3].set_title('Target vs ps_ind_08_bin')

plt.close(2)

sns.countplot('ps_ind_09_bin',data=train,ax=ax[4,0])

ax[4,0].set_title('Count ps_ind_09_bin')

sns.factorplot('ps_ind_09_bin','target',data=train,ax=ax[4,1])

ax[4,1].set_title('Target vs ps_ind_10_bin')

plt.close(2)

sns.countplot('ps_ind_10_bin',data=train,ax=ax[4,2])

ax[4,2].set_title('Count ps_ind_10_bin')

sns.factorplot('ps_ind_10_bin','target',data=train,ax=ax[4,3])

ax[4,3].set_title('Target vs ps_ind_10_bin')

plt.close(2)

sns.countplot('ps_ind_11_bin',data=train,ax=ax[5,0])

ax[5,0].set_title('Count ps_ind_11_bin')

sns.factorplot('ps_ind_11_bin','target',data=train,ax=ax[5,1])

ax[5,1].set_title('Target vs ps_ind_11_bin')

plt.close(2)

sns.countplot('ps_ind_12_bin',data=train,ax=ax[5,2])

ax[5,2].set_title('Count ps_ind_12_bin')

sns.factorplot('ps_ind_12_bin','target',data=train,ax=ax[5,3])

ax[5,3].set_title('Target vs ps_ind_12_bin')

plt.close(2)

sns.countplot('ps_ind_13_bin',data=train,ax=ax[6,0])

ax[6,0].set_title('Count ps_ind_13_bin')

sns.factorplot('ps_ind_13_bin','target',data=train,ax=ax[6,1])

ax[6,1].set_title('Target vs ps_ind_13_bin')

plt.close(2)

sns.countplot('ps_ind_14',data=train,ax=ax[6,2])

ax[6,2].set_title('Count ps_ind_14')

sns.factorplot('ps_ind_14','target',data=train,ax=ax[6,3])

ax[6,3].set_title('Target vs ps_ind_14')

plt.close(2)

sns.countplot('ps_ind_15',data=train,ax=ax[7,0])

ax[7,0].set_title('Count ps_ind_15')

sns.factorplot('ps_ind_15','target',data=train,ax=ax[7,1])

ax[7,1].set_title('Target vs ps_ind_15')

plt.close(2)

sns.countplot('ps_ind_16_bin',data=train,ax=ax[7,2])

ax[7,2].set_title('Count ps_ind_16_bin')

sns.factorplot('ps_ind_16_bin','target',data=train,ax=ax[7,3])

ax[7,3].set_title('Target vs ps_ind_16_bin')

plt.close(2)

sns.countplot('ps_ind_17_bin',data=train,ax=ax[8,0])

ax[8,0].set_title('Count ps_ind_17_bin')

sns.factorplot('ps_ind_17_bin','target',data=train,ax=ax[8,1])

ax[8,1].set_title('Target vs ps_ind_17_bin')

plt.close(2)

sns.countplot('ps_ind_18_bin',data=train,ax=ax[8,2])

ax[8,2].set_title('Count ps_ind_18_bin')

sns.factorplot('ps_ind_18_bin','target',data=train,ax=ax[8,3])

ax[8,3].set_title('Target vs ps_ind_18_bin')

plt.close(2)

plt.show()
f,ax=plt.subplots(8,4,figsize=(20,50))

sns.countplot('ps_car_01_cat',data=train,ax=ax[0,0])

ax[0,0].set_title('Count ps_car_01_cat')

sns.factorplot('ps_car_01_cat','target',data=train,ax=ax[0,1])

ax[0,1].set_title('Target vs ps_car_01_cat')

plt.close(2)

sns.countplot('ps_car_02_cat',data=train,ax=ax[0,2])

ax[0,2].set_title('Count ps_car_02_cat')

sns.factorplot('ps_car_02_cat','target',data=train,ax=ax[0,3])

ax[0,3].set_title('Target vs ps_car_02_cat')

plt.close(2)

sns.countplot('ps_car_03_cat',data=train,ax=ax[1,0])

ax[1,0].set_title('Count ps_car_03_cat')

sns.factorplot('ps_car_03_cat','target',data=train,ax=ax[1,1])

ax[1,1].set_title('Target vs ps_car_03_cat')

plt.close(2)

sns.countplot('ps_car_04_cat',data=train,ax=ax[1,2])

ax[1,2].set_title('Count ps_car_04_cat')

sns.factorplot('ps_car_04_cat','target',data=train,ax=ax[1,3])

ax[1,3].set_title('Target vs ps_car_04_cat')

plt.close(2)

sns.countplot('ps_car_05_cat',data=train,ax=ax[2,0])

ax[2,0].set_title('Count ps_car_05_cat')

sns.factorplot('ps_car_05_cat','target',data=train,ax=ax[2,1])

ax[2,1].set_title('Target vs ps_car_05_cat')

plt.close(2)

sns.countplot('ps_car_06_cat',data=train,ax=ax[2,2])

ax[2,2].set_title('Count ps_car_06_cat')

sns.factorplot('ps_car_06_cat','target',data=train,ax=ax[2,3])

ax[2,3].set_title('Target vs ps_car_06_cat')

plt.close(2)

sns.countplot('ps_car_07_cat',data=train,ax=ax[3,0])

ax[3,0].set_title('Count ps_car_07_cat')

sns.factorplot('ps_car_07_cat','target',data=train,ax=ax[3,1])

ax[3,1].set_title('Target vs ps_car_07_cat')

plt.close(2)

sns.countplot('ps_car_08_cat',data=train,ax=ax[3,2])

ax[3,2].set_title('Count ps_car_08_cat')

sns.factorplot('ps_car_08_cat','target',data=train,ax=ax[3,3])

ax[3,3].set_title('Target vs ps_car_08_cat')

plt.close(2)

sns.countplot('ps_car_09_cat',data=train,ax=ax[4,0])

ax[4,0].set_title('Count ps_car_09_cat')

sns.factorplot('ps_car_09_cat','target',data=train,ax=ax[4,1])

ax[4,1].set_title('Target vs ps_car_09_cat')

plt.close(2)

sns.countplot('ps_car_10_cat',data=train,ax=ax[4,2])

ax[4,2].set_title('Count ps_car_10_cat')

sns.factorplot('ps_car_10_cat','target',data=train,ax=ax[4,3])

ax[4,3].set_title('Target vs ps_car_10_cat')

plt.close(2)

sns.countplot('ps_car_11_cat',data=train,ax=ax[5,0])

ax[5,0].set_title('Count ps_car_11_cat')

sns.factorplot('ps_car_11_cat','target',data=train,ax=ax[5,1])

ax[5,1].set_title('Target vs ps_car_11_cat')

plt.close(2)

sns.countplot('ps_car_11',data=train,ax=ax[5,0])

ax[5,0].set_title('Count ps_car_11')

sns.factorplot('ps_car_11','target',data=train,ax=ax[5,1])

ax[5,1].set_title('Target vs ps_car_11')

plt.close(2)

sns.countplot('ps_car_12_range',data=train,ax=ax[5,2])

ax[5,2].set_title('Count ps_car_12')

sns.factorplot('ps_car_12_range','target',data=train,ax=ax[5,3])

ax[5,3].set_title('Target vs ps_car_12')

plt.close(2)

sns.countplot('ps_car_13_range',data=train,ax=ax[6,0])

ax[6,0].set_title('Count ps_car_13')

sns.factorplot('ps_car_13_range','target',data=train,ax=ax[6,1])

ax[6,1].set_title('Target vs ps_car_13')

plt.close(2)

sns.countplot('ps_car_14_range',data=train,ax=ax[6,2])

ax[6,2].set_title('Count ps_car_14')

sns.factorplot('ps_car_14_range','target',data=train,ax=ax[6,3])

ax[6,3].set_title('Target vs ps_car_14')

plt.close(2)

sns.countplot('ps_car_15_range',data=train,ax=ax[7,0])

ax[7,0].set_title('Count ps_car_15')

sns.factorplot('ps_car_15_range','target',data=train,ax=ax[7,1])

ax[7,1].set_title('Target vs ps_car_15')

plt.close(2)
f,ax=plt.subplots(2,4,figsize=(20,12))

sns.countplot('ps_reg_01_range',data=train,ax=ax[0,0])

ax[0,0].set_title('Count ps_reg_01')

sns.factorplot('ps_reg_01_range','target',data=train,ax=ax[0,1])

ax[0,1].set_title('Target vs ps_reg_01')

plt.close(2)

sns.countplot('ps_reg_02_range',data=train,ax=ax[0,2])

ax[0,2].set_title('Count ps_reg_02')

sns.factorplot('ps_reg_02_range','target',data=train,ax=ax[0,3])

ax[0,3].set_title('Target vs ps_reg_02')

plt.close(2)

sns.countplot('ps_reg_03_range',data=train,ax=ax[1,0])

ax[1,0].set_title('Count ps_reg_03')

sns.factorplot('ps_reg_03_range','target',data=train,ax=ax[1,1])

ax[1,1].set_title('Target vs ps_reg_03')

plt.close(2)

plt.show()
f,ax=plt.subplots(10,4,figsize=(20,60))

sns.countplot('ps_calc_01_range',data=train,ax=ax[0,0])

ax[0,0].set_title('Count ps_calc_01')

sns.factorplot('ps_calc_01_range','target',data=train,ax=ax[0,1])

ax[0,1].set_title('Target vs ps_calc_01')

plt.close(2)

sns.countplot('ps_calc_02_range',data=train,ax=ax[0,2])

ax[0,2].set_title('Count ps_calc_02')

sns.factorplot('ps_calc_02_range','target',data=train,ax=ax[0,3])

ax[0,3].set_title('Target vs ps_calc_02')

plt.close(2)

sns.countplot('ps_calc_03_range',data=train,ax=ax[1,0])

ax[1,0].set_title('Count ps_calc_03')

sns.factorplot('ps_calc_03_range','target',data=train,ax=ax[1,1])

ax[1,1].set_title('Target vs ps_calc_03')

plt.close(2)

sns.countplot('ps_calc_04',data=train,ax=ax[1,2])

ax[1,2].set_title('Count ps_calc_04')

sns.factorplot('ps_calc_04','target',data=train,ax=ax[1,3])

ax[1,3].set_title('Target vs ps_calc_04')

plt.close(2)

sns.countplot('ps_calc_05',data=train,ax=ax[2,0])

ax[2,0].set_title('Count ps_calc_05')

sns.factorplot('ps_calc_05','target',data=train,ax=ax[2,1])

ax[2,1].set_title('Target vs ps_calc_05')

plt.close(2)

sns.countplot('ps_calc_06',data=train,ax=ax[2,2])

ax[2,2].set_title('Count ps_calc_06')

sns.factorplot('ps_calc_06','target',data=train,ax=ax[2,3])

ax[2,3].set_title('Target vs ps_calc_06')

plt.close(2)

sns.countplot('ps_calc_07',data=train,ax=ax[3,0])

ax[3,0].set_title('Count ps_calc_07')

sns.factorplot('ps_calc_07','target',data=train,ax=ax[3,1])

ax[3,1].set_title('Target vs ps_calc_07')

plt.close(2)

sns.countplot('ps_calc_08',data=train,ax=ax[3,2])

ax[3,2].set_title('Count ps_calc_08')

sns.factorplot('ps_calc_08','target',data=train,ax=ax[3,3])

ax[3,3].set_title('Target vs ps_calc_08')

plt.close(2)

sns.countplot('ps_calc_09',data=train,ax=ax[4,0])

ax[4,0].set_title('Count ps_calc_09')

sns.factorplot('ps_calc_09','target',data=train,ax=ax[4,1])

ax[4,1].set_title('Target vs ps_calc_09')

plt.close(2)

sns.countplot('ps_calc_10',data=train,ax=ax[4,2])

ax[4,2].set_title('Count ps_calc_10')

sns.factorplot('ps_calc_10','target',data=train,ax=ax[4,3])

ax[4,3].set_title('Target vs ps_calc_10')

plt.close(2)

sns.countplot('ps_calc_11',data=train,ax=ax[5,0])

ax[5,0].set_title('Count ps_calc_11')

sns.factorplot('ps_calc_11','target',data=train,ax=ax[5,1])

ax[5,1].set_title('Target vs ps_calc_11')

plt.close(2)

sns.countplot('ps_calc_12',data=train,ax=ax[5,2])

ax[5,2].set_title('Count ps_calc_12')

sns.factorplot('ps_calc_12','target',data=train,ax=ax[5,3])

ax[5,3].set_title('Target vs ps_calc_12')

plt.close(2)

sns.countplot('ps_calc_13',data=train,ax=ax[6,0])

ax[6,0].set_title('Count ps_calc_13')

sns.factorplot('ps_calc_13','target',data=train,ax=ax[6,1])

ax[6,1].set_title('Target vs ps_calc_13')

plt.close(2)

sns.countplot('ps_calc_14',data=train,ax=ax[6,2])

ax[6,2].set_title('Count ps_calc_14')

sns.factorplot('ps_calc_14','target',data=train,ax=ax[6,3])

ax[6,3].set_title('Target vs ps_calc_14')

plt.close(2)

sns.countplot('ps_calc_15_bin',data=train,ax=ax[7,0])

ax[7,0].set_title('Count ps_calc_15_bin')

sns.factorplot('ps_calc_15_bin','target',data=train,ax=ax[7,1])

ax[7,1].set_title('Target vs ps_calc_15_bin')

plt.close(2)

sns.countplot('ps_calc_16_bin',data=train,ax=ax[7,2])

ax[7,2].set_title('Count ps_calc_16_bin')

sns.factorplot('ps_calc_16_bin','target',data=train,ax=ax[7,3])

ax[7,3].set_title('Target vs ps_calc_16_bin')

plt.close(2)

sns.countplot('ps_calc_17_bin',data=train,ax=ax[8,0])

ax[8,0].set_title('Count ps_calc_17_bin')

sns.factorplot('ps_calc_17_bin','target',data=train,ax=ax[8,1])

ax[8,1].set_title('Target vs ps_calc_17_bin')

plt.close(2)

sns.countplot('ps_calc_18_bin',data=train,ax=ax[8,2])

ax[8,2].set_title('Count ps_calc_18_bin')

sns.factorplot('ps_calc_18_bin','target',data=train,ax=ax[8,3])

ax[8,3].set_title('Target vs ps_calc_18_bin')

plt.close(2)

sns.countplot('ps_calc_19_bin',data=train,ax=ax[9,0])

ax[9,0].set_title('Count ps_calc_19_bin')

sns.factorplot('ps_calc_19_bin','target',data=train,ax=ax[9,1])

ax[9,1].set_title('Target vs ps_calc_19_bin')

plt.close(2)

sns.countplot('ps_calc_20_bin',data=train,ax=ax[9,2])

ax[9,2].set_title('Count ps_calc_20_bin')

sns.factorplot('ps_calc_20_bin','target',data=train,ax=ax[9,3])

ax[9,3].set_title('Target vs ps_calc_20_bin')

plt.close(2)

plt.show()
trace1 = go.Bar(x=bin_col, y=zero_list ,name='Zero count')

trace2 = go.Bar( x=bin_col,y=one_list, name='One count')

data = [trace1, trace2]

layout = go.Layout(barmode='stack', title='Count of 1 and 0 in binary variables')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
print('\nFeature - Individual')

for col_name in ind_col:

    unique_cat=len(train[col_name].unique())

    #dt=dtypes(col_name) 

    print("Feature '{col_name}' has {unique_cat} unique categories ".format(col_name=col_name, unique_cat=unique_cat))

print('\nFeature - Region')

for col_name in reg_col:

    unique_cat=len(train[col_name].unique())

    print("Feature '{col_name}' has {unique_cat} unique categories ".format(col_name=col_name, unique_cat=unique_cat))

print('\nFeature - Car')

for col_name in car_col:

    unique_cat=len(train[col_name].unique())

    print("Feature '{col_name}' has {unique_cat} unique categories ".format(col_name=col_name, unique_cat=unique_cat))

print('\nFeature - Calculation')

for col_name in calc_col:

    unique_cat=len(train[col_name].unique())

    print("Feature '{col_name}' has {unique_cat} unique categories ".format(col_name=col_name, unique_cat=unique_cat))  
train.head()