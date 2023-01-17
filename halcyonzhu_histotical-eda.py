# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



import os

from datetime import date 

from sklearn.metrics import f1_score

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

import gc



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

from sklearn.kernel_ridge import KernelRidge



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



from tqdm import tqdm



from itertools import product

import ast



from datetime import timedelta 

from time import time



import seaborn as sns

import random
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno  # 用于可视化缺失值分布

import scipy.stats as st
#Load data

hist_data  =  pd.read_excel('../input/history/Historical Sales.xlsx', usecols=['YM','Date','SalesRegion','SKU Code','VolumeHL'])

hist_data.head()

hist_data.isnull().sum()
#把master data和commercial planning 两张表都读进来

md = pd.read_excel('../input/master/(Replace)Product Master Data.xlsx')

md.head()

pl = pd.read_excel('../input/planning/(Replace)Commercial Planning.xlsx')

pl.head()
#这里用曦然的方法 导入planning以后 做替换处理，将SKU Code替换成Brand

pl = pl.rename(columns={'SKU Code':'Brand'})



pl['Brand'].replace([1,2,3,4], 'middle_end_2',inplace=True)

pl['Brand'].replace([5,6,7,8,9,10,11,28,37], 'high_end_2',inplace=True)

pl['Brand'].replace([34], 'high_end_1',inplace=True)

pl['Brand'].replace([12], 'high_end_3',inplace=True)

pl['Brand'].replace([44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,68,69], 'low_end_1',inplace=True)

pl['Brand'].replace([13,14,15,16,17,18,19,20,25,26,27,29,31,36,41,42,43,45], 'low_end_2',inplace=True)

pl['Brand'].replace([23,24,32,33,35,39,40,67], 'low_end_3',inplace=True)

pl['Brand'].replace([65], 'middle_end_1',inplace=True)

pl['Brand'].replace([21,22,30], 'middle_end_3',inplace=True)

pl['Brand'].replace([38], 'superhigh_end_1',inplace=True)



pl.head()
md.isnull().sum()# 0 0 0 0 

pl.isnull().sum()# price : 1374

pl['Price'].isnull()



import missingno as msno

msno.bar(pl.sample(1000))
msno.matrix(pl.sample(1000))
#先把historical data 和master data连接，之后将historical 通过月份求和求得月度的销量以后再和commercial planning连接

hm = pd.merge(hist_data,md,on='SKU Code')

#把date（也就是年月日）设成index

hm = hm.set_index(['Date'])

#hm = hm0.drop(['SKU Code'], axis=1)

#hm0.head()

#hm0.describe()

hm.head()
import scipy.stats as st

y = hm['VolumeHL']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
#看预测值（VolumeHL）的偏度和峰度

sns.distplot(hm['VolumeHL']);

print("Skewness: %f" % hm['VolumeHL'].skew())

print("Kurtosis: %f" % hm['VolumeHL'].kurt())

plt.hist(hm['VolumeHL'], orientation = 'vertical',histtype = 'bar', color ='red')

plt.show()

hm.skew(), hm.kurt()
# 数字特征

numeric_features = hm.select_dtypes(include=[np.number])

#numeric_features.columns#Index(['YM', 'SKU Code', 'VolumeHL'], dtype='object')

# 类型特征

categorical_features = hm.select_dtypes(include=[np.object])

#categorical_features.columns#Index(['SalesRegion', 'Brand', 'Package', 'Segment'], dtype='object')
#相关性分析

volume_numeric = hm[['YM', 'SKU Code', 'VolumeHL']]

correlation = volume_numeric.corr()

print(correlation['VolumeHL'].sort_values(ascending=False),'\n')

f, ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Volume',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)

del volume_numeric['VolumeHL']
#sns.pairplot()：展示变量两两之间的关系（线性或非线性，有无较为明显的相关关系）

sns.set()

columns = ['VolumeHL','YM', 'SKU Code','SalesRegion', 'Brand', 'Package', 'Segment']

sns.pairplot(hm[columns],size = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
#类别特征

for cat_fea in categorical_features:

    print(cat_fea + '特征分布如下：')

    print('{}特征有{}个不同的值'.format(cat_fea, hm[cat_fea].nunique()))

    print(hm[cat_fea].value_counts())
#查看箱型图

categorical_features =['SalesRegion', 'Brand', 'Package', 'Segment']

for c in categorical_features:

    hm[c] = hm[c].astype('category')

    if hm[c].isnull().any():

        hm[c] = hm[c].cat.add_categories(['MISSING'])

        hm[c] = hm[c].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=45)



f = pd.melt(hm, id_vars=['VolumeHL'], value_vars=categorical_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "VolumeHL")
import pandas_profiling

pfr1 = pandas_profiling.ProfileReport(hm)

pfr1.to_file("./example1.html")



pfr2 = pandas_profiling.ProfileReport(pl)

pfr2.to_file("./example2.html")
#看两个地区分别有哪些SKU有对应的数据



#可以看出两个地区（黑龙江，吉林）不是所有品类（SKU）都有当地对应的销售数据

H = sorted(set(hist_data.loc[hist_data['SalesRegion']=='Heilongjiang']['SKU Code']))

J = sorted(set(hist_data.loc[hist_data['SalesRegion']=='Jilin']['SKU Code']))

AL = sorted(set(hist_data['SKU Code']))

               

print('Heilongjiang has ',len(hist_data.loc[hist_data['SalesRegion']=='Heilongjiang']),'records.')

print('Jilin has ',len(hist_data.loc[hist_data['SalesRegion']=='Jilin']),'records.')

print('The SKUs which have data in Heilongjiang:\n ',H)

print('The SKUs which DO NOT have data in Heilongjiang:\n ',[item for item in AL if not item in H])

print('The SKUs which have data in Jilin:\n ',J)

print('The SKUs which DO NOT have data in Jilin:\n ',[item for item in AL if not item in J])
print(hm['Brand'].unique()) 
#本想用for loop循环搞个字典，发现自动出来有些地区没有一些特定的brand/segement只能笨办法一个一个看，这样也能看出来哪些地方有

#用这种方法操作的时候 SKU code也会在月度加和的时候被加进去

#【地区】——黑龙江

#LOW-END





#hl1 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'low-end 1')]——low end 1 在黑龙江地区无售

#hl1.head()



hl2 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'low-end 2')]



#hl2_merge = pd.merge(hl2,pl,on=['YM','Brand'])



hl2.head(25)

#print('hl2 has following SKU Codes:',hl2['SKU Code'].unique())



mhl2 = hl2.resample('M').sum()

mhl2.to_period('M')['VolumeHL'].head()



hl3 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'low-end 3')]

#hl3.head()



#hm1 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'middle-end 1')]

#hm1.head()



hm2 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'middle-end 2')]

#hm2.head()



hm3 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'middle-end 3')]

#hm3.head()



hh1 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'high-end 1')]

#hh1.head()



hh2 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'high-end 2')]

#hh2.head()



hh3 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'high-end 3')]

#hh3.head()



hs1 = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == 'super high-end 1')]

#hs1.head()
sns.distplot(hm['VolumeHL']);

print("Skewness: %f" % hm['VolumeHL'].skew())

print("Kurtosis: %f" % hm['VolumeHL'].kurt())

sns.distplot(hm.skew(),color='blue',axlabel ='Skewness')

sns.distplot(hm.kurt(),color='orange',axlabel ='Kurtness')

#hm['VolumeHL'].skew(), hm['VolumeHL'].kurt()
import matplotlib.pyplot as plt

plt.hist(hm['VolumeHL'], orientation = 'vertical',histtype = 'bar', color ='red')

plt.show()

plt.hist(np.log(hm['VolumeHL']),orientation = 'vertical',histtype = 'bar',color ='red')

plt.show()
#地区：黑龙江；Brand：10种



layered_hlist = []

layered_jlist = []

heilong = {}

jilin = {}



#先看地区为黑龙江的

for segment in ['low-end 1','low-end 2','low-end 3','middle-end 1','middle-end 2','middle-end 3','high-end 1','high-end 2','high-end 3','super high-end 1']:

    heilong = hm[(hm['SalesRegion']=='Heilongjiang') & (hm['Brand'] == segment)]

    layered_hlist.append(heilong)



#再看地区为吉林的

for segment in ['low-end 1','low-end 2','low-end 3','middle-end 1','middle-end 2','middle-end 3','high-end 1','high-end 2','high-end 3','super high-end 1']:

    jilin = hm[(hm['SalesRegion']=='Jilin') & (hm['Brand'] == segment)]

    layered_jlist.append(jilin)

    

print(layered_hlist)

#地区为黑龙江的20个随机SKU时间序列——用于观察不同SKU的不同销量趋势变化特征

twenty_examples = random.sample(range(1,70), 20)



fig, axs = plt.subplots(10, 2, figsize=(15, 20))

axs = axs.flatten()

ax_idx = 0

for i in twenty_examples:

    s = hm0.loc[hm0['SKU Code']==i]

    sh= s.loc[s['SalesRegion']=='Heilongjiang']['VolumeHL']

    sh.plot(title='SKU'+str(i),ax=axs[ax_idx])

    ax_idx += 1

plt.tight_layout()

plt.show()
#地区为吉林的20个随机SKU时间序列——用于观察不同SKU的不同销量趋势变化特征



twenty_examples = random.sample(range(1,70), 20)



fig, axs = plt.subplots(10, 2, figsize=(15, 20))

axs = axs.flatten()

ax_idx = 0

for i in twenty_examples:

    s = hm0.loc[hm0['SKU Code']==i]

    sh= s.loc[s['SalesRegion']=='Jilin']['VolumeHL']

    sh.plot(title='SKU'+str(i),ax=axs[ax_idx])

    ax_idx += 1

plt.tight_layout()

plt.suptitle('Jilin')

plt.show()