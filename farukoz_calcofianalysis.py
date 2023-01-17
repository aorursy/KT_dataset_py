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
bottle = pd.read_csv('/kaggle/input/calcofi/bottle.csv')

cast = pd.read_csv('/kaggle/input/calcofi/cast.csv')
bottle.columns
bottle9045 = bottle[bottle['Sta_ID']=='090.0 045.0']
bottle9045 = bottle9045[['Depthm','T_degC','Salnty','O2ml_L','PO4uM']]
bottle9045.isna().sum()
bottle9045.groupby(bottle9045.Depthm).count()
bottle9045.Depthm.value_counts()
bottle_0 = bottle9045[bottle['Depthm']==0]

bottle_0
bottle_0.isna().sum()
bottle_0.dropna(inplace=True)
bottle_0
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (5,5))

# corr() is actually pearson correlation

sns.heatmap(bottle_0.corr(),annot= True,linewidths=0.5,fmt = ".3f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
bottle9045.dropna(inplace = True)
bottle9045
desc = bottle9045.describe()

desc
Q1_dep = desc['Depthm'][4]

Q3_dep = desc['Depthm'][6]

IQR_dep = Q3_dep-Q1_dep

lower_dep = Q1_dep-1.5*IQR_dep

upper_dep = Q3_dep+1.5*IQR_dep

bottle9045[bottle9045['Depthm']<upper_dep].Depthm.values,upper_dep,lower_dep,Q1_dep,Q3_dep
Q1_deg = desc['T_degC'][4]

Q3_deg = desc['T_degC'][6]

IQR_deg = Q3_deg-Q1_deg

lower_deg = Q1_deg-1.5*IQR_deg

upper_deg = Q3_deg+1.5*IQR_deg

bottle9045[(bottle9045['T_degC']<lower_deg)|(bottle9045['T_degC']>upper_deg)].T_degC.values,upper_deg,lower_deg,Q1_deg,Q3_deg
Q1_salt = desc['Salnty'][4]

Q3_salt = desc['Salnty'][6]

IQR_salt = Q3_salt-Q1_salt

lower_salt = Q1_salt-1.5*IQR_salt

upper_salt = Q3_salt+1.5*IQR_salt

bottle9045[(bottle9045['Salnty']<lower_salt)|(bottle9045['Salnty']>upper_salt)].Salnty.values,upper_salt,lower_salt,Q1_salt,Q3_salt
Q1_o2 = desc['O2ml_L'][4]

Q3_o2 = desc['O2ml_L'][6]

IQR_o2 = Q3_o2-Q1_o2

lower_o2 = Q1_o2-1.5*IQR_o2

upper_o2 = Q3_o2+1.5*IQR_o2

bottle9045[(bottle9045['O2ml_L']<lower_o2)|(bottle9045['O2ml_L']>upper_o2)].O2ml_L.values,upper_o2,lower_o2,Q1_o2,Q3_o2
Q1_PO4uM = desc['PO4uM'][4]

Q3_PO4uM = desc['PO4uM'][6]

IQR_PO4uM = Q3_PO4uM-Q1_PO4uM

lower_PO4uM = Q1_PO4uM-1.5*IQR_PO4uM

upper_PO4uM = Q3_PO4uM+1.5*IQR_PO4uM

bottle9045[(bottle9045['PO4uM']<lower_PO4uM)|(bottle9045['PO4uM']>upper_PO4uM)].PO4uM.values,upper_PO4uM,lower_PO4uM,Q1_PO4uM,Q3_PO4uM
bottle_outlier = bottle9045[(bottle9045['T_degC']>lower_deg)&(bottle9045['T_degC']<upper_deg)]

bottle_outlier
warnings.filterwarnings("ignore")

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (5,5))

# corr() is actually pearson correlation

sns.heatmap(bottle_outlier.corr(),annot= True,linewidths=0.5,fmt = ".3f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
p1 = bottle_outlier.loc[:,["Depthm","T_degC"]].corr(method= "pearson")

p2 = bottle_outlier.Depthm.cov(bottle_outlier.T_degC)/(bottle_outlier.Depthm.std()*bottle_outlier.T_degC.std())

print('Pearson correlation: ')

print(p1)

print('Pearson correlation: ',p2)
sns.jointplot(bottle_outlier.Depthm,bottle_outlier.T_degC,kind="regg")

plt.show()
sns.jointplot(bottle_outlier.Depthm,bottle_outlier.Salnty,kind="regg")

plt.show()
sns.jointplot(bottle_outlier.Depthm,bottle_outlier.O2ml_L,kind="regg")

plt.show()
sns.jointplot(bottle_outlier.Depthm,bottle_outlier.PO4uM,kind="regg")

plt.show()
ranked_dataend = bottle_outlier.rank()

spearman_corrend = ranked_dataend.loc[:,["T_degC","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corrend)
ranked_dataend = bottle_outlier.rank()

spearman_corrend = ranked_dataend.loc[:,["Salnty","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corrend)
ranked_dataend = bottle_outlier.rank()

spearman_corrend = ranked_dataend.loc[:,["O2ml_L","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corrend)
ranked_dataend = bottle_outlier.rank()

spearman_corrend = ranked_dataend.loc[:,["PO4uM","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corrend)
mean_diff = bottle_outlier.Depthm.mean() - bottle_outlier.T_degC.mean()    # Depthm-T_degC

var_instance = bottle_outlier.Depthm.var()

var_instance1 = bottle_outlier.T_degC.var()

var_pooled = (len(bottle_outlier)*var_instance1 +len(bottle_outlier)*var_instance ) / float(len(bottle_outlier)+ len(bottle_outlier))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)
mean_diff = bottle_outlier.Depthm.mean() - bottle_outlier.Salnty.mean()    # Depthm-Salnty

var_instance = bottle_outlier.Depthm.var()

var_instance1 = bottle_outlier.Salnty.var()

var_pooled = (len(bottle_outlier)*var_instance1 +len(bottle_outlier)*var_instance ) / float(len(bottle_outlier)+ len(bottle_outlier))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)
mean_diff = bottle_outlier.Depthm.mean() - bottle_outlier.O2ml_L.mean()    # Depthm-PO4uM

var_instance = bottle_outlier.Depthm.var()

var_instance1 = bottle_outlier.O2ml_L.var()

var_pooled = (len(bottle_outlier)*var_instance1 +len(bottle_outlier)*var_instance ) / float(len(bottle_outlier)+ len(bottle_outlier))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)
mean_diff = bottle_outlier.Depthm.mean() - bottle_outlier.PO4uM.mean()    # Depthm-PO4uM

var_instance = bottle_outlier.Depthm.var()

var_instance1 = bottle_outlier.PO4uM.var()

var_pooled = (len(bottle_outlier)*var_instance1 +len(bottle_outlier)*var_instance ) / float(len(bottle_outlier)+ len(bottle_outlier))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)
bottle_outlier.groupby('Depthm')['T_degC','Salnty','O2ml_L','PO4uM'].mean()
bottle_30 = bottle9045[bottle['Depthm']==30]
bottle_30
warnings.filterwarnings("ignore")

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (5,5))

# corr() is actually pearson correlation

sns.heatmap(bottle_30.corr(),annot= True,linewidths=0.5,fmt = ".3f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
bottle.columns
bottle_new.head(50)
bottle_new.info()
bottle_new.describe()
bottle_new[bottle_new['Depthm']>5000]['Depthm'].count()
ax = bottle_new.Depthm.plot.kde()
ax = bottle_new.T_degC.plot.kde()
ax = bottle_new.Depthm.plot.box()
ax = bottle_new.T_degC.plot.box()
ax = bottle_new.Salnty.plot.box()
ax = bottle_new.O2ml_L.plot.box()
ax = bottle_new.PO4uM.plot.box()
bottle_new.corr()
bottle1=bottle_new[(bottle_new['PO4uM']>=0)&(bottle_new['O2ml_L']>=0)&(bottle_new['Salnty']>=0)&(bottle_new['T_degC']>=-20)]
bottle1.info()
bottle1.corr()
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (5,5))

# corr() is actually pearson correlation

sns.heatmap(bottle1.corr(),annot= True,linewidths=0.5,fmt = ".3f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
dep= bottle1['Depthm']

dep.describe()
dsc_dep = dep.describe()

Q1= dsc_dep[4]

Q3= dsc_dep[6]

IQR = Q3-Q1

lower_bound = Q1-1.5*IQR

upper_bound = Q3+1.5*IQR

print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")

print('Lower Outliers:', dep[dep<lower_bound])

print('Lower Outliers:', dep[dep>upper_bound])