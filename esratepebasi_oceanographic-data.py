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
data_bottle=pd.read_csv("/kaggle/input/calcofi/bottle.csv")
data_bottle.head()
data_bottle.info()  
data_bottle.columns
data_bottle.drop(['DIC1','DIC2', 'TA1', 'TA2', 'pH2', 'pH1', 'DIC Quality Comment'],axis=1,inplace=True)
data_bottle.iloc[0,3]
# data_bottle['year']=data_bottle['Depth_ID']*2  #gibi bir islemde her verinin kendi degerini str olarak iki kere yaziyor
#data_bottle['year']=float(str(data_bottle['Depth_ID'])[3:5])

# tek tek 'Depth_ID' degerini alip dondurmuyor.butun column un birlesik str yapiyor
data=data_bottle.sample(n=900)
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
p1 = data.loc[:,["P_qual","Btl_Cnt"]].corr(method= "pearson")

p2 = data.P_qual.cov(data.Btl_Cnt)/(data.P_qual.std()*data.Btl_Cnt.std())

print('Pearson correlation: ')

print(p1)

print('Pearson correlation: ',p2)
sns.jointplot(data.P_qual,data.Btl_Cnt,kind="regg")

plt.show()
data_bottle.P_qual
data_bottle.P_qual.unique()
data_bottle.P_qual.value_counts()
#data_bottle.drop(['P_qual'],axis=1,inplace=True)

data.drop(['P_qual'],axis=1,inplace=True)
data_bottle.Chlqua
data_bottle.Chlqua.unique()
data_bottle.Chlqua.value_counts()
#data.drop(['Chlqua'],axis=1,inplace=True)

data.drop(['Chlqua'],axis=1,inplace=True)
data_bottle.Phaqua.value_counts()
#data_bottle.drop(['Phaqua'],axis=1,inplace=True)

data.drop(['Phaqua'],axis=1,inplace=True)
data_bottle.PO4q.value_counts()
#data_bottle.drop(['PO4q'],axis=1,inplace=True)

data.drop(['PO4q'],axis=1,inplace=True)
data_bottle["C14A1q"].value_counts()
#data_bottle.drop(['C14A1q'],axis=1,inplace=True)

data.drop(['C14A1q'],axis=1,inplace=True)
data_bottle.C14A2q.value_counts()
#data_bottle.drop(['C14A2q'],axis=1,inplace=True)

data.drop(['C14A2q'],axis=1,inplace=True)
data_bottle.DarkAq.value_counts()
#data_bottle.drop(['DarkAq'],axis=1,inplace=True)

data.drop(['DarkAq'],axis=1,inplace=True)
data_bottle.MeanAq.value_counts()
#data_bottle.drop(['MeanAq'],axis=1,inplace=True)

data.drop(['MeanAq'],axis=1,inplace=True)
data_bottle.SiO3qu.value_counts()
#data_bottle.drop(['SiO3qu'],axis=1,inplace=True)

data.drop(['SiO3qu'],axis=1,inplace=True)
data_bottle.DarkAp.value_counts()
#data_bottle.drop(['DarkAp'],axis=1,inplace=True)

data.drop(['DarkAp'],axis=1,inplace=True)
data.info()
data.isna().sum()
data.describe().T
import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()


desc = data.T_degC.describe()

print(desc)

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print('lower bound is :',lower_bound)

print('upper bound is :',upper_bound)



print("Outliers: ",data[(data.T_degC < lower_bound) | (data.T_degC > upper_bound)].T_degC)
desc = data.T_degC.describe()

print('data describe')

print(desc)

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print('lower bound is :',lower_bound)

print('upper bound is :',upper_bound)



desc_bottle=data_bottle.T_degC.describe()

print('data_bottle describe')

print(desc_bottle)

Q1_b = desc_bottle[4]

Q3_b = desc_bottle[6]

IQR_b = Q3_b-Q1_b

lower_bound_b = Q1_b - 1.5*IQR_b

upper_bound_b = Q3_b + 1.5*IQR_b

print('lower bound is :',lower_bound_b)

print('upper bound is :',upper_bound_b)



print("Outliers: ",data_bottle[(data_bottle.T_degC < lower_bound_b) | (data_bottle.T_degC > upper_bound_b)].T_degC)
p1 = data.loc[:,["O2ml_L","T_degC"]].corr(method= "pearson")

p2 = data.T_degC.cov(data.O2ml_L)/(data.T_degC.std()*data.O2ml_L.std())

print('Pearson correlation: ')

print(p1)

print('Pearson correlation: ',p2)
sns.jointplot(data.O2ml_L,data.T_degC,kind="regg")

plt.show()
ranked_data = data.rank()

spearman_corr = ranked_data.loc[:,["O2ml_L","T_degC"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr)
data['SalntyRanked'] = data['Salnty'].rank(ascending=1)

data
mean_diff =  data.Salnty.mean()-data.T_degC.mean()    # m1 - m2

var_Salnty = data.Salnty.var()

var_T_degC = data.T_degC.var()

var_pooled = (len(data)*var_T_degC +len(data)*var_Salnty ) / float(len(data)+ len(data))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)