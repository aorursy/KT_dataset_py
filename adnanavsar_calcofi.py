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
data=pd.read_csv("/kaggle/input/calcofi/bottle.csv")
data2=pd.DataFrame.copy(data)
data2.info()
data2.Cst_Cnt.unique()
len(data2.Cst_Cnt.unique())
data2.Btl_Cnt.unique()
len(data2.Btl_Cnt.unique())
data2.groupby(["Cst_Cnt"]).Btl_Cnt.count()
data2=data2[["Cst_Cnt","Btl_Cnt","Sta_ID","Depth_ID","Depthm","T_degC","Salnty","O2ml_L"]]
data2.columns.unique()
data2.info()
data2.isnull().any()
data2.isnull().sum()
data3=data2.dropna(subset=["T_degC"])

data3.isnull().sum()
data2.dropna(subset=["Salnty","O2ml_L","T_degC"], inplace=True)

data2.isnull().sum()
data2.info()
data2.drop(columns=["Cst_Cnt"],inplace=True)
data2.head(50)
data6=data2.groupby(["Sta_ID"]).Btl_Cnt.count()
liste1=dict(data6)
print(liste1)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
data_instance = data2[data2["Sta_ID"] == "080.0 060.0"]
data_instance1=data2[data2["Sta_ID"]=="093.3 055.0"]
data_instance
data_instance1
desc = data_instance.T_degC.describe()
print(desc)
desc1=data_instance1.T_degC.describe()
print(desc1)
Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("Anything outside this range is an outlier for desc: (", lower_bound ,",", upper_bound,")")
data_instance[data_instance.T_degC < lower_bound].T_degC

print("Outliers: ",data_instance[(data_instance.T_degC < lower_bound) | (data_instance.T_degC > upper_bound)].T_degC.values)
Q_1 = desc1[4]

Q_3 = desc1[6]

IQ_R = Q_3-Q_1

lower__bound = Q_1 - 1.5*IQ_R

upper__bound = Q_3 + 1.5*IQ_R

print("Anything outside this range is an outlier for desc1: (", lower__bound ,",", upper__bound,")")
data_instance1[data_instance1.T_degC < lower__bound].T_degC

print("Outliers: ",data_instance1[(data_instance1.T_degC < lower__bound) | (data_instance1.T_degC > upper__bound)].T_degC.values)
data_instancetotal=pd.concat([data_instance,data_instance1], axis = 0)
data_instancetotal.head()
melted_data = pd.melt(data_instancetotal,id_vars = "Sta_ID",value_vars = ['T_degC'])

sns.boxplot(x = "variable", y = "value", hue="Sta_ID",data= melted_data)
import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(data_instance1.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(data_instance.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
p1 = data_instance.loc[:,["T_degC","Salnty"]].corr(method= "pearson")



print(p1)
p2 = data_instance1.loc[:,["Depthm","O2ml_L"]].corr(method= "pearson")



print(p2)
p3 = data_instance1.loc[:,["Depthm","T_degC"]].corr(method= "pearson")



print(p3)
p4 = data_instance.loc[:,["O2ml_L","T_degC"]].corr(method= "pearson")



print(p4)
sns.jointplot(data_instance.O2ml_L,data_instance.T_degC,kind="regg")

plt.show()
sns.jointplot(data_instance1.O2ml_L,data_instance1.T_degC,kind="regg")

plt.show()
sns.jointplot(data_instance1.Depthm,data_instance1.T_degC,kind="regg")

plt.show()
sns.jointplot(data_instance.Depthm,data_instance.T_degC,kind="regg")

plt.show()
sns.jointplot(data2.Depthm,data2.T_degC,kind="regg")

plt.show()
ranked_data = data_instance.rank()

spearman_corr = ranked_data.loc[:,["T_degC","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr)
ranked_data1 = data_instance1.rank()

spearman_corr1 = ranked_data1.loc[:,["T_degC","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr1)
ranked_dataend = data2.rank()

spearman_corrend = ranked_dataend.loc[:,["T_degC","Depthm"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corrend)
mean_diff = data_instance.T_degC.mean() - data_instance1.T_degC.mean()    # station1-station2

var_instance = data_instance.T_degC.var()

var_instance1 = data_instance1.T_degC.var()

var_pooled = (len(data_instance)*var_instance1 +len(data_instance1)*var_instance ) / float(len(data_instance)+ len(data_instance1))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)