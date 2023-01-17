# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns



data = pd.read_csv('/kaggle/input/calcofi/bottle.csv')

data.head(5)
#T_degCWater temperature in degree Celsius     # su derecesi

#SalntySalinity in g of salt per kg of water (g/kg)   # sudaki tuzluluk orani 

#Ikisi arasinda iliskiye bakalim..

dataSet = pd.DataFrame()

dataSet["T_degC"] = data["T_degC"]

dataSet["Salnty"] = data["Salnty"]

print(dataSet)
dataSet.describe()
mean_Salnty= np.mean(dataSet.Salnty)

mean_T_degC= np.mean(dataSet.T_degC)

print("Sudaki tuzluluk orani ortalamasi",mean_Salnty)

print("Su derecesi ortalamasi",mean_T_degC)
median_Salnty= np.median(dataSet.Salnty)

median_T_degC= np.median(dataSet.T_degC)

print("Sudaki tuzluluk orani medyani",median_Salnty)

print("Su derecesi medyani",median_T_degC)
mod_Salnty= stats.mode(dataSet.Salnty)

mod_T_degC= stats.mode(dataSet.T_degC)

print("Sudaki tuzluluk orani modu",mod_Salnty)

print("Su derecesi modu",mod_T_degC)
print("Sudaki tuzluluk orani ranj",max(dataSet.Salnty)-min(dataSet.Salnty))

print("Su derecesi ranj",max(dataSet.T_degC)-min(dataSet.T_degC))
var_Salnty= np.var(dataSet.Salnty)

var_T_degC= np.var(dataSet.T_degC)

print("Sudaki tuzluluk orani varyans",var_Salnty)

print("Su derecesi varyans",var_T_degC)
std_Salnty= np.std(dataSet.Salnty)

std_T_degC= np.std(dataSet.T_degC)

print("Sudaki tuzluluk orani standard deviation",std_Salnty)

print("Su derecesi standard deviation",std_T_degC)
desc = dataSet.Salnty.describe()

print(desc)

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("Tuzluluk orani: Alt uc deger, Ust uc deger: (", lower_bound ,",", upper_bound,")") 



dataSet[dataSet.Salnty < lower_bound].Salnty

print("Tuzluluk orani: Outliers(uc degerler): ",dataSet[(dataSet.Salnty < lower_bound) | (dataSet.Salnty > upper_bound)].Salnty.values)
desc = dataSet.T_degC.describe()

print(desc)

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("Su dereceleri: Alt uc deger, Ust uc deger: (", lower_bound ,",", upper_bound,")") 



dataSet[dataSet.T_degC < lower_bound].T_degC

print("Su dereceleri: Outliers(uc degerler): ",dataSet[(dataSet.T_degC < lower_bound) | (dataSet.T_degC > upper_bound)].T_degC.values)
# melted_data = pd.melt(dataSet,id_vars = "T_degC",value_vars = ['Salnty'])

# sns.boxplot(x = "variable", y = "value", hue="T_degC",data= melted_data)
import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (18,18))

# corr() is actually pearson correlation

sns.heatmap(dataSet.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
p1 = dataSet.corr(method= "pearson")

print('Pearson correlation: ',p1)

sns.jointplot(data.T_degC,data.Salnty,kind="regg")

plt.show()
ranked_data = dataSet.rank()   

spearman_corr = ranked_data.corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr)
mean_diff = dataSet.T_degC.mean() - dataSet.Salnty.mean()    # m1 - m2

var_T_degC = dataSet.T_degC.var()

var_Salnty = dataSet.Salnty.var()

var_pooled = (len(dataSet.Salnty)*var_Salnty +len(dataSet.T_degC)*var_T_degC ) / float(len(dataSet.Salnty)+ len(dataSet.Salnty))

effect_size = mean_diff/np.sqrt(var_pooled)

print("Effect size: ",effect_size)