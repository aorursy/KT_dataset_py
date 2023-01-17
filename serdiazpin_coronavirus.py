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
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
data_covit19_co = pd.read_csv("/kaggle/input/covit19-in-colombia/dataIns.csv")
data_covit19_co
numeric_data = data_covit19_co.select_dtypes(include = ["number"])
def groupCount(column_name, data):
    print(data.groupby(column_name).size())

def frecuency(column_name, data):
    print(100 * data.groupby(column_name).size() / data.shape[0] ) 

exclude = ["IDcaso","Fecdiagnostico", "CiudadUbicacion", "DepartDistr", "Edad", "Paissprocedencia"]
for name in data_covit19_co.columns:
    if not name in exclude:      
        print("##### Conteo #####")
        groupCount(name, data_covit19_co)
        print("##### Frecuencia #####")
        frecuency(name, data_covit19_co)        
#Promedio
print("##### promedio #####")
print(data_covit19_co.mean(axis = 0))
#Mediana
print("##### mediana #####")
print(data_covit19_co.median(axis = 0))
#Moda
print("##### moda #####")
#print(data_covit19_co.mode(axis=0, dropna=False))
def variance(column_name, data):
    print("Varianza:",data[column_name].var())
    
def standardDeviation(column_name, data):
    print("Desviación Standard:", data[column_name].std())
    
def quartile(column_name, data):    
    print("Rango Interquantile:", data[column_name].quantile(0.75) - data[column_name].quantile(0.25))
    print("Mínimo:", min(data[column_name]))
    print("Maximo:", max(data[column_name]))
    print("Cuartiles ")
    for i in np.arange(0, 1.25, 0.25):        
        print(i,"%:",data[column_name].quantile(i))

for name in numeric_data.columns:
    print("#####", name, "######")    
    variance(name, numeric_data)    
    standardDeviation(name, numeric_data)
    quartile(name, numeric_data)
def kurtosisFisher(column_name, data):
    print("Kurtosis Fisher:",kurtosis(data[column_name]))
    print("Kurtosis:",kurtosis(data[column_name], fisher = False))

print("##### skewness #####")
print(data_covit19_co.skew(axis = 0))
for name in numeric_data.columns:
    print("#####", name, "######")    
    kurtosisFisher(name, numeric_data)
def histogram(column_name, data):
    plt.figure(figsize=(10,10))
    _ = plt.hist(data[column_name], bins='auto')  
    plt.title("Histogram {} ".format(column_name) )
    plt.xticks(rotation='vertical')
    plt.show()

# No funciona solo estoy colocando metodos
def stemp(column_name, data):
    plt.figure(figsize=(10,10))
    _ = plt.stem(data[column_name], use_line_collection=True)
    plt.title("Stem {} ".format(column_name) )
    plt.xticks(rotation='vertical')
    plt.show()

def boxplot(column_name, data):
    sns.set(style="whitegrid")
    if type(data[column_name][0]) is not str:
        ax = sns.boxplot(x=data[column_name])
    
exclude = ["IDcaso","Fecdiagnostico"]
for name in data_covit19_co.columns:
    if not name in exclude:
        histogram(name, data_covit19_co)
        #stemp(name, data_covit19_co)
        boxplot(name, data_covit19_co)

print(data_covit19_co.cov())
print(data_covit19_co.corr())