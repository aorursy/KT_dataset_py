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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from datetime import datetime
data_covit19 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data_covit19
def check_null(dataset):
    print("--------Null columns--------")
    null_columns=dataset.columns[dataset.isnull().any()]
    total_null = dataset[null_columns].isnull().sum()
    print(null_columns, total_null)
    return null_columns, total_null
    
def fill_missing(dataset, value,null_columns, column = None):
    print("--------Fill missing data with columns--------")
    dataset[null_columns] =dataset[null_columns].fillna('missing')
    
null_cols, total = check_null(data_covit19)
fill_missing(data_covit19, "missing", null_cols)
null_cols, total = check_null(data_covit19)
def histogram(column_name, data):
    plt.figure(figsize=(30,30))
    labels, counts = np.unique(data[column_name], return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.title("Histogram {} ".format(column_name) )
    plt.xticks(rotation='vertical')
    plt.show()
   

# No funciona solo estoy colocando metodos
def stemp(column_name_x, column_name_y, data):
    plt.figure(figsize=(30,30))
    _ = plt.stem(data[column_name_x],data[column_name_y], use_line_collection=True)
    plt.title("Stem {} vs {} ".format(column_name_x,column_name_y) )
    plt.xticks(rotation='vertical')
    plt.show()

def boxplot(column_name, data):
    sns.set(style="whitegrid")
    if type(data[column_name][0]) is not str:
        ax = sns.boxplot(x=data[column_name])
data_covit19.describe()

data_covit19.describe(include=['object'])
columns_to_show = ['ObservationDate', 'Country/Region']
data_covit19.groupby(['Confirmed'])[columns_to_show].describe(percentiles=[])
columns_to_show = ['Last Update', 'Country/Region']
data_covit19.groupby(['Confirmed'])[columns_to_show].describe(percentiles=[])
numeric_data = data_covit19.select_dtypes(include = ["number"])
def groupCount(column_name, data):
    print(data.groupby(column_name).size())

def frecuency(column_name, data):
    print(100 * data.groupby(column_name).size() / data.shape[0] )

exclude = ["SNo"]
    
for name in data_covit19.columns:
    if not name in exclude and name in numeric_data:      
        print("##### Conteo #####")
        groupCount(name, data_covit19)
        print("##### Frecuencia #####")
        frecuency(name, data_covit19)    
        
def kurtosisFisher(column_name, data):
    print("Kurtosis Fisher:",kurtosis(data[column_name]))
    print("Kurtosis:",kurtosis(data[column_name], fisher = False))

print("##### skewness #####")
print(data_covit19.skew(axis = 0))
for name in numeric_data.columns:
    print("#####", name, "######")    
    kurtosisFisher(name, numeric_data)
exclude = ["SNo", "ObservationDate" ,"Recovered","Confirmed","Deaths","Last Update"]
for name in data_covit19.columns:
    if not name in exclude:
        print(name)
        histogram(name, data_covit19)
        boxplot(name, data_covit19)

def histogram_incomplete(column_name, data, number_elements):
    plt.figure(figsize=(30,30))
    labels, counts = np.unique(data[column_name], return_counts=True)
    counts, labels = zip(*sorted(zip(counts, labels)))
    print(counts[-number_elements:])
    print(labels[-number_elements:])
    plt.bar(labels[-number_elements:], counts[-number_elements:], align='center')
    plt.gca().set_xticks(labels[-number_elements:])
    plt.title("Histogram {} ".format(column_name) )
    plt.xticks(rotation='vertical')
    plt.show()

include = ["Country/Region", "Province/State"]
for name in data_covit19.columns:
    if name in include:
        histogram_incomplete(name, data_covit19,10)
data_covit19.cov()
data_covit19.corr()
stemp("ObservationDate", "Recovered", data_covit19)
stemp("ObservationDate", "Deaths", data_covit19)
stemp("ObservationDate", "Confirmed", data_covit19)
def plot_total(country, metric):
    dates= []
    total = []
    for date, df_date in country.groupby("ObservationDate"):
        subset_df = df_date[df_date["ObservationDate"] == date]
        dates.append(date)
        total.append(subset_df.sum()[metric])
    return dates, total

def plot_line(country_name, metric, show_state = False, show=True):
    country = data_covit19[data_covit19["Country/Region"] == country_name]
    #print(country['Province/State']["missing"])
    plt.figure(figsize=(30,30))
    if show_state:
        for region, df_region in country.groupby('Province/State'):
            if region == "missing":
                region = country_name
            plt.plot(df_region["ObservationDate"],df_region[metric], '*-',label=region)
    else:
        dates= []
        total = []
        if show:
            dates, total = plot_total(country, metric)
            plt.plot(dates,total,'*-',label=country_name+"-"+metric)
        else:
            for i in ["Deaths","Confirmed","Recovered"]:
                dates, total = plot_total(country, i)
                plt.plot(dates,total,'*-',label=country_name+"-"+i)
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.show()

plot_line("Mainland China", "Deaths")
plot_line("US", "Deaths")
plot_line("Colombia", "Deaths")
plot_line("Ecuador", "Deaths")
plot_line("Mainland China", "Confirmed")
plot_line("US", "Confirmed")
plot_line("Colombia", "Confirmed")
plot_line("Ecuador", "Confirmed")
plot_line("Mainland China", "Recovered")
plot_line("US", "Recovered")
plot_line("Colombia", "Recovered")
plot_line("Ecuador", "Recovered")
plot_line("Mainland China", "Recovered",show=False)

plot_line("US", "Recovered",show=False)

plot_line("Colombia", "Recovered",show=False)

plot_line("Ecuador", "Recovered",show=False)
plot_line("Italy", "Recovered",show=False)
plot_line("Spain", "Recovered",show=False)
#Fijar 
data_covit19_full = data_covit19
#Manejar SNo como índice
data_covit19_full.set_index("SNo", inplace=True)
data_covit19_day = data_covit19_full[data_covit19_full["ObservationDate"] == "04/20/2020"]
print(data_covit19.shape)
print(data_covit19_full.shape)
print(data_covit19_day.shape)
data_covit19.info()
#data_covit19_day.info()
def plot_totals():    
    #plt.figure(figsize=(30,30))    
    data_covit19_full.groupby('ObservationDate').sum()["Confirmed"].plot()
    data_covit19_full.groupby('ObservationDate').sum()["Deaths"].plot() 
    data_covit19_full.groupby('ObservationDate').sum()["Recovered"].plot() 
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.show()
plot_totals()
#print(data_covit19_full.describe())
data_covit19_day.describe().round()
data_covit19_day.var()
data_covit19_day.skew().round(1)
data_covit19_day.kurtosis().round(1)
print(data_covit19_full.cov())
print(data_covit19_day.cov())
print(data_covit19_full.corr())
print(data_covit19_day.corr())
#Histogramas
def histogramas (column_name, data, bins, range, median):
    plt.hist(data[column_name], bins, (0, range))
    plt.axvline(x=median, label = "median", linewidth = 1, color = "red")
    plt.title("Histogram of {} ".format(column_name))
    plt.legend()
    plt.show()
#histogramas("Confirmed", data_covit19_full, 30, 3077, 582, )
#histogramas("Deaths", data_covit19_full, 30, 30)
#histogramas("Recovered", data_covit19_full, 30, 30)
histogramas("Confirmed", data_covit19_day, 30, 3077, 582)
histogramas("Deaths", data_covit19_day, 20, 101, 9)
histogramas("Recovered", data_covit19_day, 20, 339, 48)
#Diagramas caja
def diagramasCaja (column_name, data, median, mean):
    plt.figure(figsize=(3,4))
    plt.boxplot(data[column_name])
    plt.axhline(y=mean, label = "mean", linewidth = 1, color = "blue")    
    plt.axhline(y=median, label = "median", linewidth = 1, color = "orange")    
    plt.title("Boxplot of {} ".format(column_name))
    plt.legend()
    plt.show()
    
def diagramasCaja2 (column_name, data, range):
    plt.figure(figsize=(2,4))    
    plt.boxplot(data[column_name])
    plt.title("Boxplot of {} ".format(column_name))
    if range != 0:
        plt.ylim(0, range)  
    plt.show()
diagramasCaja("Confirmed", data_covit19_day, 582, 7726)
diagramasCaja2("Confirmed", data_covit19_day, 7750)
diagramasCaja("Deaths", data_covit19_day, 9, 531)
diagramasCaja2("Deaths", data_covit19_day, 260)
diagramasCaja("Recovered", data_covit19_day, 48, 2018)
diagramasCaja2("Recovered", data_covit19_day, 850)
data_covit19_day.cov().round()
data_covit19_day.corr().round(2)
#Correlation
def correlation (column_nameA, column_nameB, data, maxA, maxB):
    plt.plot([0, maxA], [0, maxB], linewidth = 1, color = "blue")
    plt.scatter(data[column_nameA], data[column_nameB], s = 10, color = 'red')
    #Definir título y nombres de ejes
    plt.title("Scatter of {} - {}".format(column_nameA, column_nameB))
    plt.xlabel(column_nameA)
    plt.ylabel(column_nameB)
    #Mostrar leyenda y figura
    plt.show()
correlation("Confirmed", "Deaths", data_covit19_day, 253060, 24114)
correlation("Confirmed", "Recovered", data_covit19_day, 253060, 91500)
correlation("Recovered", "Deaths", data_covit19_day, 91500, 24114)
def upperWhisker(column_name, data):
    iqr = data[column_name].quantile(0.75).round() - data[column_name].quantile(0.25).round()   
    upperWhisker = (1.5 * iqr) + data[column_name].quantile(0.75).round()
    loco = data[column_name][data[column_name] > upperWhisker] 
    print(loco.shape)

upperWhisker("Confirmed", data_covit19_day)
upperWhisker("Deaths", data_covit19_day) 
upperWhisker("Recovered", data_covit19_day) 