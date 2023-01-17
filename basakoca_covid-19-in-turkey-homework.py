# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualisation

import seaborn as sns #visualisation

plt.style.use("seaborn-whitegrid")



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_tr=pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data_tr.columns
data_tr.head()
data_tr.describe()
#Defining Active Case: Active Case = confirmed - deaths - recovered



data_tr["Active"]=data_tr["Confirmed"]-data_tr["Recovered"]-data_tr["Deaths"]

data_tr.drop(columns="Province/State")

data_tr.info()
def plot_plot(variable):

    plt.figure(figsize= (9,4))

    plt.plot(data_tr[variable])

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(" {} cases distribution".format(variable))

    plt.show()
numericVar = ["Confirmed", "Recovered", "Deaths", "Active"]

for n in numericVar:

    plot_plot(n)


data_tr.corr()
#correlation map

f,ax = plt.subplots(figsize=(9,7))

sns.heatmap(data_tr.corr(), annot=True, linewidths=.5, fmt= ".1f", ax=ax)

plt.show()
# Scatter plot

data_tr.plot(kind="scatter", x="Confirmed", y="Recovered",alpha = 1,color = "green")

plt.xlabel("Confirmed")              

plt.ylabel("Recovered")

plt.title("Confirmed & Recovered")     

plt.show()
# Scatter plot

data_tr.plot(kind="scatter", x="Confirmed", y="Deaths",alpha = 1,color = "red")

plt.xlabel("Confirmed")              

plt.ylabel("Deaths")

plt.title("Confirmed & Deaths Cases")     

plt.show()
# Scatter plot

data_tr.plot(kind="scatter", x="Active", y="Deaths",alpha = 1,color = "red")

plt.xlabel("Active")              

plt.ylabel("Deaths")

plt.title("Active & Deaths Cases")     

plt.show()
def detect_outliers(df, features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 =  np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75) 

        #IQR

        IQR = Q3-Q1 

        #Outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indices

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index 

        # store indices

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)

    

    return multiple_outliers
data_tr.loc[detect_outliers(data_tr, ["Confirmed", "Recovered", "Deaths", "Active"])]