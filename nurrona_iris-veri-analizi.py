# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



from sklearn.preprocessing import LabelEncoder,OneHotEncoder



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.tail()
data.columns
data.info()
data.describe() # sadece numeric değerleri verir
def bar_plot(variable):

   

    var = data[variable]

    varValue = var.value_counts()



    

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1 = ["Species"]

for c in category1:

    bar_plot(c)
data.Species.nunique()
data.Species.unique()
data.PetalLengthCm.nunique()
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(data[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{}".format(variable))

    plt.show()
numericVar = ["Species"]

for n in numericVar:

    plot_hist(n)
data.columns
data = data.drop(["Id"], axis = 1)
data[["SepalWidthCm","Species"]].groupby(["Species"], as_index = False).mean().sort_values(by="Species", ascending = False)
data[["SepalLengthCm","Species"]].groupby(["Species"], as_index = False).mean().sort_values(by="Species", ascending = False)
data[["PetalLengthCm","Species"]].groupby(["Species"], as_index = False).mean().sort_values(by="Species", ascending = False)
data[["SepalWidthCm","Species"]].groupby(["Species"], as_index = False).mean().sort_values(by="Species", ascending = False)
def detect_outliers(df, features):

    outlier_indices = []

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3 - Q1

        #Outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df  [c] > Q3 + outlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers

data.loc[detect_outliers(data,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])]
data.isnull().sum()
data.PetalWidthCm.plot(kind ='line', color='b',label='PetalWidthCm',linewidth=1, alpha=1, grid=True, linestyle='solid')

data.PetalLengthCm.plot(color='r', label='PetalLengthCm', linewidth=1, alpha=1, grid=True, linestyle='solid')

plt.legend(loc='upper left')

plt.xlabel('ID') #x çizgimizin adı

plt.ylabel('CM') #y çizgimizin adı

plt.title('PetalWidth - PetalLenght') #başlık

plt.show()
data.SepalWidthCm.plot(kind ='line', color='b',label='SepalWidthCm',linewidth=1, alpha=1, grid=True, linestyle=':')

data.SepalLengthCm.plot(color='r', label='SepalLengthCm', linewidth=1, alpha=1, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('ID') #x çizgimizin adı

plt.ylabel('CM') #y çizgimizin adı

plt.title('Line Plot') #başlık

plt.show()
data.plot(kind='scatter', x='SepalWidthCm', y='SepalLengthCm', alpha=0.5, color='blue')

plt.xlabel('SepalWidthCm')

plt.ylabel('SepalLenghtCm') 

plt.title('Attack Scatter Plot') 
data.SepalWidthCm.plot(kind='hist', bins=50, figsize=(10,10))

plt.show()
corr = data.iloc[:,0:5].corr()

corr
data["Species"].value_counts()
data["Species"]  = [ 0 if i =="Iris-versicolor"  else 1 if  i == "Iris-virginica" else 2 for i in data["Species"] ]
plt.figure(figsize=(10,8)) 

sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='black',cmap='PuBu')

plt.show()
data.Species.unique()