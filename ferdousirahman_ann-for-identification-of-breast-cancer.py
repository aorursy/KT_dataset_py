import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

data = pd.read_csv('../input/data.csv', header=0)
def dataSetAnalysis(df):

    #view starting values of data set

    print("Dataset Head")

    print(df.head(3))

    print("=" * 30)

    

    # View features in data set

    print("Dataset Features")

    print(df.columns.values)

    print("=" * 30)

    

    # View How many samples and how many missing values for each feature

    print("Dataset Features Details")

    print(df.info())

    print("=" * 30)

    

    # view distribution of numerical features across the data set

    print("Dataset Numerical Features")

    print(df.describe())

    print("=" * 30)

    

    # view distribution of categorical features across the data set

    print("Dataset Categorical Features")

    print(df.describe(include=['O']))

    print("=" * 30)

dataSetAnalysis(data)

data.head(100)
# feature names as a list

col = data.columns       # .columns gives columns names in data 

print(col)


# y includes our labels and x includes our features

y = data.diagnosis                          # M or B 

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

x.head(100)



ax = sns.countplot(y,label="Count",)       # M = 212, B = 357

B, M = y.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)



import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

np.random.seed(sum(map(ord, "categorical")))

sns.set()

heat = data.iloc[0:33,]

x.describe()
sns.set(style="whitegrid", palette="muted")

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(20,20))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(20,20))

sns.barplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(20,20))

sns.pointplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)