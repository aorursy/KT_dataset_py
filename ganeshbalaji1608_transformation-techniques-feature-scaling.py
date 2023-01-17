import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

import warnings

warnings. simplefilter(action='ignore', category=Warning) 
data = pd.read_csv("../input/wine-components-dataset/wine.csv")
data.head(5)
data.describe()
df = data.copy()
df
df = df[['Alcohol', 'Proline', 'Alcalinity']]
def plot_Before_logerthemic(df,var):

    plt.figure(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)



    sns.distplot(df[var], color = col, label = var)

    plt.legend()



    plt.plot()
for i in df.columns:

    print(i, ":", df[i].std())
for i in df.columns:

    plot_Before_logerthemic(df,i)
df = np.log(df)
for i in df.columns:

    print(i, ":", df[i].std())
def plot_After_logerthemic(df,var):

    plt.figure(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)



    sns.distplot(df[var], color = col, label = var)

    plt.legend()



    plt.plot()
for i in df.columns:

    plot_After_logerthemic(df,i)
df = data.copy()
df = df[['Proline']]
def Trans(df,var):

    df[var + "_Reci_trans"] = 1/(df[var]+1)

    df[var + "_SquareRt_trans"] = df[var] ** (1/2)

    df[var + "_Exp_trans"] = np.exp(df[var])

    df[var + "BoxCox"], param = stats.boxcox(df[var]+1) 
Trans(df,'Proline')
df
for i in df.columns:

    print(i, ":", df[i].std())
def plot_Trans(df,var):

    plt.figure(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)



    sns.distplot(df[var], color = col, label = var)

    plt.legend()



    plt.plot()
### Because of Proline_Exp_trans has inf values, ignore them
for i in df.columns:

    if i == 'Proline_Exp_trans': 

        continue

    plot_Trans(df,i)
df = data.copy()
df = df[['Alcohol', 'Malic', 'Hue', 'Nonflavanoids']]
df
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
def norm(df,var):

    df[var + "_norm"] = scaling.fit_transform(df[[var]])
for i in df.columns:

    norm(df,i)
df.head(10)
def plot_for_norm(df,var):

    plt.figure(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[var], color = col, label = var)

    plt.legend()

    plt.subplot(1,2,2)

    stats.probplot(df[var], dist = "norm", plot = plt)



    plt.plot()
for i in df.columns:

    plot_for_norm(df,i)
## In some case after this transformation, scaling is needed. 

## In other cases scaling only needed. Based on the variable and the algorithm, transformation is made
df = data.copy()
df
df = df[['Ash', 'Magnesium', 'Flavanoids', 'Color']]
df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def stan(df,var):

    df.loc[: , var + "_scale"] = scaler.fit_transform(df[[var]])

for i in df.columns:

    stan(df,i)
def plot(df,var):

    plt.plot(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[var], color = col, label = var)

    plt.legend()

    plt.subplot(1,2,2)

    stats.probplot(df[var], dist = "norm", plot = plt)

    plt.show()    



    
for i in df.columns:

    plot(df,i)
df
df = data.copy()
df
df = df[['Alcohol', 'Phenols', 'Flavanoids']]
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
def robust(df,var):

    df[var + "_Median_quantile"] = scaler.fit_transform(df[[var]])
for i in df.columns:

    robust(df,i)
df
def plot(df,var):

    plt.plot(figsize = (10,4))

    

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[var], color = col, label = var)

    plt.legend()

    plt.subplot(1,2,2)

    stats.probplot(df[var], dist = "norm", plot = plt)

    plt.show()    



    
for i in df.columns:

    plot(df,i)