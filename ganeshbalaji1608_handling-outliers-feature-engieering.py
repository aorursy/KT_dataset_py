import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/titanic/train.csv")
data.head()
data = data[['Age', 'Fare']]
data
df = data.copy()
for i in df.columns:

    plt.figure(figsize = (10,4))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[i], kde = False, color = col, label = i)

    plt.legend()

    plt.subplot(1,2,2)

    sns.boxplot(df[i], color = col)



    plt.show()
def ran(df, var):

    upper = df[var].mean() + (3 * df[var].std())

    lower = df[var].mean() - (3 * df[var].std())

    mean = df[var].mean()

    return (upper,lower,mean)
Age_lst = ran(df,'Age')
Age_lst
Fare_lst = ran(df,"Fare")
Fare_lst
df['Age'] = np.where(df['Age']>73, 73, df['Age'])
df.loc[df['Fare'] > 181 , 'Fare'] = 181
for i in df.columns:

    plt.figure(figsize = (10,4))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[i], kde = False, color = col, label = i)

    plt.legend()

    plt.subplot(1,2,2)

    sns.boxplot(df[i], color = col)



    plt.show()
# Still Fare Having more outliers because of it doesnt following normal distribution.

# for that we use IQR
Q1, Q3 = np.percentile(df['Fare'], [25,72])
Q1
Q3
IQR = Q3 - Q1
Upper = Q3 + (3 * IQR) #Extreme Bound is selected because of It doesnt follow normal distribution

Lower = Q1 - (3 * IQR)
print(Lower)

print(Upper)
df.loc[df['Fare']>92.03, 'Fare'] = 92
for i in df.columns:

    plt.figure(figsize = (10,4))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[i], kde = False, color = col, label = i)

    plt.legend()

    plt.subplot(1,2,2)

    sns.boxplot(df[i], color = col)



    plt.show()
df = data.copy()
df = df[['Age', 'Fare']].dropna()
from scipy import stats

for i in df.columns:

    df[i + "_log"] = np.log(df[i]+1)

    df[i + "_exp"] = np.exp(df[i])

    df[i + "_sqrt"] = df[i] ** (1/2)

    df[i + "_rec"] = 1/(df[i])

    df[i + "_box"], param = stats.boxcox(df['Fare']+1)

    
df
for i in df.columns:

    if "rec" in i.lower():

        continue

    plt.figure(figsize = (10,4))

    val = np.random.randint(100000,999999)

    col = "#" + str(val)

    plt.subplot(1,2,1)

    sns.distplot(df[i], kde = False, color = col, label = i)

    plt.legend()

    plt.subplot(1,2,2)

    sns.boxplot(df[i], color = col)



    plt.show()