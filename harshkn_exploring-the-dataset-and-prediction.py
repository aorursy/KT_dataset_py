import pandas as pd

import numpy as np

%matplotlib inline
#read the data

df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

y = df[['id', 'target']]

x = df.drop('target', axis=1)
x.describe()

# x['ps_car_12']


def plot_corr_matrix(df):

    import seaborn as sns

    import matplotlib.pyplot as plt

    plt.subplots(figsize=(20,15))

    # calculate the correlation matrix

    corr = df.corr()

    # plot the heatmap

    sns.heatmap(corr)

    

    corr_matrix = df.corr().abs()

    high_corr_var=np.where(corr_matrix>0.6)

    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

    print(high_corr_var)





plot_corr_matrix(df)
import seaborn as sns

sns.countplot(x=df['ps_ind_14'], data=df[['ps_ind_12_bin']])


g = sns.regplot(x=df['ps_reg_01'], y=df['ps_reg_03'])

g.set(ylim=(-0.5, None))
g = sns.regplot(x=df['ps_car_12'], y=df['ps_car_13'])

g.set(xlim=(0, None))

g.set(ylim=(-0.5, None))
import missingno as msno



train_null = df

train_null = train_null.replace(-1, np.NaN)



msno.matrix(df=train_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2)) 
# Extract columns with null data

train_null = train_null.loc[:, train_null.isnull().any()]

print(train_null.columns)
