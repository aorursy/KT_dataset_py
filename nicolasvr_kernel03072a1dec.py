# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



df = pd.read_csv('../input/train.csv');



#Clean columns by NaN percentage 

df = df.drop(['Id','Alley'], axis=1);

count = df.count().to_frame().reset_index()

count.columns = ['A','B']

cm = count['B'].mean()

drop_below = count[count['B'] < cm*0.8]['A'].tolist()

df = df.drop(drop_below,axis=1)

df.head()



#df.count().reset_index(name='count').sort_values(['count'],ascending=True).head()
# Feature selection

##Using Pearson Correlation

plt.figure(figsize=(30,25))

cor = df.corr()

cor_target = abs(cor['SalePrice'])

cor_threshold = 0.35



ft_low_cor_target = []

for i, v in cor_target.items():

    if v < cor_threshold:

        ft_low_cor_target.append(i)    



low_cors  = cor[ft_low_cor_target]

lc_rdy = abs(low_cors.drop(low_cors))

lc_rdy

df.count()
# Heatmap 

## Map correlations between the above and below threshold variables

import seaborn as sns



plt.figure(figsize=(20,20))

sns.heatmap(lc_rdy, annot=True, cmap="RdYlGn", vmax=1)
# Remover mantener columnas que tienen alto grado de correlación con columnas de alta correlación con la variable output

low_cors_col = lc_rdy.columns.tolist()

for v in low_cors_col:

    over_th = len(lc_rdy[lc_rdy[v] > 0.6])

    if over_th > 0:

        low_cors_col.remove(v)

df = df.drop(low_cors_col,axis=1)
df.head()
df_cp = df

obj_df = df.select_dtypes(include=['object']).copy()

obj_list = obj_df.columns.tolist()

for v in obj_list:

    counts = obj_df[v].value_counts()

    #mask = obj_df[v].isin(counts[counts < (int(counts.max()*0.25))].index)

    mask = np.logical_not(obj_df[v].isin(counts[:3].index))

    df_cp[v][mask] = 'Otros'

    df_cp = pd.get_dummies(df_cp, columns=[v],drop_first=False, prefix=(v+"_"))

df_cp.head()

cor2 = df_cp.corr()

cor_target2 = abs(cor2['SalePrice'])

cor_target2.sort_values(ascending=False)



ft_low_cor_target = []

cor_threshold = 0.3

for i, v in cor_target2.items():

    if v < cor_threshold:

        ft_low_cor_target.append(i)    



low_cors  = cor2[ft_low_cor_target]

lc_rdy = abs(low_cors.drop(low_cors))

lc_rdy



df_cp.drop(ft_low_cor_target, axis=1, inplace=True)



cor2 = df_cp.corr()

cor_target2 = abs(cor2['SalePrice'])



print(cor_target2.sort_values(ascending=False))



df_cp.head(5)
df_cp.count()
df_cp['LotFrontage'].fillna(df_cp['LotFrontage'].mean(), inplace=True)

df_cp['MasVnrArea'].fillna(df_cp['MasVnrArea'].mean(), inplace=True)

df_cp['GarageYrBlt'].fillna(method="ffill", inplace=True)

df_cp.count()


#Chequear outliers

def detect_outlier(data_1):

    outliers=[]

    threshold=3

    mean_1 = np.mean(data_1)

    std_1 =np.std(data_1)

    

    

    for y in data_1:

        z_score= (y - mean_1)/std_1 

        if np.abs(z_score) > threshold:

            outliers.append(y)

    print(mean_1)

    return outliers



df_cp.std()
df_cp.hist(bins=20, figsize=(20,45), layout=(15,3))

df_cp_fn = df_cp.loc[:, df_cp.std() > .3]
df_cp_fn.hist(bins=20, figsize=(20,45), layout=(15,3))

df_cp_fn_2 = df_cp_fn.drop(['2ndFlrSF','MasVnrArea'], axis=1)
df_cp_fn_2.columns
df_cp_fn_2.describe()
#Standarization

from sklearn.preprocessing import StandardScaler



df_standar = df_cp_fn_2



def standarize_df(df,label): 

    df_mean = df[label].mean()

    df_std = df[label].std()

    df[label] = (df[label] - df_mean) / df_std



df_labels = df_cp_fn_2.drop('SalePrice', axis=1).columns.tolist()



for i in df_labels:

    standarize_df(df_standar,i)
df_standar.boxplot(['GarageArea','YearRemodAdd','YearBuilt'])
res = [k for k in df_labels if not '__' in k]

df_standar[res].hist(bins=20, figsize=(20,45), layout=(15,3))
df_standar[res].boxplot(figsize=[25,20])
Q1 = df_standar[res].quantile(0.25)

Q3 = df_standar[res].quantile(0.75)

IQR = Q3-Q1

print(IQR)
# Test quartile outlier removal, check without it first.

df_standar.head()
df_standar.columns

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

y = df_standar[['SalePrice']].values

X = df_standar.drop('SalePrice',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

ridge = Ridge(alpha=0.1,normalize=True)

ridge.fit(X_train,y_train)

ridge_pred = ridge.predict(X_test)



ridge.score(X_test,y_test)
