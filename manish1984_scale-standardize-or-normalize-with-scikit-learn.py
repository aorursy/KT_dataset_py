import numpy as np 

import pandas as pd 

from sklearn import preprocessing



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')



%matplotlib inline

matplotlib.style.use('ggplot')



np.random.seed(34)
#create columns of various distributions

df = pd.DataFrame({ 

    'beta': np.random.beta(5, 1, 1000) * 60,        # beta

    'exponential': np.random.exponential(10, 1000), # exponential

    'normal_p': np.random.normal(10, 2, 1000),      # normal platykurtic

    'normal_l': np.random.normal(10, 10, 1000),     # normal leptokurtic

})



# make bimodal distribution

first_half = np.random.normal(20, 3, 500) 

second_half = np.random.normal(-20, 3, 500) 

bimodal = np.concatenate([first_half, second_half])



df['bimodal'] = bimodal



# create list of column names to use later

col_names = list(df.columns)

print(col_names)
# plot original distribution plot

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('Original Distributions')



sns.kdeplot(df['beta'], ax=ax1)

sns.kdeplot(df['exponential'], ax=ax1)

sns.kdeplot(df['normal_p'], ax=ax1)

sns.kdeplot(df['normal_l'], ax=ax1)

sns.kdeplot(df['bimodal'], ax=ax1);
df.head()
df.mean()
df.describe()
df.plot()
normal_big = np.random.normal(1000000, 10000, (1000,1))  # normal distribution of large values

df['normal_big'] = normal_big
col_names.append('normal_big')
df['normal_big'].plot(kind='kde')
df.normal_big.mean()
# plot original distribution plot with larger value feature

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('Original Distributions')



sns.kdeplot(df['beta'], ax=ax1)

sns.kdeplot(df['exponential'], ax=ax1)

sns.kdeplot(df['normal_p'], ax=ax1)

sns.kdeplot(df['normal_l'], ax=ax1)

sns.kdeplot(df['bimodal'], ax=ax1);

sns.kdeplot(df['normal_big'], ax=ax1);
df.plot()
df.describe()
mm_scaler = preprocessing.MinMaxScaler()

df_mm = mm_scaler.fit_transform(df)

print(col_names)

df_mm = pd.DataFrame(df_mm, columns=col_names)



fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('After MinMaxScaler')



sns.kdeplot(df_mm['beta'], ax=ax1)

sns.kdeplot(df_mm['exponential'], ax=ax1)

sns.kdeplot(df_mm['normal_p'], ax=ax1)

sns.kdeplot(df_mm['normal_l'], ax=ax1)

sns.kdeplot(df_mm['bimodal'], ax=ax1)

sns.kdeplot(df_mm['normal_big'], ax=ax1);
df_mm['beta'].min()
df_mm['beta'].max()
mins = [df[col].min() for col in df.columns]

mins
maxs = [df[col].max() for col in df.columns]

maxs
mins = [df_mm[col].min() for col in df_mm.columns]

mins
maxs = [df_mm[col].max() for col in df_mm.columns]

maxs
r_scaler = preprocessing.RobustScaler()

df_r = r_scaler.fit_transform(df)



df_r = pd.DataFrame(df_r, columns=col_names)



fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('After RobustScaler')



sns.kdeplot(df_r['beta'], ax=ax1)

sns.kdeplot(df_r['exponential'], ax=ax1)

sns.kdeplot(df_r['normal_p'], ax=ax1)

sns.kdeplot(df_r['normal_l'], ax=ax1)

sns.kdeplot(df_r['bimodal'], ax=ax1)

sns.kdeplot(df_r['normal_big'], ax=ax1);
mins = [df_r[col].min() for col in df_r.columns]

mins
maxs = [df_r[col].max() for col in df_r.columns]

maxs
s_scaler = preprocessing.StandardScaler()

df_s = s_scaler.fit_transform(df)



df_s = pd.DataFrame(df_s, columns=col_names)



fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('After StandardScaler')



sns.kdeplot(df_s['beta'], ax=ax1)

sns.kdeplot(df_s['exponential'], ax=ax1)

sns.kdeplot(df_s['normal_p'], ax=ax1)

sns.kdeplot(df_s['normal_l'], ax=ax1)

sns.kdeplot(df_s['bimodal'], ax=ax1)

sns.kdeplot(df_s['normal_big'], ax=ax1);
mins = [df_s[col].min() for col in df_s.columns]

mins
maxs = [df_s[col].max() for col in df_s.columns]

maxs
n_scaler = preprocessing.Normalizer()

df_n = n_scaler.fit_transform(df)



df_n = pd.DataFrame(df_n, columns=col_names)



fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('After Normalizer')



sns.kdeplot(df_n['beta'], ax=ax1)

sns.kdeplot(df_n['exponential'], ax=ax1)

sns.kdeplot(df_n['normal_p'], ax=ax1)

sns.kdeplot(df_n['normal_l'], ax=ax1)

sns.kdeplot(df_n['bimodal'], ax=ax1)

sns.kdeplot(df_n['normal_big'], ax=ax1);
mins = [df_n[col].min() for col in df_n.columns]

mins
maxs = [df_n[col].max() for col in df_n.columns]

maxs
# Combined plot.



fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(20, 8))





ax0.set_title('Original Distributions')



sns.kdeplot(df['beta'], ax=ax0)

sns.kdeplot(df['exponential'], ax=ax0)

sns.kdeplot(df['normal_p'], ax=ax0)

sns.kdeplot(df['normal_l'], ax=ax0)

sns.kdeplot(df['bimodal'], ax=ax0)

sns.kdeplot(df['normal_big'], ax=ax0);





ax1.set_title('After MinMaxScaler')



sns.kdeplot(df_mm['beta'], ax=ax1)

sns.kdeplot(df_mm['exponential'], ax=ax1)

sns.kdeplot(df_mm['normal_p'], ax=ax1)

sns.kdeplot(df_mm['normal_l'], ax=ax1)

sns.kdeplot(df_mm['bimodal'], ax=ax1)

sns.kdeplot(df_mm['normal_big'], ax=ax1);





ax2.set_title('After RobustScaler')



sns.kdeplot(df_r['beta'], ax=ax2)

sns.kdeplot(df_r['exponential'], ax=ax2)

sns.kdeplot(df_r['normal_p'], ax=ax2)

sns.kdeplot(df_r['normal_l'], ax=ax2)

sns.kdeplot(df_r['bimodal'], ax=ax2)

sns.kdeplot(df_r['normal_big'], ax=ax2);





ax3.set_title('After StandardScaler')



sns.kdeplot(df_s['beta'], ax=ax3)

sns.kdeplot(df_s['exponential'], ax=ax3)

sns.kdeplot(df_s['normal_p'], ax=ax3)

sns.kdeplot(df_s['normal_l'], ax=ax3)

sns.kdeplot(df_s['bimodal'], ax=ax3)

sns.kdeplot(df_s['normal_big'], ax=ax3);