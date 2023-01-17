import numpy as np 

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.max_columns = 100

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_targets = df_train[['id', 'target']]

df_train.drop(['id', 'target'], axis = 1, inplace = True)
def get_info(df):

    """

    Gives some infos on columns types and number of null values

    """

    print('dataframe dimensions:', df.shape)

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

    df.replace({-1:np.nan}, inplace = True) # TAG NULL VALUES

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                             .T.rename(index={0:'null values (%)'}))

    return tab_info
tab_info = get_info(df_train)

tab_info
get_info(df_test)
tab_info = tab_info.T.reset_index()

tab_info = tab_info.sort_values('null values (%)').reset_index(drop = True)

#_____________________________________

y_axis  = tab_info['null values (%)'] 

x_label = tab_info['index']

x_axis  = tab_info.index



fig = plt.figure(figsize=(11, 4))

plt.xticks(rotation=80, fontsize = 14)

plt.yticks(fontsize = 13)

plt.bar(x_axis, y_axis)

plt.xticks(x_axis, x_label, fontsize = 12)

plt.title('Missing values (%)', fontsize = 18);
nb = sum(["cat" in s for s in df_train.columns])

print('categorical variables: {} '.format(nb))
ind = 0

for col in df_train.columns:

    if "cat" not in col: continue

    ind += 1

    fig = plt.figure(1, figsize=(11,30))

    ax1 = fig.add_subplot(nb,1,ind)    

    x_axis = list(df_train[col].value_counts().index)

    y_axis = list(df_train[col].value_counts())

    x_label = list(map(int,x_axis))

    if len(x_label) > 50:

        x_label = [s if s%2 == 0 else '' for i,s in enumerate(x_label)]

    plt.xticks(x_axis, x_label)

    ax1.bar(x_axis, y_axis, align = 'center', label = col)

    plt.legend(prop={'size': 14})

    if ind == nb: break
nb  = sum([("cat" not in s) and ('bin' not in s) for s in df_train.columns])

print('numerical variables: {} '.format(nb))
# uniform distributions:

list_cols_uniform = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03']

#____________________________

ind = 0

for col in list_cols_uniform:

    ind += 1

    fig = plt.figure(1, figsize=(11,3))

    ax1 = fig.add_subplot(1, 3, ind)    

    sns.distplot(df_train[col].dropna(), kde=False  )        

    if ind == nb: break

plt.suptitle('Uniform distributions');
# shallow distributions: 

list_cols_shallow = ['ps_car_13', 'ps_reg_03', 'ps_calc_10', 'ps_calc_14', 'ps_calc_11',

                     'ps_ind_03', 'ps_calc_13', 'ps_calc_06', 'ps_calc_07', 'ps_calc_07',

                     'ps_calc_09', 'ps_calc_12', 'ps_calc_04', 'ps_calc_05', 'ps_car_11']

#____________________________

ind = 0

for col in list_cols_shallow:

    ind += 1

    fig = plt.figure(1, figsize=(10,15))

    ax1 = fig.add_subplot(5, 3, ind)    

    sns.distplot(df_train[col].dropna(), kde=False  )        

    if ind == nb: break

fig.suptitle('Shallow distributions')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
list_cols_other   = ['ps_ind_01', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_car_12',

                     'ps_car_14', 'ps_car_15', 'ps_ind_14', 'ps_car_12']

#____________________________

ind = 0

for col in list_cols_other:

    ind += 1

    fig = plt.figure(1, figsize=(10,10))

    ax1 = fig.add_subplot(3, 3, ind)    

    sns.distplot(df_train[col].dropna(), kde=False  )        

    if ind == nb: break

fig.suptitle('Empirical distributions')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

list_cols = []

for col in df_train.columns:

    if 'bin' not in col and 'cat' not in col:

        list_cols.append(col)



        

df_corr = df_train.copy(deep=True)

df_corr['target'] = df_targets['target']

corrmat = df_corr[list_cols + ['target']].corr()
df_corr[list_cols + ['target']].dropna(how='any').corr()[:5]
corrmat[:5]
f, ax = plt.subplots(figsize=(12, 9))

k = 15 # number of variables for heatmap

cols = corrmat.nlargest(k, 'ps_reg_01')['ps_reg_01'].index

#cm = np.corrcoef(df_corr[cols].dropna(how='any').values.T)

cm = np.corrcoef(df_corr[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, linewidth = 0.1, cmap = 'coolwarm',

                 yticklabels=cols.values, xticklabels=cols.values)

f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize = 18)

plt.show()