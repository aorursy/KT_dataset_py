import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()
df_train = pd.read_csv("../input/train_set.csv", index_col="Id", low_memory=False)

df_test = pd.read_csv("../input/test_set.csv", index_col="Id", low_memory=False)
# 適切な型に変換

df_train.GIS_LAST_MOD_DTTM = df_train.GIS_LAST_MOD_DTTM.astype("str")

df_train.USECODE = df_train.USECODE.astype("str")

df_train.CMPLX_NUM = df_train.CMPLX_NUM.astype("str")

df_train.ZIPCODE = df_train.ZIPCODE.astype("str")

df_train.CENSUS_TRACT = df_train.CENSUS_TRACT.astype("str")

df_train.BLDG_NUM = df_train.BLDG_NUM.astype("str")



# 時間領域に直す

df_train.SALEDATE = pd.to_datetime(df_train.SALEDATE)

df_train["SALEDATE_YEAR"] = df_train.SALEDATE.dt.year

df_train["SALEDATE_month"] = df_train.SALEDATE.dt.month



# 前半のみ取り出す

df_train.NATIONALGRID = df_train.NATIONALGRID[df_train.NATIONALGRID.notnull()].apply(lambda x : x[:-12])



# trainデータの型を取得

list_dtypes = list(df_train.dtypes.unique())

print(list_dtypes)



# 不必要な変数を削除

del_col = ["X", "Y", "CITY", "STATE", "CMPLX_NUM", "FULLADDRESS",\

           "SQUARE", "CENSUS_BLOCK"] # X, YはほとんどLong, Latiとほぼ同じ、City, STATEは同じ内容

df_train.drop(del_col, axis=1, inplace=True)

df_test.drop(del_col, axis=1, inplace=True)





# 数値型とdatetime型のみを集める

df_train_num = df_train.select_dtypes(include=["float64", "int64"])

# カテゴリ変数を集める

df_train_cat = df_train.select_dtypes(include=["object"])
th_price = 2.4e+6

fig, ax = plt.subplots(1,2, figsize=(15,4))



df_train_num.PRICE[df_train_num.PRICE < th_price].hist(bins=50, ax=ax[0])

ax[0].set_title("Price (Low Threthold)", fontsize=15)



df_train_num.PRICE[df_train_num.PRICE >= th_price].hist(bins=50, ax=ax[1])

ax[1].set_title("Price (High Threthold)", fontsize=15);
# 下層のインデックスを取得

idx_low = df_train.PRICE.index[df_train.PRICE < th_price]

df_train_num_low = df_train.iloc[idx_low, :]
# 数値型のヒストグラム

df_train_num_low.hist(figsize=(20, 20), bins=30, xlabelsize=10, ylabelsize=10);
# PRICEと他の変数の関係

for i in range(0, len(df_train_num.columns), 4):

    sns.pairplot(height=5, data=df_train_num_low, x_vars=df_train_num.columns[i:i+4], y_vars=["PRICE"])
df_train_cat_low = df_train_cat.iloc[idx_low, :]

df_train_cat_low.nunique().sort_values()
# ユニークな要素の数

df_train_cat_count = df_train_cat_low.nunique()



list_cat = df_train_cat_count.index[df_train_cat_count < 25].values

print("ユニークな要素が多すぎる変数 {0}".format(df_train_cat_count.index[df_train_cat_count > 25].values))
# カテゴリ変数のヒストグラム

fig, axes = plt.subplots(round(len(list_cat) / 3), 3, figsize=(20, 40))



for i, ax in enumerate(fig.axes):

    if i < len(list_cat):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=20)

        ax.set_xlabel(ax.get_xlabel, fontsize=20)

        sns.countplot(x=list_cat[i], alpha=0.7, data=df_train_cat_low, ax=ax)



fig.tight_layout()
df_train.loc[:,["GIS_LAST_MOD_DTTM", "SOURCE"]].drop_duplicates()
fig, axes = plt.subplots(len(list_cat), 1, figsize=(14, 60))



for i, ax in enumerate(fig.axes):

    if i < len(list_cat):

        sns.boxplot(x=list_cat[i], y=df_train_num_low.PRICE, data=df_train_cat_low, ax=ax)

        

fig.tight_layout()
# 外れ値の除去

df_train_num_fil = df_train_num.copy()



df_train_num_fil.drop(df_train_num_fil.index[df_train_num_fil.YR_RMDL < 250], axis=0, inplace=True)

df_train_num_fil.drop(df_train_num_fil.index[df_train_num_fil.STORIES > 10], axis=0, inplace=True)

df_train_num_fil.drop(df_train_num_fil.index[df_train_num_fil.KITCHENS > 10 ], axis=0, inplace=True)

df_train_num_fil.drop(df_train_num_fil.index[(df_train_num_fil.LANDAREA == 0) |\

                                             (df_train_num_fil.LANDAREA > 25000)], axis=0, inplace=True)

df_train_num_fil.drop(df_train_num_fil.index[df_train_num_fil.PRICE > th_price], axis=0, inplace=True)

df_train_num_fil.drop(df_train_num_fil.index[df_train_num_fil.LIVING_GBA > 4000], axis=0, inplace=True)
# 数値型のヒストグラム

df_train_num_fil.hist(figsize=(20, 20), bins=30, xlabelsize=10, ylabelsize=10);
# PRICEと他の変数の関係

for i in range(0, len(df_train_num_fil.columns), 4):

    sns.pairplot(height=5, data=df_train_num_fil, x_vars=df_train_num_fil.columns[i:i+4], y_vars=["PRICE"])
df_train_num_fil.corr()["PRICE"].abs().sort_values(ascending=False)
corr = df_train_num_fil.drop("PRICE", axis=1).corr()

plt.figure(figsize=(15, 15))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 18}, square=True);
idx_high = df_train.PRICE.index[df_train.PRICE > th_price]

df_train_num_high = df_train_num.iloc[idx_high, :]

df_train_cat_high = df_train_cat.iloc[idx_high, :]
# PRICEと他の変数の関係

for i in range(0, len(df_train_num_high.columns), 4):

    sns.pairplot(height=5, data=df_train_num_high, x_vars=df_train_num_high.columns[i:i+4], y_vars=["PRICE"])
# カテゴリ変数のヒストグラム

fig, axes = plt.subplots(round(len(list_cat) / 3), 3, figsize=(20, 40))



for i, ax in enumerate(fig.axes):

    if i < len(list_cat):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=20)

        ax.set_xlabel(ax.get_xlabel, fontsize=20)

        sns.countplot(x=list_cat[i], alpha=0.7, data=df_train_cat_high, ax=ax)



fig.tight_layout()
fig, axes = plt.subplots(len(list_cat), 1, figsize=(14, 60))



for i, ax in enumerate(fig.axes):

    if i < len(list_cat):

        sns.boxplot(x=list_cat[i], y=df_train_num_high.PRICE, data=df_train_cat_high, ax=ax)

        

fig.tight_layout()
df_train_cond = df_train.query('SOURCE=="Condominium"')

df_train.columns[df_train_cond.isnull().sum() > 10000]
df_train_resi = df_train.query('SOURCE=="Residential"')

df_train.columns[df_train_resi.isnull().sum() > 10000]