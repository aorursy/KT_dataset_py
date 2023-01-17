from datetime import datetime
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import skew

from IPython.core.display import display
from tqdm import tqdm
tqdm.pandas()

print(os.listdir("../input"))
# 冒頭のpivot_tableを用いたバージョン
# 欠損値が含まれることに注意する

tick = datetime.now()
train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": np.uint32,
                                                           "mjd": np.float64,
                                                           "passband": np.uint8,
                                                           "flux": np.float32,
                                                           "flux_err": np.float32,
                                                           "detected": np.uint8})
train_meta_df = pd.read_csv('../input/training_set_metadata.csv')
tock = datetime.now()
print("load_data: {} ms".format((tock - tick).seconds * 1000 + ((tock - tick).microseconds / 1000)))

tick = datetime.now()

# pivot_tableのindexをrankを用いて作成する
train_df["rank"] = train_df.groupby(["object_id", "passband"])["mjd"].rank()

flux = train_df.pivot_table(columns=["object_id", "passband"],
                            index="rank",
                            values="flux",
                            aggfunc="mean")
dflux = train_df.pivot_table(columns=["object_id", "passband"],
                             index="rank",
                             values="flux_err",
                             aggfunc="mean")

# 列にNaNが含まれるので扱いに注意する
flux_mean = np.sum(flux*np.square(flux/dflux), axis=0)/np.sum(np.square(flux/dflux), axis=0)
flux_std = np.std(flux/flux_mean, ddof = 1, axis=0)
flux_amp = (np.max(flux, axis=0) - np.min(flux, axis=0))/flux_mean
flux_mad = np.nanmedian(np.abs((flux - np.nanmedian(flux, axis=0))/flux_mean), axis=0) # array
flux_beyond = np.sum(np.abs(flux - flux_mean) > np.std(flux, ddof = 1, axis=0), axis=0)/flux.count()
flux_skew = skew(flux, nan_policy="omit", axis=0)  # masked_array

result_df = pd.concat([flux_mean.reset_index(name="flux_mean"),
                      flux_std.reset_index(name="flux_std").iloc[:, 2:],
                      flux_amp.reset_index(name="flux_amp").iloc[:, 2:],
                      flux_beyond.reset_index(name="flux_beyond").iloc[:, 2:]], axis=1)
result_df["flux_mad"] = flux_mad
result_df["flux_skew"] = flux_skew
colnames = ["flux_mean", "flux_std", "flux_amp", "flux_beyond", "flux_mad", "flux_skew"]

for j in range(6):
    train_meta_df = train_meta_df.merge(result_df.loc[result_df.passband == j, :]
                                                 .rename(columns={colname: "{}_{}".format(colname, j) for colname in colnames})
                                                 .drop("passband", axis=1),
                                        how="left",
                                        on=["object_id"])
tock = datetime.now()
print("processing_time: {} sec".format((tock - tick).seconds))

train_meta_df.head()
# 以下のようなデータを用意する
dammy_dics = []
for i in range(5):
    for j in range(10):
        dammy_dics.append({"time": i, "category": j, "price": 10*i + j})

dammy_df = pd.DataFrame(dammy_dics)
dammy_df.head(10)
# DataFrame.pivot_table()でクロス集計表を作れる
dammy_piv = dammy_df.pivot_table(index="time",
                                 columns="category",
                                 values="price",
                                 aggfunc="sum")
display(dammy_piv)
# pivot_tableは行列として計算することができる
# 各数値を二乗する
print("piv^2")
display(np.square(dammy_piv))

# スカラーで割る"
print("piv / 10")
display(dammy_piv / 10)

# pivot_table同士を足す
print("piv + piv^2")
display(dammy_piv + np.square(dammy_piv))
# 列方向への集計
# axisを指定しないと自動的に列方向の集計になり、Seriesが返ってくる
display(dammy_piv.mean())

# pivot_tableに対してSeriesで計算するとと列方向にbroadcastされる
display(dammy_piv - dammy_piv.mean())
# "行方向への集計も可能だが"
display(dammy_piv.mean(axis=1))

# いい感じにbroadcastしてくれない
print("piv - seires")
display(dammy_piv - dammy_piv.mean(axis=1))

# 転値を使うくらいしか良い方法が思い浮かばないので良い方法があれば教えてください
print("(piv.T - series).T")
display((dammy_piv.T - dammy_piv.mean(axis=1)).T)
# piv.shift()でひとつ前の値をとれる
dammy_piv.shift(1)
# これを活用すると、ひとつ前との差分をとることができる
dammy_piv - dammy_piv.shift(1)
# rolling関数で、移動平均等をとることができる
# 以下のコードは自信を含めた三つの期間分の平均
dammy_piv.rolling(window=3, center=False).mean()
# shiftと組み合わせることで、一つ前からn個前までの平均といった特徴量を作ることができる
dammy_piv.rolling(window=3, center=False).mean().shift(1)
# cum〇〇系の関数はそれまでの合計を計算できる
# 合計
display(dammy_piv.cumsum())
# 上記までのテクニックを駆使すると、leak無しに時系列のmean_encodingができる
cum_sum = dammy_df.pivot_table(index="time",
                               columns="category",
                               values="price",
                               aggfunc="sum").cumsum()
cum_count = dammy_df.pivot_table(index="time",
                                 columns="category",
                                 values="price",
                                 aggfunc="count").cumsum()
cum_mean = cum_sum / cum_count
cum_mean_without_leakage = cum_mean.shift(1)
cum_mean_without_leakage
# データのロード
train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": np.uint32,
                                                           "mjd": np.float64,
                                                           "passband": np.uint8,
                                                           "flux": np.float32,
                                                           "flux_err": np.float32,
                                                           "detected": np.uint8})
train_meta_df = pd.read_csv('../input/training_set_metadata.csv')
test_meta_df = pd.read_csv('../input/test_set_metadata.csv')
# train_dfを集計してtrain_metaに結合したい
display(train_df.head())
display(train_meta_df.head())
print("train_meta: ", train_meta_df.shape)
print("test_meta: ", test_meta_df.shape)
print("テストデータは訓練データの{:.4}倍".format(test_meta_df.shape[0] / train_meta_df.shape[0]))
# groupby無しに毎回取り出そうとするととてつもない時間がかかるので1/100だけ計算
bands = [train_df.passband == b for b in train_df.passband.unique()]
for id_ in tqdm(train_df.object_id.unique()[:78]):
    for band in bands:
        idx = train_df[(train_df.object_id == id_) & band].index
        flux, dflux = train_df.loc[idx, "flux"], train_df.loc[idx, "flux_err"]
        train_df.loc[idx, "flux_mean"] = np.sum(flux*np.square(flux/dflux))/np.sum(np.square(flux/dflux))
        fluxm = train_df.loc[idx, "flux_mean"]

        train_df.loc[idx, "flux_std"] = np.std(flux/fluxm, ddof = 1)
        train_df.loc[idx, "flux_amp"] = (np.max(flux) - np.min(flux))/fluxm
        train_df.loc[idx, "flux_mad"] = np.median(np.abs((flux - np.median(flux))/fluxm))
        train_df.loc[idx, "flux_beyond"] = sum(np.abs(flux - fluxm) > np.std(flux, ddof = 1))/len(flux)
        train_df.loc[idx, "flux_skew"] = skew(flux)
# 2. groupbyを使って計算する
tick = datetime.now()
train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": np.uint32,
                                                           "mjd": np.float64,
                                                           "passband": np.uint8,
                                                           "flux": np.float32,
                                                           "flux_err": np.float32,
                                                           "detected": np.uint8})
train_meta_df = pd.read_csv('../input/training_set_metadata.csv')
tock = datetime.now()
print("load_data: {} ms".format((tock - tick).seconds * 1000 + ((tock - tick).microseconds / 1000)))

tick = datetime.now()

def agg_func(x):
    d = {}
    flux, dflux = x["flux"], x["flux_err"]
    flux_mean = np.sum(flux*np.square(flux/dflux))/np.sum(np.square(flux/dflux))
    d["flux_mean"] = flux_mean
    d["flux_std"] = np.std(flux/flux_mean, ddof = 1)
    d["flux_amp"] = (np.max(flux) - np.min(flux))/flux_mean
    d["flux_beyond"] = np.sum(np.abs(flux - flux_mean) > np.std(flux, ddof = 1))/flux.shape[0]
    d["flux_mad"] = np.median(np.abs((flux - np.median(flux))/flux_mean))
    d["flux_skew"] = skew(flux)
    return pd.Series(d, index = ["flux_mean", "flux_std", "flux_amp", "flux_mad", "flux_beyond", "flux_skew"])

result_df = train_df.groupby(["object_id", "passband"]).progress_apply(agg_func).reset_index()

colnames = ["flux_mean", "flux_std", "flux_amp", "flux_mad", "flux_beyond", "flux_skew"]
for j in range(6):
    train_meta_df = train_meta_df.merge(result_df.loc[result_df.passband == j, :]
                                                 .rename(columns={colname: "{}_{}".format(colname, j) for colname in colnames})
                                                 .drop("passband", axis=1),
                                        how="left",
                                        on=["object_id"])

tock = datetime.now()
tmp = print("total_processing: {} sec".format((tock - tick).seconds))
train_meta_df.head()
# 欠損値が含まれることに注意する

tick = datetime.now()
train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": np.uint32,
                                                           "mjd": np.float64,
                                                           "passband": np.uint8,
                                                           "flux": np.float32,
                                                           "flux_err": np.float32,
                                                           "detected": np.uint8})
train_meta_df = pd.read_csv('../input/training_set_metadata.csv')
tock = datetime.now()
print("load_data: {} ms".format((tock - tick).seconds * 1000 + ((tock - tick).microseconds / 1000)))

tick = datetime.now()

# pivot_tableのindexをrankを用いて作成する
train_df["rank"] = train_df.groupby(["object_id", "passband"])["mjd"].rank()

flux = train_df.pivot_table(columns=["object_id", "passband"],
                            index="rank",
                            values="flux",
                            aggfunc="mean")
dflux = train_df.pivot_table(columns=["object_id", "passband"],
                             index="rank",
                             values="flux_err",
                             aggfunc="mean")

# 列にNaNが含まれるので扱いに注意する
flux_mean = np.sum(flux*np.square(flux/dflux), axis=0)/np.sum(np.square(flux/dflux), axis=0)
flux_std = np.std(flux/flux_mean, ddof = 1, axis=0)
flux_amp = (np.max(flux, axis=0) - np.min(flux, axis=0))/flux_mean
flux_mad = np.nanmedian(np.abs((flux - np.nanmedian(flux, axis=0))/flux_mean), axis=0) # array
flux_beyond = np.sum(np.abs(flux - flux_mean) > np.std(flux, ddof = 1, axis=0), axis=0)/flux.count()
flux_skew = skew(flux, nan_policy="omit", axis=0)  # masked_array

result_df = pd.concat([flux_mean.reset_index(name="flux_mean"),
                      flux_std.reset_index(name="flux_std").iloc[:, 2:],
                      flux_amp.reset_index(name="flux_amp").iloc[:, 2:],
                      flux_beyond.reset_index(name="flux_beyond").iloc[:, 2:]], axis=1)
result_df["flux_mad"] = flux_mad
result_df["flux_skew"] = flux_skew
colnames = ["flux_mean", "flux_std", "flux_amp", "flux_beyond", "flux_mad", "flux_skew"]

for j in range(6):
    train_meta_df = train_meta_df.merge(result_df.loc[result_df.passband == j, :]
                                                 .rename(columns={colname: "{}_{}".format(colname, j) for colname in colnames})
                                                 .drop("passband", axis=1),
                                        how="left",
                                        on=["object_id"])
tock = datetime.now()
print("processing_time: {} sec".format((tock - tick).seconds))

train_meta_df.head()