# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_excel("/kaggle/input/wr9464.xlsx", header=None)

data.dtypes
data.columns = ["Индекс ВМО", "year", "month", "days",

"Минимальная температура воздуха", "Средняя температура воздуха",

"Максимальная температура воздуха", "Количество осадков"]

data
data["date"] = pd.to_datetime(data[["year", "month", "days"]])

data = data[["Индекс ВМО", "Минимальная температура воздуха", "Средняя температура воздуха",

"Максимальная температура воздуха", "Количество осадков", "date"]]

data = data.set_index("date").drop("Индекс ВМО", axis=1)
print(data.isna().sum())
nul_years = data.isnull().groupby(data.index.year).sum().sum(axis=1)

nul_years = nul_years[nul_years>0]

nul_years
data = data[data.index >= pd.datetime(1966, 1, 1)]

data
data["Размах температуры"] = data["Максимальная температура воздуха"] - data["Минимальная температура воздуха"]

data
data["есть осадки"] = data["Количество осадков"] > 0

data["cumsum"] = data["есть осадки"].cumsum().shift(1, fill_value=0) #Some magic

data["rank"] = data["cumsum"].rank(method="min").astype(int) - 1

data["№"] = np.arange(len(data))

data["дней без осадков"] = data["№"] - data["rank"]

data = data.drop(["№", "cumsum", "rank", "есть осадки"], axis=1)

data.head(30)
data.iloc[data["дней без осадков"].values.argmax()]
jan = data.groupby([data.index.year, data.index.month])["Максимальная температура воздуха"].max()

jun = data.groupby([data.index.year, data.index.month])["Средняя температура воздуха"].mean()

jan

jan.xs(1, level=1)
jun.xs(6, level=1)
data[data["Средняя температура воздуха"] < -30]
data[(data["Средняя температура воздуха"] > 25) & (data["дней без осадков"] > 3)]