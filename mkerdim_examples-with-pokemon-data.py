# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path = "/kaggle/input/pokemon-challenge"



from subprocess import check_output

print(check_output(["ls", data_path]).decode("utf-8"))



pokemon_file_path = os.path.join(data_path, "pokemon.csv")

combat_file_path = os.path.join(data_path, "combats.csv")

test_file_path = os.path.join(data_path, "tests.csv")



data = pd.read_csv(pokemon_file_path)

data.info()
f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt=".1f", ax=ax)

plt.show()
data.head(10)
data.columns
f, ax = plt.subplots(figsize=(16, 10))

data.Speed.plot(kind='line', color='g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle='-')

data.Defense.plot(kind='line', color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc="upper right")

plt.xlabel("x")

plt.ylabel("y")

plt.title("Line Plot")

plt.show()
print("Min and Max Attack values: {}, {}\nMin and Max Defense values: {}, {}".format(min(data.Attack),

                                                                                     max(data.Attack),

                                                                                     min(data.Defense),

                                                                                     max(data.Defense)))
f, ax = plt.subplots(figsize=(16, 10))

data.Attack.plot(kind="hist", bins=20, grid=True, alpha=0.6, color="orange", label="my_hist")

plt.legend(loc="upper center")

plt.show()
data.plot(kind="scatter", x="Attack", y="Defense", alpha=0.5, color="red")

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack-Defense Scatter Plot")

plt.show()
data.Speed.plot(kind="hist", bins=50, color="g", alpha=0.5, figsize=(10,10))

plt.show()
series = data["Defense"]

print(type(series))

print(series[:5])
data_frame = data[["Defense"]]

print(type(data_frame))

print(data_frame[:5])
x = series > 100

print(x)



condition1 = data.Attack > 100

condition2 = data.Defense < 150



data[condition1]
x = data.Defense > 200

print(data[x])



filtered = data[np.logical_and(data["Defense"]>150, data["Attack"]>100)]

#filtered = data[(data["Defense"]>150) & (data["Attack"]>100)]



print("Length of the filtered data: {}".format(len(filtered)))

print(filtered)



for index, value in data[["Attack"]][0:2].iterrows():

    print(index, ":", value)
data.head()

data.columns

data.info()

print(data["Type 1"].value_counts(dropna=False))
data.describe()
data.boxplot(column="Attack", by="Legendary")

plt.show()
data_new = data.head()

melted = pd.melt(frame=data_new, id_vars="Name", value_vars=["Attack", "Defense"])

melted
melted.pivot(index="Name", columns="variable", values="value")
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2], axis=0, ignore_index=True)

conc_data_row
data1 = data["Attack"].head()

data2 = data["Defense"].head()

conc_data_col = pd.concat([data1,data2], axis=1)

conc_data_col
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))

def prRed(skk): print("\033[91m{}\033[00m" .format(skk))

def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))

template = "{}: {}"



prYellow("#"*30)

_ = [prRed(template.format(i,data[i].dtypes)) if i=="Type 1" else print(template.format(i,data[i].dtypes)) for i in data.columns]

data["Type 1"] = data["Type 1"].astype("category")

prYellow("#"*30)

_ = [prCyan(template.format(i,data[i].dtypes)) if i=="Type 1" else print(template.format(i,data[i].dtypes)) for i in data.columns]

prYellow("#"*30)
data1 = data.copy()

print(data1["Type 2"].value_counts(dropna=False))

data1["Type 2"].dropna(inplace=True)

assert data1["Type 2"].notnull().all()
data2 = data.copy()

data2["Type 2"].fillna("empty", inplace=True)

assert data2["Type 2"].notnull().all()

print(data2["Type 2"].value_counts(dropna=False))

print(data.columns[1])

assert data.columns[1] == "Name"



print(data.Speed.dtypes)

assert data.Speed.dtypes == "int64"

assert data.Speed.dtypes == np.int
# data frames from dictionary



country = ["Spain", "France"]

population = ["11", "12"]

list_label = ["country", "population"]

list_col = [country, population]

print(list_col)

zipped = list(zip(list_label, list_col))

print(zipped)

data_dict = dict(zipped)

print(data_dict)



df = pd.DataFrame(data_dict)

df
df["capital"] = ["Madrid", "Paris"]

df["income"] = 0

df
data1 = data.loc[:,["Attack", "Defense", "Speed"]]

data1.plot()

plt.show()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="Attack",y="Speed")

plt.show()
data1.plot(kind="hist", bins=50, y="Speed", normed=True, range=(0,250))

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist", bins=50, y="Speed", normed=True, range=(0,250), ax=axes[0])

data1.plot(kind="hist", bins=50, y="Speed", normed=True, range=(0,250), ax=axes[1], cumulative=True, color='orange')

plt.savefig("graph.png")

plt.show()
# time series



time_list = ["2020-01-01", "2020-01-02"]

print(type(time_list))



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))

print(datetime_object)
data2 = data.copy().head()

date_list = ["2020-01-05", "2020-01-02", "2020-01-01", "2020-01-04", "2020-02-03"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

print(data2.dtypes)

data2 = data2.set_index("date")

data2
print(data2.loc["2020-01-02"])

print(data2.loc["2020-01-02":"2020-01-04"])

print(data2.dtypes)
print(data2.resample("A").mean()) # "A"->Year

print(data2.resample("M").mean()) # "M"->Month
data2.resample("M").first().interpolate("bfill")
data_path = "/kaggle/input/pokemon-challenge"

pokemon_file_path = os.path.join(data_path, "pokemon.csv")



data = pd.read_csv(pokemon_file_path)

data = data.set_index("#")

data.head()
print(data["HP"][1])

print(data.HP[1])

print(data.loc[1,["HP"]])
data[["HP", "Speed"]][0:10]
data.loc[1:8,"HP":"Defense"]
filter1 = data.HP > 120

filter2 = data.Speed <60

data[filter1 & filter2]
data.HP[data.Speed<20]
def div(n):

    return n/2



print(data.HP[0:5])

print(data.HP.apply(div)[0:5])

print(data.HP.apply(lambda n: n*2)[0:5])
data["total_power"] = data.Attack + data.Defense

data.head()
print(data.index.name)

data.index.name = "index_name"

data.head()
data2 = data.copy()

data2.index = range(100,900,1)

data2.head()
data2.index = data2["Name"]

data2.head()
data2 = data2.set_index(["Type 1", "Type 2"])

data2.head(20)
# pivot example

dic = {"treatment":["A","B","A","B"], "gender":["F","F","M","M"], "response":[11,12,13,14], "age":[42,53,24,77]}

df = pd.DataFrame(dic)

print(df)



df_pivot = df.pivot(index="treatment", columns="gender", values="response")

print(df_pivot)



df1 = df.set_index(["treatment", "gender"])

print(df1)

print(df1.unstack(level=0))

print(df1.unstack(level=1))
print(df1)

df2=df1.swaplevel(0,1)

print(df2)
df_melt = pd.melt(df,id_vars="treatment", value_vars=["age","response"])

print(df_melt)
print(df.groupby("treatment").mean())

print(df.groupby("treatment")[["age"]].mean())