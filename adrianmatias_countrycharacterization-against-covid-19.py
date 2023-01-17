%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(color_codes=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df = pd.read_csv(os.path.join(dirname, filename))

        print("\n", filename)

        print(df.head())
folder = "/kaggle/input/novel-corona-virus-2019-dataset"

filename = "covid_19_data.csv"



df = pd.read_csv(

    os.path.join(folder, filename), 

    parse_dates=["ObservationDate", "Last Update"], 

#     index_col="SNo"

)



df.head()
df.describe(include="all")
col_date = "ObservationDate"

col_country = "Country/Region"

col_province = "Province/State"

cols_count = list(df.columns[-3:])

cols_count
df["Province/State"].value_counts()
def plot_vs_time(df, col_place, place, col_count):

    df[df[col_place] == place].set_index(col_date)[col_count].plot()



plot_vs_time(

    df=df,

    col_place=col_province,

    place="Shanxi",

    col_count="Confirmed")
# cols_key = list(df.columns[1:])

cols_key = ["ObservationDate", "Country/Region"]

cols_count = list(df.columns[-3:])

df_country = df.groupby(cols_key,)[cols_count].sum().reset_index()

df_country.head()
plot_vs_time(

    df=df_country,

    col_place=col_country,

    place="Mainland China",

    col_count="Confirmed")
plot_vs_time(

    df=df_country,

    col_place=col_country,

    place="Spain",

    col_count="Confirmed")
def get_r0(df, country="Spain", col_count="Confirmed", min_n=100):

    """

    For fitting y = Aexp(rx), take the logarithm of both side gives log y = log A + Bx

    """

    df = df.copy()

    df = df[df[col_country] == country]

    df = df[df[col_count] >= min_n]



    cols = [col_date, col_count]

    df = (

        df

#         [df[col_count] >= min_n]

        .reset_index()

        .drop(columns="index")

    )

    

    df["log"] = np.log(df[col_count])

    df["log"].plot()

    return df

    (r, logA) = np.polyfit(x=df.index, y=np.log(df[col_count]), deg=1)

    return r, logA



get_r0(df_country, country="Mainland China")
def get_r0_values(xs: np.ndarray):

    (r, logA) = np.polyfit(x=range(len(xs)), y=np.log(xs), deg=1)

    return r



def get_r0_rolling(df: pd.DataFrame):

    df = df.groupby([col_country, col_date]).sum().reset_index()



    return (

        df

        .set_index(col_date)

        .groupby(col_country)

        .rolling("7D", min_periods=7)

        .apply(get_r0_values)

    )



get_r0_rolling(df)
def get_r0_log_delta(df, do_moving_avg=False):



    df = df.groupby([col_country, col_date]).sum().reset_index()

    

    df = df[df[cols_count[0]] >= 100]

    

#     return df



    df[cols_count] = (

        df[cols_count]

        .apply(lambda x: np.log(x + 1))

        .diff()

        .clip(lower=0)

    )

    

    

    

    if not do_moving_avg:

        return df



    return (

        df

        .set_index(col_date)

        .groupby(col_country)[cols_count]

        .rolling("2D", min_preriods=7)

        .mean()

        .reset_index()

    )



df_r = get_r0_log_delta(df)

df_r.tail(100)
countries = ["Mainland China", "Spain", "Italy", "South Korea", "France", "US", "Japan"]

for c in countries:

    df_r[df_r[col_country] == c].set_index(col_date)[cols_count[0]].plot(title=c)

    plt.show()
def get_top_country_list(df: pd.DataFrame):

    return (

        df

        .groupby(col_country)[cols_count[0]]

        .max()

        .sort_values(ascending=False)

    )



df_country_count = get_top_country_list(df)



df_country_count.head(20).plot(kind="barh")
def get_country_speed(df: pd.DataFrame):

#     return df

    df = df[df[cols_count[0]] > 0]

    return (

        df

        .groupby(col_country)[cols_count[0]]

        .median()

        .sort_values(ascending=False)

    )



df_country_speed = get_country_speed(df_r)

# df_country_speed

df_country_speed.head(20).plot(kind="barh")
def flatten_hierarchical_cols(df: pd.DataFrame):

    combined_levels = zip(

        df.columns.get_level_values(0),

        df.columns.get_level_values(1)

    )

    

    df.columns = map(lambda levels: "_".join(levels), combined_levels)

    return df



def describe_country(df: pd.DataFrame):

    cols_dict = dict(zip(

        cols_count,

        [c + "_r0" for c in cols_count]

    ))

    df_r = get_r0_log_delta(df)

    df_r = df_r[df_r[cols_count[0]] > 0]

    df_r = df_r.rename(columns=cols_dict)

    

    df_c = (

        df

        .groupby(col_country)[cols_count].agg(["count", "max"])

    )

    

    df_c = flatten_hierarchical_cols(df_c)



    df_c["mortality"] = df_c["Deaths_max"] / df_c["Confirmed_max"]

    df_c["recovery"] = df_c["Recovered_max"] / df_c["Confirmed_max"]

    

    df_r = get_r0_log_delta(df)

    df_r = df_r.set_index(col_country)[cols_count]

    cols_count_r = [c + "_r0" for c in cols_count]

    df_r.columns = cols_count_r

    

    df_c_r = df_r.groupby(col_country).agg(["mean", "median", "var", "min", "max", "skew"])

    

    df_c_r = flatten_hierarchical_cols(df_c_r)



    return df_c.join(df_c_r)

    



df_c = describe_country(df)

df_c.sort_values("mortality", ascending=False).head(20)
df_c.describe()
def safe_log(s: pd.Series):

    s = s.dropna()

    return np.log(s[s > 0])



def log_cols(df: pd.DataFrame, cols_log_not=[]):

    cols_log = list(

        set(df_c.columns).difference(

            set(cols_log_not)))

    for c in cols_log:

        df[c] = safe_log(df[c])

    return df



df_c = log_cols(df_c)



for c in df_c.columns[:10]:

    sns.distplot(df_c[c])

    plt.show()
df_c.describe()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
n_components_pca = 5

pipeline_cluster = Pipeline([

    ('StandardScaler', StandardScaler()),

    ('SimpleImputer', SimpleImputer(strategy="constant", fill_value=0)),

    ('PCA', PCA(n_components=n_components_pca)),

    ('KMeans', KMeans()),

])



pipeline_cluster = pipeline_cluster.fit(df_c)
# df_c["cluster"] = pipeline_cluster.predict(df_c)
# for i in df_c["cluster"].unique():

#     print(list(df_c[df_c["cluster"] == i].head(20).index))

#     print()
pipeline_pca = Pipeline(pipeline_cluster.steps[:3])

cols_pca = [f"pca_{i}" for i in range(n_components_pca)]

pipeline_pca.fit_transform(df_c)
df_pca = df_c.join(pd.DataFrame(

    pipeline_pca.fit_transform(df_c),

    columns=cols_pca,

    index=df_c.index

))
pipeline_pca.steps[-1][1].explained_variance_ratio_
df_corr = df_pca.corr()

df_corr[cols_pca[0]].sort_values(ascending=False)
df_corr[cols_pca[1]].sort_values(ascending=False)
df_corr[cols_pca[2]].sort_values(ascending=False)
interpretation_pca = [

    "component_0: duration, volume, mortality rate",

    "component_1: deaths deceleration, recovery rate",

    "component_2: recovered volume, recovered acceleration"

]
df_pca_top = df_pca.sort_values("Confirmed_max").tail(20)



def plot_components(df_pca_top, x=0, y=1):

    plt.figure(figsize=(8, 8), dpi=60)

    plt.scatter(df_pca_top[cols_pca[x]], df_pca_top[cols_pca[y]]).setfigSize=(30, 30)

    plt.xlabel(interpretation_pca[x])

    plt.ylabel(interpretation_pca[y])



    for _, (c_pcs) in df_pca_top.reset_index()[[col_country] + cols_pca].iterrows():

        plt.annotate(c_pcs[0], (c_pcs[x + 1], c_pcs[y + 1]))



plot_components(df_pca_top, x=0, y=1)
plot_components(df_pca_top, x=1, y=2)