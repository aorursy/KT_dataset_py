# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Prepare your file

parent_dir: str = os.path.join('/kaggle', 'input', 'twitter-bots-accounts')

dataset_name: str = "twitter_human_bots_dataset.csv"

dataset_path: str = os.path.join(parent_dir, dataset_name)

print(f"Dataset directory: {dataset_path}")
# Generate a Pandas DataFrame

twitter_accounts_df: pd.DataFrame = pd.read_csv(dataset_path, index_col=0)

print(f"Dataset shape {twitter_accounts_df.shape}")
# Take a look to the Data

print(f"Dataset columns: {twitter_accounts_df.columns}")

twitter_accounts_df.head()
# Functions to preprocess the DataFrame

def convert_bool_to_int(data: pd.DataFrame, boolean_cols: list):

    try:

        for col in boolean_cols:

            data[col] = data[col].astype(int)

    except Exception as e:

        print(e)

    return data



def popularity_metric(friends_count: int, followers_count: int):

    return np.round(np.log(1+friends_count) * np.log(1+followers_count), 3)





def compute_popularity_metric(row):

    return popularity_metric(friends_count=row["friends_count"],

                             followers_count=row["followers_count"])
# Preprocess boolean columns

boolean_cols: list = ["default_profile", "default_profile_image",

                      "geo_enabled", "verified"]

twitter_accounts_df = convert_bool_to_int(data=twitter_accounts_df, boolean_cols=boolean_cols)

twitter_accounts_df.head()
# Create a custom metric to measure the popularity of an input account

twitter_accounts_df["popularity"] = twitter_accounts_df.apply(compute_popularity_metric, axis=1)



# Let's show some examples of such value

twitter_accounts_df[['popularity']]
import matplotlib as mpl

import seaborn as sns

from matplotlib import pyplot as plt

from collections import OrderedDict



mpl.rcParams['font.family'] = 'sans-serif'

mpl.rcParams['figure.figsize'] = 12, 8

mpl.rcParams['font.sans-serif'] = ['Tahoma']

sns.set(font_scale=1.5)

sns.set_style("whitegrid")
# Set up some parameters for EDA

palette: str = "husl"

grouped: str = "account_type"

default_value: str = "unknown"
def get_labels_colors_from_pandas_column(df: pd.DataFrame, column: str, palette: str):

    data_labels: dict = dict()

    try:

        labels: list = df[column].unique().tolist()

        colors: list = sns.color_palette(palette, len(labels))

        data_labels: dict = dict(zip(labels, colors))

    except Exception as e:

        logger.error(e)

    return data_labels



# Retrieve labels and additional parameters to plot figures

data_labels: dict = get_labels_colors_from_pandas_column(

    df=twitter_accounts_df, column=grouped, palette=palette)

# Show labels

print(f"Unique Target values: {data_labels.keys()}")
# Functions to plot data distributions

def plot_multiple_histograms(data: pd.DataFrame,

                             grouped_col: str,

                             target_col: str,

                             data_labels: dict):

    # Plot

    plt.figure(figsize=(12, 10))

    title = "\n"

    labels: list = list(data_labels.keys())

    for j, i in enumerate(labels):

        x = data.loc[data[grouped_col] == i, target_col]

        mu_x = round(float(np.mean(x)), 3)

        sigma_x = round(float(np.std(x)), 3)

        ax = sns.distplot(x, color=data_labels.get(i), label=i, hist_kws=dict(alpha=.1),

                          kde_kws={'linewidth': 2})

        ax.axvline(mu_x, color=data_labels.get(i), linestyle='--')

        ax.set(xlabel=f"{target_col.title()}", ylabel='Density')

        title += f"Parameters {str(i)}: $G(\mu=$ {mu_x}, $\sigma=$ {sigma_x} \n"

        ax.set_title(title)

    plt.legend(title="Account Type")

    plt.grid()

    plt.tight_layout()

    plt.show()





def plot_multiple_boxplots(data: pd.DataFrame, grouped_col: str, target_col: str,

                           palette: str = "husl"):

    plt.figure(figsize=(12, 10))



    means: dict = data.groupby([grouped_col])[target_col].mean().to_dict(OrderedDict)

    counter: int = 0



    bp = sns.boxplot(x=grouped_col, y=target_col, data=data, palette=palette, order=list(means.keys()))

    bp.set(xlabel='', ylabel=f"{target_col.title()}")

    ax = bp.axes



    for k, v in means.items():

        # every 4th line at the interval of 6 is median line

        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value

        mean = round(v, 2)

        ax.text(

            counter,

            mean,

            f'{mean}',

            ha='center',

            va='center',

            fontweight='bold',

            size=10,

            color='white',

            bbox=dict(facecolor='#445A64'))

        counter += 1

    bp.figure.tight_layout()

    plt.grid()

    plt.show()
target: str = "popularity"  

# Extract histograms

plot_multiple_histograms(data=twitter_accounts_df, 

                         grouped_col=grouped,

                         data_labels=data_labels,

                         target_col=target)

# Extract Box-plots

plot_multiple_boxplots(data=twitter_accounts_df,

                       grouped_col=grouped,

                       target_col=target,

                       palette=palette)

target: str = "average_tweets_per_day"  

# Extract histograms

plot_multiple_histograms(data=twitter_accounts_df, 

                         grouped_col=grouped,

                         data_labels=data_labels,

                         target_col=target)

# Extract Box-plots

plot_multiple_boxplots(data=twitter_accounts_df,

                       grouped_col=grouped,

                       target_col=target,

                       palette=palette)
target_col: str = "verified"

twitter_accounts_df2 = twitter_accounts_df.groupby([grouped, target_col])[grouped].count().unstack(target_col)

twitter_accounts_df2.plot(kind='bar', stacked=True)
target: str = "statuses_count"  

# Extract histograms

plot_multiple_histograms(data=twitter_accounts_df, 

                         grouped_col=grouped,

                         data_labels=data_labels,

                         target_col=target)

# Extract Box-plots

plot_multiple_boxplots(data=twitter_accounts_df,

                       grouped_col=grouped,

                       target_col=target,

                       palette=palette)
# Preprocess Response variable (account type)

twitter_accounts_df[grouped] = twitter_accounts_df[grouped].astype('category')

twitter_accounts_df.dtypes
twitter_accounts_df[grouped] = twitter_accounts_df[grouped].cat.codes

twitter_accounts_df.head()

twitter_accounts_df_num: pd.DataFrame = twitter_accounts_df.copy()

twitter_accounts_df_num: pd.DataFrame = twitter_accounts_df_num._get_numeric_data()

twitter_accounts_df_num.head()

# Remove columns

drop_cols: list = ["id"]

twitter_accounts_df_num.drop(drop_cols, axis=1,inplace=True)

twitter_accounts_df_num.head()
# Compute correlation among the features and the response variable

corr: pd.DataFrame = twitter_accounts_df_num.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot = True, fmt='.1g', cmap= 'coolwarm')