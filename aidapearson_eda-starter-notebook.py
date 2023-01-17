%matplotlib inline

import glob

import json

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
def create_data_frame(raw_data):

    """

    Create a Pandas DataFrame and a list for all the latex expressions



    Parameters

    ----------

    raw_data : list

        A list that contains all the image information



    Returns

    ----------

    df: DataFrame

        A Pandas DataFrame for running the analysis

    all_latex_lst: list

        A list for all the tokens, used for creating the token distribution

    """

    data = {}

    data['latex'] = []

    data['seq_len'] = []

    data['font'] = []

    data['image_ratio'] = []

    data['image_width'] = []

    data['image_height'] = []

    all_latex_lst = []

    for image in raw_data:

        data['latex'].append(image['image_data']['full_latex_chars'])

        data['seq_len'].append(len(image['image_data']['full_latex_chars']))

        data['font'].append(image['font'])

        data['image_ratio'].append(round(image['image_data']['width'] / image['image_data']['height'],1))

        data['image_width'].append(image['image_data']['width'])

        data['image_height'].append(image['image_data']['height'])

        all_latex_lst = all_latex_lst + image['image_data']['full_latex_chars']

    df = pd.DataFrame.from_dict(data)

    return df, all_latex_lst
# Load data into a Pandas DataFrame and store all the tokens into a list.

with open(file='/kaggle/input/ocr-data/batch_1/JSON/kaggle_data_1.json') as f:

    raw_data = json.load(f)

df, all_latex_lst = create_data_frame(raw_data)
df.columns
df.describe()
def plot_dist(df, field, bins, color, xlabel, ylabel, title):

    """

    Plot an univariate distribution of observations

    

    Parameters

    ----------

    df : DataFrame

        A Panadas Dataframe used for plotting the sequence length distribution and image ratio distribution

    field : String

        A string that represents what column you would like to use for the plots

    bins : Int

        Specification of hist bins

    color : String

        The Color for the plot

    xlabel: String

        The Name for the x-axis label

    ylabel: String

        The Name for the y-axis label

    title: String

        The Title for the plot

    """

    sns.set(color_codes=True)

    fig, ax = plt.subplots(figsize=(18,6))

    sns.distplot(df[field], bins=bins, color=color, ax=ax)

    ax.set_xlabel(xlabel, fontsize=13)

    ax.set_ylabel(ylabel, fontsize=13)

    ax.set_title(title, fontsize=20)

    plt.show()
plot_dist(df=df, field='seq_len', bins=50, color='b', xlabel='Sequence Length', ylabel='Frequency', title='Sequence Length Distribution (10k Images)')
plot_dist(df=df, field='image_ratio', bins=10, color='r', xlabel='Image Ratio (Image Width / Image Height)', ylabel='Frequency', title='Image Ratio Distribution (10k Images)')
g = sns.jointplot("image_width", "image_height", data=df, kind="kde", space=0, color="r")

g.set_axis_labels("Image Width", "Image Height")
def create_count_df(df, field, index):

    """

    Create a Group DataFrame by a Series of columns



    Parameters

    ----------

    df : DataFrame

        A Pandas DataFrame used for the groupby operation

    filed: String

        A string that used to determine the groups for the groupby

    Index: String

        A string that used to compute count of group



    Returns

    ----------

    count_df: DataFrame

        A Pandas DataFrame that represents the count for each group

    """

    count=df.groupby(field)[index].count().sort_values(ascending=False)

    count_df = count.to_frame().reset_index()

    count_df.columns = [field, field + '_count']

    return count_df



def plot_count_df(df, field, random_sample, color, rotation, xlabel, ylabel, title):

    """

    Create a bar plot for the font distribution and token distribution

    

    Parameters

    ----------

    df : DataFrame

        A Panadas Dataframe used for ploting the font distribution and token distribution

    field : String

        A string that represents what column you would like to use for the plot

    random_sample : Boolean

        A boolean to specify if we are going to take a random sample or not

    color : String

        The Color for the plot

    rotation: Int

        The rotation for the x-axis label

    xlabel: String

        The Name for the x-axis label

    ylabel: String

        The Name for the y-axis label

    title: String

        The Title for the plot

    """

    fig, ax = plt.subplots(figsize=(18,6))

    if random_sample:

        df = df.sample(n=50, random_state=1)

    ax.bar(df[field], df[field + '_count'], color=color, align='center',alpha=0.5)

    ax.set_xticklabels(df[field],rotation=rotation, fontsize=13)

    ax.set_xlabel(xlabel, fontsize=13)

    ax.set_ylabel(ylabel, fontsize=13)

    ax.set_title(title, fontsize=20)

    plt.show()
font_count_df = create_count_df(df=df, field='font', index='latex')

plot_count_df(df=font_count_df, field='font', random_sample=True, color='y', rotation=90, xlabel='Font Name', ylabel='Number of Images', title='Font Distribution (50 Random Fonts of 210 Total Fonts)')
token_df = pd.DataFrame(all_latex_lst, columns =['token'])

token_df['index']=token_df.index
token_count_df = create_count_df(df=token_df, field='token', index='index')

plot_count_df(df=token_count_df, field='token', random_sample=False, color='g', rotation=90, xlabel='Token', ylabel='Number of Tokens', title='Token Distribution (10k Images)')