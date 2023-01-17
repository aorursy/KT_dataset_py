# Here is all the imports you need to run this kernel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
import seaborn as sns
import scipy
import itertools
import os
from datetime import date

pd.options.mode.chained_assignment = None 

% matplotlib inline

# For the sake of organization, let's start by defining all the plot functions.

def bar_plot(Xaxis,
             Yaxis,
             df_,
             title,
             figsize=(9, 9),
             decimals=1,
             color=None,
             palette=None):
    """
    Plot a barplot with values on the top of each bar.

    palette reference:
    https://matplotlib.org/examples/color/colormaps_reference.html

    :param Xaxis: column used to be the x axis
    :type Xaxis: str
    :param Yaxis: column used to be the y axis
    :type Xaxis: str
    :param df_: data frame
    :type df_: pd.DataFrame
    :param title: plot's title
    :type title: str
    :param path: path to save plot
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    :param color: color for all of the elements
    :type color: srt
    :param palette: matplotlib color palette
    :type palette: srt
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.barplot(x=Xaxis, y=Yaxis, data=df_, color=color, palette=palette)  # noqa
    ax.set_xlabel(Xaxis, fontsize=20)
    ax.set_ylabel(Yaxis, fontsize=20)
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=decimals),
                    (p.get_x() + p.get_width() / 2.,
                     p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontweight='bold',
                    color='black')
    fig.suptitle(title, fontsize=18, fontweight='bold')


def series_plot(place_list,
                place2series,
                time_series,
                title,
                xlabel,
                ylabel,
                path,
                figsize=(15, 12)):
    """
    Plot a time series

    :param place_list: list of places for reference (countries, continents)
    :type place_list: [str]
    :param place2series: dict mapping places to time series
    :type place2series: {str:[tuple]}
    :param time_series: list of dates
    :type time_series: [datetime.date]
    :param title: plot's title
    :type title: str
    :param xlabel: x axis label
    :type xlabel: str
    :param ylabel: y axis label
    :type ylabel: str
    :param path: path to save plot
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    """
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(linewidth=0.4)
    for place in place_list:
        series = place2series[place]
        place_y = [cdr for car, cdr in series]
        line, = ax.plot_date(x=time_series, y=place_y, ls='-', label=place)
        lines.append(line)
    plt.legend(handles=lines, loc=2, fontsize=18)
    fig.suptitle(title, fontsize=24, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.savefig(path)


def plot_stacked_bar(df_,
                     place_list,
                     base_column,
                     target_column,
                     palette,
                     value_list,
                     title,
                     figsize=(12, 6),
                     ylabel='number of responses'):
    """
    Plot a barplot stacking different bars.
    The bars are defined by a list of values from the target_column
    
    :param df_: data frame
    :type df_: pd.DataFrame
    :param place_list: list of places for reference (countries, continents)
    :type place_list: [str]
    :param base_column: df_ column used to define x axis
    :type base_column: str
    :param target_column: df_ column used to define the different bars
    :type target_column: str
    :param palette: matplotlib color palette
    :type palette: srt
    :param value_list: list of the different values appearing in target_column
    :type value_list: [str]
    :param title: plot's title
    :type title: str
    :param figsize: plot's size
    :type figsize: tuple
    :param ylabel: y axis label
    :type ylabel: str

    :return: target DataFrame
    :rtype: pd.DataFrame
    """
    all_entries = []
    for place in place_list:
        target_dict = Counter(df_[df_[base_column] == place][target_column])
        entry = [place] + [target_dict[value] for value in value_list]
        all_entries.append(entry)
    target_columns = [base_column] + value_list
    df_target = pd.DataFrame(columns=target_columns,
                             data=all_entries)
    values_number = len(value_list)
    colormap = ListedColormap(sns.color_palette(palette, values_number))
    ax = df_target.set_index(base_column)\
        .reindex(df_target.set_index(base_column).sum().sort_values().index, axis=1)\
        .plot(kind='bar',
              rot=45,
              stacked=True,
              colormap=colormap,
              figsize=figsize)
    ax.set_xlabel(base_column, fontsize=20, x=0.45, y=2)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=18, fontweight='bold')
    return df_target


def plot_stacked_bar_simpl(df_,
                           base_column,
                           values_number,
                           palette,
                           title,
                           figsize=(12, 6),
                           ylabel='number of responses'):
    """
    Plot a barplot stacking different bars.
    The bars are defined by a list of values from the target_column
    uses a already prepared df_
    
    :param df_: data frame
    :type df_: pd.DataFrame
    :param base_column: df_ column used to define x axis
    :type base_column: str
    :param values_number: number of values
    :type values_number: int
    :param palette: matplotlib color palette
    :type palette: srt
    :param title: plot's title
    :type title: str
    :param figsize: plot's size
    :type figsize: tuple
    :param ylabel: y axis label
    :type ylabel: str
    """
    df_target = df_
    colormap = ListedColormap(sns.color_palette(palette, values_number))
    ax = df_target.set_index(base_column)\
        .reindex(df_target.set_index(base_column).sum().sort_values().index, axis=1)\
        .plot(kind='bar',
              rot=45,
              stacked=True,
              colormap=colormap,
              figsize=figsize)
    ax.set_xlabel(base_column, fontsize=20, x=0.45, y=2)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    
def plot_distances(names_,
                   distances_,
                   title,
                   cmap=plt.cm.Oranges,
                   figsize=(9, 9)):
    """
    Plot a matrix with KL-distances.
    
    cmap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    
    :param names_: row/collum names
    :type names_: [str]
    :param distances_: matrix with distances
    :type distances_: np.array
    :param title: image title
    :type title: str
    :param cmap: plt color map
    :type cmap: plt.cm
    :param figsize: plot's size
    :type figsize: tuple
    """
    plt.figure(figsize=figsize)
    plt.imshow(distances_, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(names_))
    plt.xticks(tick_marks, names_, rotation=45)
    plt.yticks(tick_marks, names_)
    thresh = distances_.max() / 2.
    for i, j in itertools.product(range(distances_.shape[0]), range(distances_.shape[1])):
        plt.text(j, i, format(distances_[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if distances_[i, j] > thresh else "black")

    plt.tight_layout()
# Let's see some examples of this csv and used it to organize the countries in continents. 
df_rank = pd.read_csv("../input/ranked-users-kaggle-data/ranked_users_kaggle.csv")
df_rank.head()
country2continent  = {k:v for k,v in zip(df_rank["Country"].values, df_rank["Continent"].values)}

South_America = [i for i in list(country2continent.keys()) if country2continent[i]=="South America"]
North_America = [i for i in list(country2continent.keys()) if country2continent[i]=="North America"]
Central_America = [i for i in list(country2continent.keys()) if country2continent[i]=="Central America"]
Africa = [i for i in list(country2continent.keys()) if country2continent[i]=="Africa"]
Europe = [i for i in list(country2continent.keys()) if country2continent[i]=="Europe"]
Asia =[i for i in list(country2continent.keys()) if country2continent[i]=="Asia"]
Oceania = [i for i in list(country2continent.keys()) if country2continent[i]=="Oceania"]


continent2country = {"South America": South_America,
                     "North America": North_America,
                     "Central America": Central_America,
                     "Africa": Africa,
                     "Europe": Europe,
                     "Asia": Asia,
                     "Oceania": Oceania}

all_countries = list(set(df_rank["Country"].values))
all_continents = list(set(df_rank["Continent"].values))

all_continents = [c for c in all_continents if c!='None']
def get_top_n(df_, column, new_name, n=10, column_2="Points", mode="count"):
    """
    Get top n places (as defined in column) from DataFrame df_
    by using the values from column_2 as reference.

    :param df_: data frame
    :type df_: pd.DataFrame
    :param column: df_ column used to groub values from column_2
    :type column: str
    :param new_name: name of the new aggregated value
    :type new_name: str
    :param n: number of top places
    :type n: int
    :param column_2: df_ column with reference values 
    :type column_2: str
    :param mode: mode to aggregate values
    :type mode: str
    :return: top n DataFrame
    :rtype: pd.DataFrame
    """
    if mode == "count":
        place_counts = df_.groupby(column)[column_2].count()
    elif mode == "mean":
        place_counts = df_.groupby(column)[column_2].mean()
    elif mode == "sum":
        place_counts = df_.groupby(column)[column_2].sum()
    else:
        return None
    place_counts = place_counts.sort_values(ascending=False)
    place2count = dict(place_counts)
    place2count = {k: v for k, v in place2count.items() if k != 'None'}
    place2count_keys = list(place2count.keys())
    place2count_values = list(place2count.values())
    df_place = pd.DataFrame({column: place2count_keys, new_name: place2count_values})
    top_n = df_place.head(n)
    return top_n


top10_countries = get_top_n(df_=df_rank,
                            column='Country',
                            new_name='Number of ranked users')

top10_continents = get_top_n(df_=df_rank,
                             column='Continent',
                             new_name='Number of ranked users')

bar_plot("Country",
         'Number of ranked users',
         df_=top10_countries,
         title="Plot 1: Top 10 countries by number of ranked users",
         figsize=(12.6, 9))


bar_plot("Continent",
         'Number of ranked users',
         df_=top10_continents,
         title="Plot 2: Ranked users by continent",
         figsize=(12.6, 9))
# Let's extract the time series from the DataFrame.

df_rank["Year"] = [int(i.split("/")[2]) for i in df_rank["RegisterDate"].values]

years = list(set(df_rank["Year"].values))
years.sort()
years_d = [date(y, 1, 1) for y in years]

def get_series(df_, column, place_list, time_series):
    """
    Get a dictionary of series from a DataFrame
    using the values of column as reference
    
    :param df_: data frame
    :type df_: pd.DataFrame
    :param column: df_ column for the kind of place that will be used
                   (countries, continents)
    :type column: str
    :param place_list: list of places for reference (countries, continents)
    :type place_list: [str]
    :param time_series: list of dates
    :type time_series: [datetime.date]
    :return: dict mapping places to time series
    :rtype: {str:[tuple]}
    """
    place2series = {}
    for place in place_list:
        df_time_place = df_[df_[column] == place]
        df_time_per_place = df_time_place.groupby('Year')
        df_time_per_place = df_time_per_place.count()
        df_time_per_place.reset_index(inplace=True)
        dict_ = {k:v for k,v in zip(df_time_per_place["Year"].values, df_time_per_place["Points"].values)}
        tuples = []
        total = 0
        for year in time_series:
            if year in dict_:
                current = dict_[year]
            else:
                current = 0
            total += current
            tuples.append((year, total))
        place2series[place] = tuples
    
    return place2series

continent2series = get_series(df_=df_rank,
                              column="Continent",
                              place_list=all_continents,
                              time_series=years)

country2series = get_series(df_=df_rank,
                            column='Country',
                            place_list=all_countries,
                            time_series=years)

# To help visualization let's change the order of the continents

all_continents = ["North America",
                  "Asia",
                  "Europe",
                  "Oceania",
                  "South America",
                  "Africa",
                  "Central America"]


series_plot(place_list=all_continents,
            place2series=continent2series,
            time_series=years_d,
            title="Plot 3: Ranked users in Kaggle from 2010 to 2018",
            xlabel="Year",
            ylabel='Number of ranked users',
            path="ranked_users_region.png")
# To help visualization let's use only the countries
# with most ranked users from Asia and Europe

Asia_ = ["China",
         "Russia",
         "India",
         "Vietnam",
         "Singapore",
         "Hong Kong",
         "Taiwan",
         "Japan",
         "Israel",
         "South Korea"]

Europe_ = ["France",
           "United Kingdom",
           "Germany",
           "Ukraine",
           "Netherlands",
           "Italy",
           "Spain",
           "Poland",
           "Belgium",
           "Belarus"]

continent2country["Asia"] = Asia_
continent2country["Europe"] = Europe_

for i, continent in enumerate(all_continents):
    i += 4
    if continent != "None":
        
        series_plot(place_list=continent2country[continent],
            place2series=country2series,
            time_series=years_d,
            title="Plot {}: Ranked users in Kaggle from 2010 to 2018 ({})".format(i, continent),
            xlabel="Year",
            ylabel='Number of ranked users',
            path="ranked_users_series_{}.png".format(continent))
df_rank["Points"] = list(map(lambda x: float(x), df_rank["Points"].values))

top_countries_points = get_top_n(df_=df_rank,
                                 column='Country',
                                 new_name="Points",
                                 n=10,
                                 column_2="Points",
                                 mode="sum")

top_continents_points = get_top_n(df_=df_rank,
                                 column='Continent',
                                 new_name="Points",
                                 n=10,
                                 column_2="Points",
                                 mode="sum")

bar_plot("Country",
         "Points",
         df_=top_countries_points,
         title="Plot 11: Top 10 countries by total points",
         figsize=(12.6, 9))

bar_plot("Continent",
         "Points",
         df_=top_continents_points,
         title="Plot 12: Total points by continent",
         figsize=(12.6, 9))
# Creating two new collumns with the score fuction

df_rank["CurrentRanking"] = list(map(lambda x: int(x), df_rank["CurrentRanking"].values))
lowest_rank = np.max(df_rank["CurrentRanking"].values)
f_score = lambda x : np.abs((np.log(x/lowest_rank)) / (np.log(1/lowest_rank)))
df_rank["CurrentRankingScore"] = list(map(f_score, df_rank["CurrentRanking"].values))


df_rank["HighestRanking"] = list(map(lambda x: int(x), df_rank["HighestRanking"].values))
lowest_rank = np.max(df_rank["HighestRanking"].values)
f_score = lambda x : np.abs((np.log(x/lowest_rank)) / (np.log(1/lowest_rank)))
df_rank["HighestRankingScore"] = list(map(f_score, df_rank["HighestRanking"].values))

top_continents_HR = get_top_n(df_=df_rank,
                              column='Continent',
                              new_name="Highest ranking score",
                              n=10,
                              column_2="HighestRankingScore",
                              mode="sum")

top_continents_CR = get_top_n(df_=df_rank,
                              column='Continent',
                              new_name="Current ranking score",
                              n=10,
                              column_2="CurrentRankingScore",
                              mode="sum")


bar_plot("Continent",
         "Highest ranking score",
         df_=top_continents_HR,
         title="Plot 13: Highest ranking score by continent",
         figsize=(12.6, 9))


bar_plot("Continent",
         "Current ranking score",
         df_=top_continents_CR,
         title="Plot 14: Current ranking score by continent",
         figsize=(12.6, 9))
# Before plotting anything, let's re-organize the data.
# To do so we need to standardize some countries names, and add new countries to our country2continent dict

country2continent["Kenya"] = "Africa"
country2continent['Tunisia'] = "Africa"
country2continent['Bangladesh'] = "Asia"

country2continent_f = lambda x: country2continent[x] if x in country2continent else x

South_America = [i for i in list(country2continent.keys()) if country2continent[i]=="South America"]
North_America = [i for i in list(country2continent.keys()) if country2continent[i]=="North America"]
Africa = [i for i in list(country2continent.keys()) if country2continent[i]=="Africa"]
Europe = [i for i in list(country2continent.keys()) if country2continent[i]=="Europe"]
Asia =[i for i in list(country2continent.keys()) if country2continent[i]=="Asia"]
Oceania = [i for i in list(country2continent.keys()) if country2continent[i]=="Oceania"]


continent2country = {"South America": South_America,
                     "North America": North_America,
                     "Africa": Africa,
                     "Europe": Europe,
                     "Asia": Asia,
                     "Oceania": Oceania}

df_mult = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv',low_memory=False)

selected_question = ["Q2",
                     "Q3",
                     "Q4",
                     "Q9",
                     "Q25",
                     "Q15_Part_1",
                     "Q15_Part_2",
                     "Q15_Part_3",
                     "Q15_Part_4",
                     "Q15_Part_5",
                     "Q15_Part_6",
                     "Q15_Part_7",
                     "Q19_Part_1",
                     "Q19_Part_2",
                     "Q19_Part_3",
                     "Q19_Part_4",
                     "Q19_Part_5",
                     "Q19_Part_6",
                     "Q19_Part_7",
                     "Q19_Part_8",
                     "Q19_Part_9",
                     "Q19_Part_10",
                     "Q19_Part_11",
                     "Q19_Part_12",
                     "Q19_Part_13",
                     "Q19_Part_14",
                     "Q19_Part_15",
                     "Q19_Part_16",
                     "Q19_Part_17",
                     "Q19_Part_18",
                     "Q19_Part_19"] 

df_less = df_mult[selected_question]

# Let's simplify the answers to help visualization.

simple_questions = ["Q2",
                     "Q3",
                     "Q4",
                     "Q9",
                     "Q25"]

multiple_questions = ["Q15_Part_1",
                     "Q15_Part_2",
                     "Q15_Part_3",
                     "Q15_Part_4",
                     "Q15_Part_5",
                     "Q15_Part_6",
                     "Q15_Part_7",
                     "Q19_Part_1",
                     "Q19_Part_2",
                     "Q19_Part_3",
                     "Q19_Part_4",
                     "Q19_Part_5",
                     "Q19_Part_6",
                     "Q19_Part_7",
                     "Q19_Part_8",
                     "Q19_Part_9",
                     "Q19_Part_10",
                     "Q19_Part_11",
                     "Q19_Part_12",
                     "Q19_Part_13",
                     "Q19_Part_14",
                     "Q19_Part_15",
                     "Q19_Part_16",
                     "Q19_Part_17",
                     "Q19_Part_18",
                     "Q19_Part_19"] 

nan_dict_simple = {q: "Other" for q in simple_questions}
nan_dict_multiple = {q: " " for q in multiple_questions}

nan_dict = {**nan_dict_simple, **nan_dict_multiple}

df_less.fillna(nan_dict, inplace=True)

# Simplifying Q2

q2_simpl = {'25-29':"22-29",
            '22-24': "22-29",
            '30-34': "30-39",
            '18-21': '18-21',
            '35-39': "30-39",
            '40-44': "40-49",
            '45-49': "40-49",
            '50-54': "50-59",
            '55-59': "50-59",
            '60-69': "60+",
            '70-79': "60+",
            '80+': "60+"}


q2_simpl_f = lambda x: q2_simpl[x] if x in q2_simpl else x

df_less["Age"] = list(map(q2_simpl_f, df_less["Q2"]))

# Simplifying Q3 and adding a column for to indicate the user's continent

q3_simpl = {'United States of America': "United States",
            'Other': "None",
            'Iran, Islamic Republic of...': "Iran",
            'United Kingdom of Great Britain and Northern Ireland': "United Kingdom",
            'I do not wish to disclose my location': "None",
            'Hong Kong (S.A.R.)': "Hong Kong",
            'Viet Nam': "Vietnam",
            'Republic of Korea': "South Korea"}

q3_simpl_f = lambda x: q3_simpl[x] if x in q3_simpl else x

df_less["Country"] = list(map(q3_simpl_f, df_less["Q3"]))

df_less["Continent"] = list(map(country2continent_f, df_less["Country"]))

# Simplifying Q4

q4_simpl = {'I prefer not to answer': "Other"}
q4_simpl_f = lambda x: q4_simpl[x] if x in q4_simpl else x

df_less["FormalEducation"] = list(map(q4_simpl_f, df_less["Q4"]))

# Simplifying Q9

q9_simpl = {'200-250,000': '200,000+',
            '250-300,000': '200,000+',
            '300-400,000': '200,000+',
            '400-500,000': '200,000+',
            '500,000+': '200,000+',
            'I do not wish to disclose my approximate yearly compensation': "Other"}
q9_simpl_f = lambda x: q9_simpl[x] if x in q9_simpl else x

df_less["CurrentYearlyCompensation$USD"] = list(map(q9_simpl_f, df_less["Q9"]))

# Simplifying Q15 and adding a column to each cloud computing service

df_less["CloudComputing"] =  df_less["Q15_Part_1"] + df_less["Q15_Part_2"] + df_less["Q15_Part_3"] + df_less["Q15_Part_4"] + df_less["Q15_Part_5"] + df_less["Q15_Part_6"] + df_less["Q15_Part_7"] 

f_strip = lambda x: x.strip()

df_less["CloudComputing"] = list(map(f_strip, df_less["CloudComputing"].values))

def cloud_f(service): return lambda x: int(x.find(service) != -1)

all_services = ["Azure",
                "GCP",
                "IBM",
                "AWS",
                "Alibaba"]

all_f_cloud = list(map(cloud_f, all_services))

def apply(y): return lambda f: f(y)

def sum_f(x): return str(np.sum(list(map(apply(x), all_f_cloud))))

for service in all_services:

    df_less[service] = list(map(cloud_f(service), df_less["CloudComputing"].values))
    
df_less["CloudComputing_num"] = list(map(sum_f, df_less["CloudComputing"].values))

# Simplifying Q19 and adding a column to each machine learning framework

df_less["Framework"] = df_less["Q19_Part_1"] + df_less["Q19_Part_2"] + df_less["Q19_Part_3"] + df_less["Q19_Part_4"] + df_less["Q19_Part_5"] + df_less["Q19_Part_6"] + df_less["Q19_Part_7"] + df_less["Q19_Part_8"] + df_less["Q19_Part_9"] + \
    df_less["Q19_Part_10"] + df_less["Q19_Part_11"] + df_less["Q19_Part_12"] + df_less["Q19_Part_13"] + df_less["Q19_Part_14"] + \
    df_less["Q19_Part_15"] + df_less["Q19_Part_16"] + \
    df_less["Q19_Part_17"] + df_less["Q19_Part_18"] + df_less["Q19_Part_19"]

def f_strip(x): return x.strip()

df_less["Framework"] = list(map(f_strip, df_less["Framework"].values))

all_frameworks = ["Scikit-Learn",
                  "TensorFlow",
                  "Keras",
                  "PyTorch",
                  "Spark MLlib",
                  "H20",
                  "Fastai",
                  "Mxnet",
                  'Caret',
                  "Xgboost",
                  "mlr",
                  "Prophet",
                  "randomForest",
                  "lightgbm",
                  "catboost",
                  "CNTK",
                  "Caffe"]


def framework_f(framework): return lambda x: int(x.find(framework) != -1)

all_f = list(map(framework_f, all_frameworks))

def apply(y): return lambda f: f(y)

def sum_f(x): return str(np.sum(list(map(apply(x), all_f))))

for frame in all_frameworks:
    df_less[frame] = list(map(framework_f(frame), df_less["Framework"].values))

df_less["Framework_num"] = list(map(sum_f, df_less["Framework"].values))

q19_simpl = {'10': '10+',
             '11': '10+',
             '12': '10+',
             '13': '10+',
             '14': '10+',
             '15': '10+',
             '16': '10+',
             '17': '10+'}

q19_simpl_f = lambda x: q19_simpl[x] if x in q19_simpl else x

df_less["Framework_num"] = list(map(q19_simpl_f, df_less["Framework_num"]))

# Simplifying Q25

q25_simpl = {'I have never studied machine learning and I do not plan to': "Other",
             'I have never studied machine learning but plan to learn in the future': 'I have never studied, plan to learn in the future'}
q25_simpl_f = lambda x: q25_simpl[x] if x in q25_simpl else x

df_less["YearsUsingML"] = list(map(q25_simpl_f, df_less["Q25"]))

new_col = ["Country",
           "Continent",
           "Age",
           "FormalEducation",
           "YearsUsingML",
           "CurrentYearlyCompensation$USD",
           "CloudComputing_num",
           "Framework_num"]

new_col += all_frameworks + all_services

df = df_less[new_col]

df.drop(df.index[0], inplace=True)

top10_countries = get_top_n(df_=df,
                            column='Country',
                            column_2='YearsUsingML',
                            new_name='Number of paticipants')

top10_continents = get_top_n(df_=df,
                             column='Continent',
                             column_2='YearsUsingML',
                             new_name='Number of paticipants')

selected_continents = list(top10_continents.Continent.values)
selected_countries = list(top10_countries.Country.values)
group1 = ["Asia", "North America", "Europe"]
group2 = ["South America", "Africa", "Oceania"]

bar_plot("Country",
         'Number of paticipants',
         df_=top10_countries,
         title="Plot 15: Top 10 countries by number of paticipants in the ML & DS Survey",
         figsize=(12.6, 9))

bar_plot("Continent",
         'Number of paticipants',
         df_=top10_continents,
         title="Plot 16: Paticipants in the ML & DS Survey by continent",
         figsize=(12.6, 9))
# The functions to calculate the KL distance matrix

def smooth(array_, epsilon=0.003):
    """
    function smooth a distribution.
    
    :param array_: input array
    :type array_: np.array
    :param epsilon: smoothing value
    :type epsilon: float
    
    :return: output array
    :rtype: np.array
    """
    id_zeros = []
    for i,v in enumerate(array_):
        if v == 0:
            id_zeros.append(i)
    p_max = np.argmax(array_)
    minus_value = (len(id_zeros) * - epsilon)
    array_[p_max] += minus_value
    for i in id_zeros:
        array_[i] += epsilon
    return array_

def KLdistance(p,q):
    """
    Computes KL distance 
        
    :param p: left distribution
    :type p: np.array
    :param q: right distribution
    :type q: np.array

    :return: distance
    :rtype: float
    """
    if np.min(p)==0:
        p = smooth(p)
    if np.min(q)==0:
        q = smooth(q)
    return scipy.stats.entropy(p, q) + scipy.stats.entropy(q, p) 

def compare_distr(df_):
    """
    Function to compare the distribution for each place
    (the first column of the DataFrame) in df_
    
    :param df_: data frame
    :type df_: pd.DataFrame

    :return: first column values, distance matrix
    :rtype: [str], np.array
    """
    distr_ = []
    names_ = []
    n_lines = df_.shape[0] 
    for i in range(n_lines):
        name = list(df_.iloc[i])[0]
        names_.append(name)
        array_ = list(df_.iloc[i])[1:]
        array_ = np.array(array_) * (1 / np.sum(array_))
        distr_.append(array_)
    n = len(distr_)
    all_distances = []
    for i in range(n):
        distance = []
        for j in range(n):
            distance.append(KLdistance(distr_[i],distr_[j]))
        all_distances.append(distance)
    all_distances = np.array(all_distances)    
    return names_, all_distances
age_value =['18-21',
            '22-29',
            '30-39',
            '40-49',
            '50-59',
            '60+']

df_age = plot_stacked_bar(df_=df,
                          place_list=selected_continents,
                          base_column="Continent",
                          target_column="Age",
                          palette="Set2",
                          value_list=age_value,
                          title="Plot 17: Age (all continents)",
                          figsize=(12, 6),
                          ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group1,
                     base_column="Continent",
                     target_column="Age",
                     palette="Set2",
                     value_list=age_value,
                     title="Plot 18: Age (Group 1)",
                     figsize=(12, 6),
                     ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group2,
                     base_column="Continent",
                     target_column="Age",
                     palette="Set2",
                     value_list=age_value,
                     title="Plot 19: Age (Group 2)",
                     figsize=(12, 6),
                     ylabel='Number of responses')

names_age, distances_age = compare_distr(df_age)

plot_distances(names_=names_age,
               distances_=distances_age,
               title="Plot 20: Distances (Age)")
formal_education_values = ['Bachelor’s degree',
                           'Master’s degree',
                           'Doctoral degree',
                           'Some college/university study without earning a bachelor’s degree',
                           'Professional degree',
                           'No formal education past high school']


df_formal_education = plot_stacked_bar(df_=df,
                                       place_list=selected_continents,
                                       base_column="Continent",
                                       target_column="FormalEducation",
                                       palette="tab20c",
                                       value_list=formal_education_values,
                                       title="Plot 21: Formal education (all continents)",
                                       figsize=(12, 6),
                                       ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group1,
                     base_column="Continent",
                     target_column="FormalEducation",
                     palette="tab20c",
                     value_list=formal_education_values,
                     title="Plot 22: Formal education (Group 1)",
                     figsize=(12, 6),
                     ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group2,
                     base_column="Continent",
                     target_column="FormalEducation",
                     palette="tab20c",
                     value_list=formal_education_values,
                     title="Plot 23: Formal education (Group 2)",
                     figsize=(12, 6),
                     ylabel='Number of responses')

names_formal_education, distances_formal_education = compare_distr(df_formal_education)

plot_distances(names_=names_formal_education,
               distances_=distances_formal_education,
               title="Plot 24: Distances\n(Formal education)")
yearsML_values = ['< 1 year',
                  '1-2 years',
                  '2-3 years',
                  '3-4 years',
                  '4-5 years',
                  '5-10 years',
                  '10-15 years',
                  '20+ years',
                  'I have never studied, plan to learn in the future']

df_yearsML = plot_stacked_bar(df_=df,
                              place_list=selected_continents,
                              base_column="Continent",
                              target_column="YearsUsingML",
                              palette="tab20",
                              value_list=yearsML_values,
                              title="Plot 25: Years using machine learning (all continents)",
                              figsize=(12, 6),
                              ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group1,
                     base_column="Continent",
                     target_column="YearsUsingML",
                     palette="tab20",
                     value_list=yearsML_values,
                     title="Plot 26: Years using machine learning (Group 1)",
                     figsize=(12, 7),
                     ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group2,
                     base_column="Continent",
                     target_column="YearsUsingML",
                     palette="tab20",
                     value_list=yearsML_values,
                     title="Plot 27: Years using machine learning (Group 2)",
                     figsize=(12, 7),
                     ylabel='Number of responses')

names_yearsML, distances_yearsML = compare_distr(df_yearsML)

plot_distances(names_=names_yearsML,
               distances_=distances_yearsML,
               title="Plot 28: Distances\n(Years in ML)")
comp_values = ['0-10,000',
               '10-20,000',
               '20-30,000',
               '30-40,000',
               '40-50,000',
               '50-60,000',
               '60-70,000',
               '70-80,000',
               '80-90,000',
               '90-100,000',
               '100-125,000',
               '125-150,000',
               '150-200,000',
               '200,000+']

df_compensation = plot_stacked_bar(df_=df,
                                   place_list=selected_continents,
                                   base_column="Continent",
                                   target_column="CurrentYearlyCompensation$USD",
                                   palette="tab20b",
                                   value_list=comp_values,
                                   title="Plot 29: Yearly compensation in $USD (all continents)",
                                   figsize=(12, 6),
                                   ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group1,
                     base_column="Continent",
                     target_column="CurrentYearlyCompensation$USD",
                     palette="tab20b",
                     value_list=comp_values,
                     title="Plot 30: Yearly compensation in $USD (Group 1)",
                     figsize=(14, 12),
                     ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=group2,
                     base_column="Continent",
                     target_column="CurrentYearlyCompensation$USD",
                     palette="tab20b",
                     value_list=comp_values,
                     title="Plot 31: Yearly compensation in $USD (Group 2)",
                     figsize=(12, 8),
                     ylabel='Number of responses')

_ = plot_stacked_bar(df_=df,
                     place_list=selected_countries,
                     base_column="Country",
                     target_column="CurrentYearlyCompensation$USD",
                     palette="tab20b",
                     value_list=comp_values,
                     title="Plot 32: Yearly compensation in $USD (selected countries)",
                     figsize=(12, 8),
                     ylabel='Number of responses')

names_compensation, distances_compensation = compare_distr(df_compensation)

plot_distances(names_=names_compensation,
               distances_=distances_compensation,
               title="Plot 33: Distances\n(Yearly compensation)")
cloud_values = ['0',
                '1',
                '2',
                '3',
                '4',
                '5',
                "6"]


Azure = df.groupby("Continent")["Azure"].sum().to_frame().reset_index()
IBM = df.groupby("Continent")["IBM"].sum().to_frame().reset_index()
AWS = df.groupby("Continent")["AWS"].sum().to_frame().reset_index()
GCP = df.groupby("Continent")["GCP"].sum().to_frame().reset_index()
Alibaba = df.groupby("Continent")["Alibaba"].sum().to_frame().reset_index()


df_cloud = pd.merge(Azure, IBM, on=['Continent'])
df_cloud = pd.merge(df_cloud, AWS, on=['Continent'])
df_cloud = pd.merge(df_cloud, GCP, on=['Continent'])
df_cloud = pd.merge(df_cloud, Alibaba, on=['Continent'])

df_cloud["Total"] = df_cloud["Azure"] + df_cloud["IBM"] + df_cloud["AWS"] + df_cloud["GCP"] + df_cloud["Alibaba"]
df_cloud.sort_values(by="Total", inplace=True, ascending=False)
df_cloud.drop("Total", axis=1, inplace=True)
df_cloud.drop(axis=0, index=3, inplace=True)
df_cloud.reset_index(drop=True, inplace=True)


df_cloud_n = plot_stacked_bar(df_=df,
                              place_list=selected_continents,
                              base_column="Continent",
                              target_column="CloudComputing_num",
                              palette="tab20",
                              value_list=cloud_values,
                              title="Plot 34: Number of used cloud computing plataforms (all continents)",
                              figsize=(12, 6),
                              ylabel='Number of responses')


names_cloud_n, distances_cloud_n = compare_distr(df_cloud_n)

plot_distances(names_=names_cloud_n,
               distances_=distances_cloud_n,
               title="Plot 35: Distances\n(Number of cloud services)")


plot_stacked_bar_simpl(df_=df_cloud,
                       base_column="Continent",
                       values_number=5,
                       palette="tab20c",
                       title="Plot 36: Cloud computing plataforms by continent (all continents)",
                       figsize=(12, 6),
                       ylabel='number of responses')

names_cloud, distances_cloud = compare_distr(df_cloud)

plot_distances(names_=names_cloud,
               distances_=distances_cloud,
               title="Plot 37: Distances\n(Type of cloud services)")
framework_values = ['0',
                    '1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7',
                    '8',
                    '9',
                    '10+']


ScikitLearn = df.groupby("Continent")["Scikit-Learn"].sum().to_frame().reset_index()
TensorFlow = df.groupby("Continent")["TensorFlow"].sum().to_frame().reset_index()
Keras = df.groupby("Continent")["Keras"].sum().to_frame().reset_index()
PyTorch = df.groupby("Continent")["PyTorch"].sum().to_frame().reset_index()
SparkMLlib = df.groupby("Continent")["Spark MLlib"].sum().to_frame().reset_index()

H20 = df.groupby("Continent")["H20"].sum().to_frame().reset_index()
Fastai = df.groupby("Continent")["Fastai"].sum().to_frame().reset_index()
Mxnet = df.groupby("Continent")["Mxnet"].sum().to_frame().reset_index()
Caret = df.groupby("Continent")["Caret"].sum().to_frame().reset_index()

Xgboost = df.groupby("Continent")["Xgboost"].sum().to_frame().reset_index()
mlr = df.groupby("Continent")["mlr"].sum().to_frame().reset_index()
Prophet = df.groupby("Continent")["Prophet"].sum().to_frame().reset_index()
randomForest = df.groupby("Continent")["randomForest"].sum().to_frame().reset_index()

lightgbm = df.groupby("Continent")["lightgbm"].sum().to_frame().reset_index()
catboost = df.groupby("Continent")["catboost"].sum().to_frame().reset_index()
CNTK = df.groupby("Continent")["CNTK"].sum().to_frame().reset_index()
Caffe = df.groupby("Continent")["Caffe"].sum().to_frame().reset_index()

df_frame = pd.merge(ScikitLearn, TensorFlow, on=['Continent'])
df_frame = pd.merge(df_frame, Keras, on=['Continent'])
df_frame = pd.merge(df_frame, PyTorch, on=['Continent'])
df_frame = pd.merge(df_frame, SparkMLlib, on=['Continent'])
df_frame = pd.merge(df_frame, H20, on=['Continent'])
df_frame = pd.merge(df_frame, Fastai, on=['Continent'])

df_frame = pd.merge(df_frame, Mxnet, on=['Continent'])
df_frame = pd.merge(df_frame, Caret, on=['Continent'])
df_frame = pd.merge(df_frame, Xgboost, on=['Continent'])
df_frame = pd.merge(df_frame, mlr, on=['Continent'])
df_frame = pd.merge(df_frame, Prophet, on=['Continent'])
df_frame = pd.merge(df_frame, randomForest, on=['Continent'])
df_frame = pd.merge(df_frame, lightgbm, on=['Continent'])
df_frame = pd.merge(df_frame, catboost, on=['Continent'])
df_frame = pd.merge(df_frame, CNTK, on=['Continent'])
df_frame = pd.merge(df_frame, Caffe, on=['Continent'])


df_frame["Total"] = df_frame["Scikit-Learn"] + df_frame["TensorFlow"] + df_frame["Keras"] + df_frame["Spark MLlib"] + df_frame["PyTorch"] + df_frame["H20"] + df_frame["Fastai"] + df_frame["Mxnet"] + \
    df_frame['Caret'] + df_frame["Xgboost"] + df_frame["mlr"] + df_frame["Prophet"] + df_frame["randomForest"] + \
    df_frame["lightgbm"] + df_frame["catboost"] + \
    df_frame["CNTK"] + df_frame["Caffe"]


df_frame.sort_values(by="Total", inplace=True, ascending=False)
df_frame.drop("Total", axis=1, inplace=True)
df_frame.drop(axis=0, index=3, inplace=True)
df_frame.reset_index(drop=True, inplace=True)

df_framework = plot_stacked_bar(df_=df,
                                place_list=selected_continents,
                                base_column="Continent",
                                target_column="Framework_num",
                                palette="tab20b",
                                value_list=framework_values,
                                title="Plot 38: Number of frameworks used (all continents)",
                                figsize=(12, 6),
                                ylabel='Number of responses')

names_framework, distances_framework = compare_distr(df_framework)

plot_distances(names_=names_framework,
               distances_=distances_framework,
               title="Plot 39: Distances\n(Number of frameworks)")

plot_stacked_bar_simpl(df_=df_frame,
                       base_column="Continent",
                       values_number=17,
                       palette="tab20b",
                       title="Plot 40: Frameworks by continent (all continents)",
                       figsize=(15, 7),
                       ylabel='number of responses')

names_frame, distances_frame = compare_distr(df_frame)

plot_distances(names_=names_frame,
               distances_=distances_frame,
               title="Plot 41: Distances\n(Types of frameworks)")
# Let's sum the KL divergence for each type of group mentioned above

def get_distance_groups(input_names, input_matrix):
    """
    Get the sum of the KL divergence for 4 types of groups:

    Group 1 = [Asia, North America, Europe]
    Group 2 = [South America, Africa, Oceania]
    Group 3 = [North America, Europe, Oceania]
    Group 4 = [Africa, Asia, South America]

    :param input_names: list of continents
    :type input_names: [str]
    :param input_matrix: matrix with KL divergences
    :type input_matrix: np.array
    :return: matrix with KL divergences for groups 1, 2, 3 and 4
    :rtype: np.array, np.array, np.array, np.array
    """
    together = list(zip(input_names, input_matrix))
    together.sort()

    g1 = ["Asia", "Europe", 'North America']
    g1_i_sorted = [1, 2, 3]
    g2 = ["Africa", 'Oceania', 'South America']
    g2_i_sorted = [0, 4, 5]
    g3 = ["Europe", 'North America', 'Oceania']
    g3_i_sorted = [2, 3, 4]
    g4 = ["Africa", 'Asia', 'South America']
    g4_i_sorted = [0, 1, 5]

    g1_i = [i for i, v in enumerate(input_names) if v in g1]
    g2_i = [i for i, v in enumerate(input_names) if v in g2]
    g3_i = [i for i, v in enumerate(input_names) if v in g3]
    g4_i = [i for i, v in enumerate(input_names) if v in g4]

    g1_d = [together[i][1][g1_i] for i in g1_i_sorted]
    g2_d = [together[i][1][g2_i] for i in g2_i_sorted]
    g3_d = [together[i][1][g3_i] for i in g3_i_sorted]
    g4_d = [together[i][1][g4_i] for i in g4_i_sorted]

    return np.array(g1_d), np.array(g2_d), np.array(g3_d), np.array(g4_d)


all_distances = [(names_frame, distances_frame),
                 (names_framework, distances_framework),
                 (names_cloud, distances_cloud),
                 (names_cloud_n, distances_cloud_n),
                 (names_compensation, distances_compensation),
                 (names_yearsML, distances_yearsML),
                 (names_formal_education, distances_formal_education),
                 (names_age, distances_age)]


g1_d = np.zeros((3, 3))
g2_d = np.zeros((3, 3))
g3_d = np.zeros((3, 3))
g4_d = np.zeros((3, 3))

for names, distances in all_distances:
    g1_d_, g2_d_, g3_d_, g4_d_ = get_distance_groups(names, distances)
    g1_d += g1_d_
    g2_d += g2_d_
    g3_d += g3_d_
    g4_d += g4_d_


print("Group 1 = {:.2f}".format(np.sum(g1_d)))
print("Group 2 = {:.2f}".format(np.sum(g2_d)))
print("Group 3 = {:.2f}".format(np.sum(g3_d)))
print("Group 4 = {:.2f}".format(np.sum(g4_d)))