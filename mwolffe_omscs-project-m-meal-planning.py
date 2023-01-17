# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting library

from matplotlib import rcParams # formatting params

from scipy.stats import ttest_ind



# fix formatter to auto-layout

rcParams.update({'figure.autolayout': True})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Helper functions for plotting and creating frequency maps. 

def plot_bar_chart_for_categorical_data_frequency(categorical_results_column, title):

    category_to_count_list = list(construct_occurance_count_map(categorical_results_column).items())

    category_to_count_list.sort(key = lambda x: x[0])

    categories = [x[0] for x in category_to_count_list]

    counts = [x[1] for x in category_to_count_list]

    sequence = np.arange(len(categories))

    plt.bar(sequence, counts, align="center", alpha=0.5)

    plt.xticks(sequence, categories)

    plt.ylabel('Count')

    plt.title(title)

    plt.show()



def construct_occurance_count_map(categorical_results_column):

    category_count = {key:0 for key in set(categorical_results_column)}

    for instance in categorical_results_column:

        category_count[instance] += 1

    return category_count



def plot_and_describe_series(series, title):

    plot_bar_chart_for_categorical_data_frequency(series, title)

    print(series.describe())



def is_number(string):

    try:

        return float(string) != None

    except ValueError:

        return False
def describe_numeric_series(series):

    series = series[series.apply(lambda x: is_number(x))]

    print(series.apply(lambda x: float(x)).describe())
# meal_planner_results = pd.read_csv("../input/meal_planners.csv")

meal_planner_results_filtered = pd.read_csv("../input/omscsprojectmmealplanners/meal_planners_filtered.csv")



not_meal_planner_results_filtered = pd.read_csv("../input/project-m-not-meal-planners/not_meal_planners_filtered.csv")
meal_planner_results_filtered.keys()
# User evaulation section



# Age

plot_and_describe_series(meal_planner_results_filtered["Select your age:"], "What is your age?")



# What devices do they have?

describe_numeric_series(meal_planner_results_filtered["has iphone"])

describe_numeric_series(meal_planner_results_filtered["has ipad"])

describe_numeric_series(meal_planner_results_filtered["has android"])

describe_numeric_series(meal_planner_results_filtered["has android tablet"])
### Goal Section



plot_bar_chart_for_categorical_data_frequency(meal_planner_results_filtered["Coded Goal"], "What is your primary goal when meal planning?")



# Cost 

plot_and_describe_series(meal_planner_results_filtered["Rate your agreement with the following statement:  Cost is an important factor when deciding what I will cook for the week."], "The Importance of Cost")



# Variety

plot_and_describe_series(meal_planner_results_filtered["Rate your agreement with the following statement:  Trying out new recipes is an important factor when deciding what I will cook for the week."], "The Importance of Variety")



# Diet

plot_and_describe_series(meal_planner_results_filtered["Rate your agreement with the following statement:  Maintaining dietary restrictions is an important factor when deciding what I will cook for the week."], "The Importance of Dietary Restrictions")



# Health 

plot_and_describe_series(meal_planner_results_filtered["Rate your agreement with the following statement:  Healthiness is an important factor when deciding what I will cook for the week."], "The Importance of Health")



# Time

plot_and_describe_series(meal_planner_results_filtered["Rate your agreement with the following statement:  Time to prepare is an important factor when deciding what I will cook for the week."], "The Importance of Time")

# Task section



# How do people keep track of what they will cook each evening. 

plot_and_describe_series(meal_planner_results_filtered["Unnamed: 14"], "How do you currently keep track of the recipe you are cooking on a specific day?")



# How long does each task take you?

describe_numeric_series(meal_planner_results_filtered["About how long in minutes does it typically take you to select recipes for the week? "])

describe_numeric_series(meal_planner_results_filtered["About how long in minutes does it typically take you to create a shopping list?"])

describe_numeric_series(meal_planner_results_filtered["About how long in minutes does it typically take you to shop for the week?"])



# Do users have multiple recipe sources

describe_numeric_series(meal_planner_results_filtered["Coded - does the response have multiple sources of recipes"])
# Not Meal Planners Analysis

not_meal_planner_results_filtered.keys()
plot_and_describe_series(not_meal_planner_results_filtered["Coded - Why do you not meal plan?"], "Why do you not meal plan?")
plot_and_describe_series(not_meal_planner_results_filtered["Coded - Have you tried meal planning?"], "Have you tried meal planning in the past?")