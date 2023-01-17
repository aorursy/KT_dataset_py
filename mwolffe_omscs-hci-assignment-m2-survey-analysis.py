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
raw_survey_results = pd.read_csv("../input/wolffe_survey_results_coded.csv")

raw_survey_results # display survey results
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
male_survey_filter = raw_survey_results["Q2"] == "Male"

male_survey_results = raw_survey_results[male_survey_filter]

female_survey_filter = raw_survey_results["Q2"] == "Female"

female_survey_results = raw_survey_results[female_survey_filter]
plot_and_describe_series(raw_survey_results["Q1"], "Age")

plot_and_describe_series(male_survey_results["Q1"], "Male Age")

plot_and_describe_series(female_survey_results["Q1"], "Female Age")
plot_and_describe_series(raw_survey_results["Q2"], "Gender")
plot_and_describe_series(raw_survey_results["Q4"], "How many nights a week do you cook?")

plot_and_describe_series(male_survey_results["Q4"], "Male - How many nights a week do you cook?")

plot_and_describe_series(female_survey_results["Q4"], "Female - How many nights a week do you cook?")



# Conduct a t-test to determine whether it's statistically significant

ttest_ind(male_survey_results["Q4"], female_survey_results["Q4"])
plot_and_describe_series(raw_survey_results["Q5"], "How many people are you cooking for each week?")

plot_and_describe_series(male_survey_results["Q5"], "How many people are you cooking for each week?")

plot_and_describe_series(female_survey_results["Q5"], "How many people are you cooking for each week?")
plot_and_describe_series(raw_survey_results["Q7"], "What is the most important thing to you when planning your meals?")
plot_and_describe_series(raw_survey_results["Q6Coded"], "Does the respondent meal plan?")

plot_and_describe_series(male_survey_results["Q6Coded"].map({"T": "Yes", "F": "No"}), "Male - Does the respondent meal plan?")

plot_and_describe_series(female_survey_results["Q6Coded"].map({"T": "Yes", "F": "No"}), "Female - Does the respondent meal plan?")



# I coded this as T == yes, meal plans and F = no, does not meal plan, bad for getting t-values. Map to 1 for mealplans 0 for does not. 

print("Male Remapped")

male_meal_plan_remap_results = male_survey_results["Q6Coded"].map({"T": 1, "F": 0})

print(male_meal_plan_remap_results.describe())

print("Female Remapped")

female_meal_plan_remap_results = female_survey_results["Q6Coded"].map({"T": 1, "F": 0})

print(female_meal_plan_remap_results.describe())



# Conduct a t-test to determine whether it's statistically significant

ttest_ind(male_meal_plan_remap_results, female_meal_plan_remap_results)
plot_and_describe_series(raw_survey_results["Q12"], "How often do you meal plan?")

plot_and_describe_series(male_survey_results["Q12"], "Male - How often do you meal plan?")

plot_and_describe_series(female_survey_results["Q12"], "Female - How often do you meal plan?")
plot_and_describe_series(raw_survey_results["Q15"], "Have you ever used a meal planning app?")
plot_and_describe_series(raw_survey_results["Q7"], "What is most important to you when deciding what to cook each week?")