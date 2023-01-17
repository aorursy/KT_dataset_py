# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
projects_2018 = pd.read_csv('../input/ks-projects-201801.csv')
projects_2018
from matplotlib import pyplot as plt
def plot_pie(data, show_plot=True, figsize=(11,12), chart_title=None):
    plt.figure(figsize=figsize)
    plt.title(chart_title)
    plt.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        shadow=False,
        startangle=140
    )
    if show_plot:
        plt.show()
    else:
        return plt
def plot_bar(title, xlabel, ylabel, x_list, y_list, figure_size: tuple, font_size, x_label_size: int, y_label_size: int, show_plot: bool=True, label_bars:bool=True):
    bar_chart = plt.figure(figsize=figure_size)
    bar_chart.bar(x_list, y_list)
    bar_chart.xticks(x_list, fontsize=x_label_size, rotation=30)
    bar_chart.xlabel(xlabel, fontsize=x_label_size)
    bar_chart.ylabel(ylabel, fontsize=y_label_size)
    bar_chart.title(title)
    if show_plot:
        bar_chart.show()
    return bar_chart
by_category = projects_2018.groupby('category').count()['main_category']
bar_plot = plot_bar(
    title='Categorical Distribution of data according to Secondary Categories',
    xlabel='Category',
    ylabel='No. of Kickstarter Projects',
    x_list=by_category.index,
    y_list=by_category.values,
    figure_size=(160, 10),
    font_size=12,
    x_label_size=12,
    y_label_size=12,
    show_plot=True,
    label_bars=True
)
by_major_category = projects_2018.groupby('main_category').count()['category']
plt = plot_bar(
    title='Categorical Distribution of data according to Main Categories',
     xlabel='Category',
     ylabel='No. of Kickstarter Projects',
     x_list=by_major_category.index,
     y_list=by_major_category.values,
     figure_size=(15, 10),
     font_size=12,
     x_label_size=12,
     y_label_size=12,
     show_plot=True,
     label_bars=True
)
# Pie Charts showing subcategory distributions for each main category.
# print(projects_2018['main_category'].unique())
pub_category_data = projects_2018[(projects_2018.main_category == 'Publishing')].groupby('category').count()['ID']
plot_pie(pub_category_data, figsize=(14, 13), chart_title='Subcategories under Publishing({})'.format(np.sum(pub_category_data.values)))
f_v_category_data = projects_2018[(projects_2018.main_category == 'Film & Video')].groupby('category').count()['ID']
plot_pie(f_v_category_data, figsize=(14, 13), chart_title='Subcategories under Film & Video({})'.format(np.sum(f_v_category_data.values)))
webseries = projects_2018[(projects_2018.category == 'Webseries')].groupby('state').count()['ID']
plot_pie(webseries, figsize=(14, 13), chart_title='Current States of Webseries Projects(' + str(np.sum(webseries.values)) + ')')
# Creates pie charts for the subcategory-wise distribution of data in all categories.
for category_name in projects_2018['main_category'].unique():
    category_data = projects_2018[(projects_2018.main_category == category_name)].groupby('category').count()['ID']
    plot_pie(category_data, figsize=(14, 13), chart_title='Subcategories under ' + category_name)
category_wise_state_data = pd.DataFrame()
count = 0
for category_name in projects_2018['main_category'].unique():
    category_data = projects_2018[(projects_2018.main_category == category_name)].groupby('state').count()['ID']
    category_wise_state_data[category_name] = category_data
category_wise_state_data.fillna(
    value=0,
    inplace=True
)
by_state = projects_2018.groupby('main_category').count()['ID']
plot_pie(by_state, chart_title='Categorical Distribution of Kickstarted projects')
successful_projects = projects_2018[(projects_2018.state == 'successful')].groupby('main_category').count()['ID']
plot_pie(successful_projects, chart_title='Category Wise Percentage Distribution of Successful Projects')
tech_projects = projects_2018[(projects_2018.main_category == 'Technology')]
tech_projects_by_category = tech_projects.groupby('category').count()['ID']
plot_pie(tech_projects_by_category)
failed_tech_projects = tech_projects[(tech_projects.state == 'failed')]
failed_tech_projects_by_category = failed_tech_projects.groupby('category').count()['ID']
plot_pie(failed_tech_projects_by_category)
successful_tech_projects = tech_projects[(tech_projects.state == 'successful')]
successful_tech_projects_by_category = successful_tech_projects.groupby('category').count()['ID']
plot_pie(successful_tech_projects_by_category, chart_title='Successful Tech Projects Distribution')
category_totals = np.sum(category_wise_state_data)
percentages = category_wise_state_data / category_totals
barWidth = 0.85
x_list = range(0, len(category_wise_state_data.columns))
plt.figure(figsize=(15,10))
green = percentages.iloc[0]
brown = percentages.iloc[1]
blue = percentages.iloc[2]
red = percentages.iloc[3]
gray = percentages.iloc[4]
orange = percentages.iloc[5]
green_bar = plt.bar(x_list, green, color='#b5ffb9', edgecolor='white', width=barWidth)
brown_bar = plt.bar(x_list, brown, bottom=green ,color='#8b4513', edgecolor='white', width=barWidth)
blue_bar = plt.bar(x_list, blue, bottom=[i + j for i, j in zip(green, brown)], color='#a3acff', edgecolor='white', width=barWidth)
red_bar = plt.bar(x_list, red, bottom=[i + j + k for i, j, k in zip(green, brown, blue)], color='#ff0000', edgecolor='white', width=barWidth)
gray_bar = plt.bar(x_list, gray, bottom=[i + j + k + l for i, j, k, l in zip(green, brown, blue, red)], color='#d3d3d3', edgecolor='white', width=barWidth)
orange_bar = plt.bar(x_list, orange, bottom=[i + j + k + l + m for i, j, k, l, m in zip(green, brown, blue, red, gray)], color='#f9bc86', edgecolor='white', width=barWidth)
plt.xticks(x_list, category_wise_state_data.columns, rotation=30)
# plt.yticks(range(0,1), category_wise_state_data.columns, rotation=30)
plt.legend([green_bar, brown_bar, blue_bar, red_bar, gray_bar, orange_bar], category_wise_state_data.index)
plt.title('Stacked Percentage Bar Chart Distribution of Projects according to Main Category\n Each bar shows the Percentage of Projects in Each State')
plt.show()