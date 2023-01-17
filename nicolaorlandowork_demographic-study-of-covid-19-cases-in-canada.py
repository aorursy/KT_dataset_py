# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import Counter

import math



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



from collections import Counter

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print("Load the dataset ...")

file_path="/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/public-covid-19-cases-canada.csv"

df_input=pd.read_csv(file_path)

print("Looking at the input data ...")

print(df_input.head())
# Iterative replacement of elements in the input df to make it more readable 

df_input=df_input.replace({'has_travel_history': {'t': 'has_traveled', 'f': 'no_travels'}})

df_input=df_input.replace({'locally_acquired': {'Close contact': 'Close Contact'}})

df_input=df_input.replace({'age': {'61': '60-69', 

                                   '50': '50-59',

                                   '10-19': '<20',

                                   '<1': '<20',

                                   '<10': '<20',

                                   '2': '<20',

                                   '<18': '<20'}}) 



# For plotting convenience sorting bt age

df_input=df_input.sort_values(by='age')



print(df_input.head())
def group_two_columns(dataframe,first_feature,second_feature,manipulated_data,title): 

    grouped_data=dataframe.groupby([first_feature,second_feature])[manipulated_data].count().reset_index(name=title)

    return grouped_data



df_age_sex=group_two_columns(df_input,'age','sex','report_week','ValueM')

df_age_province=group_two_columns(df_input,'age','province','report_week','ValueM')

df_sex_province=group_two_columns(df_input,'sex','province','report_week','ValueM')

df_age_has_travel_history=group_two_columns(df_input,'age','has_travel_history','report_week','ValueM')

df_sex_has_travel_history=group_two_columns(df_input,'sex','has_travel_history','report_week','ValueM')



# See how it looks 

print(df_age_sex.head())
# Clean-up the datasets, basic manipulation on the input data

df_cleaned_base=df_input

df_cleaned_age=df_cleaned_base[df_cleaned_base.age != 'Not Reported']

df_cleaned_sex=df_cleaned_base[df_cleaned_base.sex != 'Not Reported']

df_age_sex=df_age_sex[df_age_sex.age != 'Not Reported'] 

df_age_sex=df_age_sex[df_age_sex.sex != 'Not Reported'] 

df_age_province=df_age_province[df_age_province.age != 'Not Reported']

df_sex_province=df_sex_province[df_sex_province.sex != 'Not Reported']

df_age_has_travel_history=df_age_has_travel_history[df_age_has_travel_history.age != 'Not Reported']

df_sex_has_travel_history=df_sex_has_travel_history[df_sex_has_travel_history.sex != 'Not Reported']

# Check for NaNs

df_cleaned_locally_acquired=df_cleaned_base[df_cleaned_base.locally_acquired == df_cleaned_base.locally_acquired]

df_cleaned_has_travel_history=df_cleaned_base[df_cleaned_base.has_travel_history == df_cleaned_base.has_travel_history]



# See how it looks 

print(df_age_sex.head())
# Split the datasets for projection of scatter plots

df_sex_province_male=df_sex_province[df_sex_province.sex == 'Male']

df_sex_province_female=df_sex_province[df_sex_province.sex == 'Female']

df_sex_has_travel_history_male=df_sex_has_travel_history[df_sex_has_travel_history.sex == 'Male']

df_sex_has_travel_history_female=df_sex_has_travel_history[df_sex_has_travel_history.sex == 'Female']

df_age_sex_male=df_age_sex[df_age_sex.sex == 'Male'] 

df_age_sex_female=df_age_sex[df_age_sex.sex == 'Female'] 



# Look for two examples of datasets 

print(df_sex_province_female.head(100))

print(df_sex_province_male.head(100))

def compare_dataframes_and_fillup(dataframe_first, dataframe_second, labeling_feature, default_value, additional_columns):

    for i, row_first in dataframe_first.iterrows():

        found_label=False

        label_first=dataframe_first[labeling_feature].ix[i]

        for j, row_second in dataframe_second.iterrows():

            label_second=dataframe_second[labeling_feature].ix[j]

            # Not checking for unmatching cases, we need at least one label to match, the found_label will be true

            if label_first == label_second: 

                found_label=True

        if not found_label: 

            dataframe_second=dataframe_second.append({additional_columns[0] : additional_columns[1], labeling_feature : label_first , 'ValueM' : 0} , ignore_index=True)

    return dataframe_second
# Now compare splitted dataframes to ensure they have some data content or fill up missing gaps 

df_sex_province_female=compare_dataframes_and_fillup(df_sex_province_male, df_sex_province_female, 'province', 0, ['sex','Female'])

df_sex_province_male=compare_dataframes_and_fillup(df_sex_province_female, df_sex_province_male, 'province', 0, ['sex','Male'])



# Look for complemented datasets 

print(df_sex_province_female.head(100))

print(df_sex_province_male.head(100))
def make_text_freq_plot(dataframe_feature,color,plot_title,is_log_y,y_axis_name,is_grid,width,legend,tick_size=0):

    counter=Counter(dataframe_feature)

    keys_names=counter.keys()

    keys_counts=counter.values()

    # Convert to list and evaluate sqrt for error plotting 

    keys_counts_to_list=list(keys_counts)

    errors=[]

    for element in keys_counts_to_list:

        err=math.sqrt(float(element))

        errors.append(err)

    indexes=np.arange(len(keys_names))

    plt.bar(indexes, keys_counts, width, color=color, linewidth=0.5,edgecolor='black',label=legend,yerr=errors)

    ax=plt.axes()

    plt.xticks(indexes, keys_names)

    if is_log_y:

        plt.yscale('log')        

    plt.ylabel(y_axis_name)

    if tick_size != 0: 

        for tick in ax.xaxis.get_major_ticks():

            tick.label.set_fontsize(tick_size) 

        for tick in ax.yaxis.get_major_ticks():

            tick.label.set_fontsize(tick_size) 

    if is_grid: 

        plt.grid(True,axis='y')

    plt.legend()

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()

    plt.savefig(plot_title)

    plt.close()
# Plots with plain datasets 

make_text_freq_plot(df_input.health_region,'lavender','health_region.png',False,'counts',True,0.8,'Cases per health region',5.0)

make_text_freq_plot(df_input.province,'lavender','province.png',False,'counts',True,0.8,'Cases per province')

# Re-order df_input 

df_input=df_input.sort_values(by='date_report')

make_text_freq_plot(df_input.date_report,'lavender','date_report.png',False,'counts',True,0.8,'Date of report',7.0)
# Plots with cleaned up datasets

make_text_freq_plot(df_cleaned_sex.sex,'lavender','sex_cleaned.png',False,'counts',True,0.8,'Cases per sex')

make_text_freq_plot(df_cleaned_age.age,'lavender','age_cleaned.png',False,'counts',True,0.8,'Cases per age group')

make_text_freq_plot(df_cleaned_has_travel_history.has_travel_history,'lavender','has_travel_hystory_cleaned.png',False,'counts',True,0.8,'Cases for individuals with/without travel history')

make_text_freq_plot(df_cleaned_locally_acquired.locally_acquired,'lavender','locally_acquired_cleaned.png',False,'counts',True,0.8,'Transmission')
def make_scatter_plot(dataframe,first_feature,second_feature,third_feature,z_value,plot_title,scaling_factor): 

    tick_size=8

    dataframe[z_value]=scaling_factor*dataframe[z_value].astype(float)

    ax=plt.axes()

    for tick in ax.xaxis.get_major_ticks():

        tick.label.set_fontsize(tick_size) 

    for tick in ax.yaxis.get_major_ticks():

        tick.label.set_fontsize(tick_size) 

    plt.scatter(first_feature,second_feature,100,dataframe.ValueM,alpha=0.5,lw=0.0,cmap=plt.cm.viridis)

    plt.xticks(rotation=90)

    plt.colorbar()

    plt.show()

    plt.tight_layout()

    plt.savefig(plot_title)

    plt.clf()

    plt.close()



# Scatter plots

make_scatter_plot(df_age_sex,df_age_sex.age,df_age_sex.sex,df_age_sex.ValueM,'ValueM',"2d_age_sex.png",1)

make_scatter_plot(df_age_province,df_age_province.age,df_age_province.province,df_age_province.ValueM,'ValueM',"2d_age_province.png",1)

make_scatter_plot(df_sex_province,df_sex_province.sex,df_sex_province.province,df_sex_province.ValueM,'ValueM',"2d_sex_province.png",1)

make_scatter_plot(df_age_has_travel_history,df_age_has_travel_history.age,df_age_has_travel_history.has_travel_history,df_age_has_travel_history.ValueM,'ValueM',"2d_age_has_travel_history.png",1)
# Similar to the previous but handles the data differently 

def make_text_freq_plot_splitted(dataframe_feature_first,dataframe_feature_second,labels,color,plot_title,is_log_y,y_axis_name,is_grid,width,legend,title_first,title_second,tick_size=0):

    labels=labels.to_numpy()

    indexes=np.arange( len(labels) )

    feature_first=dataframe_feature_first.to_numpy()

    feature_second=dataframe_feature_second.to_numpy()

    errors_first=[]

    errors_second=[]

    # The first and second lists have some size, cross check

    if len(feature_first) != len(feature_second): 

        print('len(feature_first) != len(feature_second) this function should not be used')

    for i in range(len(feature_first)):

        element_first=feature_first[i]

        element_second=feature_second[i]

        errors_first.append( math.sqrt(float(element_first)) )

        errors_second.append( math.sqrt(float(element_second)) )

    fig, ax=plt.subplots()

    # Distancing the histograms for some bin by 0.015*2

    rects1=ax.bar(indexes -0.015 - width/2, feature_first, width, label=title_first, color=color[0], linewidth=0.5,edgecolor='black',yerr=errors_first)

    rects2=ax.bar(indexes +0.015 + width/2, feature_second, width, label=title_second, color=color[1], linewidth=0.5,edgecolor='black',yerr=errors_second)

    ax.set_xticks(np.arange(len(labels)))

    ax.set_xticklabels(labels=labels)

    plt.ylabel(y_axis_name)

    if is_grid: 

        plt.grid(True,axis='y')

    ax.legend()

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()

    plt.savefig(plot_title)

    plt.close()
# Sex splitted histograms

make_text_freq_plot_splitted(df_sex_has_travel_history_male.ValueM,df_sex_has_travel_history_female.ValueM,df_sex_has_travel_history_male.has_travel_history,['palegreen','moccasin'],'1d_sex_has_travel_history.png',False,'counts',True,0.35,'Has travel history split by sex','Male','Female')

make_text_freq_plot_splitted(df_age_sex_male.ValueM,df_age_sex_female.ValueM,df_age_sex_male.age,['palegreen','moccasin'],'1d_df_age_sex.png',False,'counts',True,0.35,'Age split by sex','Male','Female')

make_text_freq_plot_splitted(df_sex_province_male.ValueM,df_sex_province_female.ValueM,df_sex_province_male.province,['palegreen','moccasin'],'1d_sex_province.png',False,'counts',True,0.35,'Province split by sex','Male','Female')