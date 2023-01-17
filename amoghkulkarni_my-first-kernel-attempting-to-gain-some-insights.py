# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/Suicides in India 2001-2012.csv', 'r') as fp:
    df = pd.read_csv(fp)
df.head()
for type_code in df['Type_code'].unique():
    print("{0}: {1}".format(type_code, df[df['Type_code'] == type_code].size))
for type_code in df['Type_code'].unique():
    type_code_series = df[df['Type_code'] == type_code]
    
    # Print the name and the number of datapoints in the series
    print("{0}: {1}".format(type_code, type_code_series.size))
    
    # Check how many different types of values are there in the Type column
    for type_value in type_code_series['Type'].unique():
        type_value_series = type_code_series[type_code_series['Type'] == type_value]
        
        # Print the name and the number of datapoints in the series 
        print("\t{0}: {1}".format(type_value, type_value_series.size))
for age in df['Age_group'].unique():
    age_series = df[df['Age_group'] == age]
    print("{0}: {1}".format(age, age_series.size))
df[(df['State'] == 'A & N Islands') 
   & (df['Year'] == 2001) 
   & (df['Type_code'] == 'Causes') 
   & (df['Type'] == 'Illness (Aids/STD)')]
def get_data_from_df(dataframe, selector_dict):
    df_to_return = dataframe
    for key, val in selector_dict.items():
        df_to_return = df_to_return[df_to_return[key] == val]
    return df_to_return

get_data_from_df(df, {'State': 'Maharashtra',
                      'Year': 2001,
                      'Type_code': 'Professional_Profile',
                      'Type': 'Farming/Agriculture Activity',
                      'Gender': 'Male'})
years = []
total_suicides_in_agriculture = []
total_suicides_in_non_agriculture = []
for year in df['Year'].unique():
    returned_df = get_data_from_df(df, {'Year': year,
                                        'Type_code': 'Professional_Profile',
                                        'Type': 'Farming/Agriculture Activity'})
    total_suicides_count = returned_df['Total'].sum() 
    years.append(year)
    total_suicides_in_agriculture.append(total_suicides_count)

for year in df['Year'].unique():
    total_suicides_count = 0
    for profession_type in df[df['Type_code'] == 'Professional_Profile']['Type'].unique():
        if profession_type is not 'Farming/Agriculture Activity':
            returned_df = get_data_from_df(df, {'Year': year,
                                                'Type_code': 'Professional_Profile', 
                                                'Type': profession_type})
            total_suicides_count += returned_df['Total'].sum()
    total_suicides_in_non_agriculture.append(total_suicides_count)

ind = np.arange(len(years))
plot1 = plt.bar(x=ind, height=total_suicides_in_non_agriculture)
plot2 = plt.bar(x=ind, height=total_suicides_in_agriculture, bottom=total_suicides_in_non_agriculture)
plt.title('Total suicides in India between 2001 to 2012')
plt.xticks(ind, years, rotation='vertical')
plt.yticks(np.arange(0, 200000, 20000))
plt.legend((plot1[0], plot2[0]), ('Suicides in non-agriculture sectors', 'Suicides in agriculture sector'))
from operator import add  
total_suicides = list(map(add, total_suicides_in_agriculture, total_suicides_in_non_agriculture))
print(total_suicides)

agri_to_non_agri_ratios = list(map(lambda x, y: float(x)/float(y) * 100, total_suicides_in_agriculture, total_suicides_in_non_agriculture))
print(agri_to_non_agri_ratios)

plot3 = plt.plot(ind, agri_to_non_agri_ratios, color='g')
plt.title('Ratio of agricultural suicides to non-agricultural ones in India between 2001 to 2012')
plt.xticks(ind, years, rotation='vertical')
plt.yticks(np.arange(0, 20, 2))
plt.show()
state_wise_suicides = []
for state in df['State'].unique():
    total_suicides_in_agriculture = []
    total_suicides_in_non_agriculture = []
    
    for year in df['Year'].unique():
        returned_df = get_data_from_df(df, {'Year': year,
                                            'Type_code': 'Professional_Profile',
                                            'Type': 'Farming/Agriculture Activity', 
                                            'State': state})
        total_suicides_count = returned_df['Total'].sum() 
        years.append(year)
        total_suicides_in_agriculture.append(total_suicides_count)

    for year in df['Year'].unique():
        total_suicides_count = 0
        for profession_type in df[df['Type_code'] == 'Professional_Profile']['Type'].unique():
            if profession_type is not 'Farming/Agriculture Activity':
                returned_df = get_data_from_df(df, {'Year': year,
                                                    'Type_code': 'Professional_Profile', 
                                                    'Type': profession_type, 
                                                    'State': state})
                total_suicides_count += returned_df['Total'].sum()
        total_suicides_in_non_agriculture.append(total_suicides_count)
    
    total_suicides = list(map(add, total_suicides_in_agriculture, total_suicides_in_non_agriculture))
    agri_to_non_agri_ratios = list(map(lambda x, y: float(x)/float(y) * 100 if y != 0 else 0, total_suicides_in_agriculture, total_suicides_in_non_agriculture))
    
    this_state_suicides = {
        'State': state,
        'agri': total_suicides_in_agriculture,
        'non_agri': total_suicides_in_non_agriculture,
        'total': total_suicides,
        'ratios': agri_to_non_agri_ratios
    }
    state_wise_suicides.append(this_state_suicides)
    
    # Show the graph
    plot4 = plt.plot(ind, agri_to_non_agri_ratios, label=state)

plt.title('State-wise, trends of agricultural suicides to non-agricultural ones in India between 2001 to 2012')
plt.xticks(ind, years, rotation='vertical')
plt.show()
state_indices = []
best_ratios = []
worst_ratios = []
f, ax = plt.subplots(figsize=(20,20))
for state_data in state_wise_suicides:
    state_name = state_data['State']
    
    # Calculate the best and the worst for every state and plot it 
    state_indices.append(state_name)
    
    best_ratios.append(min(state_data['ratios']))
    worst_ratios.append(max(state_data['ratios']))
    
# Show the graph
plot5 = plt.plot(state_indices, best_ratios)
plot6 = plt.plot(state_indices, worst_ratios)

plt.title('State-wise, Ratios of agricultural suicides to non-agricultural ones in India between 2001 to 2012')
plt.xticks(np.arange(len(state_indices)), state_indices, rotation='vertical')
plt.legend((plot5[0], plot6[0]), ('Best ratios in suicides in agriculture sectors', 'Worst ratios in suicides in agriculture sector'))
plt.show()  
import seaborn as sbr
x_states = []
y_ratios = []
f, ax = plt.subplots(figsize=(20,20))
for state_data in state_wise_suicides:
    for state_data_ratio in state_data['ratios']:
        x_states.append(state_data['State'])
        y_ratios.append(state_data_ratio)
bx = sbr.boxplot(ax=ax, x=x_states,y=y_ratios, palette="Set3")
bx = bx.set_xticklabels(state_indices, rotation=90)
from scipy import stats
states_all = []
professions_all = []
years_all = []
for state in df['State'].unique():
    states_all.append(state)
for profession in df[df['Type_code'] == 'Professional_Profile']['Type'].unique():
    professions_all.append(profession)
for year in df['Year'].unique():
    years_all.append(year)

state_vs_professions_slopes = []
for state in states_all:
    state_data = []
    for profession in professions_all:
        total_suicides_in_profession = []
        total_suicides_in_all_professions = []
        
        for year in years_all:
            returned_df = get_data_from_df(df, {'Year': year, 
                                                'Type_code': 'Professional_Profile', 
                                                'Type': profession, 
                                                'State': state})
            total_suicides_in_profession.append(returned_df['Total'].sum())
        for year in years_all:
            total_suicides_count = 0
            for other_profession in professions_all:
                returned_df = get_data_from_df(df, {'Year': year, 
                                                    'Type_code': 'Professional_Profile', 
                                                    'Type': other_profession, 
                                                    'State': state})
                total_suicides_count += returned_df['Total'].sum()
            total_suicides_in_all_professions.append(total_suicides_count)
        
        pro_to_all_ratios = list(map(lambda x, y: float(x)/float(y) * 100 if y != 0 else 0, total_suicides_in_profession, total_suicides_in_all_professions))
        
        # Calculate the slope of the values of pro_to_non_all_ratios and save them 
        slope_for_a_profession, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(years_all)), pro_to_all_ratios)
        
        state_data.append(slope_for_a_profession)
    state_vs_professions_slopes.append(state_data)

f, ax = plt.subplots(figsize=(20, 20))
hm = sbr.heatmap(state_vs_professions_slopes, xticklabels=professions_all, yticklabels=states_all, ax=ax, cmap="YlGnBu", linecolor="black", linewidths='.5')
Chhattisgarh_others_total = []
Chhattisgarh_all_total = []

for year in years_all:
    returned_df = get_data_from_df(df, {'Year': year, 
                                        'Type_code': 'Professional_Profile', 
                                        'Type': 'Others (Please Specify)', 
                                        'State': 'Chhattisgarh'})
    Chhattisgarh_others_total.append(returned_df['Total'].sum())
for year in years_all:
    total_suicides_count = 0
    for other_profession in professions_all:
        returned_df = get_data_from_df(df, {'Year': year, 
                                            'Type_code': 'Professional_Profile', 
                                            'Type': other_profession, 
                                            'State': 'Chhattisgarh'})
        total_suicides_count += returned_df['Total'].sum()
    Chhattisgarh_all_total.append(total_suicides_count)

others_to_nonothers_ratios = list(map(lambda x, y: float(x)/float(y) * 100 if y != 0 else 0, Chhattisgarh_others_total, Chhattisgarh_all_total))

# Calculate the slope of the values of pro_to_non_pro_ratios and save them 
slope_for_others, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(years_all)), others_to_all_ratios)

# Show the graph
plot7 = plt.plot(np.arange(len(years_all)), others_to_all_ratios)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope_for_others * x_vals
plot8 = plt.plot(x_vals, y_vals, '--')

plt.title('Chhattisgarh: Trend of ratios of other suicides to non-other ones between 2001 to 2012')
plt.xticks(np.arange(len(years_all)), years_all, rotation='vertical')
plt.legend((plot7[0], plot8[0]), ('The values of ratio', 'Line fit'))
plt.show()  






