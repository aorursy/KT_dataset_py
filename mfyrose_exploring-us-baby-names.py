import pandas as pd
import seaborn as sns
import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/NationalNames.csv')
data.head()
data.info()
names_dict = dict()

for index, row in data.iterrows():
    if row['Name'] not in names_dict:
        names_dict[row['Name']] = row['Count']
    else:
        names_dict[row['Name']] += row['Count']
top_25 = Counter(names_dict).most_common(25)
print('Top 25 most popular names:')
for pair in top_25:
    print(pair)
top_25 = Counter(names_dict).most_common(25)
print('Top 25 most popular names:')
for pair in top_25:
    print(pair[0])
def most_popular_names_data_transform():
    years = range(1880, 2015)
    female_names = []
    female_counts = []
    male_names = []
    male_counts = []
    
    for year in years:
        curr_year_data = data[data['Year'] == year]
                              
        curr_year_data_female = curr_year_data[curr_year_data['Gender'] == 'F']
        res = curr_year_data_female.loc[curr_year_data_female['Year'].idxmax()]
        female_names.append(res['Name'] + ' (' + str(year) + ')')
        female_counts.append(res['Count'])
                              
        curr_year_data_male = curr_year_data[curr_year_data['Gender'] == 'M']
        res = curr_year_data_male.loc[curr_year_data_male['Year'].idxmax()]
        male_names.append(res['Name'] + ' (' + str(year) + ')')
        male_counts.append(res['Count'])
                                    
    return (female_names, male_names, female_counts, male_counts)
female_names, male_names, female_counts, male_counts = most_popular_names_data_transform()
sns.set(style="whitegrid")
sns.set_color_codes("pastel")

f, axes = plt.subplots(1, 2, figsize=(12, 30))

sns.barplot(x=female_counts, y=female_names, color="r", ax=axes[0])
axes[0].set(ylabel="Name (Year)", xlabel="Count", title="Most popular female names in US")

sns.barplot(x=male_counts, y=male_names, color="b", ax=axes[1])
axes[1].set(ylabel="", xlabel="Count", title="Most popular male names in US")

plt.tight_layout(w_pad=4)
print('Top 25 rare names:')
for pair in Counter(names_dict).most_common()[:-25:-1]:
    print(pair[0])
def average_length_data_transform():
    years = []
    female_average_length = []
    female_average_name_length = dict()
    male_average_length = []
    male_average_name_length = dict()
    
    for index, row in data.iterrows():
        if row['Gender'] == 'F':
            curr_year = row['Year']
            curr_name_length = len(row['Name'])
            if curr_year not in female_average_name_length:
                female_average_name_length[curr_year] = [curr_name_length, 1]
            else:
                female_average_name_length[curr_year][0] += curr_name_length
                female_average_name_length[curr_year][1] += 1
        else:
            curr_year = row['Year']
            curr_name_length = len(row['Name'])
            if curr_year not in male_average_name_length:
                male_average_name_length[curr_year] = [curr_name_length, 1]
            else:
                male_average_name_length[curr_year][0] += curr_name_length
                male_average_name_length[curr_year][1] += 1
    
    for key, value in female_average_name_length.items():
        years.append(key)
        female_average_length.append(float(value[0]) / value[1])
        
    for key, value in male_average_name_length.items():
        years.append(key)
        male_average_length.append(float(value[0]) / value[1])
        
    return (female_average_length, female_average_name_length, male_average_length, male_average_name_length)
female_average_length, female_average_name_length, male_average_length, male_average_name_length = average_length_data_transform()
years = range(1880, 2015)
f, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([1880, 2014])

plt.plot(years, female_average_length, label='Average length of female names', color='r')
plt.plot(years, male_average_length, label='Average length of male names', color='b')

ax.set_ylabel('Length of name')
ax.set_xlabel('Year')
ax.set_title('Average length of names')
legend = plt.legend(loc='best', frameon=True, borderpad=1, borderaxespad=1)
top_in_each_year = dict()
years = range(1880, 2015)

for each_year in years:
    each_year_data = data[data['Year'] == each_year]
    top_in_each_year[each_year] = dict()
    for index, row in each_year_data.iterrows():            
        top_in_each_year[each_year][row['Name']] = row['Count']
all_sum = []
top_25_sum = []
for year, names_in_year in top_in_each_year.items():
    all_sum.append(sum(Counter(names_in_year).values()))
    top_25 = Counter(names_in_year).most_common(25)
    sum_temp = 0
    for pair in top_25:
        sum_temp += pair[1]
    top_25_sum.append(sum_temp)
percent_unique_names = np.array(top_25_sum).astype(float) / np.array(all_sum) * 100
f, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([1880, 2014])

plt.plot(years, percent_unique_names, label='Percent of unique names', color='black')

ax.set_ylabel('Percent of unique names')
ax.set_xlabel('Year')
ax.set_title('Percent of unique names')
legend = plt.legend(loc='best', frameon=True, borderpad=1, borderaxespad=1)