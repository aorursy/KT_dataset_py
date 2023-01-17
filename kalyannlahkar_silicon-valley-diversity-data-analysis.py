import pandas as pd

import numpy as np



#loading the dataset from csv file

df = pd.read_csv('../input/Reveal_EEO1_for_2016.csv')

print(len(df.axes[0]))
#data cleaning

modif_df = df[df['count'] != '0']

modif_df = modif_df[modif_df['count'] != 'na']



modif_df = modif_df[modif_df['gender'] != '']



modif_df = modif_df[modif_df['race'] != 'Overall_totals']



modif_df = modif_df[modif_df['job_category'] != 'Previous_totals']

modif_df = modif_df[modif_df['job_category'] != 'Totals']



modif_df['count'] = modif_df['count'].astype('int64')

modif_df.head()

male_df = modif_df[modif_df['gender'] == 'male']

num_of_males = modif_df.loc[modif_df['gender'] == 'male', 'count'].sum()

print(num_of_males)
def get_attribute_count_by_value(attribute_name, value, df):

    return df.loc[df[attribute_name] == value, 'count'].sum()



print(get_attribute_count_by_value('gender', 'female', modif_df))
#plotting gender wise workforce data as a pandas dataframe using matplotlib

%matplotlib inline

import matplotlib as plt

plt.rc('figure', figsize=(50, 50))

font_options = {'family' : 'sans-serif',

                'weight' : 'normal',

                'size' : 36}

plt.rc('font', **font_options)



unique_companies = modif_df['company'].unique()

df2 = pd.DataFrame(columns=['percent of males', 'percent of females'], index=unique_companies)



for company in unique_companies:

    

    new_df = modif_df[modif_df['company'] == company]

    

    num_of_males = get_attribute_count_by_value('gender', 'male', new_df)

    num_of_females = get_attribute_count_by_value('gender', 'female', new_df)

    total = num_of_males + num_of_females

    

    percentage_of_males = float(num_of_males/total)*100

    percentage_of_females = float(num_of_females/total)*100

    

    df2.loc[company] = [percentage_of_males, percentage_of_females]



df2.sort_values(['percent of females'], axis=0, ascending=False, inplace=True)

ax = df2.plot.barh(alpha=0.75)

ax.set(xlabel = "Percentage", ylabel = "Company", title = "Percentage of workforce by gender")

for p in ax.patches:

    ax.annotate(np.round(p.get_width(),decimals=2), \

                (p.get_x() + p.get_width(), p.get_y()), \

                ha='left', va='center', xytext=(0, 10), \

                textcoords='offset points')
#plotting race wise workforce data as a pandas dataframe using matplotlib

%matplotlib inline

import matplotlib.pyplot as plt

font_options = {'family' : 'sans-serif',

                'weight' : 'normal',

                'size' : 14}

plt.rc('font', **font_options)





unique_companies = modif_df['company'].unique()

unique_races = modif_df['race'].unique()



df3 = pd.DataFrame(columns=unique_races, index=unique_companies)



for company in unique_companies:

    company_df = modif_df[modif_df['company'] == company]

    total_for_company = company_df['count'].sum()

    

    race_percents = []

    for race in unique_races:

        percent = float(company_df.loc[company_df['race'] == race, 'count'].sum() / total_for_company)*100

        race_percents.append(percent)

    

    df3.loc[company] = race_percents



for index, row in df3.iterrows():

    

    ax = row.plot.barh(alpha=0.75)

    ax.set(xlabel = "Percentage of workforce", \

           ylabel = "Races", \

           title = index)

    legend = ax.legend(loc='upper right', borderpad=1, labelspacing=1)

    legend = legend.remove()



    for p in ax.patches:

        ax.annotate(np.round(p.get_width(),decimals=2), \

                    (p.get_x() + p.get_width(), p.get_y()), \

                    ha='left', va='center', xytext=(0, 5), \

                    textcoords='offset points')

    plt.show()

    
