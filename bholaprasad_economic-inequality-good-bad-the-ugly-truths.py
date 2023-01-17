# Import the libraries

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns 

# to show all the columns

pd.options.display.max_columns=999 



# Import the 2019 dataset 

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)

# assign the first row as column 

df_2019.columns = df_2019.iloc[0]

# Now drop the first row 

df_2019=df_2019.drop([0])



# Replacing the ambigious countries name with Standard names

df_2019['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)



# create a mapping of the long column names that we are going to 

# use for our analysis

col_rename_dict_2019 = {'What is your age (# years)?': 'Age',

                  'What is your gender? - Selected Choice':'Gender',

                  'In which country do you currently reside?': 'Country',

                  'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'Highest Education',

                  'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'Current Role',

                  'What is your current yearly compensation (approximate $USD)?': 'Yearly Compensation',

                  'What is the size of the company where you are employed?': 'Company Size',

                  'Approximately how many individuals are responsible for data science workloads at your place of business?':'Team Size'}



# Rename the columns 

df_2019.rename(columns=col_rename_dict_2019, inplace=True)



# create a new column to turn 25 compensation categories to 5 sub-categories

mapping_dict = {

    '$0-999':'Very Low Wage',

        '1,000-1,999':'Very Low Wage',

        '2,000-2,999':'Very Low Wage',

        '3,000-3,999': 'Very Low Wage',

        '4,000-4,999': 'Very Low Wage',

        '5,000-7,499': 'Low Wage',

        '7,500-9,999': 'Low Wage',

        '10,000-14,999':'Low Wage',

        '15,000-19,999':'Low Wage',

        '20,000-24,999': 'Low Wage',

        '25,000-29,999': 'Medium Wage',

        '30,000-39,999': 'Medium Wage',

        '40,000-49,999':'Medium Wage',

        '50,000-59,999':'Medium Wage',

        '60,000-69,999':'Medium Wage',

        '70,000-79,999':'High Wage',

        '80,000-89,999':'High Wage',

        '90,000-99,999':'High Wage',

        '100,000-124,999':'High Wage',

        '125,000-149,999':'High Wage',

        '150,000-199,999':'Very High Wage',

        '200,000-249,999':'Very High Wage',

        '250,000-299,999':'Very High Wage',

        '300,000-500,000':'Very High Wage',

        '> $500,000':'Very High Wage'

}



df_2019['Yearly Compensation Category'] = df_2019['Yearly Compensation'].map(mapping_dict)



# Mapping Higher Education

mapping_dict = {

    'Highest Education':{

        'Some college/university study without earning a bachelor’s degree':'Some college/university',

        'No formal education past high school':'High School'

    }

}



df_2019 = df_2019.replace(mapping_dict)





# Only select the columns that we need for our analysis

df_2019_subset = df_2019[['Country','Gender','Age','Highest Education','Current Role','Yearly Compensation',

                          'Yearly Compensation Category','Company Size','Team Size']]



# select only the rows where gender is Female or Male

female_male_2019 = df_2019_subset[(df_2019_subset['Gender']=='Female')| (df_2019_subset['Gender']=='Male')]



# Create a two way table 

comp_pivot = female_male_2019.pivot_table(index='Gender',columns='Yearly Compensation Category', aggfunc='size')

# calculate conditional percentage 

comp_cond_perc = comp_pivot.apply(lambda x: round(x/ comp_pivot.sum(axis=1)*100, 2))

# reset the index

comp_cond_perc.reset_index(inplace=True)



# Melt the columns to tidy the data 

comp_cond_perc = comp_cond_perc.melt(id_vars=['Gender'],var_name='Yearly Compensation Category', value_name='Percentages')



# create a list according to which we want the data to be plotted

order_list = ['Very Low Wage','Low Wage','Medium Wage','High Wage','Very High Wage']



# create the plot 

g = sns.catplot(x="Yearly Compensation Category", y="Percentages", hue="Gender", data=comp_cond_perc,

                kind="bar", order=order_list, height=7, aspect=1.2, palette='muted')

plt.title('Yearly Compensation 2019 By Gender', fontsize=17)

g.set_axis_labels(y_var="Conditional Percentages");
# create a list of top 6 countries

top_6_country_list= list(female_male_2019['Country'].value_counts()[:6].index )

# select only the data for these countries

top_6_country_data = female_male_2019[female_male_2019['Country'].isin(top_6_country_list)]



# Create a two way table 

comp_pivot_country = top_6_country_data.pivot_table(index=['Country','Gender'], 

                                                     columns='Yearly Compensation Category', aggfunc='size').fillna(0)



# calculate conditional percentage 

comp_cond_perc = comp_pivot_country.apply(lambda x: round(x/ comp_pivot_country.sum(axis=1)*100, 2))

# reset the index

comp_cond_perc.reset_index(inplace=True)

# tidy the data 

comp_cond_perc = comp_cond_perc.melt(id_vars=['Country','Gender'],

                                     var_name='Yearly Compensation Category', value_name='Percentages')



# Plot the data 

g = sns.catplot(x="Yearly Compensation Category", y="Percentages", hue="Gender", data=comp_cond_perc,

                kind="bar", order=order_list, col='Country',col_wrap=2,palette='muted',aspect=1.1)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Yearly Compensation 2019 Top 6 Countries', fontsize=17) 

g.set_axis_labels(y_var="Conditional Percentages");
comp_pivot = female_male_2019.pivot_table(index=['Gender','Current Role'], 

                                          columns='Yearly Compensation Category', aggfunc='size')

comp_cond_perc = comp_pivot.apply(lambda x: round(x/comp_pivot.sum(axis=1)*100, 2))

comp_cond_perc.reset_index(inplace=True)

comp_cond_perc = comp_cond_perc.melt(id_vars=['Gender','Current Role'], 

                                     var_name='Yearly Compensation Category', value_name='Percentages')

g = sns.catplot(x='Yearly Compensation Category', y='Percentages', hue='Gender', data=comp_cond_perc,

               kind='bar',order=order_list, col='Current Role', col_wrap=2, palette='muted',aspect=1.1)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Yearly Compensation 2019 By Profession', fontsize=17)

g.set_axis_labels(y_var="Conditional Percentages");
comp_pivot = female_male_2019.pivot_table(index=['Gender','Company Size'], 

                                          columns='Yearly Compensation Category', aggfunc='size')

comp_cond_perc = comp_pivot.apply(lambda x: round(x/comp_pivot.sum(axis=1)*100, 2))

comp_cond_perc.reset_index(inplace=True)

comp_cond_perc = comp_cond_perc.melt(id_vars=['Gender','Company Size'], 

                                     var_name='Yearly Compensation Category', value_name='Percentages')



g = sns.catplot(x='Yearly Compensation Category', y='Percentages', hue='Gender', data=comp_cond_perc,

               kind='bar',order=order_list, col='Company Size', col_wrap=2, palette='muted',aspect=1.1)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Yearly Compensation 2019 By Company Size', fontsize=17)

g.set_axis_labels(y_var="Conditional Percentages");
comp_pivot = female_male_2019.pivot_table(index=['Gender','Age'], 

                                          columns='Yearly Compensation Category', aggfunc='size')

comp_cond_perc = comp_pivot.apply(lambda x: round(x/comp_pivot.sum(axis=1)*100, 2))

comp_cond_perc.reset_index(inplace=True)

comp_cond_perc = comp_cond_perc.melt(id_vars=['Gender','Age'], 

                                     var_name='Yearly Compensation Category', value_name='Percentages')



g = sns.catplot(x='Yearly Compensation Category', y='Percentages', hue='Gender', data=comp_cond_perc,

               kind='bar',order=order_list, col='Age', col_wrap=3, palette='muted',aspect=1.1)



plt.subplots_adjust(top=0.9)

g.fig.suptitle('Yearly Compensation 2019 Based On Age', fontsize=17)

g.set_axis_labels(y_var="Conditional Percentages");
# Female 25-29 in the very high wage category

female_25_29_very_high = female_male_2019[(female_male_2019['Gender']=='Female') & (

    female_male_2019['Age']=='25-29') & (female_male_2019['Yearly Compensation Category']=='Very High Wage')]

female_25_29_very_high
# Female age between 55-59 and high wage 

female_55_59_high_wage = female_male_2019[(female_male_2019['Gender']=='Female') & (

    female_male_2019['Age']=='55-59') & (female_male_2019['Yearly Compensation Category']=='High Wage')]

female_55_59_high_wage
comp_pivot = female_male_2019.pivot_table(index=['Gender','Highest Education'], 

                                          columns='Yearly Compensation Category', aggfunc='size')

comp_cond_perc = comp_pivot.apply(lambda x: round(x/comp_pivot.sum(axis=1)*100, 2))

comp_cond_perc.reset_index(inplace=True)

comp_cond_perc = comp_cond_perc.melt(id_vars=['Gender','Highest Education'], 

                                     var_name='Yearly Compensation Category', value_name='Percentages')

g = sns.catplot(x='Yearly Compensation Category', y='Percentages', hue='Gender', data=comp_cond_perc,

               kind='bar',order=order_list, col='Highest Education', col_wrap=2, palette='muted',aspect=1.1)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Yearly Compensation 2019 Based On Education', fontsize=17)

g.set_axis_labels(y_var="Conditional Percentages");