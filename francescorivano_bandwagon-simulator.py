%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

multiple_choice = pd.read_csv('../input/multipleChoiceResponses.csv')
multiple_choice.head()
questions = []
for column in multiple_choice.columns:
    questions.append((column, multiple_choice.loc[0, column]))
questions[:5]

multiple_choice = multiple_choice.iloc[1:]

columns_with_other = multiple_choice.filter(regex=".OTHER.").columns


for col in columns_with_other:
    print(multiple_choice[col].unique()[:5], "...", multiple_choice[col].unique()[-5:] )

multiple_choice[columns_with_other] = multiple_choice[columns_with_other].apply(pd.to_numeric)
multiple_choice.rename(index=str, columns={'Time from Start to Finish (seconds)': 'TIME'}, inplace=True)
multiple_choice['TIME'] = multiple_choice['TIME'].apply(pd.to_numeric)
multiple_choice_replace_minus_one =  multiple_choice[columns_with_other].replace(-1, 0)
total_characters = multiple_choice_replace_minus_one.sum(axis=1)
multiple_choice['TOTAL_CHARACTERS'] = total_characters
multiple_choice['TYPING_SPEED'] = multiple_choice['TOTAL_CHARACTERS'] / multiple_choice['TIME']

our_winner = multiple_choice.sort_values(['TYPING_SPEED'], ascending=False)
print(( "Our winner is user number {} from {}, with a typing speed "\
      +"of {} characters per second!\n").format(our_winner.index[0],
                                                our_winner.iloc[0].Q3,
                                                our_winner.iloc[0].loc['TYPING_SPEED']))

novelist = multiple_choice.sort_values(['TOTAL_CHARACTERS'], ascending=False)
print(( "Wow, this one's a novelist. Here is user number {} from {}, who typed "\
      +"{} characters to answer their questions.\n").format(novelist.index[0],
                                                            novelist.iloc[0].Q3,
                                                            novelist.iloc[0].loc['TOTAL_CHARACTERS']))

taking_it_easy = multiple_choice.sort_values(['TIME'], ascending=False)
print(("Better to take it easy if it's a survey with no benefits whatsoever.\n"\
       +"Here is user number {} from {}, who took "\
       +"{} seconds,\nAKA {} hours, "\
       +"or {} days, to answer their questions.").format(taking_it_easy.index[0],
                                                         taking_it_easy.iloc[0].Q3,
                                                         taking_it_easy.iloc[0].loc['TIME'],
                                                         taking_it_easy.iloc[0].loc['TIME'] / 3600,
                                                         taking_it_easy.iloc[0].loc['TIME'] / 86400))

aggregate_values = multiple_choice.groupby(['Q3']).agg({'TYPING_SPEED': ['mean', 'max']})
aggregate_values.sort_values([('TYPING_SPEED', 'max')], ascending=False).head(10)
print(multiple_choice.Q3.unique().tolist()[:10], 'and so on, it\'s 58 Countries.')
# Why
country_code_dict = {
    'United States of America': 'USA', 'Indonesia': 'IDN', 'India': 'IND',
    'Colombia': 'COL', 'Chile': 'CHL', 'Turkey': 'TUR', 'Hungary': 'HUN',
    'Ireland': 'IRL', 'France': 'FRA', 'Argentina': 'ARG', 'Japan': 'JAP',
    'Nigeria': 'NGA', 'Spain': 'ESP', 'Iran, Islamic Republic of...': 'IRN',
    'United Kingdom of Great Britain and Northern Ireland': 'GBR',
    'Poland': 'POL', 'Kenya': 'KEN', 'Denmark': 'DNK', 
    'Netherlands': 'NLD', 'China': 'CHN', 'Australia': 'AUS',
    'Sweden': 'SWE', 'Ukraine': 'UKR', 'Canada': 'CAN',
    'Russia': 'RUS', 'Austria': 'AUT', 'Italy': 'ITA',
    'Mexico': 'MEX', 'Germany': 'DEU', 'Singapore': 'SGP',
    'Brazil': 'BRA', 'Switzerland': 'CHE', 'Tunisia': 'TUN',
    'South Africa': 'ZAF','South Korea': 'KOR', 'Pakistan': 'PAK',
    'Malaysia': 'MYS', 'Hong Kong (S.A.R.)': 'HKG', 'Egypt': 'EGY',
    'Portugal': 'PRT', 'Thailand': 'THA', 'Morocco' : 'MAR',
    'Czech Republic': 'CZE', 'Romania': 'ROU', 'Israel': 'ISR',
    'Philippines': 'PHL', 'Bangladesh': 'BGD', 'Belarus': 'BLR',
    'Viet Nam': 'VNM', 'Belgium': 'BEL', 'New Zealand': 'NZL',
    'Norway': 'NOR', 'Finland': 'FIN', 'Greece': 'GRC', 'Peru': 'PER',
    'Republic of Korea': 'ROK'
}
multiple_choice['COUNTRY_CODE'] = multiple_choice['Q3'].replace(country_code_dict).fillna('Unknown')

useful_entries = multiple_choice[multiple_choice['COUNTRY_CODE'].isin(country_code_dict.values())][['TIME', 'Q3', 'COUNTRY_CODE', 'TYPING_SPEED', 'TOTAL_CHARACTERS']]
useful_entries.head()    
aggregate_values_clean = useful_entries.groupby(['COUNTRY_CODE']).agg({'TYPING_SPEED': ['mean', 'max']})
from plotly.offline import init_notebook_mode, iplot

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

init_notebook_mode(connected=True)

reverse_countries_dict = {country_code: country_name for country_name, country_code in country_code_dict.items()}

locations = pd.Series(aggregate_values_clean.index)
print(aggregate_values_clean.index)
text = locations.replace(reverse_countries_dict)

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = locations,
        z = aggregate_values_clean[('TYPING_SPEED', 'max')].astype(float),
        #locationmode = 'country names',
        text = text,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Characters per second")
        ) ]

layout = dict(
        title = 'Fastest Kaggle Users by Country',
        geo = dict(
            scope='world',
            projection=dict( type='Stereographic'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig,validate=False)
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = locations,
        z = aggregate_values_clean[('TYPING_SPEED', 'mean')].astype(float),
        #locationmode = 'country names',
        text = text,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Characters per second")
        ) ]

layout = dict(
        title = 'Best Typing Mean by Country',
        geo = dict(
            scope='world',
            projection=dict( type='Stereographic'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig,validate=False)
print(multiple_choice.Q4.unique()) # To understand what we're going to replace next, labels-wise
import seaborn as sns
import numpy as np
# nan values could be "No answer" ones, but we don't know; let's get rid of them
no_nan = multiple_choice.dropna(subset=['Q4'])

fontdict = {'fontsize': 20, 'fontweight' : 'bold'}

catpl = sns.factorplot(x="Q4", kind="count", data=no_nan, orient='h', aspect=1.5)
labels=('Doctoral degree', 'Bachelor’s degree', 'Master’s degree',
        'Professional degree', 'Some college/university',
        'No answer', 'High school')
plt.xticks(np.arange(0, 7, 1), labels, rotation=-80)
plt.title('Kagglers and Their Education', fontdict=fontdict)
plt.ylabel('')

# Let's be minimally minimalistic, and maybe write a function instead of copying and pasting
# This junk

def minimalistifier(subplot):
    for key, spine in subplot.spines.items():
        spine.set_visible(False)
    for tick in subplot.yaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    for tick in subplot.xaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    subplot.tick_params(bottom=False, left=False)

minimalistifier(catpl.ax)
country_continent_dict = {
    'United States of America': 'North America', 'Indonesia': 'Asia', 'India': 'Asia',
    'Colombia': 'South America', 'Chile': 'South America', 'Turkey': 'Asia', 'Hungary': 'Europe',
    'Ireland': 'Europe', 'France': 'Europe', 'Argentina': 'South America', 'Japan': 'Asia',
    'Nigeria': 'Africa', 'Spain': 'Europe', 'Iran, Islamic Republic of...': 'Asia',
    'United Kingdom of Great Britain and Northern Ireland': 'Europe',
    'Poland': 'Europe', 'Kenya': 'Africa', 'Denmark': 'Europe', 
    'Netherlands': 'Europe', 'China': 'Asia', 'Australia': 'Oceania',
    'Sweden': 'Europe', 'Ukraine': 'Europe', 'Canada': 'North America',
    'Russia': 'Europe', 'Austria': 'Europe', 'Italy': 'Europe',
    'Mexico': 'North America', 'Germany': 'Europe', 'Singapore': 'Asia',
    'Brazil': 'South America', 'Switzerland': 'Europe', 'Tunisia': 'Africa',
    'South Africa': 'Africa','South Korea': 'Asia', 'Pakistan': 'Asia',
    'Malaysia': 'Asia', 'Hong Kong (S.A.R.)': 'Asia', 'Egypt': 'Africa',
    'Portugal': 'Europe', 'Thailand': 'Asia', 'Morocco' : 'Africa',
    'Czech Republic': 'Europe', 'Romania': 'Europe', 'Israel': 'Asia',
    'Philippines': 'Asia', 'Bangladesh': 'Asia', 'Belarus': 'Europe',
    'Viet Nam': 'Asia', 'Belgium': 'Europe', 'New Zealand': 'Oceania',
    'Norway': 'Europe', 'Finland': 'Europe', 'Greece': 'Europe', 'Peru': 'South America',
    'Republic of Korea': 'Asia', 'I do not wish to disclose my location': 'Unknown', 'Other': 'Unknown'
}

multiple_choice['CONTINENT'] = multiple_choice['Q3'].replace(country_continent_dict)
multiple_choice['CONTINENT'].value_counts()
catpl = sns.factorplot(x="CONTINENT", kind="count", data=multiple_choice, orient='h', aspect=1.5)

plt.xticks(rotation=-80)
plt.title('Kagglers and Their Continents', fontdict=fontdict)
plt.ylabel('')
plt.xlabel('')

# Let's be minimally minimalistic, again; copy and paste works like a charm
minimalistifier(catpl.ax)
values_by_country = multiple_choice['Q3'].value_counts()
top_10_countries = values_by_country[:10]
other_countries = pd.Series([values_by_country[10:].sum()])
other_countries.index = ['Other Countries']
#top_10_countries = top_10_countries.append(other_countries)
top_10_and_others = pd.concat([top_10_countries, other_countries], axis=0)
top_10_and_others
# Thank you, people who put "Other" as their 'country'; let's fix this; oh, apparently
# Kaggle did it to preserve their privacy

values_by_country = multiple_choice['Q3'].value_counts()
top_10_countries = pd.concat([values_by_country[:3], values_by_country[4:11]], axis=0)
other_countries = pd.Series([values_by_country[10:].sum() + values_by_country[3]])
other_countries.index = ['Other Countries']
#top_10_countries = top_10_countries.append(other_countries)
top_10_and_others = pd.concat([top_10_countries, other_countries], axis=0)
top_10_and_others
# This will save us some time
def its_always_barplots_anyway(x, y, hue=None, xlabel='', ylabel='', title='', rotation=0, bar_text_col='white', bar_text=False, mantissa=1):
    plt.figure(figsize=(15, 8))
    barpl = sns.barplot(x=x, y = y, hue=hue)
    plt.xticks(rotation=rotation)
    plt.title(title, fontweight='bold', fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if bar_text:
        ax = plt.gca()
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height()*2/5, '%.{}f'.format(mantissa) % float(p.get_height()), 
                    fontsize=14, fontweight='bold', color=bar_text_col, ha='center', va='bottom')
    minimalistifier(barpl)

labels=('USA', 'India', 'China', 'Russia', 'Brazil', 'Germany',
        'UK', 'Canada', 'France', 'Japan', 'Other\nCountries')
its_always_barplots_anyway(top_10_and_others.index, top_10_and_others, xlabel='', ylabel='', title='Kagglers by Country', rotation=0, bar_text=False)
plt.xticks(np.arange(0, 11, 1), labels)
# I wrote this in order to avoid insanity. You can get these values by calling 'unique()'
order= ['18-21', '22-24', '25-29', '30-34', '35-39',
        '40-44', '45-49', '50-54', '55-59', '60-69',
        '70-79', '80+']

catpl = sns.factorplot(x='Q2', kind='count',
                       data=multiple_choice, 
                       order=order,
                       size=5, aspect=1.5)
plt.xticks(rotation=-80)
plt.title('Kagglers by Age', fontdict=fontdict)
plt.ylabel('')


minimalistifier(catpl.ax)
# We basically count how many of each 'kind' there are, then we make a pandas Series out
# of these values and multiply by 100 and divide by the total amount of participants
# in order to find the percentages

males = multiple_choice[multiple_choice.Q1 == 'Male']['Q1'].shape[0]
females = multiple_choice[multiple_choice.Q1 == 'Female']['Q1'].shape[0]
other_1 = multiple_choice[multiple_choice.Q1 == 'Prefer not to say']['Q1'].shape[0]
other_2 = multiple_choice[multiple_choice.Q1 == 'Prefer to self-describe']['Q1'].shape[0]
other = other_1 + other_2

males_females = pd.Series([males, females, other]) * 100 / 23859
males_females.index = ['Males', 'Females', 'Other']
print(males_females.index)
barpl = sns.barplot(y=males_females.index,
                    x=males_females.values,
                    orient='h')

plt.title('Percentages of Kagglers by Gender', fontdict=fontdict)
plt.ylabel('')
plt.xlabel('')

minimalistifier(barpl)
multiple_choice.Q6.value_counts()
Q6_dictionary = {'Data Scientist': 'Data Science',
                 'Data Analyst': 'Data Science',
                 'Data Engineer': 'Data Science',
                 'Data Journalist': 'Data Science',
                 'Marketing Analyst': 'Marketing-Business',
                 'Business Analyst': 'Marketing-Business',
                 'Student': 'Students-Researchers',
                 'Research Scientist': 'Students-Researchers',
                 'Research Assistant': 'Students-Researchers'}
multiple_choice.Q1.astype('category', inplace=True)
multiple_choice['Q1_SIMPLIFIED'] = multiple_choice[multiple_choice.Q1.isin(['Male', 'Female'])]['Q1']
multiple_choice.Q6.astype('category', inplace=True)
multiple_choice['Q6_SIMPLIFIED'] = multiple_choice.Q6.replace(Q6_dictionary) 

catpl = sns.factorplot(y='Q6_SIMPLIFIED', hue='Q1_SIMPLIFIED',
                       data=multiple_choice, kind='count', orient='h',
                       aspect=0.8, size=7, legend=None)


plt.title(' Kagglers by Occupation and Gender (Simplified)', fontdict=fontdict)
plt.ylabel('')
plt.xlabel('')

minimalistifier(catpl.ax)
multiple_choice_dummies = pd.get_dummies(multiple_choice, columns=['Q1_SIMPLIFIED'])
grouped_with_dummies = multiple_choice_dummies.groupby(['Q6_SIMPLIFIED']).agg({'Q1_SIMPLIFIED_Male': 'sum', 'Q1_SIMPLIFIED_Female': 'sum'})
grouped_with_dummies['males_females_ratio']= (grouped_with_dummies['Q1_SIMPLIFIED_Male'] / 
                       grouped_with_dummies['Q1_SIMPLIFIED_Female'])
grouped_with_dummies
its_always_barplots_anyway(grouped_with_dummies.index, grouped_with_dummies.males_females_ratio, xlabel='', ylabel='',
                           title='Men-to-Women Ratio for each Occupation (Simplified)', rotation=-65, bar_text=True)
plt.yticks([])
dummies_gender_salary = pd.get_dummies(multiple_choice_dummies, columns=['Q9'])
Q9_columns = dummies_gender_salary.filter(regex="Q9_.").columns.tolist()
Q9_columns
# Let's sort
Q9_columns = ['Q9_0-10,000', 'Q9_10-20,000', 'Q9_20-30,000', 'Q9_30-40,000',
              'Q9_40-50,000', 'Q9_50-60,000', 'Q9_60-70,000', 'Q9_70-80,000',
              'Q9_80-90,000', 'Q9_90-100,000', 'Q9_100-125,000', 'Q9_125-150,000',
              'Q9_150-200,000', 'Q9_200-250,000', 'Q9_250-300,000', 'Q9_300-400,000',
              'Q9_400-500,000', 'Q9_I do not wish to disclose my approximate yearly compensation']
grouped_dummies_gender_salary = dummies_gender_salary.groupby(['Q3', 'Q1_SIMPLIFIED_Male']).agg({x: 'sum' for x in Q9_columns})
grouped_dummies_gender_salary
grouped_dummies_gender_salary.index
for column in Q9_columns:
    males = grouped_dummies_gender_salary.loc[('United States of America', 0)][column]
    females = grouped_dummies_gender_salary.loc[('United States of America', 1)][column]
    print(('For the $ {} bracket the percentage of females is: {:2.2%} ').format(column[3:], males/ (females + males)))


values = {}
for column in Q9_columns:
    males = grouped_dummies_gender_salary.loc[('United States of America', 1)][column]
    females = grouped_dummies_gender_salary.loc[('United States of America', 0)][column]
    values[column[3:]] = (females/(females + males) * 100)
values['Undisclosed'] = values.pop('I do not wish to disclose my approximate yearly compensation')
values_series = pd.Series(values)

its_always_barplots_anyway(values_series.index, values_series.values, xlabel='', ylabel='',
                           title='Percentage of Women in the USA by Salary Bracket', rotation=-90, bar_text=True, mantissa=1)
def women_percentage_country(country_name):
    import math
    values = {}
    for column in Q9_columns:
        males = grouped_dummies_gender_salary.loc[(country_name, 1)][column]
        females = grouped_dummies_gender_salary.loc[(country_name, 0)][column]
        ratio = (females/(females + males) * 100) 
        values[column[3:]] = 0 if math.isnan(ratio) else ratio # Not all Countries have someone in certain brackets, let's divide by zero responsibly
    values['Undisclosed'] = values.pop('I do not wish to disclose my approximate yearly compensation')
    values_series = pd.Series(values)
    
    its_always_barplots_anyway(values_series.index, values_series.values, xlabel='', ylabel='',
                           title='Percentage of Women in {} by Salary Bracket'.format(country_name), rotation=-90, bar_text=True)
countries = aggregate_values.sort_values([('TYPING_SPEED', 'max')], ascending=False).head(10).index
for country in countries:
    women_percentage_country(country)
import math
# Most of the Countries you'd expect in this list are still in the European Union, but less than 50 people from them answered the survey
# and they are therefore missing as a distinct Country in the survey answers.
# The European Union is still going strong, those who claim otherwise are nothing but enemies of the European Union and will be treated accordingly.
european_union = ['Austria', 'Italy', 'Belgium', 
                  'Sweden', 'Netherlands', 'Czech Republic',
                  'Denmark', 'Poland', 'Portugal','Hungary',
                  'Finland', 'Romania', 'France', 'Ireland',
                  'Germany', 'Greece', 'Spain',
                  'United Kingdom of Great Britain and Northern Ireland']

for column in Q9_columns:
    for country in european_union:
        males = grouped_dummies_gender_salary.loc[(country, 0)][column]
        females = grouped_dummies_gender_salary.loc[(country, 1)][column]

values = {}
males_total = 0
females_total = 0
for column in Q9_columns:
    for country in european_union:
        males = grouped_dummies_gender_salary.loc[(country, 1)][column]
        females = grouped_dummies_gender_salary.loc[(country, 0)][column]
        males_total += males
        females_total += females
        if column[3:]+'_males' in values:
            values[column[3:]+'_males'] += males

        else:
            values[column[3:]+'_males'] = males
            
        if column[3:]+'_females' in values:
            values[column[3:]+'_females'] += females
        else:
            values[column[3:]+'_females'] = females
        
print(values)
for column in Q9_columns:
    ratio = values[column[3:]+'_females'] / (values.pop(column[3:]+'_females') + values.pop(column[3:]+'_males')) * 100
    values[column[3:]] = 0 if math.isnan(ratio) else ratio

values['Undisclosed'] = values.pop('I do not wish to disclose my approximate yearly compensation')
values_series = pd.Series(values)

its_always_barplots_anyway(values_series.index, values_series.values, xlabel='', ylabel='',
                           title='Percentage of Women in the European Union by Salary Bracket', rotation=-90, bar_text=True)
print(values)
# Bonus:
print('\n\n{} European Union people answered the survey.'.format(males_total + females_total))
women_percentage_country('United States of America')
women_percentage_country('India')
questions
 
Q16_columns = multiple_choice.filter(regex="Q16_.*[0-9]").columns.tolist()
column_names_for_decent_human_beings = {'Q16_Part_1': 'Python', 'Q16_Part_2': 'R', 'Q16_Part_3': 'SQL',
                                        'Q16_Part_4': 'Bash', 'Q16_Part_5': 'Java', 'Q16_Part_6': 'Javascript/Typescript',
                                        'Q16_Part_7': 'Visual Basic/VBA', 'Q16_Part_8': 'C/C++', 'Q16_Part_9': 'MATLAB',
                                        'Q16_Part_10': 'Scala', 'Q16_Part_11': 'Julia', 'Q16_Part_12': 'Go',
                                        'Q16_Part_13': 'C#/.NET', 'Q16_Part_14': 'PHP', 'Q16_Part_15': 'Ruby',
                                        'Q16_Part_16': 'SAS/STATA', 'Q16_Part_17': 'None', 'Q16_Part_18': 'Other'}
column_names_for_decent_human_beings = {key:'Uses ' + value for key, value in column_names_for_decent_human_beings.items()}
multiple_choice.rename(column_names_for_decent_human_beings, axis=1, inplace=True)
multiple_choice.head()
languages_columns = list(column_names_for_decent_human_beings.values())
val_dict = {np.nan : 0, 'Python': 1, 'R': 1, 'SQL': 1, 'Bash': 1, 'Java': 1,
            'Javascript/Typescript': 1, 'Visual Basic/VBA': 1, 'C/C++': 1,
            'MATLAB': 1, 'Scala': 1, 'Julia': 1, 'Go': 1, 'C#/.NET': 1, 'PHP': 1,
            'Ruby': 1, 'SAS/STATA': 1, 'None': 1, 'Other': 1}
multiple_choice[languages_columns] = multiple_choice[languages_columns].replace(val_dict).apply(pd.to_numeric)
multiple_choice.head()
groupby_programming_languages = multiple_choice.groupby(['Q2']).agg({col:['sum', 'mean'] for col in languages_columns})
groupby_programming_languages
x = groupby_programming_languages.index
y = 100 - groupby_programming_languages[('Uses Bash', 'mean')] * 100
hue = None
xlabel = ''
ylabel = ''
title = 'Command Line Illiteracy by Age Bracket'
rotation = 0
its_always_barplots_anyway(x, y, hue, xlabel, ylabel, title, rotation, 'white', bar_text=True)
x = groupby_programming_languages.index
y = groupby_programming_languages[('Uses SQL', 'mean')] * 100
its_always_barplots_anyway(x,y, title='Usage of SQL by Age Bracket', xlabel='', ylabel='', rotation=0, bar_text=True)
x = groupby_programming_languages.index
y = groupby_programming_languages[('Uses Python', 'mean')] * 100
its_always_barplots_anyway(x,y, None, title='Usage of Python by Age Bracket', xlabel='', ylabel='', rotation=0, bar_text=True)
plt.figure(figsize=(20,20))
sns.heatmap(multiple_choice[languages_columns].corr())
multiple_choice[languages_columns].corr()
multiple_choice[languages_columns].corr().iloc[0]
multiple_choice[languages_columns].corr().iloc[1]
groupby_income_programming_languages = multiple_choice.groupby(['Q9']).agg({col:['sum', 'mean'] for col in languages_columns}).iloc[:-1]
groupby_income_programming_languages
groupby_income_programming_languages.index
groupby_income_programming_languages = groupby_income_programming_languages.loc[['0-10,000', '10-20,000', '20-30,000',  '30-40,000', '40-50,000',
                                                                            '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
                                                                            '100-125,000', '125-150,000', '150-200,000', '200-250,000',
                                                                            '250-300,000', '300-400,000', '400-500,000', '500,000+']]
groupby_income_programming_languages
groupby_income_languages_means = multiple_choice.groupby(['Q9']).agg({col:'mean' for col in languages_columns}).iloc[:-1]
groupby_income_languages_means = groupby_income_languages_means.loc[['0-10,000', '10-20,000', '20-30,000',  '30-40,000', '40-50,000',
                                                                     '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
                                                                     '100-125,000', '125-150,000', '150-200,000', '200-250,000',
                                                                     '250-300,000', '300-400,000', '400-500,000', '500,000+']]
groupby_income_languages_means
dataf = groupby_income_languages_means.iloc[-9:]
dataf = pd.DataFrame(dataf).T

for col in dataf.columns:
    x = dataf.index
    y = dataf[col] * 100
    its_always_barplots_anyway(x=x, y=y, title='Usage of Programming Languages in the ${} Bracket'.format(col), xlabel='', ylabel='', rotation=-90, bar_text_col='gold', bar_text=True)
multiple_choice['TOTAL_LANGUAGES'] = multiple_choice[languages_columns].sum(axis=1, skipna=True)
multiple_choice.TOTAL_LANGUAGES.value_counts()
multiple_choice_salary_dummies = pd.get_dummies(multiple_choice[['Q4', 'Q9']], columns=['Q4', 'Q9'])
multiple_choice_salary_dummies.corr().iloc[:7, 7:]  # The indices deal with redundant values we're not interested in