import pandas as pd

import numpy as np

import pprint

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')

path= "../input/police-fatalities-in-the-us-from-2000-to-2020/police_fatalities.csv"

data = pd.read_csv(path)

data.head()
data.shape
# print the columns

print("Column names: ")

pp = pprint.PrettyPrinter(indent=4)

pp.pprint(data.columns.tolist())
def get_columns_stats(df: pd.DataFrame)-> None:

    col_with_na = [col for col in df.columns if data[col].isnull().sum() > 0]

    col_without_na = [col for col in df.columns if data[col].isnull().sum() == 0]

    print(f"We have {len(col_with_na)} columns with missing values and {len(col_without_na)} without missing values")

    print()

    print("Variable with missing values")

    print()

    print(f'{"Variable":<65} {"Number of missings":<20} {"Percent of missings":<20}')

    print()

    for col in col_with_na:

        print(f'{col:<65} {df[col].isnull().sum():<20} {np.round(data[col].isnull().mean()*100, 3)}%')

    print()

    print("variable without missing values")

    print(col_without_na)
get_columns_stats(data)
columns = ['id','name', 'age', 'gender', 'race', 'race_with_imputations', 'imputation_probability','url_image_of_deceased', 'date_of_injury_resulting_in_death', 'address_of_injury', 'city_of_death', 'state_of_death',

          'zip_code_of_death', 'county_of_death', 'full_address', 'latitude', 'longitude', 'agency_responsible_for_death', 'cause_of_death',

          'description_circumstances_surrounding_death', 'dispositions', 'intentional_use_of_force', 'link_news_article_or_photo',

          'symptoms_of_mental_illness', 'video', 'date_and_description', 'unique_id_formula', 'unique_id', 'year']

data.columns = columns
pp.pprint(data.columns.tolist())
get_columns_stats(data)
# make list of numerical variables

num_vars = [var for var in data.columns if data[var].dtypes != 'O']



print('Number of numerical variables: ', len(num_vars))

print()

pp.pprint(num_vars)

print()

# visualise the numerical variables

get_columns_stats(data[num_vars])

print()

data[num_vars].head()
data[data.id.isnull()]
data[data.unique_id.isnull()]
# convert id to float

#print(data['id'].astype(float).values == data['unique_id'].values)
data[data.id == 'Victor Sanchez Ancira']
# check if values in id and unique_id are the same

print((data[~data.index.isin([24866,28334])].id.astype(float).values == 

      data[~data.index.isin([24866,28334])].unique_id.astype(float).values).all())
print(f"Number of Unique id: {len(data['id'].unique())}")

print(f"Number of Unique unique_id: {len(data['unique_id'].unique())}")
data.drop(['id', 'unique_id_formula'], axis=1, inplace=True)
pp.pprint(data.columns.tolist())

# some useful function

import itertools

import seaborn as sns

from typing import Tuple

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





def top_n_bar(df: pd.DataFrame, col: str, n: int, figsize:Tuple[int, int]=(13, 10), s_elt:float = 0.6) -> None:

    out_series = df[col].value_counts()

    total_size = sum(out_series.tolist())

    out = dict(itertools.islice(out_series.items(), n)) 

    pd_df = pd.DataFrame(list(out.items()))

    pd_df.columns =[col, "Count"] 

    plt.figure(figsize=figsize)

    ax = sns.barplot(y=pd_df.index, x=pd_df.Count, orient='h')

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    ax.set(xlabel="Count", ylabel=col)

    ax.set_yticklabels(pd_df[col])

    for i, v in enumerate(pd_df["Count"].iteritems()):

        ax.text(v[1] ,i, "{:,}".format(v[1]), color='m', va ='top', rotation=0)

        ax.text(v[1]+ s_elt ,i, "({:0.2f}%)".format((v[1]/total_size)*100), color='m', va ='top', rotation=0)

    plt.tight_layout()

    plt.show()





def check_variable(df: pd.DataFrame, col: str): 

    number_unique = df[col].nunique()

    print(f'variable {col} has {number_unique} unique values')

    if number_unique< 200:

        print(f'These are the {col} values')

        print(f'{df[col].unique()}')
check_variable(data, 'zip_code_of_death')
top_n_bar(data, 'zip_code_of_death', 20, (13, 10))
check_variable(data, 'date_of_injury_resulting_in_death')
data['date_of_injury_resulting_in_death'][:10]
data['date_month'], data['date_day'], data['date_year'] = zip(*data['date_of_injury_resulting_in_death'].apply(lambda x: x.split('/')))
pp.pprint(data.columns.tolist())
check_variable(data, 'date_month')
top_n_bar(data, 'date_month', 12, (13, 10), s_elt=100.0)
check_variable(data, 'date_day')
top_n_bar(data, 'date_day', 31, (13, 10), s_elt=34.0)
check_variable(data, 'date_year')
data[data['date_year'] == '2100']
data.drop(labels=28334, inplace=True)

check_variable(data, 'date_year')
data.date_year.isnull().sum()
top_n_bar(data, 'date_year', 20, s_elt=65.0)
check_variable(data, 'year')
data.year.isnull().sum()
(data[~data.year.isnull()]['year'].values == data[~data.year.isnull()]['date_year'].astype('float').values).all()
temporal_vars = ['date_day', 'date_month', 'date_year', 'date_of_injury_resulting_in_death']

cat_vars = [var for var in data.columns if data[var].dtypes == 'O' and var not in temporal_vars]
print('Number of categorical variables: ', len(cat_vars))
get_columns_stats(data[cat_vars])
check_variable(data, 'age')
unspecific_age = ['20s-30s', '18-25', '25-30', '40-50', '46/53', '45 or 49', '40s', '30s', '50s', '70s', '60s']

age_less_than_a_year = ['3 months', '6 months', '9 months', '10 months', '2 months', '7 months', '8 months', '4 months', '3 days', '11 mon','7 mon' ]
data[data['age'].isin(unspecific_age + age_less_than_a_year)].shape[0]
age_imputation_dict = {

    '20s-30s':25,

    '40-50': 45,

    '18-25':22,

    '46/53':46,

    '45 or 49': 45, 

    '40s': 45,

    '30s':35,

    '50s': 55,

    '60s': 65,

    '18 months': 2,

    '70s': 75,

    '25-30': 27,

    '20s':25,

    '25`': 25, 

    '55.':55

}
data["age"].replace(age_imputation_dict, inplace=True)

data['age']= data['age'].apply(lambda x: 0.0 if x in age_less_than_a_year else float(x))

check_variable(data, 'age')
top_n_bar(data, 'age', 50, (10, 12), s_elt=34.0)
figure(num=None, figsize=(15, 9), dpi=80, facecolor='w', edgecolor='k')



sns.distplot(data['age']);
figure(num=None, figsize=(15, 9), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(y="age", data=data)
check_variable(data, 'gender')
top_n_bar(data, 'gender', 5, s_elt=1400.0)
check_variable(data, 'race')
data[data['race'] == 'HIspanic/Latino']
data.loc[27546, 'race'] = 'Hispanic/Latino'
check_variable(data, 'race')
top_n_bar(data, 'race', 7, s_elt=400.0)
plt.rcParams["figure.figsize"] = [20, 10]

cm = data.groupby(["date_year", "race"])["race"].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
print('race before 2010')

top_n_bar(data[data['year'] <= 2009], 'race', 7, s_elt=400.0)
print('race after 2009')

top_n_bar(data[data['year'] > 2009], 'race', 7, s_elt=400.0)
check_variable(data, 'city_of_death')
top_n_bar(data, 'city_of_death', 20, s_elt=12.0)
check_variable(data, 'state_of_death')
top_n_bar(data, 'state_of_death', 51, s_elt=160.0)
check_variable(data, 'county_of_death')
top_n_bar(data, 'county_of_death', 30, s_elt=45.0)
check_variable(data, 'agency_responsible_for_death')
top_n_bar(data, 'agency_responsible_for_death', 20, s_elt=20.0)
check_variable(data, 'cause_of_death')
top_n_bar(data, 'cause_of_death', 16, s_elt=1050.0)
check_variable(data, 'intentional_use_of_force')
top_n_bar(data, 'intentional_use_of_force', 11, s_elt=1050.0)
data.loc[data[data['intentional_use_of_force'].isin(['Yes','Intentional Use of Force'])].index, 'intentional_use_of_force'] = 'Yes'

data.loc[data[data['intentional_use_of_force'].isin(['Intenional Use of Force, Deadly', 'Intentional Use of Force, Deadly'])].index, 'intentional_use_of_force'] = 'Intentional Use of Force, Deadly'

data.loc[data[data['intentional_use_of_force'].isin(['Vehicle/Pursuit', 'Vehicle','Pursuit'])].index, 'intentional_use_of_force'] = 'Vehicle/Pursuit'

top_n_bar(data, 'intentional_use_of_force', 11, s_elt=1050.0)
check_variable(data, 'symptoms_of_mental_illness')
top_n_bar(data, 'symptoms_of_mental_illness', 5, s_elt=1050.0)
check_variable(data, 'dispositions')
top_n_bar(data, 'dispositions', 30,  s_elt=600)






def plotbox(df, x, y, figsize=(15, 9), orientation='v'):

    figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    IQRs =  df.groupby([x])[y].quantile(0.75) -  df.groupby([x])[y].quantile(0.25)

    medians = df.groupby([x])[y].median()

    percentile_25th = df.groupby([x])[y].quantile(0.25)

    percentile_75th = df.groupby([x])[y].quantile(0.75)

    lower_boundary = df.groupby([x])[y].min()

    upper_boundary = percentile_75th + 1.5*IQRs

    max_value = df.groupby([x])[y].max()

    order = df.groupby([x])[y].sum().index



    

    

    if orientation == 'v':

        box_plot = sns.boxplot(x=x,y=y,data=df)

        vertical_offset_median = df[y].median() * 0.05 #

        vertical_offset_percentile_25th = df[y].quantile(0.25) * 0.05 

        vertical_offset_percentile_75th = df[y].quantile(0.75) * -0.05 

        vertical_lower_boundary_offset = df[y].min()*0.05

        vertical_max_boundary_offset = df[y].max()*0.05

        vertical_upper_boundary_offset = (df[y].quantile(0.75) - (df[y].quantile(0.75)  - (df[y].quantile(0.75) -  df[y].quantile(0.25))*1.5))*0.05

    

        for xtick in box_plot.get_xticks():

            box_plot.text(xtick,medians[xtick] + vertical_offset_median,medians[xtick], 

                horizontalalignment='center',size='x-small',color='w',weight='semibold')

            box_plot.text(xtick,percentile_25th[xtick] + vertical_offset_percentile_25th,percentile_25th[xtick], 

                horizontalalignment='center',size='x-small',color='w',weight='semibold')

            box_plot.text(xtick,percentile_75th[xtick] + vertical_offset_percentile_75th,percentile_75th[xtick], 

                horizontalalignment='center',size='x-small',color='w',weight='semibold')

            box_plot.text(xtick + 0.25,lower_boundary[xtick] + vertical_lower_boundary_offset,lower_boundary[xtick], 

                horizontalalignment='center',size='small',color='b',weight='semibold')

            box_plot.text(xtick -0.25,upper_boundary[xtick] + vertical_upper_boundary_offset,upper_boundary[xtick], 

                horizontalalignment='center',size='x-small',color='b',weight='semibold')

            box_plot.text(xtick,max_value[xtick] + vertical_max_boundary_offset,max_value[xtick], 

                horizontalalignment='center',size='x-small',color='b',weight='semibold')

    else:

        box_plot = sns.boxplot(x=y,y=x,data=df, order=order)

        vertical_offset_median = df[y].median()*0.05 #

        vertical_offset_percentile_25th = df[y].quantile(0.25)*0 

        vertical_offset_percentile_75th = df[y].quantile(0.75)*0

        vertical_lower_boundary_offset = df[y].min()*0

        vertical_max_boundary_offset = df[y].max()*0

        vertical_upper_boundary_offset = (df[y].quantile(0.75) - (df[y].quantile(0.75)  - (df[y].quantile(0.75) -  df[y].quantile(0.25))*1.5))*0

    

        

        for ytick in box_plot.get_yticks():

            box_plot.text(medians[ytick], ytick + 0.05, medians[ytick], 

                verticalalignment='center',size='x-small',color='b',weight='semibold')

            box_plot.text(percentile_25th[ytick], ytick + 0.05,percentile_25th[ytick], 

                verticalalignment='center',size='x-small',color='b',weight='semibold')

            box_plot.text(percentile_75th[ytick], ytick + 0.05, percentile_75th[ytick], 

                verticalalignment='center',size='x-small',color='b',weight='semibold')

            box_plot.text(lower_boundary[ytick],ytick + 0.05, lower_boundary[ytick], 

                verticalalignment='center',size='small',color='b',weight='semibold')

            box_plot.text(upper_boundary[ytick],ytick + 0.05, upper_boundary[ytick], 

                verticalalignment='center',size='x-small',color='b',weight='semibold')

            box_plot.text(max_value[ytick], ytick + 0.05, max_value[ytick], 

                verticalalignment='center',size='x-small',color='b',weight='semibold')

            

    plt.show()

    

plotbox(data, x='date_year',y='age')
plotbox(data, x='race',y='age', orientation='h')
for year in data.date_year.unique():

    print(year)

    plotbox(data[data.date_year == year], x='race',y='age', orientation='h')
## median death age by race each year

median_df = data.groupby(['date_year', 'race']).age.median().reset_index()

ax = sns.lineplot(x="date_year", y="age", hue='race',estimator=None, lw=3,

                  data=median_df [median_df ['race'] != 'Race unspecified'])

ax.set_title('Median Age Death By Race')
print('African American/Black median age range')

print(median_df[median_df.race == 'African-American/Black']['age'].max() - median_df[median_df.race == 'African-American/Black']['age'].min())

print('Hispanic/Latino median age range')

print(median_df[median_df.race == 'Hispanic/Latino']['age'].max() - median_df[median_df.race == 'Hispanic/Latino']['age'].min())

print('European-American/White median age range')

print(median_df[median_df.race == 'European-American/White']['age'].max() - median_df[median_df.race == 'European-American/White']['age'].min())

print('Native American/Alaskan median age range')

print(median_df[median_df.race == 'Native American/Alaskan']['age'].max() - median_df[median_df.race == 'Native American/Alaskan']['age'].min())

print('Middle Eastern median age range')

print(median_df[median_df.race == 'Middle Eastern']['age'].max() - median_df[median_df.race == 'Middle Eastern']['age'].min())

print('Asian/Pacific Islander median age range')

print(median_df[median_df.race == 'Asian/Pacific Islander']['age'].max() - median_df[median_df.race == 'Asian/Pacific Islander']['age'].min())
plotbox(data, x='gender',y='age', orientation='h')
for year in data.date_year.unique():

    print(year)

    plotbox(data[data.date_year == year], x='gender',y='age', orientation='h')
## median death age by race each year

median_df_gender = data.groupby(['date_year', 'gender']).age.median().reset_index()

ax = sns.lineplot(x="date_year", y="age", hue='gender',estimator=None, lw=3,

                  data=median_df_gender)

ax.set_title('Median Age Death By gender')
len(data[data.gender == 'Transgender'])
data[data.gender == 'Transgender']
print('Transgender median age range')

print(median_df_gender[median_df_gender.gender == 'Transgender']['age'].max() - median_df_gender[median_df_gender.gender == 'Transgender']['age'].min())
print('Male median age range')

print(median_df_gender[median_df_gender.gender == 'Male']['age'].max() - median_df_gender[median_df_gender.gender == 'Male']['age'].min())
print('Female median age range')

print(median_df_gender[median_df_gender.gender == 'Female']['age'].max() - median_df_gender[median_df_gender.gender == 'Female']['age'].min())
plotbox(data, x='cause_of_death', y='age', orientation='h')
for year in data.date_year.unique():

    print(year)

    plotbox(data[data.date_year == year], x='cause_of_death',y='age', orientation='h')
my_colors = ['#6D7815', 

                    '#49392F',

                    '#4924A1', 

                    '#A1871F', 

                    '#9B6470',  

                    '#7D1F1A',  

                    '#9C531F', 

                    '#6D5E9C',  

                    '#493963', 

                    '#638D8D',  

                    '#6D6D4E', 

                    '#682A68', 

                    '#A13959', 

                    '#D1C17D',

                    '#445E9C',

                    '#44685E'

             ]



median_df = data.groupby(['date_year', 'cause_of_death']).age.median().reset_index()

ax = sns.lineplot(x="date_year", y="age", hue='cause_of_death',estimator=None, lw=3,

                  data=median_df, palette=my_colors )

ax.set_title('Median Age Death By Cause of death')
plotbox(data, x='intentional_use_of_force', y='age', orientation='h')
for year in data.date_year.unique():

    print(year)

    plotbox(data[data.date_year == year], x='intentional_use_of_force',y='age', orientation='h')
median_df = data.groupby(['date_year', 'intentional_use_of_force']).age.median().reset_index()

ax = sns.lineplot(x="date_year", y="age", hue='intentional_use_of_force',estimator=None, lw=3,

                  data=median_df)

ax.set_title('Median Age Death By intentional_use_of_force')
plotbox(data, x='symptoms_of_mental_illness', y='age', orientation='h')
for year in data.date_year.unique():

    print(year)

    plotbox(data[data.date_year == year], x='symptoms_of_mental_illness',y='age', orientation='h')
median_df = data.groupby(['date_year', 'symptoms_of_mental_illness']).age.median().reset_index()

ax = sns.lineplot(x="date_year", y="age", hue='symptoms_of_mental_illness',estimator=None, lw=3,

                  data=median_df)

ax.set_title('Median Age Death By symptoms_of_mental_illness')
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'agency_responsible_for_death', 5, s_elt=2, figsize=(10, 5))
top_5_agency_year = []

for year in data.date_year.unique():

    out_series = data[data['date_year'] == year]['agency_responsible_for_death'].value_counts()

    total_size = sum(out_series.tolist())

    out = dict(itertools.islice(out_series.items(), 5))

    top_5_agency_year.append(list(out.keys()))

top_agency_list = [agency for agency_year in top_5_agency_year for agency in agency_year]

top_agency_list = list(set(top_agency_list))

for agency in top_agency_list:

  print(agency)

  top_n_bar(data[data['agency_responsible_for_death'] == agency], 'date_year', 20, s_elt=2, figsize=(10, 5))
for race in data['race'].unique():

    print(race)

    if race == 'Middle Eastern':

        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, figsize=(12, 5))

    elif race in ['European-American/White', 'Native American/Alaskan']:

        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, s_elt=7, figsize=(12, 5))

    else:

        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, s_elt=10, figsize=(12, 5))

        

  

for gender in ['Male', 'Female','Transgender']:

    print(gender)

    if gender in ['Transgender']:

      top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, figsize=(10, 5))

    elif gender == 'Female': 

        top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, s_elt=10, figsize=(10, 5))

    else:

        top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, s_elt=30, figsize=(10, 5))
for cause_of_death in data['cause_of_death'].unique():

    print(cause_of_death)

    if cause_of_death in ['Vehicle']:

        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))

    elif cause_of_death in ['Gunshot']:

        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, s_elt=25, figsize=(10, 5))

    else:

        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, figsize=(10, 5))

        
for use_of_force in ['Vehicle/Pursuit', 'Intentional Use of Force, Deadly', 'Suicide',

       'Yes', 'No', 'Undetermined', 'Unknown']:

    print(use_of_force)

    if use_of_force in ['Undetermined', 'Unknown']:

        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, figsize=(10, 5))

    elif use_of_force == 'Intentional Use of Force, Deadly':

        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, s_elt=25, figsize=(10, 5))

    

    else:

        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))

  
for mental_illness in ['No', 'Drug or alcohol use', 'Unknown', 'Yes']:

    print(mental_illness)

    top_n_bar(data[data['symptoms_of_mental_illness'] == mental_illness], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))

        
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'gender', 5, s_elt=60, figsize=(10, 5))
plt.rcParams["figure.figsize"] = [20, 10]

cm = data.groupby(["date_year", "gender"])["gender"].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'race', 5, s_elt=25, figsize=(10, 5))
plt.rcParams["figure.figsize"] = [20, 10]

cm = data[data.race != 'Race unspecified'].groupby(["date_year", "race"])["race"].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'cause_of_death', 5, s_elt=25, figsize=(10, 5))
plt.rcParams["figure.figsize"] = [20, 10]

cm = data.groupby(["date_year", 'cause_of_death'])['cause_of_death'].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'symptoms_of_mental_illness', 5, s_elt=50, figsize=(10, 5))
plt.rcParams["figure.figsize"] = [20, 10]

cm = data.groupby(["date_year", 'symptoms_of_mental_illness'])['symptoms_of_mental_illness'].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
for year in data['date_year'].unique():

  print(year)

  top_n_bar(data[data['date_year'] == year], 'intentional_use_of_force', 5, s_elt=55, figsize=(10, 5))
plt.rcParams["figure.figsize"] = [20, 10]

cm = data.groupby(["date_year", 'intentional_use_of_force'])['intentional_use_of_force'].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
import re

import string

import numpy as np



from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk import word_tokenize

from collections import Counter

from wordcloud import WordCloud
import nltk

nltk.download('punkt')

nltk.download('stopwords')
def process_text(text):

    """

    process the text data

    """

    stopwords_english = stopwords.words('english')

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    text = text.lower()

    text_tokens = word_tokenize(text)



    texts_clean = []

    for word in text_tokens:

        if (word not in stopwords_english and  # remove stopwords

                word not in string.punctuation

              and len(word) > 2):  

            texts_clean.append(word)



    return texts_clean
corpus_df = data[['unique_id', 'description_circumstances_surrounding_death', 'date_and_description']]
corpus_df.date_and_description[0]
corpus_df.description_circumstances_surrounding_death[0]
corpus_df.date_and_description[1]
corpus_df.description_circumstances_surrounding_death[1]
corpus_text =corpus_df.description_circumstances_surrounding_death.to_list()
text_tokens_list = [process_text(token) for token in corpus_text]
text_tokens = [token for token_list in text_tokens_list for token in token_list]
frequency_text = Counter(text_tokens)
wc = WordCloud()

wc.generate_from_frequencies(frequencies=dict(frequency_text))

plt.imshow(wc) 
text_df = pd.Series(dict(frequency_text))

text_df = pd.DataFrame(list(dict(frequency_text).items()),columns=['words', 'count'], index=np.arange(len(frequency_text)))
fig, ax = plt.subplots(figsize=(10, 15))



# Plot horizontal bar graph

text_df.sort_values(by='count')[-50:].plot.barh(x='words',

                      y='count',

                      ax=ax,

                      color="green")



ax.set_title("Common words")





plt.show()
names_text = data.name.to_list()

names_tokens_list =  [process_text(token) for token in names_text]

names_tokens = [token for token_list in names_tokens_list for token in token_list]
frequency_names = Counter(names_tokens)
frequency_names.most_common(10)
len(frequency_names)
wc = WordCloud()

wc.generate_from_frequencies(frequencies=dict(frequency_names))

plt.imshow(wc) 
name_df = pd.Series(dict(frequency_names))

name_df = pd.DataFrame(list(dict(frequency_names).items()),columns=['name', 'count'], index=np.arange(len(frequency_names)))
fig, ax = plt.subplots(figsize=(10, 15))



# Plot horizontal bar graph

name_df.sort_values(by='count')[-50:].plot.barh(x='name',

                      y='count',

                      ax=ax,

                      color="green")



ax.set_title("Common names")





plt.show()
police_df = data[data.link_news_article_or_photo.str.contains('officer-killed', na=False)]

police_df.head()
police_df.link_news_article_or_photo.tolist()
plt.rcParams["figure.figsize"] = [20, 10]

cm = police_df.groupby(["date_year", "gender"])["gender"].count()

cm = cm.unstack(fill_value=0)

cm.plot.bar()
top_n_bar(police_df, 'gender', 4)
top_n_bar(police_df, 'race', 4)
import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import math
#Create the map

map_plot = folium.Map(location=[48, -102], tiles='cartodbpositron', zoom_start=3)



# Add points to the map

mc = MarkerCluster()

for idx, row in data.iterrows():

    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):

        mc.add_child(Marker([row['latitude'], row['longitude']]))

map_plot.add_child(mc)



# # Display the map

map_plot