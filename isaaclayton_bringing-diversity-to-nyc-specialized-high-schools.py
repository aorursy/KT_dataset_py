# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 
import re #regular expressions
import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns #plotting
sns.set()
import scipy
import math
import json
#import branca
from typing import List, Tuple, Dict, Union
from textwrap import wrap
import ipywidgets as widgets
from IPython.display import Markdown
from statsmodels.sandbox.stats.multicomp import multipletests

pd.set_option('display.max_columns', None)

HSDir_2017 = pd.read_csv('../input/nyc-high-school-directory/2017-doe-high-school-directory.csv')
#Creating a dictionary of specialized high schools and their district borough numbers, a helpful ID variable
specialized_dict = {dbn:school for dbn, school in HSDir_2017.query('specialized==1')[['dbn', 'school_name']].values}

for school in specialized_dict.values():
    print(school)
#Removing Fiorello from the specialized high school dictionary
specialized_dict.pop('03M485', None)
#Import the datasets
demographics_df = pd.read_csv('../input/2013-2018-demographic-snapshot-district/2013_-_2018_Demographic_Snapshot_School.csv')
demographics_df.head()
ethnicity_list = ['ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 'OTHER']
gender_list = ['FEMALE', 'MALE']
disability_list = ['STUDENTS WITH DISABILITIES', 'STUDENTS WITH NO DISABILITIES']
ELL_list = ['ENGLISH LANGUAGE LEARNERS', 'NOT ENGLISH LANGUAGE LEARNERS']
poverty_list = ['POVERTY', 'NO POVERTY']
demographic_dict = {'Ethnicity': ethnicity_list, 'Gender': gender_list,
                    'Disabilities': disability_list, 'Poverty': poverty_list,
                    'English Language Learners': ELL_list}
#PREPROCESSING

#Making the Year column an integer
demographics_df['Year'] = demographics_df['Year'].str.slice(0,4).astype('int64')

#Adding a column for Borough
demographics_df['borough'] = demographics_df['DBN'].str.slice(2,3)
demographics_df.loc[demographics_df['borough']=='X','borough'] = 'Bronx'
demographics_df.loc[demographics_df['borough']=='K','borough'] = 'Brooklyn'
demographics_df.loc[demographics_df['borough']=='Q','borough'] = 'Queens'
demographics_df.loc[demographics_df['borough']=='M','borough'] = 'Manhattan'
demographics_df.loc[demographics_df['borough']=='R','borough'] = 'Staten Island'

#Changing 'No Data' results in the Economic Need Index to be np.NaN
demographics_df.loc[demographics_df['Economic Need Index']=='No Data', 'Economic Need Index'] = np.NaN

#Changing percentage columns to float type
for column in [column for column in demographics_df.columns if '%' in column] + ['Economic Need Index']:
    demographics_df.loc[-demographics_df[column].isnull(), column] = \
    demographics_df.loc[-demographics_df[column].isnull(), column].str.slice(0,-1).astype('float64')/100
    
#Making all column names in dataset capitalized
demographics_df.columns = demographics_df.columns.str.upper()

#Rename "MULTIPLE RACE CATEGORIES NOT REPRESENTED" to "OTHER" to keep succinct
demographics_df.rename(columns = {'# MULTIPLE RACE CATEGORIES NOT REPRESENTED': '# OTHER',
                                        '% MULTIPLE RACE CATEGORIES NOT REPRESENTED': '% OTHER',
                                        'SCHOOL NAME': 'SCHOOL_NAME'}, inplace=True)

#Adding columns for demographic inverses
demographics_df['# NO POVERTY'] = demographics_df['TOTAL ENROLLMENT'] - demographics_df['# POVERTY']
demographics_df['% NO POVERTY'] = 1 - demographics_df['% POVERTY']
    
demographics_df['# NOT ENGLISH LANGUAGE LEARNERS'] = demographics_df['TOTAL ENROLLMENT'] - demographics_df['# ENGLISH LANGUAGE LEARNERS']  
demographics_df['% NOT ENGLISH LANGUAGE LEARNERS'] = 1 - demographics_df['% ENGLISH LANGUAGE LEARNERS']

demographics_df['# STUDENTS WITH NO DISABILITIES'] = demographics_df['TOTAL ENROLLMENT'] - demographics_df['# STUDENTS WITH DISABILITIES']  
demographics_df['% STUDENTS WITH NO DISABILITIES'] = 1 - demographics_df['% STUDENTS WITH DISABILITIES']  

#Updating specialized school names in the datasets
for DBN, school_name in specialized_dict.items():
    demographics_df.loc[demographics_df['DBN']==DBN, 'SCHOOL_NAME'] = school_name
    
#Year range to work with
years = (2013,2017)

#Only specialized schools
specialized_df = demographics_df.query('DBN in @specialized_dict.keys()').copy()
specialized_df.set_index(['SCHOOL_NAME', 'YEAR'], inplace=True)
def percentage_table(date_range: Tuple[int, int], demographic_list: List[str], demographic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a view of each schools' demographic percentages for 2 years
    
    Args:
        date_range (Tuple[int,int]): The year range of demographic data to include.
        demographic_list (List[str]): A list of demographic categories.
        demographic_df (pd.DataFrame): A pandas dataframe of demographic values.

    Returns:
        pd.DataFrame: demographic dataframe filtered view.
    """
    df = demographic_df.query('YEAR in [2013,2017]')[[' '.join(['%', demographic]) for demographic in demographic_list]]

    overall_df = demographic_df.query('YEAR in [2013,2017]')[[' '.join(['#', demographic]) for demographic in demographic_list]]\
                               .groupby(level=1).agg(sum)\
                               .apply(lambda x: x/x.sum(), axis=1)\
                               .set_index(pd.MultiIndex.from_product([['OVERALL'], [2013,2017]]))\
                               .sort_index(axis=1)
                
    return df.append(overall_df.rename(columns = dict(zip(overall_df.columns, df.columns))))


def highlight_rows(x):
    colors = []
    for school in x.index.get_level_values(0):
        if x[school,x.index.unique(level='YEAR')[1]] > x[school,x.index.unique(level='YEAR')[0]]:
            colors.append('background-color: lightgreen')
        elif x[school,x.index.unique(level='YEAR')[1]] < x[school,x.index.unique(level='YEAR')[0]]:
            colors.append('background-color: #FDB5A6')
        else:
            colors.append('')
    return colors


outputs = [widgets.Output() for _ in demographic_dict.keys()]
tab = widgets.Tab(children = outputs)
for i, demographic in enumerate(demographic_dict.keys()):
    tab.set_title(i, demographic)
display(tab)    

for i, demographic in enumerate(demographic_dict.keys()):
    with outputs[i]:
        display(Markdown('### {}'.format(demographic)));
        display(percentage_table(years, demographic_dict[demographic], specialized_df).style.apply(highlight_rows, axis=0));
def log(n: float) -> float:
    """
    Allows math.log to return 0 for log(0) instead of undefined for the purpose of calculating a Shannon Index.
    """
    if n==0:
        return 0
    else:
        return math.log(n)
    
    
def shannon_index(categories: List[int]) -> Tuple[float, float, int]: 
    """
    Calculates Shannon Index numbers for a demographic list.
    
    Args:
        categories (List[int]): A list of demographic categories.

    Returns:
        Tuple[float, float, int]: (Shannon Index expected value, Shannon Index variance, total number of observations)
    """
    N = sum(categories)
    expected_value = -sum((x/N)*log(x/N) for x in categories)
    variance = ((sum((x/N)*(log(x/N)**2) for x in categories) - ((sum((x/N)*log(x/N) for x in categories))**2))/N) + ((len(categories)-1)/(2*(N**2)))
    return (expected_value, variance, N)


def shannon_list(date_range: Tuple[int, int], demographic_list: List[str], demographic_df: pd.DataFrame) -> List[Union[float, str]]:
    """
    Creates a list of the yearly overall Shannon Index information for the specialized high schools.
    
    Args:
        date_range (Tuple[int,int]): The year range of demographic data to include.
        demographic_list (List[str]): A list of demographic categories.
        demographic_df (pd.DataFrame): A pandas dataframe of demographic values.

    Returns:
        List[int, float, float, int]: A list of school year, Shannon expected value, Shannon variance, 
                                      and total number of observations as the value.
    """
    shannon = []
    for year in range(date_range[0], date_range[1]+1):
        shannon.append([year, *shannon_index(list(demographic_df.xs(year, level='YEAR')[[' '.join(['#', demographic]) for demographic in demographic_list]].sum().values))])
    return shannon


def hill_graphs(date_range: Tuple[int, int], demographic_list: List[str], demographic_name: str, demographic_df: pd.DataFrame) -> None:
    """
    Creates a set of bar charts and lineplots visualizing the change in diversity for a date range
    
    Args:
        date_range (Tuple[int,int]): The year range of demographic data to include.
        demographic_list (List[str]): A list of demographic categories.
        demographic_name (str): The name of the demographic to be included in graph text.
        demographic_df (pd.DataFrame): A pandas dataframe of demographic values.

    Returns:
        None: Multiple graphs.
    """
    shannon = shannon_list(date_range, demographic_list, demographic_df)
    
    fig = plt.figure(figsize=(20, 8*(len(specialized_dict)+1)))
    grid = plt.GridSpec(len(specialized_dict)+1, 3, hspace=0.4, wspace=0.4)

    barcharts = fig.add_subplot(grid[i,:2])
    barcharts.set_ylabel('Percentage')
    barcharts.set_ylim(0,1)
    barcharts.set_title(demographic_name + ' Distribution')
    
    demographic_df.query('YEAR in @date_range')[[' '.join(['#', demographic]) for demographic in demographic_list]]\
                   .groupby(level=1).sum().apply(lambda x: x/x.sum(), axis=1)\
                   .sort_index().T\
                   .plot(kind='bar', ax=barcharts, rot=0)


    lineplots = fig.add_subplot(grid[i,2:])
    lineplots.set_ylabel('Effective ' + demographic_name + ' Number')
    lineplots.set_xlabel('Year')
    lineplots.set_ylim(1, len(demographic_list))
    lineplots.set_title('Hill Numbers Over Time')



    lineplots.plot(*zip(*[[x[0], math.exp(x[1])] for x in shannon]), color="red")

    
    
    plt.show(fig);

    
for i, demographic in enumerate(demographic_dict.keys()):
    display(Markdown('### {}'.format(demographic)))
    hill_graphs(years, demographic_dict[demographic], demographic, specialized_df);
def t_test(a: float, var_a: float, N_a: int, b: float, var_b: float, N_b: int) -> Tuple[float, int, float]:
    """
    Calculates the Hutcheson t-test statistics to compare two Shannon Indices
    
    Args:
        a (float): Shannon Index #1
        var_a (float): Shannon Index #1 variance
        N_a (float): total observations #1
        b (float): Shannon Index #2
        var_b (float): Shannon Index #2 variance
        N_b: total observations #2

    Returns:
        Tuple[float, int, float]: A tuple of (t-statistic, degrees of freedom, p-value)
    """
    t = (a - b)/math.sqrt(var_a + var_b)
    df = math.ceil(((var_a + var_b)**2)/(((var_a**2)/N_a)+((var_b**2)/N_b)))
    return (t, df, 1 - scipy.stats.t.cdf(math.fabs(t), df))

def shannon_t_tests(date_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Creates table displaying multiple t-tests adjusted with the Benjamini-Hochberg procedure
    
    Args:
        date_range (Tuple[int,int]): The year range of demographic data to include.
        
    Returns:
        None: A pandas dataframe.
    """
    significance_df = pd.DataFrame(data=None, columns = ['Name', "Shannon Index " + str(date_range[0]), "Shannon Index " + str(date_range[1]), 't', 'df', 'pval'])
    for i, demographic in enumerate(demographic_dict.keys()):
        shannon = shannon_list(date_range, demographic_dict[demographic], specialized_df)
        significance_df.loc[i] = (demographic, shannon[0][1], shannon[-1][1], *t_test(*shannon[0][1:],*shannon[-1][1:]))
    is_reject, corrected_pvals, _, _ = multipletests(significance_df["pval"], alpha=0.05, method='fdr_bh')
    significance_df["reject"] = is_reject
    significance_df["adj_pval"] = corrected_pvals
    return significance_df
shannon_t_tests(years)
school_explorer = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
registration = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')

#Processing for school_explorer dataset

#Capitalize column names
school_explorer.columns = school_explorer.columns.str.upper()

#Change percent columns from strings to usable integers
for column in [column for column in school_explorer.columns if ('%' in column) or ('PERCENT' in column) or ('RATE' in column)]:
    school_explorer.loc[-school_explorer[column].isnull(), column] = school_explorer.loc[-school_explorer[column].isnull(), column].str.strip('%').astype('float64')/100
    school_explorer[column] = school_explorer[column].astype('float64')
    
#Change dollars to floats
school_explorer['SCHOOL INCOME ESTIMATE'] = school_explorer['SCHOOL INCOME ESTIMATE'].str.replace('[\$, ]', '').astype('float64')

#Change location code column name to DBN
school_explorer.rename(columns = {'LOCATION CODE': 'DBN'}, inplace=True)

school_explorer.head()
#testing 2015/2016 vs. 2016/2017
testing = pd.merge(school_explorer[['PERCENT BLACK', 'DBN']], demographics_df.query('YEAR in (2015,2016)')[['% BLACK', 'DBN', 'YEAR']], on='DBN')
for year in (2015,2016):
    print('Year {} Error: {}'.format(year, (testing.loc[testing['YEAR']==year, 'PERCENT BLACK'] - testing.loc[testing['YEAR']==year, '% BLACK']).sum()))
del testing
redundant_columns = ['SCHOOL NAME', 'ECONOMIC NEED INDEX']
redundant_columns.extend(['PERCENT ' + demographic for demographic in ['ELL', 'ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 'BLACK / HISPANIC']])

school_explorer = pd.merge(school_explorer[[column for column in school_explorer.columns if column not in redundant_columns]], 
         demographics_df.query('YEAR==2015'), on='DBN')
school_explorer['PERCENT BLACK/HISPANIC'] = school_explorer['% BLACK'] + school_explorer['% HISPANIC']
test_results = pd.read_csv('../input/nyc-ela-and-math-results-20152016/3-8_ELA_AND_MATH_RESEARCHER_FILE_2016.csv')

#Dropping nan values
test_results.replace('-', np.nan, inplace=True)
test_results.dropna(subset=[column for column in test_results.columns if ('_COUNT' in column) or ('_PCT' in column)], inplace=True)

#Converting percent columns into float64 types
percentage_columns = [column for column in test_results.columns if 'PCT' in column]
for column in percentage_columns:
    test_results[column] = test_results[column].str.strip('%').astype('float64')/100
    
#Create grade column
test_results['GRADE'] = test_results['ITEM_DESC'].str.slice(6,7).astype('int64')
    
#Get test averages
test_results['SUBGROUP_AVERAGE'] = test_results[percentage_columns[:-2]].apply(lambda x: sum(percentage * score for percentage, score in zip(x, range(1,5))), axis=1)
subgroup_avgs = test_results.groupby(['BEDSCODE', 'GRADE', 'SUBGROUP_NAME']).sum()['SUBGROUP_AVERAGE'].reset_index()
subgroup_avgs.rename(columns = {'SUBGROUP_AVERAGE': 'SUBGROUP_AVERAGE_TOTAL'}, inplace=True)
test_results = test_results.merge(subgroup_avgs, on=['BEDSCODE', 'GRADE', 'SUBGROUP_NAME'])
del subgroup_avgs
test_results['TOTAL_TESTED'] = test_results['TOTAL_TESTED'].astype('int64')
avg_df = test_results.groupby(['BEDSCODE', 'ITEM_SUBJECT_AREA']).apply(lambda x: sum(count * score for count, score in zip(x['TOTAL_TESTED'], x['SUBGROUP_AVERAGE']))/x['TOTAL_TESTED'].sum()).to_frame()
test_results = test_results.join(avg_df, on=['BEDSCODE', 'ITEM_SUBJECT_AREA'])
test_results = test_results.merge(test_results.groupby(['BEDSCODE', 'ITEM_SUBJECT_AREA']).first().unstack()[0].reset_index().copy(),
                   on='BEDSCODE')
test_results.rename(columns = {'ELA': 'ELA_SCHOOL_AVERAGE', 'Mathematics': 'MATH_SCHOOL_AVERAGE'}, inplace=True)
del avg_df

#Drop unnecessary columns
test_results.drop([0, 'SY_END_DATE', 'ITEM_DESC', 'MEAN_SCALE_SCORE'], axis=1, inplace=True)

#Merge
school_explorer = pd.merge(school_explorer[[column for column in school_explorer.columns if 'GRADE' not in column and 'AVERAGE' not in column]], 
                           test_results, left_on='SED CODE', right_on='BEDSCODE')

del test_results

#Just 8th grade results are wanted as they are the ones taking the SHSAT
school_explorer = school_explorer.query('GRADE==8')
from scipy import stats

fig = plt.figure(figsize=(10,10))
NYSTP_results = school_explorer.query('SUBGROUP_NAME=="All Students"').set_index(['BEDSCODE', 'ITEM_SUBJECT_AREA']).unstack()['SUBGROUP_AVERAGE'].dropna()
plt.scatter(NYSTP_results['ELA'],NYSTP_results['Mathematics'])
plt.title('NYSTP Results for the 2015/2016 School Year')
plt.xlabel('English Language Arts results')
plt.ylabel('Mathematics results');
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

def elbow_method(df: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """
    Creates graph to find optimal k-value for k-means clustering and returns a standardized dataframe.
         
    Args:
        df (pd.DataFrame): Dataframe to perform elbow method and standardization on
        kmax (int): Number of k values to try

    Returns:
        pd.DataFrame: A standardized version of df
    """
    df_norm = stats.zscore(df)
    k_values = []
    k_range = range(1,kmax)
    for k in k_range:
        kmeanModel = KMeans(n_clusters=k).fit(df_norm)
        kmeanModel.fit(df_norm)
        k_values.append(sum(np.min(cdist(df_norm, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_norm.shape[0])

    plt.plot(k_range, k_values, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squares')
    plt.title('Elbow Method for finding an optimal k')
    plt.show()
    return df_norm
NYSTP_results_norm = elbow_method(NYSTP_results, 10)
def cluster(df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Returns k-means labels.
         
    Args:
        df (pd.DataFrame): Dataframe to cluster
        n (int): Number of clusters

    Returns:
        pd.DataFrame: A standardized version of df
    """
    kmeans = KMeans(n_clusters=n, random_state=0)
    fit = kmeans.fit(df)
    labels = kmeans.predict(df)
    return labels

labels = cluster(NYSTP_results_norm, 3)
colmap = {1: '#a6cee3', 2: '#1f78b4', 3: '#b2df8a', 4: '#33a02c', 5: '#fb9a99',
          6: '#e31a1c', 7: '#fdbf6f', 8: '#ff7f00', 9: '#cab2d6', 10: '#6a3d9a'}
colors = list(map(lambda x: colmap[x+1], labels))

fig = plt.figure(figsize=(10, 10))
plt.scatter(NYSTP_results['ELA'],NYSTP_results['Mathematics'], color=colors, alpha=0.5, edgecolor='k')
plt.title('NYSTP Results for the 2015/2016 School Year Clustered')
plt.xlabel('English Language Arts results')
plt.ylabel('Mathematics results');
NYSTP_results['TEST_CLUSTER'] = labels
school_explorer = school_explorer.merge(NYSTP_results['TEST_CLUSTER'].reset_index(), on='BEDSCODE')
school_explorer['HIGH_PERFORMER'] = (school_explorer['TEST_CLUSTER'] == 2)
offers = pd.read_csv('../input/2015-2016-shsat-results/2015-2016_SHSAT_Admissions_Test_Offers_By_Sending_School.csv')

#Rename columns
offers.rename(columns = {'Feeder School DBN': 'DBN', 'Feeder School Name': 'SCHOOL_NAME',
                   'Count of Students in HS Admissions': 'OLD_COUNT',
                   'Count of Testers': 'TESTED', 'Count of Offers': 'OFFERED'}, inplace=True)
offers.set_index('DBN', inplace=True)

#Add actual student enrollments
eligible_testers = demographics_df.query('YEAR==2015')[['GRADE 8', 'GRADE 9', 'DBN']].set_index('DBN').apply(lambda x: pd.Series({'ELIGIBLE': x.sum()}), axis=1)
offers = pd.merge(offers, eligible_testers,
                left_index=True, right_index=True)
offers.drop('OLD_COUNT', axis=1, inplace=True)

#Make columns float64 type
for column in ['TESTED', 'OFFERED']:
    offers.loc[offers[column]=='0-5',column] = np.nan
    offers[column] = offers[column].astype('float64')

#Calculate relevant percentages
offers['% TESTED'] = (offers['TESTED']/offers['ELIGIBLE']).fillna(0)
offers['% SUCCESSFUL'] = offers['OFFERED']/offers['TESTED']

#Merge offers into school_explorer dataset
school_explorer = pd.merge(school_explorer.query('GRADE==8'), offers.drop('SCHOOL_NAME', axis=1), on='DBN')
tested_df = school_explorer.groupby('BEDSCODE').first()[['% TESTED', 'PERCENT BLACK/HISPANIC']].dropna()
fig = plt.figure(figsize=(10, 10))
plt.scatter(tested_df['% TESTED'],tested_df['PERCENT BLACK/HISPANIC'])
plt.title('Scatterplot Comparing SHSAT Test Participation and Proportion of Black/Hispanic Students')
plt.xlabel('Percent Tested')
plt.ylabel('Proportion of Black and Hispanic Students');
tested_df_norm = elbow_method(tested_df, 10)
labels = cluster(tested_df_norm, 5)

colors = list(map(lambda x: colmap[x+1], labels))

fig = plt.figure(figsize=(10, 10))
plt.scatter(tested_df['% TESTED'],tested_df['PERCENT BLACK/HISPANIC'], color=colors, alpha=0.5, edgecolor='k');
plt.title('Scatterplot Comparing SHSAT Test Participation and Proportion of Black/Hispanic Students \n with K-means Clusters')
plt.xlabel('Percent Tested')
plt.ylabel('Proportion of Black and Hispanic Students');
tested_df['DIVERSITY_CLUSTER'] = labels
school_explorer = school_explorer.merge(tested_df['DIVERSITY_CLUSTER'].reset_index(), on='BEDSCODE')
school_explorer['UNDERTESTED'] = (school_explorer['DIVERSITY_CLUSTER'] == 0)
display(school_explorer.query('HIGH_PERFORMER & UNDERTESTED').groupby('BEDSCODE').first()[['SCHOOL_NAME', 'SUBGROUP_AVERAGE_TOTAL', '% TESTED']])
school_explorer.query('HIGH_PERFORMER & UNDERTESTED').groupby('BEDSCODE').first().shape[0]
m = folium.Map(
    location=[40.777488, -73.879681],
    tiles='Stamen Toner',
    zoom_start=11,)

for index,row in school_explorer.query('HIGH_PERFORMER & UNDERTESTED').groupby('BEDSCODE').first().iterrows():
    folium.Marker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        popup=folium.Popup(row['SCHOOL_NAME'], parse_html=True)
    ).add_to(m)
m