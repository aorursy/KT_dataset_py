import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None) 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline

import math
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
survey_df = pd.read_csv('../input/survey_results_public.csv')
survey_schema = pd.read_csv('../input/survey_results_schema.csv', index_col='Column')
survey_df_india = survey_df.loc[survey_df['Country']=='India', :].copy(deep=True)
survey_df_germany = survey_df.loc[survey_df['Country']=='Germany', :].copy(deep=True)
survey_df_uk = survey_df.loc[survey_df['Country']=='United Kingdom', :].copy(deep=True)
survey_df_us = survey_df.loc[survey_df['Country']=='United States', :].copy(deep=True)
count = pd.DataFrame(survey_df['Country'].value_counts()[:10].copy(deep=True))
percentage = pd.DataFrame(survey_df['Country'].value_counts(normalize=True)[:10].copy(deep=True))

count.columns = ['Count']
percentage.columns = ['Percentage']
percentage['Percentage'] *= 100

top_responders_countries = pd.concat([count, percentage], axis=1)

top_responders_countries.columns.name = '#Responders'
top_responders_countries
from wordcloud import WordCloud

country = survey_df["Country"].value_counts()[:100].reset_index()
wrds = country["index"].str.replace(" ","")
wc = WordCloud(background_color='white', colormap=cm.viridis, scale=5).generate(" ".join(wrds))
plt.figure(figsize=(16,8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of countries based on the number of responders:", fontdict={'size':22, 'weight': 'bold'});
country_df_dict = {'World': survey_df, 'India': survey_df_india, 'US': survey_df_us,
                   'UK': survey_df_uk, 'Germany': survey_df_germany}
def what_is(name_list):
    '''
    Gives a description of each item present in `name_list` based on
    `survey_schema`
    :param name_list: A list of the feature names whose description is required
    
    Returns: A list containing one description string per item in `name_list`
    '''
    what_is_list = [name+': '+str(survey_schema.loc[name, 'QuestionText']) for name in name_list]
    return what_is_list
def response_overall(feature, normalize=True):
    '''
    Gives the overall response stats for `feature` in different countries.
    :param feature: String storing the column name whose overall description is
        required
    
    Returns: A pandas.DataFrame object with unique feature values as the columns
        and countries as the index
    '''
    df = pd.DataFrame(columns=survey_df[feature].value_counts().index)
    df.loc['World', :] = survey_df[feature].value_counts(normalize=normalize) * 100
    df.loc['India', :] = survey_df_india[feature].value_counts(normalize=normalize) * 100
    df.loc['US', :] = survey_df_us[feature].value_counts(normalize=normalize) * 100
    df.loc['UK', :] = survey_df_uk[feature].value_counts(normalize=normalize) * 100
    df.loc['Germany', :] = survey_df_germany[feature].value_counts(normalize=normalize) * 100
    return df
def get_trues(col):
    '''
    Helper function to store the frequency of True values for a 
    feature in its related col
    :param col: A Pandas DataFrame column generated upon calling
        `describe()` upon a boolean feature
    '''
    if col['top'] == False:
        col['top'] = True
        col['freq'] = col['count'] - col['freq']
def generate_expanded_features(feature, df):
    '''
    Helper function to generate a list of expanded feature names for 
    Multiple Options Correct type feature.
    
    :param feature: Parent feature name
    :param df: Pandas DataFrame object to which the parent feature belongs
    Returns: A list of generated feature names where each feature is of
        type -> parent+"_"+value
    '''
    values_set = set()
    values = [item for item in survey_df[feature].unique() if isinstance(item, str)]
    for entry in values:
        for item in entry.split(';'):
            values_set.add(item)
    return [feature+'_'+value for value in values_set]
def response_overall_moc(feature, normalize=True):
    '''
    Gives the overall response stats for a Multiple Options Correct type
    feature for different countries.
    :param feature: String storing the column name whose overall description is
        required
    
    Returns: A pandas.DataFrame object with unique feature values as the columns
        and countries as the index
    '''
    features_expanded = generate_expanded_features(feature=feature, df=survey_df)

    spread_features_all(moc_parent=feature, printable=False)

    features_overall_df = pd.DataFrame(columns=features_expanded)

    for country, country_df in country_df_dict.items():
        features_df = country_df[features_expanded].describe(include='all')
        features_df.apply(get_trues, axis=0)
        features_df.loc['percentage', :] = (features_df.loc['freq', :] / features_df.loc['count', :]) * 100
        features_df.rename(index={'top': 'true_count'}, inplace=True)
        for tool in features_expanded:
            if normalize:
                features_overall_df.loc[country, tool] = features_df.loc['percentage', tool]
            else:
                features_overall_df.loc[country, tool] = features_df.loc['freq', tool]

    columns = list(features_overall_df.columns)
    features_overall_df.columns = [cname.split('_')[1] for cname in columns]
    
    return features_overall_df
def update_countries_dict():
    global country_df_dict
    country_df_dict = {'World': survey_df, 'India': survey_df_india, 'US': survey_df_us,
                       'UK': survey_df_uk, 'Germany': survey_df_germany}
def spread_features(df, moc_parent, printable=True):
    '''
    Handles the Multiple Options Correct type features by spreading out
    each possible entry into a different column.
    :param df: Pandas.DataFrame object upon which we need to perform the
        operation
    :param moc_parent: The Multiple Options Correct type feature
    :param printable: Boolean; True to print the running info, False otherwise
    
    Returns: List of newly generated column names
    '''
    features_set = set()
    values = [item for item in survey_df[moc_parent].unique() if isinstance(item, str)]
    for entry in values:
        for item in entry.split(';'):
            features_set.add(item)

    if printable:
        print(features_set)

    for feature in features_set:
        df.loc[~df[moc_parent].isnull(), moc_parent+'_'+feature] = \
            df.loc[~df[moc_parent].isnull(), :] \
                     .apply(lambda row: feature in row[moc_parent], axis=1)

    if printable:
        for feature in features_set:
            print(df[moc_parent+'_'+feature].value_counts())
            
    return [moc_parent+'_'+feature for feature in features_set]
def spread_features_all(moc_parent, printable=True):
    '''
    Spread the passed feature in all the countries' DataFrames.
    :param moc_parent: The Multiple Options Correct type feature
    :param printable: Boolean; True to print the running info, False otherwise
    '''
    spread_features(survey_df, moc_parent, printable)
    global survey_df_india, survey_df_us, survey_df_germany, survey_df_uk
    survey_df_india = survey_df.loc[survey_df['Country']=='India', :].copy(deep=True)
    survey_df_germany = survey_df.loc[survey_df['Country']=='Germany', :].copy(deep=True)
    survey_df_uk = survey_df.loc[survey_df['Country']=='United Kingdom', :].copy(deep=True)
    survey_df_us = survey_df.loc[survey_df['Country']=='United States', :].copy(deep=True)
    update_countries_dict()
def plot_sequential(df, feature, order=None, colormap=cm.viridis, horizontal=False):
    '''
    Function to plot feature with sequential feature values.
    :param df: Pandas.DataFrame object containing values to be plotted
        [likely one returned from response_overall()]
    :param feature: The feature name that is being plotted
    :param order: The order in which we want to plot the feature values
    :param colormap: matplotlib.cm object that provides the colormap for plotting
    :param horizontal: Boolean specifying the orientation of the barplot
    
    Returns the plotted axis
    '''
    if order is None:
        order = list(df.columns)
    country_order = ['World', 'India', 'US', 'UK', 'Germany']
    title = what_is([feature])[0]
    if horizontal:
        ax = df.loc[country_order[::-1], order] \
                .plot.barh(figsize=(16, 8), stacked=True, colormap=colormap)
        ax.set_xlabel("Percentage", fontdict={'size':16});
        ax.set_xlim(0, 100)
        ax.set_ylabel("Responses", fontdict={'size':16});
        ax.set_title(title, fontdict={'weight': 'bold'});
        sns.despine()
        return ax
    else:
        ax = df.loc[country_order, order] \
                .plot.bar(figsize=(16, 8), stacked=True, colormap=colormap)
        ax.set_ylabel("Percentage", fontdict={'size':16});
        ax.set_xlabel("Responses", fontdict={'size':16});
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_title(title, fontdict={'weight': 'bold'});
        sns.despine(bottom=True);
        return ax
def highlighter(row):
    '''
    Function to be passed to `DataFrame.style.apply()`. 
    Highlights the rows of the DataFrame: row corresponding to 'World' data with blue 
    and the one corresponding to 'India' data with orange
    
    :param row: A pandas Series representing the row
    
    Returns: A list storing the background colors for each cell in the row
    '''
    if row.name == 'India':
        return ['background: orange' for i in row]
    elif row.name == 'World':
        return ['background: lightblue' for i in row]
    else:
        return ['' for i in row]
salary_nan_df = survey_df.loc[survey_df['ConvertedSalary'].isnull(), :]
percentage = round((salary_nan_df.shape[0] / survey_df.shape[0]) * 100, 
                   2)
print(str(percentage)+"% ("+str(salary_nan_df.shape[0])+") of responders have not filled in their salary.")
for col in ['ConvertedSalary', 'Salary']:
    survey_df.loc[(survey_df['ConvertedSalary'].isnull()) & \
                  (survey_df['Student'] == 'Yes, full-time'), col] = 0
    survey_df.loc[(survey_df['ConvertedSalary'].isnull()) & \
                  (survey_df['LastNewJob'] == "I've never had a job"), col] = 0
salary_nan_df = survey_df.loc[survey_df['ConvertedSalary'].isnull(), :]
percentage = round((salary_nan_df.shape[0] / survey_df.shape[0]) * 100, 
                   2)
print(str(percentage)+"% ("+str(salary_nan_df.shape[0])+") of responders have not filled in their salary.")
salary_percentile = pd.DataFrame(survey_df.loc[:, 'ConvertedSalary'].quantile(list(np.linspace(0.9, 0.99, 10))))
salary_percentile.index.name = "Percentiles"
salary_percentile.index = salary_percentile.index * 100
salary_percentile
MAX_LIM = survey_df.loc[:, 'ConvertedSalary'].quantile(0.95)

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(35, 20));

sns.kdeplot(survey_df_india.loc[survey_df_india['ConvertedSalary']<MAX_LIM, 'ConvertedSalary'], ax=axes[0], shade=True);
axes[0].set_title("India", fontdict={'weight': 'bold', 'size': 24});

sns.kdeplot(survey_df.loc[survey_df['ConvertedSalary']<MAX_LIM, 'ConvertedSalary'], ax=axes[1], shade=True);
axes[1].set_title("World", fontdict={'weight': 'bold', 'size': 24});

for ax in axes:
    ax.set_xlabel("Yearly Salary (converted to USD assuming 50 working weeks)", 
                  fontdict={'weight': 'bold', 'size': 24});
    ax.tick_params(axis='both', labelsize=20);
    ax.set_xlim(left=0, right=200000);

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(30, 10))

sns.kdeplot(survey_df_us.loc[survey_df_us['ConvertedSalary']<MAX_LIM, 'ConvertedSalary'], ax=axes[0], shade=True);
axes[0].set_title("United States", fontdict={'weight': 'bold', 'size': 16});

sns.kdeplot(survey_df_germany.loc[survey_df_germany['ConvertedSalary']<MAX_LIM, 'ConvertedSalary'], ax=axes[1], shade=True);
axes[1].set_title("Germany", fontdict={'weight': 'bold', 'size': 16});

sns.kdeplot(survey_df_uk.loc[survey_df_uk['ConvertedSalary']<MAX_LIM, 'ConvertedSalary'], ax=axes[2], shade=True);
axes[2].set_title("United Kingdom", fontdict={'weight': 'bold', 'size': 16});
for ax in axes:
    ax.set_xlabel("Yearly Salary (converted to USD assuming 50 working weeks)", 
                  fontdict={'weight': 'bold', 'size': 16});
    ax.tick_params(axis='both', labelsize=16);
    ax.set_ylim(0, 0.00005);
    ax.set_xlim(left=0, right=200000);
salary_describe = (survey_df.loc[survey_df['Country'].isin(['India', 'United States', 'Germany', 'United Kingdom'])]
                   .groupby('Country')['ConvertedSalary'].describe())
salary_describe.loc['World', :] = survey_df['ConvertedSalary'].describe()
salary_describe.columns.name = 'Salary description'
salary_describe.loc[['World', 'India', 'United States', 'United Kingdom', 'Germany'], 
                    ['count', 'mean', '25%', '50%', '75%']].style.apply(highlighter, axis=1)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8));
salary_describe['mean'].plot.bar(ax=ax);
ax.set_ylabel("Mean salary (converted to USD)", fontdict={'size':18});
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_xlabel("Country", fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=18);

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5), sharey=True)
for ax_id, percentile in zip([0, 1, 2], ['25%', '50%', '75%']):
    salary_describe[percentile].plot.bar(ax=axes[ax_id])
    axes[ax_id].set_title(percentile)
axes[0].set_ylabel("Salary (converted to USD)", fontdict={'size': 16});
for ax in axes:
    ax.tick_params(axis='both', labelsize=18);
unemployment_dict = {}

unemployment_df = survey_df.loc[(survey_df['JobSearchStatus'] == 'I am actively looking for a job'), :]
unemployment_dict['World'] = (unemployment_df.shape[0] / survey_df.shape[0]) * 100
unemployment_df_india = survey_df_india.loc[(survey_df_india['JobSearchStatus'] == 'I am actively looking for a job'), :]
unemployment_dict['India'] = (unemployment_df_india.shape[0] / survey_df_india.shape[0]) * 100
unemployment_df_us = survey_df_us.loc[(survey_df_us['JobSearchStatus'] == 'I am actively looking for a job'), :]
unemployment_dict['US'] = (unemployment_df_us.shape[0] / survey_df_us.shape[0]) * 100
unemployment_df_uk = survey_df_uk.loc[(survey_df_uk['JobSearchStatus'] == 'I am actively looking for a job'), :]
unemployment_dict['UK'] = (unemployment_df_uk.shape[0] / survey_df_uk.shape[0]) * 100
unemployment_df_germany = survey_df_germany.loc[(survey_df_germany['JobSearchStatus'] == 'I am actively looking for a job'), :]
unemployment_dict['Germany'] = (unemployment_df_germany.shape[0] / survey_df_germany.shape[0]) * 100
fig = pd.Series(unemployment_dict).plot.bar(figsize=(16, 8));
fig.set_ylabel("Percentage", fontdict={'size':16});
fig.set_title("Actively looking for a job", fontdict={'size':20});
fig.set_xticklabels(fig.get_xticklabels(), rotation=0);
fig.tick_params(axis='both', labelsize=18);
pd.Series(unemployment_dict)
student_df = response_overall('Student')
student_df.columns.name = 'Student?'
student_df.index.name = 'Country'
plot_sequential(df=student_df, feature='Student', order=['No', 'Yes, part-time', 'Yes, full-time']);
student_df.style.apply(highlighter, axis=1)
salary_describe = (survey_df.loc[survey_df['Country'].isin(['India', 'United States', 'Germany', 'United Kingdom']) & \
                                 survey_df['ConvertedSalary']>0]
                   .groupby('Country')['ConvertedSalary'].describe())
salary_describe.loc['World', :] = survey_df['ConvertedSalary'].describe()
salary_describe.columns.name = 'Salary description'
salary_describe.loc[['World', 'India', 'US', 'UK', 'Germany'], ['count', 'mean', '25%', '50%', '75%']].style.apply(highlighter, axis=1)
salary_describe.loc[:, '90%'] = \
    survey_df.loc[survey_df['Country'].isin(['India', 'United States', 'Germany', 'United Kingdom']) & \
                  survey_df['ConvertedSalary']>0] \
                  .groupby('Country')['ConvertedSalary'].quantile(0.9)
salary_describe.loc['World', '90%'] = survey_df['ConvertedSalary'].quantile(0.9)
salary_describe.loc[['World', 'India', 'United States', 'United Kingdom', 'Germany'], 
                    ['count', 'mean', '25%', '50%', '75%', '90%']].style.apply(highlighter, axis=1)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8));
salary_describe.loc[['World', 'India', 'United States', 'United Kingdom', 'Germany'], 'mean'].plot.bar(ax=ax);
ax.set_ylabel("Salary (converted to USD)", fontdict={'size':18});
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_xlabel("Country", fontdict={'size': 18});
ax.set_title("Mean", fontdict={'size': 20, 'weight': 'bold'});

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5), sharey=True)
for ax_id, percentile in zip([0, 1, 2], ['25%', '50%', '75%']):
    salary_describe.loc[['World', 'India', 'United States', 'United Kingdom', 'Germany'], percentile].plot.bar(ax=axes[ax_id])
    axes[ax_id].set_title(percentile, fontdict={'size': 18, 'weight': 'bold'});
    axes[ax_id].tick_params(axis='both', labelsize=18);
    axes[ax_id].set_xlabel("Country", fontdict={'size': 18})
axes[0].set_ylabel("Salary (converted to USD)", fontdict={'size': 16});
MAX_LIM = survey_df.loc[:, 'ConvertedSalary'].quantile(0.95)

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(35, 20));

sns.kdeplot(survey_df_india.loc[(survey_df_india['ConvertedSalary']<MAX_LIM) & \
                                (survey_df_india['ConvertedSalary']>0), 'ConvertedSalary'], 
            ax=axes[0], shade=True);
axes[0].set_title("India", fontdict={'weight': 'bold', 'size': 26});

sns.kdeplot(survey_df.loc[(survey_df['ConvertedSalary']<MAX_LIM) & \
                          (survey_df['ConvertedSalary']>0), 'ConvertedSalary'], 
            ax=axes[1], shade=True);
axes[1].set_title("World", fontdict={'weight': 'bold', 'size': 26});

for ax in axes:
    ax.set_xlabel("Yearly Salary (converted to USD assuming 50 working weeks)", 
                  fontdict={'weight': 'bold', 'size': 24});
    ax.tick_params(axis='both', labelsize=20);
    ax.set_xlim(left=0, right=200000);

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(30, 10))

sns.kdeplot(survey_df_us.loc[(survey_df_us['ConvertedSalary']<MAX_LIM) & \
                             (survey_df_us['ConvertedSalary']>0), 'ConvertedSalary'], 
            ax=axes[0], shade=True);
axes[0].set_title("United States", fontdict={'weight': 'bold', 'size': 18});

sns.kdeplot(survey_df_germany.loc[(survey_df_germany['ConvertedSalary']<MAX_LIM) & \
                                  (survey_df_germany['ConvertedSalary']>0), 'ConvertedSalary'], 
            ax=axes[1], shade=True);
axes[1].set_title("Germany", fontdict={'weight': 'bold', 'size': 18});

sns.kdeplot(survey_df_uk.loc[(survey_df_uk['ConvertedSalary']<MAX_LIM) & \
                             (survey_df_uk['ConvertedSalary']>0), 'ConvertedSalary'], 
            ax=axes[2], shade=True);
axes[2].set_title("United Kingdom", fontdict={'weight': 'bold', 'size': 18});

for ax in axes:
    ax.set_xlabel("Yearly Salary (converted to USD assuming 50 working weeks)", fontdict={'size': 16, 'weight': 'bold'});
    ax.tick_params(axis='both', labelsize=16);
    ax.set_ylim(0, 0.00006);
    ax.set_xlim(left=0, right=200000);
formal_education_df = response_overall('FormalEducation')
formal_education_df.columns.name = 'Highest level of formal education'
formal_education_df.index.name = 'Country'
formal_education_df.sort_values(by='India', axis=1, inplace=True)
ax = formal_education_df.T.plot.barh(figsize=(12, 20))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right');
ax.set_title(what_is(['FormalEducation'])[0].split(':')[1], fontdict={'size': 18, 'weight': 'bold'});
ax.set_xlim(0, 100);
ax.set_xlabel("Percentage", fontdict={'size': 20});
ax.set_ylabel("Highest level of formal education", fontdict={'size': 20});
ax.tick_params(axis='both', labelsize=18);
ax.legend(prop={'size': 16});
formal_education_df.loc[:, list(reversed(formal_education_df.columns))].style.apply(highlighter, axis=1)
undergrad_major_df = response_overall('UndergradMajor')
undergrad_major_df.columns.name = 'Undergrad major in college'
undergrad_major_df.index.name = 'Country'
undergrad_major_df.sort_values(by='India', axis=1, inplace=True)
ax = undergrad_major_df.T.plot.barh(figsize=(12, 20))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left');
ax.set_xlim(0, 100);
ax.set_ylabel("Undergrad Majors", fontdict={'size': 20});
ax.set_xlabel("Percentage", fontdict={'size': 20});
ax.set_title("Undergrad Majors in college", fontdict={'size': 18, 'weight': 'bold'});
ax.tick_params(axis='both', labelsize=16);
ax.legend(prop={'size': 16});
undergrad_major_df.loc[:, list(reversed(undergrad_major_df.columns))].style.apply(highlighter, axis=1)
what_is(['EducationTypes'])
education_types_df = response_overall_moc('EducationTypes')
education_types_df.columns.name = 'Non-degree education types'
education_types_df.index.name = 'Country'
education_types_df.sort_values(by='India', axis=1, inplace=True)
ax = education_types_df.T.plot.barh(figsize=(12, 20));
ax.set_title("Non-degree education", fontdict={'weight': 'bold'});
ax.set_xlim(0, 100);
ax.set_xlabel('Percentage', fontdict={'size': 18});
ax.set_ylabel('Non-degree education types', fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=18);
ax.legend(prop={'size': 16});
education_types_df.loc[:, list(reversed(education_types_df.columns))].style.apply(highlighter, axis=1)
ax = (undergrad_major_df.loc[:, ['Fine arts or performing arts (ex. graphic design, music, studio art)',
                                 'A health science (ex. nursing, pharmacy, radiology)',
                                 'A social science (ex. anthropology, psychology, political science)',
                                 'A humanities discipline (ex. literature, history, philosophy)',
                                 'A natural science (ex. biology, chemistry, physics)',
                                 'A business discipline (ex. accounting, finance, marketing)']]
     .T.plot.barh(figsize=(12, 16)));
ax.set_ylabel("Undergrad Majors", fontdict={'size': 18});
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_title("Percentage of people from less technical fields using StackOverflow", fontdict={'size': 18, 'weight': 'bold'});
ax.tick_params(axis='both', labelsize=14);
ax.legend(prop={'size': 16});
education_parents_df = response_overall("EducationParents")
education_parents_df.sort_values(by='India', axis=1, inplace=True)
ax = education_parents_df.T.plot.barh(figsize=(12, 20));
ax.set_title(what_is(['EducationParents'])[0], fontdict={'weight': 'bold'});
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=16);
ax.legend(prop={'size': 16});
education_parents_df.loc[:, list(reversed(education_parents_df.columns))].style.apply(highlighter, axis=1)
what_is(['Hobby'])
hobby_df = response_overall('Hobby')
hobby_df.columns.name = 'Code as a hobby?'
hobby_df.index.name = 'Country'
hobby_df.style.apply(highlighter, axis=1)
ax = hobby_df.plot.bar(figsize=(16, 8), color=['green', 'red'])
ax.set_title("Do you code as a hobby", fontdict={'weight': 'bold'});
ax.set_ylim(0, 100);
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xlabel("");
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.tick_params(axis='both', labelsize=16);
ax = response_overall('OpenSource').loc[:, ['Yes']].plot.bar(figsize=(16, 8), legend=False);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xlabel("Country", fontdict={'size': 18});
ax.set_title(what_is(['OpenSource'])[0], fontdict={'size': 16, 'weight': 'bold'});
what_is(['EducationTypes'])
ax = education_types_df.loc[:, ['Contributed to open source software']].plot.bar(figsize=(16, 8), legend=False);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xlabel("Country", fontdict={'size': 18});
ax.set_title("Percentage of people who marked contributing to open-source as non-degree education", 
             fontdict={'size': 16, 'weight': 'bold'});
os_df = pd.DataFrame(columns=education_types_df.index)
os_df.loc['Contribute to open-source', :] = response_overall('OpenSource').loc[:, 'Yes']
os_df.loc['Marked open-source contribution as non-degree education', :] = education_types_df.loc[:, 'Contributed to open source software']
ax = os_df.T.plot.bar(figsize=(16, 8));
ax.set_xlabel("");
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.tick_params(axis='both', labelsize=14);
ax.set_title("Contributing to open-source vs. learning from open-source contributions", fontdict={'weight': 'bold'});
age_df = response_overall('Age')
age_df.columns.name = 'Age'
age_df.index.name = 'Country'
plot_sequential(df=age_df, feature='Age',
                order=['Under 18 years old', '18 - 24 years old', '25 - 34 years old', '35 - 44 years old',
                       '45 - 54 years old', '55 - 64 years old', '65 years or older'],
                colormap=cm.inferno_r);
age_df.loc[:, ['Under 18 years old', '18 - 24 years old', '25 - 34 years old', '35 - 44 years old',
                       '45 - 54 years old', '55 - 64 years old', '65 years or older']].style.apply(highlighter, axis=1)
years_coding_df = response_overall('YearsCoding')
years_coding_df.columns.name = 'Years Coding'
years_coding_df.index.name = 'Country'
plot_sequential(df=years_coding_df, feature='YearsCoding',
                order=['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years',
                       '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years'],
                colormap=cm.inferno_r);
years_coding_df.loc[:, ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years',
                       '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']].style.apply(highlighter, axis=1)
years_coding_prof_df = response_overall('YearsCodingProf')
years_coding_prof_df.columns.name = 'Years Coding Professionally'
years_coding_prof_df.index.name = 'Country'
plot_sequential(df=years_coding_prof_df, feature='YearsCodingProf',
                order=['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years',
                       '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years'],
                colormap=cm.inferno_r);
years_coding_prof_df.loc[:, ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years',
                       '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']].style.apply(highlighter, axis=1)
competitive_df = response_overall('AgreeDisagree2')
competitive_df.columns.name = 'Competing with peers'
competitive_df.index.name = 'Country'
plot_sequential(df=competitive_df, feature='AgreeDisagree2', 
                order=['Strongly agree', 'Agree', 'Neither Agree nor Disagree', 'Disagree', 'Strongly disagree'],
                colormap=cm.viridis_r);
competitive_df[['Strongly agree', 'Agree', 'Neither Agree nor Disagree', 'Disagree', 'Strongly disagree']].style.apply(highlighter, axis=1)
hope_five_years_df = response_overall('HopeFiveYears')
hope_five_years_df.columns.name = 'Hope 5 years from now'
hope_five_years_df.index.name = 'Country'
order_hopes = ["Working in a different or more specialized technical role than the one I'm in now",
               "Doing the same work",
               "Working as a product manager or project manager",
               "Working as an engineering manager or other functional manager",
               "Retirement",
               "Working as a founder or co-founder of my own company",
               "Working in a career completely unrelated to software development"]
ax = hope_five_years_df.loc[:, order_hopes].plot.bar(figsize=(16, 8), 
                                                     stacked=True, 
                                                     legend=True,
                                                     colormap=cm.Set2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_title(what_is(['HopeFiveYears'])[0], fontdict={'weight': 'bold'});
hope_five_years_df.style.apply(highlighter, axis=1)
dev_type_count_df = response_overall_moc('DevType')
dev_type_count_df.columns.name = 'Developer Type'
dev_type_count_df.index.name = 'Country'
dev_type_count_df.sort_values(by='India', axis=1, inplace=True)
ax = dev_type_count_df.T.plot.barh(figsize=(16, 24));
ax.set_title("Developer Type", fontdict={'size': 20, 'weight': 'bold'});
ax.set_xlabel("Percentage", fontdict={'size': 20});
ax.set_ylabel("Developer Types", fontdict={'size': 20});
ax.tick_params(axis='both', labelsize=16);
ax.legend(prop={'size': 18});
dev_type_count_df.loc[:, list(reversed(dev_type_count_df.columns))].style.apply(highlighter, axis=1)
mob_dev_count = (pd.DataFrame(survey_df.loc[survey_df['DevType_Mobile developer'] == True, :]
                 .groupby('Country')['DevType_Mobile developer'].count())
                 .unstack()
                 .reset_index().loc[:, ['Country', 0]]
                 .sort_values(axis=0, by=0, ascending=False))
            
mob_dev_count.columns = ['Country', 'Count']
mob_dev_count.reset_index(drop=True, inplace=True)
import squarify
import random

plt.figure(figsize=(18, 9))
cmap = cm.get_cmap(name='Oranges')
color = [cmap(random.random()) for i in range(50)]

ax = squarify.plot(label=mob_dev_count.iloc[:50, :]['Country'], 
                   sizes=mob_dev_count.iloc[:50, :]['Count'], 
                   value=mob_dev_count.iloc[:50, :]['Count'], 
                   color=color);
ax.set_title("Top 50 countries with the max number of mobile developers", fontdict={'weight': 'bold'});
ax.set_axis_off()
fig, (ax_age, ax_exp) = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

survey_df_india.loc[survey_df_india['DevType'] == 'Mobile developer', 'Age']\
.value_counts(normalize=True)\
.plot.pie(ax=ax_age, legend=True, labels=None, colormap=cm.Paired, autopct='%1.1f%%');
ax_age.set_title("Distribution of Indian mobile developers according to age", 
                 fontdict={'size': 18, 'weight': 'bold'});
#ax_age.legend(prop={'size': 16});

survey_df_india.loc[survey_df_india['DevType'] == 'Mobile developer', 'YearsCoding']\
.value_counts(normalize=True)\
.plot.pie(ax=ax_exp, legend=True, labels=None, colormap=cm.Accent, autopct='%1.1f%%');
ax_exp.set_title("Distribution of Indian mobile developers according to coding experience", 
                 fontdict={'size': 18, 'weight': 'bold'});
#ax_exp.legend(prop={'size': 16});
ide_df = response_overall_moc('IDE')
ide_df.columns.name = 'IDE'
ide_df.index.name = 'Country'
ide_df.sort_values(by='India', axis=1, inplace=True)
ax = ide_df.T.plot.barh(figsize=(16, 24))
ax.set_xlim(0, 100);
ax.set_title(what_is(['IDE'])[0], fontdict={'size': 14, 'weight': 'bold'})
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("IDE", fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=14);
ide_df.loc[:, list(reversed(ide_df.columns))].style.apply(highlighter, axis=1)
platform_worked_with_df = response_overall_moc('PlatformWorkedWith')
platform_worked_with_df.columns.name = 'Platforms Worked With'
platform_worked_with_df.index.name = 'Country'
platform_desire_df = response_overall_moc('PlatformDesireNextYear')
platform_desire_df.columns.name = 'Platforms desire next year'
platform_desire_df.index.name = 'Country'
def get_prefs(worked_with_df, desire_df, country):
    preferences = pd.DataFrame(columns=desire_df.columns)
    preferences.loc['Worked', :] = worked_with_df.loc[country, :]
    preferences.loc['Desire', :] = desire_df.loc[country, :]
    preferences.loc['Interest', :] = preferences.loc['Desire', :] - \
                                                     preferences.loc['Worked', :]
    preferences.sort_values(by='Worked', axis=1, ascending=False, inplace=True)
    
    return preferences
def plot_preferences(preferences_df, ax, title, ylim, tick_labelsize, xlabel):
    preferences_df.loc[['Worked', 'Desire'], :].T.plot.line(ax=ax, style='*-')
    ax.set_xticks(range(len(list(preferences_df.columns))));
    ax.set_xticklabels(list(preferences_df.columns), rotation=90);
    ax.set_ylim(0, ylim);
    ax.set_title(title, fontdict={'size': 24, 'weight': 'bold'});
    ax.set_ylabel("Percentage of responders", fontdict={'size': 22});
    ax.set_xlabel(xlabel, fontdict={'size': 22});
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    ax.legend(prop={'size': 18});
def show_preferences(worked_with_df, desire_df, xlabel, ylim=100, tick_labelsizes=(14, 12)):
    preferences_df_dict = {}
    preferences_df_dict['India'] = get_prefs(worked_with_df, desire_df, 'India')
    preferences_df_dict['World'] = get_prefs(worked_with_df, desire_df, 'World')
    preferences_df_dict['US'] = get_prefs(worked_with_df, desire_df, 'US')
    preferences_df_dict['UK'] = get_prefs(worked_with_df, desire_df, 'UK')
    preferences_df_dict['Germany'] = get_prefs(worked_with_df, desire_df, 'Germany')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    plot_preferences(preferences_df_dict['India'], axes[0], title='India',
                     ylim=ylim, tick_labelsize=tick_labelsizes[0], xlabel=xlabel)
    plot_preferences(preferences_df_dict['World'], axes[1], title='World',
                     ylim=ylim, tick_labelsize=tick_labelsizes[0], xlabel=xlabel)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 8))
    plot_preferences(preferences_df_dict['US'], axes[0], title='US',
                     ylim=ylim, tick_labelsize=tick_labelsizes[1], xlabel=xlabel)
    plot_preferences(preferences_df_dict['UK'], axes[1], title='UK',
                     ylim=ylim, tick_labelsize=tick_labelsizes[1], xlabel=xlabel)
    plot_preferences(preferences_df_dict['Germany'], axes[2], title='Germany',
                     ylim=ylim, tick_labelsize=tick_labelsizes[1], xlabel=xlabel)
show_preferences(worked_with_df=platform_worked_with_df, desire_df=platform_desire_df, 
                 xlabel='Platforms', ylim=60, tick_labelsizes=(16, 14))
platform_worked_with_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
platform_desire_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
platform_preferences_df = get_prefs(platform_worked_with_df, platform_desire_df, 'India').sort_values(by='Interest', axis=1, ascending=False)
platform_preferences_df.columns.name = 'Platforms: Indian devs'
platform_preferences_df
languages_worked_with_df = response_overall_moc('LanguageWorkedWith')
languages_worked_with_df.columns.name = 'Languages Worked With'
languages_worked_with_df.index.name = 'Country'
languages_desire_df = response_overall_moc('LanguageDesireNextYear')
languages_desire_df.columns.name = 'Languages Desire Next Year'
languages_desire_df.index.name = 'Country'
show_preferences(worked_with_df=languages_worked_with_df, desire_df=languages_desire_df, 
                 xlabel='Languages', ylim=100, tick_labelsizes=(16, 14))
languages_worked_with_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
languages_desire_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
language_preferences_df = get_prefs(languages_worked_with_df, languages_desire_df, 'India').sort_values(by='Interest', axis=1, ascending=False)
language_preferences_df.columns.name = 'Languages: Indian devs'
language_preferences_df
framework_worked_with_df = response_overall_moc('FrameworkWorkedWith')
framework_worked_with_df.columns.name = 'Framework Worked With'
framework_worked_with_df.index.name = 'Country'
framework_desire_df = response_overall_moc('FrameworkDesireNextYear')
framework_desire_df.columns.name = 'Framework Desire Next Year'
framework_desire_df.index.name = 'Country'
show_preferences(worked_with_df=framework_worked_with_df, desire_df=framework_desire_df, 
                 xlabel='Frameworks', ylim=60, tick_labelsizes=(16, 14))
framework_worked_with_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
framework_desire_df.sort_values(by='India', axis=1, ascending=False).style.apply(highlighter, axis=1)
framework_preferences_df = get_prefs(framework_worked_with_df, framework_desire_df, 'India').sort_values(by='Interest', axis=1, ascending=False)
framework_preferences_df.columns.name = 'Frameworks: Indian devs'
framework_preferences_df
comm_tools = response_overall_moc('CommunicationTools')
comm_tools.columns.name = 'Communication Tools'
comm_tools.index.name = 'Country'
comm_tools.sort_values(by='India', axis=1, inplace=True)
ax = comm_tools.T.plot.barh(figsize=(14, 24))
ax.set_xlim(0, 100);
ax.set_title(what_is(['CommunicationTools'])[0].split(':')[0], fontdict={'size': 14, 'weight': 'bold'})
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Communication tools", fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=18);
ax.legend(prop={'size': 16});
comm_tools.loc[:, list(reversed(comm_tools.columns))].style.apply(highlighter, axis=1)
methodology_df = response_overall_moc('Methodology')
methodology_df.columns.name = 'Programming methodology'
methodology_df.index.name = 'Country'
methodology_df.sort_values(by='India', axis=1, inplace=True)
ax = methodology_df.T.plot.barh(figsize=(16, 24))
ax.set_xlim(0, 100);
ax.set_title(what_is(['Methodology'])[0], fontdict={'size': 14, 'weight': 'bold'})
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Programming methodology", fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=18);
ax.legend(prop={'size': 16});
methodology_df.loc[:, list(reversed(methodology_df.columns))].style.apply(highlighter, axis=1)
ethics_responsible_df = response_overall('EthicsResponsible')
ethics_responsible_df.columns.name = 'Responsibility of unethical code'
ethics_responsible_df.index.name = 'Country'
ax = ethics_responsible_df.T.plot.bar(figsize=(16, 8))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
ax.set_ylim(0, 100);
ax.set_title(what_is(['EthicsResponsible'])[0], fontdict={'size': 16, 'weight': 'bold'});
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xlabel("");
ax.legend(prop={'size': 16});
ax.tick_params(axis='both', labelsize=13);
ethics_responsible_df.style.apply(highlighter, axis=1)
ethics_choice_df = response_overall('EthicsChoice')
ethics_choice_df.columns.name = 'Write Unethical Code?'
ethics_choice_df.index.name = 'Country'
ax = plot_sequential(ethics_choice_df, feature='EthicsChoice', colormap=cm.magma_r);
ax.tick_params(axis='both', labelsize=18);
ethics_choice_df.style.apply(highlighter, axis=1)
ethical_implications_df = response_overall('EthicalImplications')
ethical_implications_df.columns.name = 'Consider ethical implications of your code?'
ethical_implications_df.index.name = 'Country'
ax = plot_sequential(ethical_implications_df, 'EthicalImplications', colormap=cm.magma_r);
ax.tick_params(axis='both', labelsize=18);
ethical_implications_df.style.apply(highlighter, axis=1)
hackathon_reasons_df = response_overall_moc('HackathonReasons')
hackathon_reasons_df.columns.name = 'Hackathon Reasons'
hackathon_reasons_df.index.name = 'Country'
hackathon_reasons_df.style.apply(highlighter, axis=1)
what_is(['AIDangerous', 'AIInteresting'])
list(survey_df['AIDangerous'].value_counts().index)
dangerous_ai_df = response_overall('AIDangerous')
interesting_ai_df = response_overall('AIInteresting')
automation = pd.DataFrame(columns=['Interesting', 'Dangerous'], index=list(dangerous_ai_df.index))
automation.loc[:, 'Interesting'] = interesting_ai_df \
    .loc[:, 'Increasing automation of jobs']
automation.loc[:, 'Dangerous'] = dangerous_ai_df \
    .loc[:, 'Increasing automation of jobs']

automation.columns.name = 'Automation of Jobs'
automation.index.name = 'Country'
ax = automation.plot.bar(figsize=(16, 6))
ax.set_title('Increasing automation of jobs', fontdict={'weight': 'bold'});
ax.set_ylabel('Percentage', fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=12);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
automation.style.apply(highlighter, axis=1)
imp_decisions = pd.DataFrame(columns=['Interesting', 'Dangerous'], index=list(dangerous_ai_df.index))
imp_decisions.loc[:, 'Interesting'] = interesting_ai_df \
    .loc[:, 'Algorithms making important decisions']
imp_decisions.loc[:, 'Dangerous'] = dangerous_ai_df \
    .loc[:, 'Algorithms making important decisions']

imp_decisions.columns.name = 'Algorithms making important decisions'
imp_decisions.index.name = 'Country'
ax = imp_decisions.plot.bar(figsize=(16, 6))
ax.set_title('Algorithms making important decisions', fontdict={'weight': 'bold'});
ax.set_ylabel('Percentage', fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=12);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
imp_decisions.style.apply(highlighter, axis=1)
singularity = pd.DataFrame(columns=['Interesting', 'Dangerous'], index=list(dangerous_ai_df.index))
singularity.loc[:, 'Interesting'] = interesting_ai_df \
    .loc[:, 'Artificial intelligence surpassing human intelligence ("the singularity")']
singularity.loc[:, 'Dangerous'] = dangerous_ai_df \
    .loc[:, 'Artificial intelligence surpassing human intelligence ("the singularity")']

singularity.columns.name = 'Singularity'
singularity.index.name = 'Country'
ax = singularity.plot.bar(figsize=(16, 6))
ax.set_title('Artificial intelligence surpassing human intelligence ("the singularity")', fontdict={'weight': 'bold'});
ax.set_ylabel('Percentage', fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=12);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
singularity.style.apply(highlighter, axis=1)
fairness = pd.DataFrame(columns=['Interesting', 'Dangerous'], index=list(dangerous_ai_df.index))
fairness.loc[:, 'Interesting'] = interesting_ai_df \
    .loc[:, 'Evolving definitions of "fairness" in algorithmic versus human decisions']
fairness.loc[:, 'Dangerous'] = dangerous_ai_df \
    .loc[:, 'Evolving definitions of "fairness" in algorithmic versus human decisions']

fairness.columns.name = 'Evolving definitions of fairness'
fairness.index.name = 'Country'
ax = fairness.plot.bar(figsize=(16, 6))
ax.set_title('Evolving definitions of "fairness" in algorithmic versus human decisions', fontdict={'weight': 'bold'});
ax.set_ylabel('Percentage', fontdict={'size': 18});
ax.tick_params(axis='both', labelsize=12);
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
fairness.style.apply(highlighter, axis=1)
responsibility_ai = response_overall("AIResponsible")
responsibility_ai.columns.name = 'Responsibility for AI'
responsibility_ai.index.name = 'Country'

ax = responsibility_ai.plot.bar(figsize=(16, 6), stacked=True, colormap=cm.Set3);
ax.set_title(what_is(['AIResponsible'])[0], fontdict={'weight': 'bold'});
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
responsibility_ai.style.apply(highlighter, axis=1)
peer_mentoring_sys = response_overall("HypotheticalTools1")
peer_mentoring_sys.columns.name = 'Peer mentoring system'
peer_mentoring_sys.index.name = 'Country'
peer_mentoring_sys.style.apply(highlighter, axis=1)
ax = plot_sequential(df=peer_mentoring_sys, feature='HypotheticalTools1', 
                    order=['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                           'Extremely interested'],
                    colormap=cm.magma, horizontal=True)
ax.tick_params(axis='both', labelsize=16);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Responses", fontdict={'size': 18});
ax.legend(prop={'size': 14});
ax.set_title("Interest in a peer mentoring system", fontdict={'size': 18, 'weight': 'bold'});
newbie_private_area = response_overall("HypotheticalTools2")
newbie_private_area.columns.name = 'Newbie private area'
newbie_private_area.index.name = 'Country'
newbie_private_area.style.apply(highlighter, axis=1)
ax = plot_sequential(df=newbie_private_area, feature='HypotheticalTools2', 
                order=['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                       'Extremely interested'],
                colormap=cm.magma, horizontal=True)
ax.tick_params(axis='both', labelsize=16);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Responses", fontdict={'size': 18});
ax.legend(prop={'size': 14});
ax.set_title("Interest in a private area for people new to programming", fontdict={'size': 18, 'weight': 'bold'});
prog_blog = response_overall("HypotheticalTools3")
prog_blog.columns.name = 'Programming blog platform'
prog_blog.index.name = 'Country'
prog_blog.style.apply(highlighter, axis=1)
ax = plot_sequential(df=prog_blog, feature='HypotheticalTools3', 
                    order=['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                           'Extremely interested'],
                    colormap=cm.magma, horizontal=True)
ax.tick_params(axis='both', labelsize=16);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Responses", fontdict={'size': 18});
ax.legend(prop={'size': 14});
ax.set_title("Interest in a programming oriented blog platform", fontdict={'size': 18, 'weight': 'bold'});
job_review = response_overall("HypotheticalTools4")
job_review.columns.name = 'Job review system'
job_review.index.name = 'Country'
job_review.style.apply(highlighter, axis=1)
ax = plot_sequential(df=job_review, feature='HypotheticalTools4', 
                order=['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                       'Extremely interested'],
                colormap=cm.magma, horizontal=True)
ax.tick_params(axis='both', labelsize=16);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Responses", fontdict={'size': 18});
ax.legend(prop={'size': 14});
ax.set_title("Interest in an employer or job review system", fontdict={'size': 18, 'weight': 'bold'});
career_growth_qa = response_overall("HypotheticalTools5")
career_growth_qa.columns.name = 'Career growth Q&A'
career_growth_qa.index.name = 'Country'
career_growth_qa.style.apply(highlighter, axis=1)
ax = plot_sequential(df=career_growth_qa, feature='HypotheticalTools5', 
                order=['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                       'Extremely interested'],
                colormap=cm.magma, horizontal=True)
ax.tick_params(axis='both', labelsize=16);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_ylabel("Responses", fontdict={'size': 18});
ax.legend(prop={'size': 14});
ax.set_title("Interest in an area for Q&A related to career growth", fontdict={'size': 18, 'weight': 'bold'});
def get_tools_vs_years_coding(tool):
    df = survey_df.groupby('YearsCoding')[tool].value_counts(normalize=True).unstack()
    df.columns = df.columns.get_level_values(0)
    df *= 100
    return df.loc[['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years',
                           '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years'], 
                  ['Not at all interested', 'A little bit interested', 'Somewhat interested', 'Very interested',
                           'Extremely interested']]
hypo1_df = get_tools_vs_years_coding('HypotheticalTools1')
hypo2_df = get_tools_vs_years_coding('HypotheticalTools2')
hypo3_df = get_tools_vs_years_coding('HypotheticalTools3')
hypo4_df = get_tools_vs_years_coding('HypotheticalTools4')
hypo5_df = get_tools_vs_years_coding('HypotheticalTools5')
titles = [desc.split('.')[-1] for desc in what_is(["HypotheticalTools"+str(i) for i in range(1, 6)])]

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 6))
sns.heatmap(hypo1_df, ax=axes[0], annot=True, vmin=0, vmax=35, cmap=cm.viridis)
sns.heatmap(hypo2_df, ax=axes[1], annot=True, vmin=0, vmax=35, cmap=cm.viridis)
axes[0].set_title(titles[0], fontdict={'weight': 'bold'});
axes[1].set_title(titles[1], fontdict={'weight': 'bold'});
for ax in axes:
    ax.tick_params(axis='both', labelsize=16);
    ax.set_ylabel("YearsCoding", fontdict={'size': 18});
    ax.set_xlabel("", fontdict={'size': 18});

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20, 6))
sns.heatmap(hypo3_df, ax=axes[0], annot=True, vmin=0, vmax=35, cmap=cm.viridis)
sns.heatmap(hypo4_df, ax=axes[1], annot=True, vmin=0, vmax=35, cmap=cm.viridis)
sns.heatmap(hypo5_df, ax=axes[2], annot=True, vmin=0, vmax=35, cmap=cm.viridis)
axes[0].set_title(titles[2], fontdict={'weight': 'bold'});
axes[1].set_title(titles[3], fontdict={'weight': 'bold'});
axes[2].set_title(titles[4], fontdict={'weight': 'bold'});
for ax in axes:
    ax.tick_params(axis='both', labelsize=14);
    ax.set_ylabel("YearsCoding", fontdict={'size': 16});
    ax.set_xlabel("", fontdict={'size': 18});
ad_blocker_df = response_overall('AdBlocker')
ax = ad_blocker_df.T.plot.barh(figsize=(16, 6));
ax.set_xlim(0, 100);
ax.set_xlabel("Percentage", fontdict={'size': 18});
ax.set_title(what_is(['AdBlocker'])[0], fontdict={'weight': 'bold'});
ax.tick_params(axis='both', labelsize=16);
ax.legend(prop={'size': 16});
ax.set_ylabel("");
ad_blocker_df.columns.name = "Use Ad Blocker?"
ad_blocker_df.index.name = 'Country'
ad_blocker_df.style.apply(highlighter, axis=1)
os = response_overall('OperatingSystem')
os.sort_values(by='India', axis=1, inplace=True)
os.columns.name = 'Operating System'
os.index.name = 'Country'
ax = os.T.plot.barh(figsize=(16, 8));
ax.set_title("Operating System", fontdict={'weight': 'bold'});
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right');
os.style.apply(highlighter, axis=1)
so_visit_df = response_overall("StackOverflowVisit")
so_visit_df.columns.name = 'StackOverflow visits frequency'
so_visit_df.index.name = 'Country'
plot_sequential(so_visit_df, 'StackOverflowVisit',
                order=['Multiple times per day', 'Daily or almost daily', 'A few times per week',
                       'A few times per month or weekly', 'Less than once per month or monthly',
                       'I have never visited Stack Overflow (before today)'],
                colormap=cm.plasma_r)
so_visit_df.loc[:, ['Multiple times per day', 'Daily or almost daily', 'A few times per week',
                    'A few times per month or weekly', 'Less than once per month or monthly',
                    'I have never visited Stack Overflow (before today)']].style.apply(highlighter, axis=1)
exercise_df = response_overall("Exercise")
exercise_df.columns.name = '#Exercise / week'
exercise_df.index.name = 'Country'
plot_sequential(exercise_df, 'Exercise', colormap=cm.Oranges,
                order=['Daily or almost every day', '3 - 4 times per week', '1 - 2 times per week',
                       "I don't typically exercise"])
exercise_df.loc[:, ['Daily or almost every day', '3 - 4 times per week', '1 - 2 times per week',
                    "I don't typically exercise"]].style.apply(highlighter, axis=1)
sexual_orientation_df = response_overall_moc('SexualOrientation')
sexual_orientation_df.columns.name = 'Sexual Orientation'
sexual_orientation_df.index.name = 'Country'
ax = sexual_orientation_df.T.plot.bar(figsize=(16, 8));
ax.set_ylim(0, 100);
ax.set_ylabel("Percentage", fontdict={'size': 18});
ax.set_title("Sexual Orientation", fontdict={'weight': 'bold'});
sexual_orientation_df.style.apply(highlighter, axis=1)